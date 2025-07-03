use std::{
    borrow::Cow,
    cmp::Reverse,
    collections::BinaryHeap,
    fmt::{self, Debug},
    marker::PhantomData,
};

use heed::{RoTxn, RwTxn};
use min_max_heap::MinMaxHeap;
use nohash::{BuildNoHashHasher, NoHashHasher};
use papaya::HashMap;
use rand::{distributions::WeightedIndex, prelude::Distribution, Rng};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use roaring::RoaringBitmap;
use smallvec::{smallvec, SmallVec};

use crate::{
    key::Key,
    node::{Item, Links, Node},
    ordered_float::OrderedFloat,
    parallel::{ImmutableItems, ImmutableLinks},
    writer::{BuildOption, FrozzenReader},
    Database, Distance, Error, ItemId, Result,
};

// TODO:
// - add dedicated 0th layer with M0 and fix corresponding code

type ScoredLink = (OrderedFloat, ItemId);

/// State with stack-allocated graph edges
struct NodeState<const M: usize> {
    links: SmallVec<[ScoredLink; M]>,
}
impl<const M: usize> Debug for NodeState<M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // from [crate::unaligned_vector]
        struct Number(f32);
        impl fmt::Debug for Number {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{:0.3}", self.0)
            }
        }
        let mut list = f.debug_list();

        for &(OrderedFloat(dist), id) in &self.links {
            let tup = (id, Number(dist));
            list.entry(&tup);
        }

        list.finish()
    }
}

/// A struct to fetch nodes and links from lmdb
struct LmdbReader<'a, D> {
    database: &'a Database<D>,
    rtxn: &'a RoTxn<'a>,
    //lru
}
impl<'a, D: Distance> LmdbReader<'a, D> {
    pub fn get_item(&self, item_id: ItemId) -> Result<Item<'a, D>> {
        let key = Key::item(0, item_id);

        // key is a `Key::item` so returned result must be a Node::Item
        Ok(self.database.get(self.rtxn, &key)?.ok_or(Error::missing_key(key))?.item().unwrap())
    }

    pub fn get_links(&self, item_id: ItemId, level: usize) -> Result<Links<'a>> {
        let key = Key::links(0, item_id, level as u8);

        // key is a `Key::links` so returned result must be a Node::Links
        Ok(self.database.get(self.rtxn, &key)?.ok_or(Error::missing_key(key))?.links().unwrap())
    }
}

pub struct HnswBuilder<D, const M: usize, const M0: usize> {
    assign_probas: Vec<f32>,
    ef_construction: usize,
    max_level: usize,
    entrypoints: Vec<ItemId>,
    pub layers: Vec<HashMap<ItemId, NodeState<M0>>>,
    distance: PhantomData<D>,
}

impl<D: Distance, const M: usize, const M0: usize> HnswBuilder<D, M, M0> {
    pub fn new(opts: &BuildOption) -> Self {
        let assign_probas = Self::get_default_probas();

        Self {
            assign_probas,
            ef_construction: opts.ef_construction,
            max_level: 0,
            entrypoints: vec![],
            layers: vec![],
            distance: PhantomData,
        }
    }

    /// build quantiles from an x ~ exp(1/ln(m))
    fn get_default_probas() -> Vec<f32> {
        let mut assign_probas = Vec::with_capacity(M);
        // NOTE: breaks when M=1 ...
        let level_factor = 1.0 / (M as f32).ln();
        let mut level = 0;
        loop {
            // P(L<x<L+1) = P(x<L+1) - P(x<L)
            // = 1-exp(-λ(L+1)) - (1-exp(-λL)) = exp(-λL)*(1-exp(-λ))
            let proba = ((level as f32) * (-1.0 / level_factor)).exp()
                * (1.0 - (-1.0 / level_factor).exp());
            if proba < 1e-09 {
                break;
            }
            assign_probas.push(proba);
            level += 1;
        }
        assign_probas
    }

    // can probably even be u8's ...
    fn get_random_level<R>(&mut self, rng: &mut R) -> usize
    where
        R: Rng + ?Sized,
    {
        let dist = WeightedIndex::new(&self.assign_probas).unwrap();
        dist.sample(rng) as usize
    }

    pub fn build<R>(
        &mut self,
        to_insert: RoaringBitmap,
        database: Database<D>,
        wtxn: &mut RwTxn,
        rng: &mut R,
    ) -> Result<()>
    where
        R: Rng + ?Sized,
    {
        // generate a random level for each point
        let mut levels: Vec<_> = to_insert
            .iter()
            .map(|item_id| {
                let level = self.get_random_level(rng);
                self.max_level = self.max_level.max(level);
                (item_id, level)
            })
            .collect();

        levels.sort_unstable_by(|(_, a), (_, b)| b.cmp(a));
        for _ in 0..=self.max_level{
            self.layers.push(HashMap::new());
        }

        // pointers to vector & links in the lmdb
        let items = ImmutableItems::new(wtxn, database, &to_insert, 0)?;
        let nb_links = database.len(wtxn)? + to_insert.len();
        let links = ImmutableLinks::new(wtxn, database, 0, nb_links)?;
        let lmdb = FrozzenReader { items: &items, links: &links };

        for (item_id, level) in levels.iter().take_while(|(_, l)| *l == self.max_level) {
            self.entrypoints.push(*item_id);
        }

        levels.into_par_iter().for_each(|(id, lvl)| {
            self.insert(id, lvl, &lmdb);
        });

        // write single threaded to lmdb
        for lvl in 0..=self.max_level {
            for (item_id, node_state) in &self.layers[lvl].pin() {
                let key = Key::links(0, *item_id, lvl as u8);
                let links = Links {
                    links: Cow::Owned(RoaringBitmap::from_iter(
                        node_state.links.iter().map(|(_, i)| *i),
                    )),
                };
                let node_edges = database.put(wtxn, &key, &Node::Links(links))?;
            }
        }

        Ok(())
    }

    fn entry_point(&self) -> ItemId {
        return self.entrypoints[0];
    }

    fn insert<'a>(&self, query: ItemId, level: usize, lmdb: &FrozzenReader<'a, D>) -> Result<()> {
        let mut eps = vec![self.entry_point()];

        let q = lmdb.get_item(query)?;

        // Greedy search with: ef = 1
        for lvl in (level + 1..=self.max_level).rev() {
            let neighbours = self.explore_layer(&q, &eps, lvl, 1, lmdb)?;
            let closest = neighbours.peek_min().map(|(_, n)| *n).expect("No neighbor was found");
            eps = vec![closest];
        }

        // Beam search with: ef = ef_construction
        for lvl in (0..=level.min(self.max_level)).rev() {
            self.create_node(query, lvl);

            let mut neighbours = self.explore_layer(&q, &eps, lvl, self.ef_construction, lmdb)?;
            eps.clear();

            // FIXME: limit neighbors with algo 4
            let mut m_nearest_iter = neighbours.drain_asc().into_iter().take(M0);
            while let Some((dist, n)) = m_nearest_iter.next() {
                // add links in both directions
                self.add_link(query, (dist, n), lvl);
                self.add_link(n, (dist, query), lvl);
                eps.push(n);
            }
        }

        Ok(())
    }

    fn create_node(&self, item_id: ItemId, level: usize) {
        self.layers[level].pin().insert(item_id, NodeState { links: smallvec![] });
    }

    /// Returns only the Id's of our neighbours.
    fn get_neighbours<'a>(
        &self,
        lmdb: &FrozzenReader<'a, D>,
        item_id: ItemId,
        level: usize,
    ) -> Result<Vec<ItemId>> {
        // FIXME: do some if-let chaining
        // also should this be a Result<Option<Vec<_>> not Result<Vec<_>> ?
        if level >= self.layers.len() {
            let Links { links } = lmdb.get_links(item_id, level)?;
            return Ok(links.iter().collect());
        }

        // O(1)
        match self.layers[level].pin().get(&item_id) {
            Some(node_state) => return Ok(node_state.links.iter().map(|(_, i)| *i).collect()),

            // O(log n)
            None => {
                let Links { links } = lmdb.get_links(item_id, level)?;
                Ok(links.iter().collect())
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn explore_layer<'a>(
        &self,
        query: &Item<D>,
        eps: &[ItemId],
        level: usize,
        ef: usize,
        lmdb: &FrozzenReader<'a, D>,
    ) -> Result<MinMaxHeap<ScoredLink>> {
        let mut candidates = BinaryHeap::new();
        let mut res = MinMaxHeap::with_capacity(ef);
        let mut visited = RoaringBitmap::new();

        // Register all entry points as visited and populate candidates
        for &ep in eps {
            let ve = lmdb.get_item(ep)?;
            let dist = D::distance(query, &ve);

            candidates.push((Reverse(OrderedFloat(dist)), ep));
            res.push((OrderedFloat(dist), ep));
            visited.push(ep);
        }

        while let Some((Reverse(OrderedFloat(f)), c)) = candidates.pop() {
            // stopping criteria
            if let Some((OrderedFloat(f_max), _)) = res.peek_max() {
                if f > *f_max {
                    break;
                }
            }

            // Get neighborhood of candidate either from self or LMDB
            let proximity = self.get_neighbours(lmdb, c, level)?;

            // can we par_iter distance computations ?
            for point in proximity {
                if !visited.insert(point) {
                    continue;
                }
                let dist = D::distance(query, &lmdb.get_item(point)?);
                candidates.push((Reverse(OrderedFloat(dist)), point));

                // optimized insert & removal maintaining original len
                if res.len() == ef {
                    let _ = res.push_pop_max((OrderedFloat(dist), point));
                } else {
                    let _ = res.push((OrderedFloat(dist), point));
                }
            }
        }

        Ok(res)
    }

    /// Tries to add link between two elements, returns a bool indicating if the link was created
    /// or not.
    fn add_link(&self, p: ItemId, q: ScoredLink, level: usize) {
        // prevent links to self
        if p == q.1 {
            return;
        }

        let map = self.layers[level].pin();

        // 'pure' links update function
        let _add_link = |node_state: &NodeState<M0>| {
            let mut links = node_state.links.clone();
            let cap = if level == 0 { M0 } else { M };

            if links.len() < cap {
                links.push(q);
                return NodeState { links };
            }

            // This is safe cause we would have returned already above if links.len() == 0
            if q.0 > links.last().unwrap().0 {
                return NodeState { links };
            }

            // pop first to avoid moving smallvec to heap
            let _ = links.pop();

            match links.binary_search(&q) {
                Ok(index) => links.insert(index, q),
                Err(index) => links.insert(index, q),
            }

            NodeState { links }
        };

        map.update_or_insert_with(p, _add_link, || NodeState { links: smallvec![] });
    }

    fn search<'a>(
        &self,
        query: &Item<D>,
        k: usize,
        ef: usize,
        lmdb: &FrozzenReader<'a, D>,
    ) -> Result<Vec<ScoredLink>> {
        let mut eps = vec![self.entry_point()];

        // search layers L->1 with ef=1
        for lvl in (1..=self.max_level).rev() {
            let neighbours = self.explore_layer(&query, &eps, lvl, 1, &lmdb)?;
            let closest = neighbours.peek_min().map(|(_, n)| n).expect("No neighbor was found");
            eps = vec![*closest];
        }

        // search layer 0 with ef=ef
        let mut neighbours = self.explore_layer(&query, &eps, 0, ef, &lmdb)?;

        let mut nns = Vec::with_capacity(k);
        while let Some(sp) = neighbours.pop_min() {
            nns.push(sp);
            if nns.len() == k {
                break;
            }
        }

        Ok(nns)
    }
}

#[cfg(test)]
mod tests {
    use super::HnswBuilder;
    use crate::{
        distance::Cosine,
        key::Key,
        node::{Item, Node},
        ordered_float::OrderedFloat,
        writer::BuildOption,
        Database,
    };
    use heed::EnvOpenOptions;
    use rand::{rngs::StdRng, thread_rng, Rng, SeedableRng};
    use roaring::RoaringBitmap;
    use std::{collections::HashMap, time::Instant};

    #[ignore = "just cause"]
    #[test]
    // should be like: https://www.pinecone.io/learn/series/faiss/hnsw/
    fn check_distribution_shape() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut hnsw = HnswBuilder::<Cosine, 32, 48>::new(&BuildOption::default());

        let mut bins = HashMap::new();
        (0..10000).into_iter().for_each(|_| {
            let level = hnsw.get_random_level(&mut rng);
            *bins.entry(level).or_insert(0) += 1;
        });

        dbg!("{:?}", bins);
    }

    #[test]
    fn test_build() {
        let env = unsafe {
            EnvOpenOptions::new()
                .map_size(1024 * 1024 * 1024 * 2) // 2GiB
                .open("./")
        }
        .unwrap();

        let mut wtxn = env.write_txn().unwrap();
        let db: Database<Cosine> = env.create_database(&mut wtxn, None).unwrap();

        // insert a few vectors
        // let mut rng = StdRng::seed_from_u64(42);
        let mut rng = thread_rng();
        let mut opts = BuildOption::default();
        opts.ef_construction = 400;
        let mut hnsw: HnswBuilder<Cosine, 8, 16> = HnswBuilder::new(&opts);

        let vecs: Vec<Vec<f32>> =
            (0..50000).map(|_| (0..784).map(|_| rng.gen()).collect()).collect();
        let backup_vecs = vecs.clone();
        // dbg!("{:?}", &vecs);

        let mut to_insert = RoaringBitmap::new();
        for (item_id, vec) in vecs.into_iter().enumerate() {
            let item = Item::new(vec);
            db.put(&mut wtxn, &Key::item(0, item_id as u32), &Node::Item(item)).unwrap();

            // update build bitmap
            to_insert.insert(item_id as u32);
        }

        let now = Instant::now();
        hnsw.build(to_insert, db, &mut wtxn, &mut rng);
        wtxn.commit().unwrap();
        println!("build; {:?}", now.elapsed());

        // for (i, l) in hnsw.layers.iter().enumerate() {
        //     println!("layer: {i}");
        //     println!("hnsw state: {:?}", l);
        // }

        // search, doing everything from lmdb
        // let mut hnsw2: HnswBuilder<Cosine, 8, 16> = HnswBuilder::new(&BuildOption::default());
        // hnsw2.entrypoints = hnsw.entrypoints.clone();
        // let query: Vec<_> = (0..784).map(|_| rng.gen()).collect();
        // let q_item = Item::new(query.clone());
        // // dbg!("query = {}", &query);
        //
        // let now = Instant::now();
        // let rtxn = env.read_txn().unwrap();
        // let nns = hnsw2.search(&q_item, 10, 10, &db, &rtxn).unwrap();
        // println!("search; {:?}", now.elapsed());
        //
        // // check now
        // fn l2_norm(vec: &[f32]) -> f32 {
        //     vec.iter().map(|x| x * x).sum::<f32>().sqrt()
        // }
        //
        // let query_norm = l2_norm(&query);
        // let mut opt: Vec<_> = backup_vecs
        //     .iter()
        //     .enumerate()
        //     // .map(|(i, v)| {
        //     //     let dist: f32 = v.iter().zip(query.iter()).map(|(a, b)| (a - b).powi(2)).sum();
        //     //     (dist, i as u32)
        //     // })
        //     .map(|(i, v)| {
        //         let dot: f32 = v.iter().zip(query.iter()).map(|(a, b)| a * b).sum();
        //         let denom = l2_norm(v) * query_norm;
        //         let cosine_sim = dot / denom.max(1e-6); // avoid division by zero
        //         (0.5 - 0.5 * cosine_sim, i as u32)
        //     })
        //     .collect();
        //
        // opt.sort_by_key(|(d, _)| OrderedFloat(*d));
        //
        // println!("{:?}", &opt[..nns.len()]);
        // println!("{:?}", &nns);
        //
        // let mut recall = 0;
        // let nearest = RoaringBitmap::from_iter(opt.iter().take(nns.len()).map(|(_, i)| *i));
        // let retrieved = RoaringBitmap::from_iter(nns.iter().map(|(_, i)| *i));
        //
        // println!("recall: {}", ((nearest & retrieved).len() as f64) / (nns.len() as f64));
    }
}
