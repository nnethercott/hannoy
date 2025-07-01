use std::{
    cmp::Reverse,
    collections::BinaryHeap,
    fmt::{self, Debug},
    marker::PhantomData,
};

use heed::{RoTxn, RwTxn};
use min_max_heap::MinMaxHeap;
use nohash::IntMap;
use ordered_float::OrderedFloat;
use rand::{distributions::WeightedIndex, prelude::Distribution, Rng};
use roaring::RoaringBitmap;
use smallvec::{smallvec, SmallVec};

use crate::{
    key::Key,
    node::{Item, Links},
    writer::BuildOption,
    Database, Distance, Error, ItemId, Result,
};

// TODO:
// - add dedicated 0th layer with M0 and fix corresponding code

type ScoredLink = (OrderedFloat<f32>, ItemId);

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

struct DbHelper<'a, D> {
    database: &'a Database<D>,
    rtxn: &'a RoTxn<'a>,
}
impl<'a, D: Distance> DbHelper<'a, D> {
    pub fn get_item(&self, item_id: ItemId) -> Result<Item<'a, D>> {
        let key = Key::item(0, item_id);
        Ok(self.database.get(self.rtxn, &key)?.ok_or(Error::missing_key(key))?.item().unwrap())
    }

    pub fn get_links(&self, item_id: ItemId, level: usize) -> Result<Links<'a>> {
        let key = Key::links(0, item_id, level as u8);
        Ok(self.database.get(self.rtxn, &key)?.ok_or(Error::missing_key(key))?.links().unwrap())
    }
}

pub struct HnswBuilder<D, const M: usize, const M0: usize> {
    assign_probas: Vec<f32>,
    ef_construction: usize,
    max_level: usize,
    entrypoints: Vec<ItemId>,
    pub layers: Vec<IntMap<ItemId, NodeState<M0>>>,
    // last: IntMap<ItemId, NodeState<M>>,
    distance: PhantomData<D>,
}

impl<D: Distance, const M: usize, const M0: usize> HnswBuilder<D, M, M0> {
    pub fn new(opts: &BuildOption) -> Self {
        let assign_probas = Self::get_default_probas();
        dbg!("{}", assign_probas.len());

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
        database: &Database<D>,
        wtxn: &mut RwTxn,
        rng: &mut R,
    ) where
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

        levels.sort_by(|(_, a), (_, b)| b.cmp(a));
        println!("levels={:?}", &levels);

        for _ in 0..=self.max_level {
            self.layers.push(IntMap::default());
        }

        // perform the insert into hnsw graph
        for (item_id, level) in levels.into_iter() {
            if level == self.max_level {
                self.entrypoints.push(item_id);
            }
            self.insert(item_id, level, database, wtxn).unwrap();
        }

        // TODO: persist in db
    }

    fn entry_point(&self) -> ItemId {
        return self.entrypoints[0];
    }

    fn insert(
        &mut self,
        query: ItemId,
        level: usize,
        database: &Database<D>,
        rtxn: &RoTxn,
    ) -> Result<()> {
        let lmdb = DbHelper { database, rtxn };
        let mut eps = vec![self.entry_point()];

        // Greedy search with: ef = 1
        for lvl in (level + 1..=self.max_level).rev() {
            let mut neighbours = self.explore_layer(&query, &eps, lvl, 1, &lmdb)?;
            let closest = neighbours.pop_min().map(|(_, n)| n).expect("No neighbor was found");
            eps = vec![closest];
        }

        // Beam search with: ef = ef_construction
        for lvl in (0..=level.min(self.max_level)).rev() {
            self.create_node(query, lvl);

            let mut neighbours =
                self.explore_layer(&query, &eps, lvl, self.ef_construction, &lmdb)?;

            // FIXME: limit neighbors with algo 4; right now we have ef_construction many

            eps.clear();
            while let Some((dist, n)) = neighbours.pop_min() {
                // add links in both directions
                self.add_link(query, (dist, n), lvl);
                self.add_link(n, (dist, query), lvl);

                // Push each near point to the search list for next layer
                eps.push(n);
            }
        }

        Ok(())
    }

    fn create_node(&mut self, item_id: ItemId, level: usize) {
        self.layers[level].insert(item_id, NodeState { links: smallvec![] });
    }

    /// Returns only the Id's of our neighbours,
    fn get_or_create_neighbours(
        &mut self,
        lmdb: &DbHelper<'_, D>,
        item_id: ItemId,
        level: usize,
    ) -> Result<Vec<ItemId>> {
        // O(1)
        match self.layers[level].get(&item_id) {
            Some(node_state) => return Ok(node_state.links.iter().map(|(_, i)| *i).collect()),

            // O(log n)
            None => {
                let Links { links } = lmdb.get_links(item_id, level)?;

                // allow for new links to be established with existing item
                self.create_node(item_id, level);

                Ok(links.iter().collect())
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn explore_layer(
        &mut self,
        q: &ItemId,
        eps: &[ItemId],
        level: usize,
        ef: usize,
        lmdb: &DbHelper<'_, D>,
    ) -> Result<MinMaxHeap<ScoredLink>> {
        let mut candidates = BinaryHeap::new();
        let mut res = MinMaxHeap::with_capacity(ef);
        let mut visited = RoaringBitmap::new();

        let vq = lmdb.get_item(*q)?;

        // Register all `eps` as visited and populate candidates
        for &ep in eps {
            let ve = lmdb.get_item(ep)?;
            let dist = D::distance(&vq, &ve);

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

            // Get neighborhood of candidate either from `self` or LMDB
            let proximity = self.get_or_create_neighbours(lmdb, c, level)?;

            // can we par_iter distance computations ?
            for &point in proximity.iter() {
                if !visited.insert(point) {
                    continue;
                }
                // distance between QUERY and point, not between neighbor and point
                let dist = D::distance(&vq, &lmdb.get_item(point)?);

                res.push((OrderedFloat(dist), point));
                candidates.push((Reverse(OrderedFloat(dist)), point));

                if res.len() > ef {
                    // NOTE: could just `res.push_pop_max()` with a single resize ...
                    let _ = res.pop_max();
                }
            }
        }

        Ok(res)
    }

    fn add_link(&mut self, p: ItemId, q: ScoredLink, level: usize) -> Result<()> {
        // prevent links to self
        if p == q.1 {
            return Ok(());
        }

        // Get links for p
        let links = &mut self.layers[level].get_mut(&p).ok_or(crate::Error::NotInIntMap(p))?.links;
        let cap = if level == 0 { M0 } else { M };

        if links.len() < cap {
            links.push(q);
            links.sort_unstable();
            return Ok(());
        }

        // Avoid doing work unless necessary
        if q.0 > links.last().unwrap().0 {
            return Ok(());
        }

        // pop first to avoid moving smallvec to heap
        let _ = links.pop();

        match links.binary_search(&q) {
            Ok(index) => links.insert(index, q),
            Err(index) => links.insert(index, q),
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::HnswBuilder;
    use crate::{
        distance::{Cosine, Euclidean},
        key::Key,
        node::{Item, Node},
        writer::BuildOption,
        Database,
    };
    use heed::EnvOpenOptions;
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use roaring::RoaringBitmap;
    use std::collections::HashMap;

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
        let db: Database<Euclidean> = env.create_database(&mut wtxn, None).unwrap();

        // insert a few vectors
        let mut rng = StdRng::seed_from_u64(42);
        let mut hnsw: HnswBuilder<Euclidean, 2, 3> = HnswBuilder::new(&BuildOption::default());

        let vecs: Vec<Vec<f32>> = (0..6).map(|_| (0..2).map(|_| rng.gen()).collect()).collect();
        dbg!("{:?}", &vecs);

        let mut to_insert = RoaringBitmap::new();
        for (item_id, vec) in vecs.into_iter().enumerate() {
            let item = Item::new(vec);
            db.put(&mut wtxn, &Key::item(0, item_id as u32), &Node::Item(item)).unwrap();

            // update build bitmap
            to_insert.insert(item_id as u32);
        }

        hnsw.build(to_insert, &db, &mut wtxn, &mut rng);

        for (i, l) in hnsw.layers.iter().enumerate() {
            println!("layer: {i}");
            println!("hnsw state: {:?}", l);
        }
    }
}
