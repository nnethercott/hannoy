use std::{
    borrow::Cow,
    cmp::Reverse,
    collections::BinaryHeap,
    f32,
    fmt::{self, Debug},
    marker::PhantomData,
};

use heed::{RoTxn, RwTxn};
use min_max_heap::MinMaxHeap;
use nohash::IntMap;
use rand::{distributions::WeightedIndex, prelude::Distribution, Rng};
use roaring::RoaringBitmap;
use smallvec::{smallvec, SmallVec};

use crate::{
    key::Key,
    node::{Item, Links, Node},
    ordered_float::OrderedFloat,
    writer::BuildOption,
    Database, Distance, Error, ItemId, Result,
};

// TODO:
// - add dedicated 0th layer with M0 and fix corresponding code
// - add a NodeState.links() method or something

pub(crate) type ScoredLink = (OrderedFloat, ItemId);

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
    index: u16,
    rtxn: &'a RoTxn<'a>,
    //lru
}
impl<'a, D: Distance> LmdbReader<'a, D> {
    pub fn get_item(&self, item_id: ItemId) -> Result<Item<'a, D>> {
        let key = Key::item(self.index, item_id);

        // key is a `Key::item` so returned result must be a Node::Item
        Ok(self.database.get(self.rtxn, &key)?.ok_or(Error::missing_key(key))?.item().unwrap())
    }

    pub fn get_links(&self, item_id: ItemId, level: usize) -> Result<Links<'a>> {
        let key = Key::links(self.index, item_id, level as u8);

        // key is a `Key::links` so returned result must be a Node::Links
        Ok(self.database.get(self.rtxn, &key)?.ok_or(Error::missing_key(key))?.links().unwrap())
    }
}

pub struct HnswBuilder<D, const M: usize, const M0: usize> {
    assign_probas: Vec<f32>,
    ef_construction: usize,
    pub max_level: usize,
    pub entry_points: Vec<ItemId>,
    pub layers: Vec<IntMap<ItemId, NodeState<M0>>>,
    // last: IntMap<ItemId, NodeState<M>>,
    distance: PhantomData<D>,
}

impl<D: Distance, const M: usize, const M0: usize> HnswBuilder<D, M, M0> {
    pub fn new(opts: &BuildOption) -> Self {
        let assign_probas = Self::get_default_probas();

        Self {
            assign_probas,
            ef_construction: opts.ef_construction,
            max_level: 0,
            entry_points: vec![],
            layers: vec![],
            distance: PhantomData,
        }
    }

    /// build quantiles from an x ~ exp(1/ln(m))
    fn get_default_probas() -> Vec<f32> {
        let mut assign_probas = Vec::with_capacity(M);
        let level_factor = 1.0 / (M as f32 + f32::EPSILON).ln();
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
        index: u16,
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

        for _ in 0..=self.max_level {
            self.layers.push(IntMap::default());
        }

        for (item_id, level) in levels.into_iter() {
            if level == self.max_level {
                self.entry_points.push(item_id);
            }
            self.insert(item_id, level, database, index, wtxn)?;
        }

        // write single threaded to lmdb
        for lvl in 0..=self.max_level {
            for (item_id, node_state) in &self.layers[lvl] {
                let key = Key::links(0, *item_id, lvl as u8);
                let links = Links {
                    links: Cow::Owned(RoaringBitmap::from_iter(
                        node_state.links.iter().map(|(_, i)| *i),
                    )),
                };
                let node_edges = database.put(wtxn, &key, &Node::Links(links))?;
            }
        }

        // println!("{:?}", &self.layers);

        Ok(())
    }

    fn entry_point(&self) -> ItemId {
        return self.entry_points[0];
    }

    fn insert(
        &mut self,
        query: ItemId,
        level: usize,
        database: &Database<D>,
        index: u16,
        rtxn: &RoTxn,
    ) -> Result<()> {
        let lmdb = LmdbReader { database, index, rtxn };
        let mut eps = vec![self.entry_point()];

        let q = lmdb.get_item(query)?;

        // Greedy search with: ef = 1
        for lvl in (level + 1..=self.max_level).rev() {
            let neighbours = self.explore_layer(&q, &eps, lvl, 1, &lmdb)?;
            let closest = neighbours.peek_min().map(|(_, n)| *n).expect("No neighbor was found");
            eps = vec![closest];
        }

        // Beam search with: ef = ef_construction
        for lvl in (0..=level.min(self.max_level)).rev() {
            self.create_node(query, lvl);

            let mut neighbours = self.explore_layer(&q, &eps, lvl, self.ef_construction, &lmdb)?;

            eps.clear();
            for (dist, n) in self.select_heuristic(neighbours, level, false, &lmdb)? {
                // add links in both directions
                self.add_link(query, (dist, n), lvl, &lmdb)?;
                self.add_link(n, (dist, query), lvl, &lmdb)?;
                eps.push(n);
            }
        }

        Ok(())
    }

    fn create_node(&mut self, item_id: ItemId, level: usize) {
        self.layers[level].insert(item_id, NodeState { links: smallvec![] });
    }

    /// Returns only the Id's of our neighbours.
    fn get_neighbours(
        &self,
        lmdb: &LmdbReader<'_, D>,
        item_id: ItemId,
        level: usize,
    ) -> Result<Vec<ItemId>> {
        // FIXME: should this be a Result<Option<Vec<_>> not Result<Vec<_>> ?
        if level >= self.layers.len() {
            let Links { links } = lmdb.get_links(item_id, level)?;
            return Ok(links.iter().collect());
        }

        // O(1)
        match self.layers[level].get(&item_id) {
            Some(node_state) => return Ok(node_state.links.iter().map(|(_, i)| *i).collect()),

            // O(log n)
            None => {
                let Links { links } = lmdb.get_links(item_id, level)?;
                Ok(links.iter().collect())
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn explore_layer(
        &self,
        query: &Item<D>,
        eps: &[ItemId],
        level: usize,
        ef: usize,
        lmdb: &LmdbReader<'_, D>,
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

        while let Some(&(Reverse(OrderedFloat(f)), c)) = candidates.peek() {
            let &(OrderedFloat(f_max), _) = res.peek_max().unwrap();
            if f > f_max {
                break;
            }
            let (_, c) = candidates.pop().unwrap(); // Now safe to pop

            // Get neighborhood of candidate either from self or LMDB
            let proximity = self.get_neighbours(lmdb, c, level)?;
            for point in proximity {
                if !visited.insert(point) {
                    continue;
                }
                let dist = D::distance(query, &lmdb.get_item(point)?);

                if res.len() < ef || dist < f_max {
                    candidates.push((Reverse(OrderedFloat(dist)), point));

                    // optimized insert & removal maintaining original len
                    if res.len() == ef {
                        let _ = res.push_pop_max((OrderedFloat(dist), point));
                    } else {
                        let _ = res.push((OrderedFloat(dist), point));
                    }
                }
            }
        }

        Ok(res)
    }

    /// Tries to add link between two elements, returns a bool indicating if the link was created
    /// or not.
    fn add_link(
        &mut self,
        p: ItemId,
        q: ScoredLink,
        level: usize,
        lmdb: &LmdbReader<'_, D>,
    ) -> Result<()> {
        // prevent links to self
        if p == q.1 {
            return Ok(());
        }

        // Get links for node p. If the node comes from lmdb we create a new NodeState with empty
        // neighbours bitmap to track new HNSW graph edges.
        let node_state = (self.layers[level].entry(p)).or_insert(NodeState { links: smallvec![] });
        let links = &mut node_state.links;

        let cap = if level == 0 { M0 } else { M };

        if links.len() < cap {
            links.push(q);
            links.sort_unstable();
            return Ok(());
        }

        // else select heuristic
        let mut links_tmp = links.clone();
        links_tmp.push(q);
        drop(links);

        let new_links =
            self.select_heuristic(MinMaxHeap::from_iter(links_tmp), level, false, lmdb)?;
        self.layers[level].entry(p).and_modify(|s| s.links = SmallVec::from_iter(new_links));

        Ok(())
    }

    /// Naively choosing the nearest neighbours performs poorly on clustered data since we can never
    /// escape our local neighbourhood.
    fn select_heuristic(
        &self,
        mut candidates: MinMaxHeap<ScoredLink>,
        level: usize,
        keep_discarded: bool,
        lmdb: &LmdbReader<'_, D>,
    ) -> Result<Vec<ScoredLink>> {
        let mut selected: Vec<ScoredLink> = Vec::with_capacity(M0);
        let mut discared = vec![];

        let cap = if level == 0 { M0 } else { M };

        while let Some((dist_to_query, c)) = candidates.pop_min() {
            if selected.len() == cap {
                break;
            }

            // ensure we're closer to the query than we are to other candidates
            // TODO: make this more rust like, `try_fold` or something
            let mut ok_to_add = true;
            for i in selected.iter().map(|(_, i)| *i) {
                let d = D::distance(&lmdb.get_item(c)?, &lmdb.get_item(i)?);
                if OrderedFloat(d) < dist_to_query {
                    ok_to_add = false;
                    break;
                }
            }

            if ok_to_add {
                selected.push((dist_to_query, c));
            } else if keep_discarded {
                discared.push((dist_to_query, c));
            }
        }

        while keep_discarded && selected.len() < cap && discared.len() > 0 {
            selected.push(discared.remove(0));
        }

        Ok(selected)
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
}
