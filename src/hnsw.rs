use std::{cmp::Reverse, collections::BinaryHeap, marker::PhantomData};

use heed::RoTxn;
use min_max_heap::MinMaxHeap;
use nohash::IntMap;
use ordered_float::OrderedFloat;
use rand::{distributions::WeightedIndex, prelude::Distribution, Rng};
use roaring::RoaringBitmap;
use smallvec::{smallvec, SmallVec};
use tracing::{debug, info};

use crate::{
    key::Key,
    node::{Node, NodeCodec},
    writer::BuildOption,
    Database, Distance, ItemId, Result,
};

type Link = (OrderedFloat<f32>, ItemId);

/// State with stack-allocated graph edges
#[derive(Debug)]
struct NodeState<const M: usize> {
    // next is always ourselves in the subsequent layer
    links: SmallVec<[Link; M]>,
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
        rtxn: &RoTxn,
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

        // perform the insert
        for (item_id, level) in levels.into_iter() {
            if level == self.max_level {
                self.entrypoints.push(item_id);
            }
            self.insert(item_id, level, database, rtxn).unwrap();
        }
    }

    /// Fetches only vector info from lmdb using special codec
    fn get_db_item<'a>(
        &'a self,
        item_id: &ItemId,
        database: &Database<D>,
        rtxn: &'a RoTxn<'a>,
    ) -> Result<Node<'a, D>> {
        database
            .remap_data_type::<NodeCodec<D>>()
            .get(rtxn, &Key::item(0, *item_id))?
            .ok_or(crate::Error::InvalidItemGet)
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
        let mut eps = vec![self.entry_point()];

        // Greedy search with: ef = 1
        for lvl in (level + 1..=self.max_level).rev() {
            let mut neighbours = self.explore_layer(&query, &eps, lvl, 1, database, rtxn)?;
            let closest = neighbours.pop_min().map(|(_, n)| n).expect("No neighbor was found");
            eps = vec![closest];
        }

        // Beam search with: ef = ef_construction
        for lvl in (0..=level.min(self.max_level)).rev() {
            self.create_node(query, lvl);

            let mut neighbours =
                self.explore_layer(&query, &eps, lvl, self.ef_construction, database, rtxn)?;

            // FIXME: limit neighbors with algo 4; right now we have ef_construction many

            eps.clear();
            while let Some((dist, n)) = neighbours.pop_min() {
                // add links in both directions
                self.add_link(query, (dist, n), lvl, database, rtxn);
                self.add_link(n, (dist, query), lvl, database, rtxn);

                // Push each near point to the search list for next layer
                eps.push(n);
            }
        }

        Ok(())
    }

    fn create_node(&mut self, item_id: ItemId, level: usize) {
        self.layers[level].insert(item_id, NodeState { links: smallvec![] });
    }

    #[allow(clippy::too_many_arguments)]
    fn explore_layer(
        &self,
        q: &ItemId,
        eps: &[ItemId],
        level: usize,
        ef: usize,
        database: &Database<D>,
        rtxn: &RoTxn,
    ) -> Result<MinMaxHeap<Link>> {
        let mut candidates = BinaryHeap::new();
        let mut res = MinMaxHeap::with_capacity(ef);
        let mut visited = RoaringBitmap::new();

        let vq = self.get_db_item(q, database, rtxn)?;

        // Register all `eps` as visited and populate candidates
        for ep in eps {
            let ve = self.get_db_item(ep, database, rtxn)?;
            let dist = D::distance(&vq, &ve);

            candidates.push((Reverse(OrderedFloat(dist)), *ep));
            res.push((OrderedFloat(dist), *ep));
            visited.push(*ep);
        }

        while let Some((Reverse(OrderedFloat(f)), c)) = candidates.pop() {
            // stopping criteria
            if let Some((OrderedFloat(f_max), _)) = res.peek_max() {
                if f > *f_max {
                    break;
                }
            }

            // Get neighborhood of candidate
            let proximity = match self.layers[level].get(&c) {
                Some(node) => &node.links,
                None => unreachable!(),
            };

            // can we par_iter distance computations ?
            for &(_, point) in proximity.iter() {
                if !visited.insert(point) {
                    continue;
                }
                // distance between QUERY and point, not between neighbor and point
                let dist = D::distance(&vq, &self.get_db_item(&point, &database, rtxn)?);

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

    // FIXME: make the Link type a struct so its more readable
    fn add_link(
        &mut self,
        p: ItemId,
        q: Link,
        level: usize,
        database: &Database<D>,
        rtxn: &RoTxn,
    ) -> Result<()> {
        // prevent links to self
        if p == q.1{
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
        distance::Cosine,
        key::Key,
        node::{DbItem, HnswNode},
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
        (0..1000000).into_iter().for_each(|_| {
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
        let mut rng = StdRng::seed_from_u64(42);
        let mut hnsw: HnswBuilder<Cosine, 2, 4> = HnswBuilder::new(&BuildOption::default());

        let mut to_insert = RoaringBitmap::new();
        for item_id in 0u32..10 {
            let vec: Vec<f32> = (0..2).map(|_| rng.gen()).collect();
            let item = HnswNode::new(vec);
            db.put(&mut wtxn, &Key::item(0, item_id), &DbItem::Item(item)).unwrap();

            // update build bitmap
            to_insert.insert(item_id);
        }

        hnsw.build(to_insert, &db, &wtxn, &mut rng);

        for (i,l) in hnsw.layers.iter().enumerate(){
            println!("layer: {i}");
            println!("hnsw state: {:?}", l);
        }
    }
}
