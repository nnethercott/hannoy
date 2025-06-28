use std::{
    cmp::Reverse,
    collections::BinaryHeap,
    marker::PhantomData,
    sync::atomic::{AtomicUsize, Ordering},
};

use heed::{RoTxn, RwTxn};
use min_max_heap::MinMaxHeap;
use nohash::IntMap;
use ordered_float::OrderedFloat;
use rand::{distributions::WeightedIndex, prelude::Distribution, Rng};
use roaring::RoaringBitmap;

use crate::{
    key::Key,
    node::{DbItem, Node, NodeCodec},
    writer::BuildOption,
    Database, Distance, ItemId, Result,
};

struct NodeState {
    pub level: usize,
    // neigbours in my layer
    pub links: RoaringBitmap,
    // who i connect to in the next layer, during search
    pub next: ItemId,
}

pub(crate) struct HnswBuilder<D> {
    m: usize,
    assign_probas: Vec<f32>,
    ef_construction: usize,
    max_level: usize,
    entrypoints: Vec<ItemId>,
    layers: IntMap<ItemId, NodeState>,
    metric: PhantomData<D>,
}

impl<D: Distance> HnswBuilder<D> {
    pub fn new(opts: &BuildOption) -> Self {
        let assign_probas = Self::get_default_probas(opts.m);

        Self {
            m: opts.m,
            assign_probas,
            ef_construction: opts.ef_construction,
            max_level: 0,
            entrypoints: vec![],
            layers: IntMap::default(),
            metric: PhantomData,
        }
    }

    /// build quantiles from an x ~ exp(1/ln(m))
    fn get_default_probas(m: usize) -> Vec<f32> {
        let mut assign_probas = Vec::with_capacity(m);
        let level_factor = 1.0 / (m as f32).ln();
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

    pub fn build<R>(&mut self, to_insert: RoaringBitmap, wtxn: &RwTxn, rng: &mut R)
    where
        R: Rng + ?Sized,
    {
        // generate a random level for each point
        let mut levels: Vec<_> = (0..to_insert.len())
            .into_iter()
            .map(|_| {
                let level = self.get_random_level(rng);
                self.max_level = self.max_level.max(level);
                level
            })
            .collect();

        // 1. sort levels and indices by level asc
        // 2. insert sequential
        todo!()
    }

    fn get_db_item<'a>(
        &'a self,
        item_id: ItemId,
        database: &Database<D>,
        rtxn: &'a RoTxn<'a>,
    ) -> Result<Node<'a, D>> {
        database
            .remap_data_type::<NodeCodec<D>>()
            .get(rtxn, &Key::item(0, item_id))?
            .ok_or(crate::Error::InvalidItemGet)
    }

    fn entry_point(&self, q: ItemId) -> ItemId {
        return self.entrypoints[0];
    }

    fn insert(
        &mut self,
        query: ItemId,
        level: usize,
        database: &Database<D>,
        rtxn: &RoTxn,
    ) -> Result<()> {
        let mut ep = self.entry_point(query);

        // greedy search with: ef = 1
        let start = (level + 1).max(self.max_level);
        for l in (start..=self.max_level).rev() {
            let mut neighbours = self.search_single_layer(query, ep, l, 1, database, rtxn)?;
            ep = neighbours
                .pop_min()
                .map(|(_, n)| n)
                .expect("Not a single nearest neighbor was found");
            // set to ep.next
        }

        // beam search with: ef = ef_construction
        for l in (0..=level.min(self.max_level)).rev() {
            let mut neighbours =
                self.search_single_layer(query, ep, l, self.ef_construction, database, rtxn)?;

            // FIXME: limit neighbors as a fn(self.m(layer)) ...

            while let Some((OrderedFloat(f), n)) = neighbours.pop_min() {
                // add links in both directions
                self.add_link(query, n, database, rtxn);
                self.add_link(n, query, database, rtxn);
                //helper.add_link_in_layer(item_id, query)
            }
        }

        // match level{
        //      Level::NonZero(n) =>{
        //          if n > self.max_level{
        //              set new one
        //          }
        //          else{
        //              bleh
        //          }
        //      },
        //      Level::Zero => {
        //          do something
        //      }
        // }

        Ok(())
    }

    // FIXME: return a vec of item ids instead of a minmax heap
    #[allow(clippy::too_many_arguments)]
    fn search_single_layer(
        &self,
        q: ItemId,
        ep: ItemId,
        level: usize,
        ef: usize,
        database: &Database<D>,
        rtxn: &RoTxn,
    ) -> Result<MinMaxHeap<(OrderedFloat<f32>, ItemId)>> {
        let mut candidates = BinaryHeap::new();
        let mut res = MinMaxHeap::with_capacity(ef);
        let mut visited = RoaringBitmap::new();

        let vq = self.get_db_item(q, database, rtxn)?;
        let ve = self.get_db_item(ep, database, rtxn)?;
        let dist = D::distance(&vq, &ve);

        // Register `ep` as visited
        candidates.push((Reverse(OrderedFloat(dist)), ep));
        res.push((OrderedFloat(dist), ep));
        visited.push(ep);

        while let Some((Reverse(OrderedFloat(f)), c)) = candidates.pop() {
            // stopping criteria
            if let Some((OrderedFloat(f_max), _)) = res.peek_max() {
                if f > *f_max {
                    break;
                }
            }

            // Get neighborhood and insert into candidates
            let proximity = match self.layers.get(&c) {
                Some(s) => &s.links,
                None => &RoaringBitmap::new(),
            };

            // wonder if we can par_iter this ?
            for point in proximity.iter() {
                if !visited.insert(point) {
                    continue;
                }
                let dist = D::distance(&vq, &self.get_db_item(point, &database, rtxn)?);

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

    // TODO: optimize; avoid recalculating distances
    // probably store the neighbors as a vec<(u32, f32)>
    fn add_link(
        &mut self,
        p: ItemId,
        q: ItemId,
        database: &Database<D>,
        rtxn: &RoTxn,
    ) -> Result<()> {
        // Add the new link
        self.layers.get_mut(&p).ok_or(crate::Error::NotInIntMap(p))?.links.insert(q);

        // Only evict neighbor if we're over capacity
        if self.layers.get(&p).ok_or(crate::Error::NotInIntMap(p))?.links.len() <= self.m as u64 {
            return Ok(());
        }

        let links_snapshot: Vec<ItemId> = self.layers.get(&p).unwrap().links.iter().collect();

        let src = self.get_db_item(p, &database, &rtxn)?;
        let mut minheap = BinaryHeap::new();

        for item_id in links_snapshot.into_iter() {
            let dest = self.get_db_item(item_id, &database, &rtxn)?;
            let d = D::distance(&src, &dest);
            minheap.push((Reverse(OrderedFloat(d)), item_id));
        }
        debug_assert!(minheap.len() > self.m);

        let mut new_neighbors = RoaringBitmap::new();
        for _ in 0..self.m {
            if let Some((_, item_id)) = minheap.pop() {
                new_neighbors.push(item_id);
            }
        }
        // Update links
        self.layers.get_mut(&p).unwrap().links = new_neighbors;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::HnswBuilder;
    use crate::{distance::Cosine, writer::BuildOption};
    use rand::{rngs::StdRng, SeedableRng};
    use std::collections::HashMap;

    #[test]
    // should be like: https://www.pinecone.io/learn/series/faiss/hnsw/
    fn check_distribution_shape() {
        let mut opts = BuildOption::default();
        opts.m = 32;
        let mut rng = StdRng::seed_from_u64(42);
        let mut hnsw = HnswBuilder::<Cosine>::new(&opts);

        let mut bins = HashMap::new();
        (0..1000000).into_iter().for_each(|_| {
            let level = hnsw.get_random_level(&mut rng);
            *bins.entry(level).or_insert(0) += 1;
        });

        dbg!("{:?}", bins);
    }
}
