use std::{
    cmp::Reverse,
    collections::BinaryHeap,
    marker::PhantomData,
    sync::atomic::{AtomicUsize, Ordering},
};

use heed::{RoTxn, RwTxn};
use min_max_heap::MinMaxHeap;
use ordered_float::OrderedFloat;
use rand::{distributions::WeightedIndex, prelude::Distribution, Rng};
use roaring::RoaringBitmap;

use crate::{
    key::Key,
    node::{DbItem, Item},
    writer::BuildOption,
    Database, Distance, ItemId, Result,
};

// TODO: this should be the struct from node.rs
struct HnswNode {
    pub level: usize,
    // neigbours in my layer
    pub links: RoaringBitmap,
    // who i connect to in the next layer, during search
    pub next: ItemId,
}

// could be worth to call build with a db ref, then build a helper struct storing that ...
pub(crate) struct HnswBuilder<D> {
    m: usize,
    assign_probas: Vec<f32>,
    ef_construction: usize,
    max_level: AtomicUsize,
    entrypoints: Vec<ItemId>,
    // TODO: this might need to become a hashmap cause we don't push in a linear order
    nodes: Vec<HnswNode>,
    metric: PhantomData<D>,
}

impl<D: Distance> HnswBuilder<D> {
    pub fn new(opts: &BuildOption) -> Self {
        let assign_probas = Self::get_default_probas(opts.m);

        Self {
            m: opts.m,
            assign_probas,
            ef_construction: opts.ef_construction,
            max_level: AtomicUsize::new(0),
            entrypoints: vec![],
            nodes: vec![],
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
        let levels: Vec<_> = (0..to_insert.len())
            .into_iter()
            .map(|_| {
                let level = self.get_random_level(rng);
                self.max_level.fetch_max(level, Ordering::Relaxed);
                level
            })
            .collect();

        let max_level = self.max_level.load(Ordering::Relaxed);

        // 1. sort levels and indices by level asc
        // 2. insert sequential
    }

    fn get_db_item<'a>(
        &'a self,
        item_id: ItemId,
        database: &Database<D>,
        rtxn: &'a RoTxn<'a>,
    ) -> Result<Item<'a, D>> {
        match database.get(rtxn, &Key::item(0, item_id))?.unwrap() {
            DbItem::Item(item) => Ok(item),
            _ => Err(crate::Error::InvalidItemGet),
        }
    }

    fn search_layer_build(
        &mut self,
        query: ItemId,
        level: usize, // turn this into enum and implement range iter on that
        database: &Database<D>,
        rtxn: &RoTxn,
    ) -> Result<()> {
        let mut ep = 0;
        let max_level = self.max_level.load(Ordering::Relaxed);

        // greedy search with: ef = 1
        for l in (level + 1..=max_level).rev() {
            let mut neighbours = self.search_single_layer(query, ep, l, 1, database, rtxn)?;
            ep = neighbours
                .pop_min()
                .map(|(_, n)| n)
                .expect("Not a single nearest neighbor was found");
        }

        // beam search with: ef = ef_construction
        for l in (0..=level.min(max_level)).rev() {
            let mut neighbours =
                self.search_single_layer(query, ep, l, self.ef_construction, database, rtxn)?;

            // FIXME: limit neighbors as a fn(self.m(layer)) ...
            while let Some((_, item_id)) = neighbours.pop_min(){
                // add links in both directions
                self.add_link_in_layer(query, item_id, database, rtxn);
                self.add_link_in_layer(item_id, query, database, rtxn);
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

    // TODO: clean this a bit
    // NOTE: won't work right now cause self.nodes has no guarantee on order ...
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

        // i don't like this, fix later
        let v_ref = self.get_db_item(q, database, rtxn)?;
        let w = self.get_db_item(ep, database, rtxn)?;
        let dist = D::distance(&v_ref, &w);

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
            let proximity = &self.nodes[c as usize].links;

            for point in proximity.iter() {
                if !visited.insert(point) {
                    continue;
                }
                let dist = D::distance(&v_ref, &self.get_db_item(point, &database, rtxn)?);

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

    /// items bitmap of graph links -- does this need to be bidirectional ?
    /// could also do a  (~furthest | new) & links
    fn add_link_in_layer(
        &mut self,
        p: ItemId,
        q: ItemId,
        database: &Database<D>,
        rtxn: &RoTxn,
    ) -> Result<()> {
        // Add the new link
        self.nodes[p as usize].links.insert(q);

        // Might not need to evict other links
        if self.nodes[p as usize].links.len() <= self.m as u64 {
            return Ok(());
        }

        let links_snapshot: Vec<ItemId> = self.nodes[p as usize].links.iter().collect();

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
        self.nodes[p as usize].links = new_neighbors;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::HnswBuilder;
    use crate::writer::BuildOption;
    use rand::{rngs::StdRng, SeedableRng};
    use std::collections::HashMap;

    #[test]
    // should be like: https://www.pinecone.io/learn/series/faiss/hnsw/
    fn check_distribution_shape() {
        let mut opts = BuildOption::default();
        opts.m = 32;
        let mut rng = StdRng::seed_from_u64(42);
        let mut hnsw = HnswBuilder::new(&opts);

        let mut bins = HashMap::new();
        (0..1000000).into_iter().for_each(|_| {
            let level = hnsw.get_random_level(&mut rng);
            *bins.entry(level).or_insert(0) += 1;
        });

        dbg!("{:?}", bins);
    }
}
