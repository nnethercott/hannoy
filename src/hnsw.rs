use std::{
    cmp::Reverse,
    collections::BinaryHeap,
    marker::PhantomData,
    sync::atomic::{AtomicU16, Ordering},
};

use heed::{RoTxn, RwTxn};
use ordered_float::OrderedFloat;
use rand::{distributions::WeightedIndex, prelude::Distribution, Rng};
use roaring::RoaringBitmap;

use crate::{
    key::Key,
    node::{DbItem, Item},
    writer::BuildOption,
    Database, Distance, ItemId, Result,
};

pub(crate) struct HnswBuilder {
    m: usize,
    assign_probas: Vec<f32>,
    ef_construction: usize,
    max_level: AtomicU16, // maybe one day we'll do concurrent stuff
    entrypoints: Vec<ItemId>,
    links: Vec<RoaringBitmap>,
}

impl HnswBuilder {
    pub fn new(opts: &BuildOption) -> Self {
        let assign_probas = Self::get_default_probas(opts.m);

        Self {
            m: opts.m,
            assign_probas,
            ef_construction: opts.ef_construction,
            max_level: AtomicU16::new(0),
            entrypoints: vec![],
            links: vec![],
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

    fn get_random_level<R>(&mut self, rng: &mut R) -> u16
    where
        R: Rng + ?Sized,
    {
        let dist = WeightedIndex::new(&self.assign_probas).unwrap();
        dist.sample(rng) as u16
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
    }

    fn insert_node(&self, node: ItemId, wtxn: &RwTxn) {
        // self.search(node) -> neighborhood
        // for n in neighbors{
        //    self.add_link(node, n, wtxn);
        // }
        todo!()
    }

    fn search(&self) {
        todo!()
    }

    /// function to update an items bitmap of graph links
    fn add_link<D: Distance>(
        &mut self,
        p: ItemId,
        q: ItemId,
        database: &Database<D>,
        wtxn: &RoTxn,
    ) -> Result<()> {
        let links = &mut self.links[p as usize];
        links.push(q);

        if links.len() < self.m as u64 {
            return Ok(());
        }

        let src = match database.get(wtxn, &Key::item(0, p))?.unwrap() {
            DbItem::Item(item) => item,
            _ => unreachable!(),
        };

        let mut minheap = BinaryHeap::new();
        for item_id in links.iter() {
            let dest = match database.get(wtxn, &Key::item(0, item_id))?.unwrap() {
                DbItem::Item(item) => item,
                _ => unreachable!(),
            };
            let d = D::distance(&src, &dest);
            minheap.push((Reverse(OrderedFloat(d)), item_id));
        }
        debug_assert!(minheap.len() > self.m);

        // TODO: turn Vec<RoaringBitmap> into Vec<Struct> which stores neighbors bitmap and current
        // furthest neighbor id
        let mut new_neighbors = RoaringBitmap::new();
        for _ in 0..self.m {
            if let Some((_, item_id)) = minheap.pop() {
                new_neighbors.push(item_id);
            }
        }
        *links = new_neighbors;

        Ok(())

        // self.select_heuristic()
    }

    fn select_heuristic(&self) {
        todo!()
    }
}

// struct NodeDistFarther {
//     id: ItemId,
//     dist: OrderedFloat<f32>,
// }
//
// impl PartialOrd for NodeDistFarther {
//     fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
//         Reverse(self.dist.partial_cmp(&other.dist))
//     }
// }

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
