use std::{
    cmp::{Ordering, Reverse},
    sync::atomic::{AtomicU16, Ordering},
};

use heed::{RoTxn, RwTxn};
use ordered_float::OrderedFloat;
use rand::{distributions::WeightedIndex, prelude::Distribution, Rng};
use roaring::RoaringBitmap;

use crate::{writer::BuildOption, ItemId};

pub(crate) struct HnswBuilder {
    assign_probas: Vec<f32>,
    ef_construction: usize,
    max_level: AtomicU16,
    entrypoints: Vec<ItemId>,
}

impl HnswBuilder {
    pub fn new(opts: &BuildOption) -> Self {
        let assign_probas = Self::get_default_probas(opts.m);

        Self {
            assign_probas,
            ef_construction: opts.ef_construction,
            max_level: AtomicU16::new(0),
            entrypoints: vec![],
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
