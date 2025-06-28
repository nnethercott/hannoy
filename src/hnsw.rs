use std::{cmp::Reverse, collections::BinaryHeap, marker::PhantomData};

use heed::{RoTxn, RwTxn};
use min_max_heap::MinMaxHeap;
use nohash::IntMap;
use ordered_float::OrderedFloat;
use rand::{distributions::WeightedIndex, prelude::Distribution, Rng};
use roaring::RoaringBitmap;
use smallvec::{smallvec, SmallVec};

use crate::{
    key::Key,
    node::{DbItem, Node, NodeCodec},
    writer::BuildOption,
    Database, Distance, ItemId, Result,
};

type Link = (OrderedFloat<f32>, ItemId);

/// State with stack-allocated graph edges
struct NodeState<const M: usize> {
    next: Option<ItemId>,
    links: SmallVec<[Link; M]>,
}

pub(crate) struct HnswBuilder<D, const M: usize, const M0: usize> {
    assign_probas: Vec<f32>,
    ef_construction: usize,
    max_level: usize,
    entrypoints: Vec<ItemId>,
    layers: Vec<IntMap<ItemId, NodeState<M0>>>,
    last: IntMap<ItemId, NodeState<M>>,
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
            last: IntMap::default(),
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

        // Register point for given layer
        // NOTE: this may be bad ! think about it more
        self.layers[level].insert(query, NodeState { next: None, links: smallvec![] });

        // Greedy search with: ef = 1
        for lvl in (level + 1..=self.max_level).rev() {
            let mut neighbours = self.search_single_layer(&query, &eps, lvl, 1, database, rtxn)?;
            let closest = neighbours.pop_min().map(|(_, n)| n).expect("No neighbor was found");

            // Set ep to closest.next
            eps = vec![self.layers[lvl].get(&closest).unwrap().next.unwrap()];
        }

        // Beam search with: ef = ef_construction
        for lvl in (0..=level.min(self.max_level)).rev() {
            let mut neighbours =
                self.search_single_layer(&query, &eps, lvl, self.ef_construction, database, rtxn)?;

            // FIXME: limit neighbors as a fn(self.m(layer)) ...

            eps.clear();
            while let Some((dist, n)) = neighbours.pop_min() {
                // add links in both directions
                self.add_link(query, (dist, n), lvl, database, rtxn);
                self.add_link(n, (dist, query), lvl, database, rtxn);

                // Push each closest point's `next` to search queue for next level
                eps.push(self.layers[lvl].get(&n).unwrap().next.unwrap());
            }

            self.create_virtual_node(query, lvl + 1);
        }

        Ok(())
    }

    fn create_virtual_node(&mut self, item_id: ItemId, level: usize) {}

    #[allow(clippy::too_many_arguments)]
    fn search_single_layer(
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

        // Register all `eps` as visited
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

            // Get neighborhood and insert into candidates
            let proximity = match self.layers[level].get(&c) {
                Some(node) => &node.links,
                None => continue,
            };

            // wonder if we can par_iter this ?
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

    fn add_link(
        &mut self,
        p: ItemId,
        q: Link,
        level: usize,
        database: &Database<D>,
        rtxn: &RoTxn,
    ) -> Result<()> {
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
    use crate::{distance::Cosine, writer::BuildOption};
    use rand::{rngs::StdRng, SeedableRng};
    use std::collections::HashMap;

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
}
