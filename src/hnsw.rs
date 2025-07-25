use heed::RwTxn;
use min_max_heap::MinMaxHeap;
use papaya::HashMap;
use rand::{distributions::WeightedIndex, prelude::Distribution, Rng};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use roaring::{bitmap, RoaringBitmap};
use slice_group_by::GroupBy;
use std::{
    borrow::Cow,
    cmp::Reverse,
    collections::BinaryHeap,
    f32,
    fmt::{self, Debug},
    marker::PhantomData,
    panic,
};
use tinyvec::{array_vec, ArrayVec};
use tracing::debug;

use crate::{
    key::Key,
    node::{Item, Links, Node},
    ordered_float::OrderedFloat,
    parallel::{ImmutableItems, ImmutableLinks},
    stats::BuildStats,
    writer::{BuildOption, FrozzenReader},
    Database, Distance, Error, ItemId, Result,
};

// TODO:
// - add dedicated 0th layer with M0 and fix corresponding code
// - add a NodeState.links() method or something

pub(crate) type ScoredLink = (OrderedFloat, ItemId);

/// State with stack-allocated graph edges
struct NodeState<const M: usize> {
    links: ArrayVec<[ScoredLink; M]>,
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

pub struct HnswBuilder<D, const M: usize, const M0: usize> {
    assign_probas: Vec<f32>,
    ef_construction: usize,
    pub max_level: usize,
    pub entry_points: Vec<ItemId>,
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
            entry_points: Vec::new(),
            layers: vec![],
            distance: PhantomData,
        }
    }

    pub fn with_entry_points(mut self, entry_points: Vec<ItemId>) -> Self {
        self.entry_points = entry_points;
        self
    }

    pub fn with_max_level(mut self, max_level: usize) -> Self {
        self.max_level = max_level;
        self
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
        mut to_insert: RoaringBitmap,
        mut to_delete: RoaringBitmap,
        database: Database<D>,
        index: u16,
        wtxn: &mut RwTxn,
        rng: &mut R,
    ) -> Result<BuildStats<D>>
    where
        R: Rng + ?Sized,
    {
        let mut build_stats = BuildStats::new();

        // Generate a random level for each point
        let mut local_max_level = usize::MIN;
        let mut levels: Vec<_> = to_insert
            .iter()
            .map(|item_id| {
                let level = self.get_random_level(rng);
                local_max_level = local_max_level.max(level);
                (item_id, level)
            })
            .collect();

        // If re-indexing new points and a random level is higher than before, then we need to clear the
        // previous `entry_points`. However, to ensure the old graph gets updated we need to
        // schedule these ids for re-indexing, otherwise we end up building a completely isolated
        // sub-graph.
        levels.extend(self.entry_points.iter().map(|&id| (id, self.max_level)));
        to_insert |= RoaringBitmap::from_iter(self.entry_points.iter());

        if local_max_level > self.max_level {
            self.entry_points.clear();
        }

        self.max_level = self.max_level.max(local_max_level);
        for _ in 0..=self.max_level {
            self.layers.push(HashMap::new());
        }

        levels.sort_unstable_by(|(_, a), (_, b)| b.cmp(a));

        let upper_layer: Vec<_> = levels
            .iter()
            .take_while(|(_, l)| *l == self.max_level)
            .filter(|&(item_id, _)| !self.entry_points.contains(item_id))
            .collect();

        for &(item_id, _) in upper_layer {
            self.entry_points.push(item_id);
            self.add_in_layers_below(item_id, self.max_level);
        }

        // setup concurrent lmdb reader
        let items = ImmutableItems::new(wtxn, database, index)?;
        let nb_links = database.len(wtxn)? + to_insert.len();
        let links = ImmutableLinks::new(wtxn, database, index, nb_links)?;
        let lmdb = FrozzenReader { index, items: &items, links: &links };

        let level_groups: Vec<_> = levels.linear_group_by(|(_, la), (_, lb)| la == lb).collect();

        // insert layers L...0 multi-threaded
        // FIXME: fix error handling here, previously was source of bug
        level_groups.into_iter().for_each(|grp| {
            grp.into_par_iter().for_each(|&(item_id, lvl)| {
                self.insert(item_id, lvl, &lmdb, &build_stats).unwrap();
            });

            build_stats.layer_dist.insert(grp[0].1, grp.len());
        });

        self.maybe_patch_old_links(&lmdb, to_delete)?;

        // single-threaded write to lmdb
        for lvl in 0..=self.max_level {
            for (item_id, node_state) in &self.layers[lvl].pin() {
                let key = Key::links(index, *item_id, lvl as u8);
                let links = Links {
                    links: Cow::Owned(RoaringBitmap::from_iter(
                        node_state.links.iter().map(|(_, i)| *i),
                    )),
                };

                database.put(wtxn, &key, &Node::Links(links))?;
            }
        }

        // NOTE: this is only true for first-time builds. could be true if we add another `self.layers`
        // for links to "old" items
        // assert_eq!(
        //     self.layers.iter().map(|m| m.len()).sum::<usize>(),
        //     build_stats.layer_dist.iter().map(|(lvl, cnt)| { (lvl + 1) * cnt }).sum::<usize>()
        // );

        build_stats.compute_mean_degree(wtxn, &database, index);
        Ok(build_stats)
    }

    fn insert<'a>(
        &self,
        query: ItemId,
        level: usize,
        lmdb: &FrozzenReader<'a, D>,
        build_stats: &BuildStats<D>,
    ) -> Result<()> {
        let mut eps = Vec::from_iter(self.entry_points.clone());

        let q = lmdb.get_item(query)?;

        // Greedy search with: ef = 1
        for lvl in (level + 1..=self.max_level).rev() {
            let neighbours = self.explore_layer(&q, &eps, lvl, 1, lmdb, build_stats)?;
            let closest = neighbours.peek_min().map(|(_, n)| *n).expect("No neighbor was found");
            eps = vec![closest];
        }

        self.add_in_layers_below(query, level);

        // Beam search with: ef = ef_construction
        for lvl in (0..=level).rev() {
            let neighbours = self
                .explore_layer(&q, &eps, lvl, self.ef_construction, lmdb, build_stats)?
                .into_vec();

            eps.clear();
            for (dist, n) in self.select_sng(neighbours, level, false, lmdb)? {
                // add links in both directions
                self.add_link(query, (dist, n), lvl, lmdb)?;
                self.add_link(n, (dist, query), lvl, lmdb)?;
                eps.push(n);

                build_stats.incr_link_count(2);
            }
        }

        Ok(())
    }

    /// During incremental updates we store a working copy of potential links to the new items. At
    /// the end of indexing we need to merge the old and new links and prune ones pointing to
    /// deleted items.
    /// Algorithm 4 from FreshDiskANN paper.
    fn maybe_patch_old_links(
        &self,
        lmdb: &FrozzenReader<D>,
        to_delete: RoaringBitmap,
    ) -> Result<()> {
        let links_in_db =
            lmdb.links.iter().map(|((id, lvl), v)| ((id, lvl as usize), v.into_owned()));

        for ((id, lvl), links) in links_in_db {
            let del_subset = &links & &to_delete;
            let map_guard = self.layers[lvl].pin();
            let mut new_links =
                map_guard.get(&id).map(|state| state.links.to_vec()).unwrap_or(vec![]);

            // no work to be done, continue
            if del_subset.is_empty() && new_links.is_empty() {
                continue;
            }

            // NOTE: are bitmaps like sets e.g. do we have deduplication ?
            let mut bitmap = RoaringBitmap::new();
            for item_id in del_subset.iter() {
                bitmap.extend(lmdb.get_links(item_id, lvl)?.iter());
            }
            bitmap |= links;
            bitmap -= &to_delete;
            bitmap
                .into_iter()
                .map(|other| {
                    let dist = D::distance(&lmdb.get_item(id)?, &lmdb.get_item(other)?);
                    Ok::<ScoredLink, Error>((OrderedFloat(dist), other))
                })
                .collect::<Result<Vec<_>>>()
                .map(|nl| new_links.extend(nl));

            // finally prune and update
            let pruned = self.select_sng(new_links, lvl, false, lmdb)?;
            let _ = map_guard.insert(id, NodeState { links: ArrayVec::from_iter(pruned) });
        }

        Ok(())
    }

    /// Rather than simply insert, we'll make it a no-op so we can re-insert the same item without
    /// overwriting it's links in mem. This is useful in cases like Vanama build.
    fn add_in_layers_below(&self, item_id: ItemId, level: usize) {
        for level in 0..=level {
            self.layers[level].pin().get_or_insert(item_id, NodeState { links: array_vec![] });
        }
    }

    /// Returns only the Id's of our neighbours. Always check lmdb first.
    fn get_neighbours<'a>(
        &self,
        lmdb: &FrozzenReader<'a, D>,
        item_id: ItemId,
        level: usize,
        build_stats: &BuildStats<D>,
    ) -> Result<Vec<ItemId>> {
        let mut res = Vec::new();

        // O(1) from frozzenreader
        if let Ok(Links { links }) = lmdb.get_links(item_id, level) {
            build_stats.incr_lmdb_hits();
            res.extend(links.iter());
        }

        // O(1) from self.layers
        match self.layers[level].pin().get(&item_id) {
            Some(node_state) => res.extend(node_state.links.iter().map(|(_, i)| *i)),
            None => {
                if res.is_empty() {
                    unreachable!(
                        "the links for `item_id` must exist in either self.layers, lmdb, or both"
                    )
                }
            }
        }

        Ok(res)
    }

    #[allow(clippy::too_many_arguments)]
    fn explore_layer<'a>(
        &self,
        query: &Item<D>,
        eps: &[ItemId],
        level: usize,
        ef: usize,
        lmdb: &FrozzenReader<'a, D>,
        build_stats: &BuildStats<D>,
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

        while let Some(&(Reverse(OrderedFloat(f)), _)) = candidates.peek() {
            let &(OrderedFloat(f_max), _) = res.peek_max().unwrap();
            if f > f_max {
                break;
            }
            let (_, c) = candidates.pop().unwrap(); // Now safe to pop

            // Get neighborhood of candidate either from self or LMDB
            let proximity = self.get_neighbours(lmdb, c, level, build_stats)?;
            for point in proximity {
                if !visited.insert(point) {
                    continue;
                }
                // If the item isn't in the frozzen reader it must have been deleted from the index,
                // in which case its OK not to explore it
                let item = match lmdb.get_item(point) {
                    Ok(item) => item,
                    Err(Error::MissingKey { index, mode: _, item, layer: _ }) => {
                        // debug!("item {item} was deleted from index {index}!");
                        panic!("item {item} was deleted from index {index}!");
                    }
                    Err(e) => panic!("{}", e),
                };
                let dist = D::distance(query, &item);

                if res.len() < ef || dist < f_max {
                    candidates.push((Reverse(OrderedFloat(dist)), point));

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

    /// Tries to add a new link between nodes in a single direction.
    // TODO: prevent duplicate links the other way. I think this arises ONLY for entrypoints since
    // we pre-emptively add them in each layer before
    fn add_link<'a>(
        &self,
        p: ItemId,
        q: ScoredLink,
        level: usize,
        lmdb: &FrozzenReader<'a, D>,
    ) -> Result<()> {
        if p == q.1 {
            return Ok(());
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

            let new_links = self
                .select_sng(links.to_vec(), level, false, lmdb)
                .map(ArrayVec::from_iter)
                .unwrap_or_else(|_| node_state.links.clone());

            NodeState { links: new_links }
        };

        map.update_or_insert_with(p, _add_link, || NodeState {
            links: array_vec!([ScoredLink; M0] => q),
        });
        Ok(())
    }

    /// Naively choosing the nearest neighbours performs poorly on clustered data since we can never
    /// escape our local neighbourhood. "Sparse Neighbourhood Graph" (SNG) condition sufficient for
    /// quick convergence.
    fn select_sng(
        &self,
        mut candidates: Vec<ScoredLink>,
        level: usize,
        keep_discarded: bool,
        lmdb: &FrozzenReader<'_, D>,
    ) -> Result<Vec<ScoredLink>> {
        let cap = if level == 0 { M0 } else { M };
        candidates.sort_by(|a, b| b.cmp(a));

        let mut selected: Vec<ScoredLink> = Vec::with_capacity(cap);
        let mut discared = vec![];

        while let Some((dist_to_query, c)) = candidates.pop() {
            if selected.len() == cap {
                break;
            }

            // ensure we're closer to the query than we are to other candidates
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
    use crate::{distance::Cosine, writer::BuildOption};

    use rand::{rngs::StdRng, SeedableRng};

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
}
