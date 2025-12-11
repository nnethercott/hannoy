use std::borrow::Cow;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::f32;
use std::fmt::{self, Debug};
use std::marker::PhantomData;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::{AtomicUsize, Ordering};

use heed::RwTxn;
use min_max_heap::MinMaxHeap;
use papaya::HashMap;
use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;
use rand::Rng;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use roaring::RoaringBitmap;
use tinyvec::{array_vec, ArrayVec};
use tracing::{debug, instrument};

use crate::key::Key;
use crate::node::{Item, Links, Node};
use crate::ordered_float::OrderedFloat;
use crate::parallel::{ImmutableItems, ImmutableLinks};
use crate::progress::{AtomicInsertItemsStep, HannoyBuild};
use crate::stats::BuildStats;
use crate::writer::{BuildOption, FrozenReader};
use crate::{Database, Distance, Error, ItemId, Result, CANCELLATION_PROBING};

pub(crate) type ScoredLink = (OrderedFloat, ItemId);

/// State with stack-allocated graph edges
pub struct NodeState<const M: usize> {
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

pub struct HnswBuilder<'a, D, const M: usize, const M0: usize> {
    assign_probas: Vec<f32>,
    ef_construction: usize,
    alpha: f32,
    cancel: &'a (dyn Fn() -> bool + 'a + Sync + Send),
    pub max_level: usize,
    pub entry_points: Vec<ItemId>,
    pub layers: Vec<HashMap<ItemId, NodeState<M0>>>,
    distance: PhantomData<D>,
}

impl<'a, D: Distance, const M: usize, const M0: usize> HnswBuilder<'a, D, M, M0> {
    pub fn new<P: steppe::Progress>(opts: &'a BuildOption<P>) -> Self {
        let assign_probas = Self::get_default_probas();
        Self {
            assign_probas,
            ef_construction: opts.ef_construction,
            alpha: opts.alpha,
            cancel: &opts.cancel,
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
        dist.sample(rng)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn build<R, P>(
        &mut self,
        mut to_insert: RoaringBitmap,
        to_delete: &RoaringBitmap,
        database: Database<D>,
        index: u16,
        wtxn: &mut RwTxn,
        rng: &mut R,
        options: &BuildOption<P>,
    ) -> Result<BuildStats<D>>
    where
        R: Rng + ?Sized,
        P: steppe::Progress,
    {
        let mut build_stats = BuildStats::new();

        let items = ImmutableItems::new(wtxn, database, index, options)?;
        let links = ImmutableLinks::new(wtxn, database, index, database.len(wtxn)?, options)?;
        let lmdb = FrozenReader { index, items: &items, links: &links };

        // Generate a random level for each point
        let mut cur_max_level = usize::MIN;
        let mut levels: Vec<_> = to_insert
            .iter()
            .map(|item_id| {
                let level = self.get_random_level(rng);
                cur_max_level = cur_max_level.max(level);
                (item_id, level)
            })
            .collect();

        let ok_eps = self.prepare_levels_and_entry_points(
            &mut levels,
            cur_max_level,
            to_delete,
            &lmdb,
            options,
        )?;
        to_insert |= ok_eps;

        let level_groups: Vec<_> = levels.chunk_by(|(_, la), (_, lb)| la == lb).collect();

        // Insert layers L...0 multi-threaded
        options.progress.update(HannoyBuild::BuildingTheGraph);
        let (item_ctr, insert_step) = AtomicInsertItemsStep::new(to_insert.len());
        options.progress.update(insert_step);
        let cancel_index = AtomicUsize::new(0);

        level_groups.iter().for_each(|grp| {
            build_stats.layer_dist.insert(grp[0].1, grp.len());
        });

        level_groups.into_iter().try_for_each(|grp| {
            grp.into_par_iter().try_for_each(|&(item_id, lvl)| {
                if cancel_index.fetch_add(1, Relaxed) % CANCELLATION_PROBING == 0 && (self.cancel)()
                {
                    Err(Error::BuildCancelled)
                } else {
                    self.insert(item_id, lvl, &lmdb, &build_stats)?;
                    item_ctr.fetch_add(1, Relaxed);
                    Ok(())
                }
            })?;
            Ok(()) as Result<(), Error>
        })?;

        self.fill_gaps_from_deleted(&lmdb, to_delete, options)?;

        // Single-threaded write to lmdb
        options.progress.update(HannoyBuild::WritingTheItems);
        let mut cancellation_index = 0;

        for (lvl, map) in self.layers.iter().enumerate() {
            let map_guard = map.pin();

            for (item_id, node_state) in &map_guard {
                if cancellation_index % CANCELLATION_PROBING == 0 && (self.cancel)() {
                    return Err(Error::BuildCancelled);
                }

                let key = Key::links(index, *item_id, lvl as u8);
                let links = Links {
                    links: Cow::Owned(RoaringBitmap::from_iter(
                        node_state.links.iter().map(|(_, i)| *i),
                    )),
                };

                database.put(wtxn, &key, &Node::Links(links))?;
                cancellation_index += 1;
            }
        }

        build_stats.compute_mean_degree(wtxn, &database, index)?;
        Ok(build_stats)
    }

    /// This function resolves several nasty edge cases that can occur, namely : deleted
    /// or partially deleted entrypoints, new indexed points assigned to higher layers, ensuring
    /// entry points are present on all layers before build
    #[instrument(skip(self, options, lmdb, levels))]
    fn prepare_levels_and_entry_points<P>(
        &mut self,
        levels: &mut Vec<(u32, usize)>,
        cur_max_level: usize,
        to_delete: &RoaringBitmap,
        lmdb: &FrozenReader<D>,
        options: &BuildOption<P>,
    ) -> Result<RoaringBitmap>
    where
        P: steppe::Progress,
    {
        debug!("Resolving entry points in (maybe incremental) build");
        options.progress.update(HannoyBuild::ResolveGraphEntryPoints);

        let old_eps = RoaringBitmap::from_iter(self.entry_points.iter());
        let mut new_eps = &old_eps - to_delete;
        let del_eps = &old_eps & to_delete;

        // If any old entry points were deleted we need to replace them with points from the
        // previous graph
        let mut l = self.max_level;
        for _ in del_eps.iter() {
            loop {
                for result in lmdb.links.iter_layer(l as u8) {
                    let ((item_id, _), _) = result?;

                    if !to_delete.contains(item_id) && new_eps.insert(item_id) {
                        break;
                    }
                }
                l = match l.checked_sub(1) {
                    Some(new_level) => new_level,
                    None => break,
                };
            }
        }

        // Case 1: if we delted some entrypoints but were unable to replace them this must
        // mean we've deleted the entire previous graph, so we reset the height.
        if !del_eps.is_empty() & (new_eps.len() != old_eps.len()) {
            self.max_level = 0;
        }

        // Schedule old entry point ids for re-indexing, otherwise we end up building a completely
        // isolated sub-graph.
        levels.extend(new_eps.iter().map(|id| (id, self.max_level)));
        levels.sort_unstable_by(|(_, a), (_, b)| b.cmp(a));

        // Case 2: if the new build puts points on higher levels than before, then we have new hnsw
        // entrypoints.
        if cur_max_level > self.max_level {
            new_eps.clear();
            self.entry_points.clear();
            self.max_level = cur_max_level;
        }

        let upper_layer: Vec<_> = levels.iter().take_while(|(_, l)| *l == self.max_level).collect();
        for _ in 0..=self.max_level {
            self.layers.push(HashMap::new());
        }
        for &(item_id, _) in upper_layer {
            new_eps.insert(item_id);
            self.add_in_layers_below(item_id, self.max_level);
        }

        self.entry_points = new_eps.iter().collect();
        Ok(new_eps)
    }

    fn insert(
        &self,
        query: ItemId,
        level: usize,
        lmdb: &FrozenReader<'_, D>,
        build_stats: &BuildStats<D>,
    ) -> Result<()> {
        let mut eps = Vec::from_iter(self.entry_points.clone());

        let q = lmdb.get_item(query)?;

        // Greedy search with: ef = 1
        for lvl in (level + 1..=self.max_level).rev() {
            let neighbours = self.walk_layer(&q, &eps, lvl, 1, lmdb, build_stats)?;
            let closest = neighbours.peek_min().map(|(_, n)| *n).expect("No neighbor was found");
            eps = vec![closest];
        }

        self.add_in_layers_below(query, level);

        // Beam search with: ef = ef_construction
        for lvl in (0..=level).rev() {
            let neighbours =
                self.walk_layer(&q, &eps, lvl, self.ef_construction, lmdb, build_stats)?.into_vec();

            eps.clear();
            for (dist, n) in self.robust_prune(neighbours, level, self.alpha, lmdb)? {
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
    fn fill_gaps_from_deleted<P>(
        &mut self,
        lmdb: &FrozenReader<D>,
        to_delete: &RoaringBitmap,
        options: &BuildOption<P>,
    ) -> Result<()>
    where
        P: steppe::Progress,
    {
        debug!("Repairing connections to deleted items, and linking old and new graphs");
        options.progress.update(HannoyBuild::PatchOldNewDeletedLinks);

        let links_in_db: Vec<_> = lmdb
            .links
            .iter()
            .map(|result| {
                result.map(|((id, lvl), v)| {
                    // Resize the layers if necessary. We must do this to accomodate links from
                    // previous builds that exist on levels larger than our current one.
                    if self.layers.len() <= lvl as usize {
                        self.layers.resize_with(lvl as usize + 1, HashMap::new);
                    }
                    ((id, lvl as usize), v.into_owned())
                })
            })
            .collect();

        let cancel_index = AtomicUsize::new(0);

        links_in_db.into_par_iter().try_for_each(|result| {
            if cancel_index.fetch_add(1, Ordering::Relaxed) % CANCELLATION_PROBING == 0
                && (self.cancel)()
            {
                return Err(Error::BuildCancelled);
            }
            let ((id, lvl), links) = result?;

            // Since we delete links AFTER a build (we need to do this to apply diskann-approach
            // for patching), links belonging to deleted items may still be present. We don't
            // care about patching them.
            if to_delete.contains(id) {
                return Ok(());
            }
            let del_subset = &links & to_delete;

            // This is safe because we resized layers above.
            let map_guard = self.layers[lvl].pin();
            let mut new_links = map_guard.get(&id).map(|s| s.links.to_vec()).unwrap_or_default();

            let mut bitmap = RoaringBitmap::new();
            for item_id in del_subset.iter() {
                bitmap.extend(lmdb.get_links(item_id, lvl).unwrap_or_default().iter());
            }
            bitmap |= links;
            bitmap -= to_delete;
            debug_assert!(bitmap.is_disjoint(to_delete));

            //  Case 1: Union of [on_disk, current_build, deleted_extension] is small enough
            let thresh = if lvl == 0 { M0 } else { M };
            if (bitmap.len() as usize) + new_links.len() <= thresh {
                // NOTE: pairwise distance is no longer relevant
                let mut entries: Vec<_> =
                    bitmap.iter().map(|node_id| (OrderedFloat(0.0f32), node_id)).collect();
                entries.extend(new_links);

                let _ = map_guard.insert(id, NodeState { links: ArrayVec::from_iter(entries) });
                return Ok(());
            }

            // Case 2: Some old links may be popped to fill gaps from deleted nodes
            let curr = &lmdb.get_item(id)?;

            for other in bitmap {
                let dist = D::distance(curr, &lmdb.get_item(other)?);
                new_links.push((OrderedFloat(dist), other));
            }
            let pruned = self.robust_prune(new_links, lvl, self.alpha, lmdb)?;
            let _ = map_guard.insert(id, NodeState { links: ArrayVec::from_iter(pruned) });
            Ok(())
        })?;

        Ok(())
    }

    /// Rather than simply insert, we'll make it a no-op so we can re-insert the same item without
    /// overwriting it's links in mem. This is useful in cases like Vanama build.
    fn add_in_layers_below(&self, item_id: ItemId, level: usize) {
        for level in 0..=level {
            let Some(map) = self.layers.get(level) else { break };
            map.pin().get_or_insert(item_id, NodeState { links: array_vec![] });
        }
    }

    /// Returns only the Id's of our neighbours. Always check lmdb first.
    #[instrument(level = "trace", skip(self, lmdb))]
    fn get_neighbours(
        &self,
        lmdb: &FrozenReader<'_, D>,
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
        let Some(map) = self.layers.get(level) else { return Ok(res) };

        let pinned = map.pin();
        match pinned.get(&item_id) {
            Some(node_state) => res.extend(node_state.links.iter().map(|(_, i)| *i)),
            None => {
                // lazily add this entry so he can get updated later
                pinned.insert(item_id, NodeState { links: array_vec![] });
            }
        }

        Ok(res)
    }

    #[allow(clippy::too_many_arguments)]
    #[instrument(name = "walk_layer", skip(self, lmdb, query))]
    fn walk_layer(
        &self,
        query: &Item<D>,
        eps: &[ItemId],
        level: usize,
        ef: usize,
        lmdb: &FrozenReader<'_, D>,
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
            visited.insert(ep);
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
                    Err(Error::MissingKey { .. }) => continue,
                    Err(e) => return Err(e),
                };
                let dist = D::distance(query, &item);

                if res.len() < ef || dist < f_max {
                    candidates.push((Reverse(OrderedFloat(dist)), point));

                    if res.len() == ef {
                        let _ = res.push_pop_max((OrderedFloat(dist), point));
                    } else {
                        res.push((OrderedFloat(dist), point));
                    }
                }
            }
        }

        Ok(res)
    }

    /// Tries to add a new link between nodes in a single direction.
    // TODO: prevent duplicate links the other way. I think this arises ONLY for entrypoints since
    // we pre-emptively add them in each layer before
    fn add_link(
        &self,
        p: ItemId,
        q: ScoredLink,
        level: usize,
        lmdb: &FrozenReader<'_, D>,
    ) -> Result<()> {
        if p == q.1 {
            return Ok(());
        }

        let Some(map) = self.layers.get(level) else { return Ok(()) };
        let map_guard = map.pin();

        // 'pure' links update function
        let _add_link = |node_state: &NodeState<M0>| {
            let mut links = node_state.links;
            let cap = if level == 0 { M0 } else { M };

            if links.len() < cap {
                links.push(q);
                return NodeState { links };
            }

            let new_links = self
                .robust_prune(links.to_vec(), level, self.alpha, lmdb)
                .map(ArrayVec::from_iter)
                .unwrap_or_else(|_| node_state.links);

            NodeState { links: new_links }
        };

        map_guard.update_or_insert_with(p, _add_link, || NodeState {
            links: array_vec!([ScoredLink; M0] => q),
        });

        Ok(())
    }

    /// Naively choosing the nearest neighbours performs poorly on clustered data since we can never
    /// escape our local neighbourhood. "Sparse Neighbourhood Graph" (SNG) condition sufficient for
    /// quick convergence.
    fn robust_prune(
        &self,
        mut candidates: Vec<ScoredLink>,
        level: usize,
        alpha: f32,
        lmdb: &FrozenReader<'_, D>,
    ) -> Result<Vec<ScoredLink>> {
        let cap = if level == 0 { M0 } else { M };
        candidates.sort_by(|a, b| b.cmp(a));
        let mut selected: Vec<ScoredLink> = Vec::with_capacity(cap);

        while let Some((dist_to_query, c)) = candidates.pop() {
            if selected.len() == cap {
                break;
            }

            // ensure we're closer to the query than we are to other candidates
            let mut ok_to_add = true;
            for i in selected.iter().map(|(_, i)| *i) {
                let d = D::distance(&lmdb.get_item(c)?, &lmdb.get_item(i)?);
                if OrderedFloat(d * alpha) < dist_to_query {
                    ok_to_add = false;
                    break;
                }
            }

            if ok_to_add {
                selected.push((dist_to_query, c));
            }
        }

        Ok(selected)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use rand::rngs::StdRng;
    use rand::SeedableRng;

    use super::HnswBuilder;
    use crate::distance::Cosine;
    use crate::writer::BuildOption;

    #[ignore = "just cause"]
    #[test]
    // should be like: https://www.pinecone.io/learn/series/faiss/hnsw/
    fn check_distribution_shape() {
        let mut rng = StdRng::seed_from_u64(42);
        let build_option = BuildOption::default();
        let mut hnsw = HnswBuilder::<Cosine, 32, 48>::new(&build_option);

        let mut bins = HashMap::new();
        (0..10000).for_each(|_| {
            let level = hnsw.get_random_level(&mut rng);
            *bins.entry(level).or_insert(0) += 1;
        });

        dbg!("{:?}", bins);
    }
}
