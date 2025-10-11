use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::marker;
use std::num::NonZeroUsize;

use heed::types::DecodeIgnore;
use heed::RoTxn;
use min_max_heap::MinMaxHeap;
use roaring::RoaringBitmap;

use crate::distance::Distance;
use crate::hnsw::ScoredLink;
use crate::internals::KeyCodec;
use crate::item_iter::ItemIter;
use crate::metadata::Metadata;
use crate::node::{Item, Links};
use crate::ordered_float::OrderedFloat;
use crate::unaligned_vector::UnalignedVector;
use crate::version::{Version, VersionCodec};
use crate::{Database, Error, ItemId, Key, MetadataCodec, Node, Prefix, PrefixCodec, Result};

/// A good default value for the `ef` parameter.
const DEFAULT_EF_SEARCH: usize = 100;

#[cfg(not(windows))]
const READER_AVAILABLE_MEMORY: &str = "HANNOY_READER_PREFETCH_MEMORY";

#[cfg(not(test))]
/// The threshold at which linear search is used instead of the HNSW algorithm.
const LINEAR_SEARCH_THRESHOLD: u64 = 1000;
#[cfg(test)]
/// Note that for tests purposes, we use set this threshold
/// to zero to make sure we test the HNSW algorithm.
const LINEAR_SEARCH_THRESHOLD: u64 = 0;

/// Options used to make a query against an hannoy [`Reader`].
pub struct QueryBuilder<'a, D: Distance> {
    reader: &'a Reader<D>,
    candidates: Option<&'a RoaringBitmap>,
    count: usize,
    ef: usize,
}

impl<'a, D: Distance> QueryBuilder<'a, D> {
    /// Returns the closests items from `item`.
    ///
    /// See also [`Self::by_vector`].
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use hannoy::{Reader, distances::Euclidean};
    /// # let (reader, rtxn): (Reader<Euclidean>, heed::RoTxn) = todo!();
    /// reader.nns(20).by_item(&rtxn, 5);
    /// ```
    pub fn by_item(&self, rtxn: &RoTxn, item: ItemId) -> Result<Option<Vec<(ItemId, f32)>>> {
        self.reader.nns_by_item(rtxn, item, self, || false).map(|res| match res {
            Some(Completion::Done(items)) => Some(items),
            Some(Completion::Cancelled(_)) => {
                unreachable!("cancellation only possible using by_item_with_cancellation")
            }
            None => None,
        })
    }

    /// Returns as many nearest neighbours to the query as possible before `cancel_fn` evaluates to
    /// true, and indicates whether or not search terminated early.
    ///
    /// See also [`Self::by_vector_with_cancellation`].
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use hannoy::{Reader, distances::Euclidean};
    /// # let (reader, rtxn): (Reader<Euclidean>, heed::RoTxn) = todo!();
    /// use std::time::{Instant, Duration};
    ///
    /// let later = Instant::now().checked_add(Duration::from_secs(1)).unwrap();
    /// let cancel_fn = || Instant::now() > later;
    /// let (nns, did_cancel) = reader.nns(20).by_item_with_cancellation(&rtxn, 5, cancel_fn)?.unwrap();
    /// # Ok::<(), hannoy::Error>(())
    /// ```
    pub fn by_item_with_cancellation(
        &self,
        rtxn: &RoTxn,
        item: ItemId,
        cancel_fn: impl Fn() -> bool,
    ) -> Result<Option<(Vec<(ItemId, f32)>, bool)>> {
        self.reader.nns_by_item(rtxn, item, self, cancel_fn).map(|res| match res {
            Some(Completion::Done(done)) => Some((done, false)),
            Some(Completion::Cancelled(cancelled)) => Some((cancelled, true)),
            None => None,
        })
    }

    /// Returns the closest items from the provided `vector`.
    ///
    /// See also [`Self::by_item`].
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use hannoy::{Reader, distances::Euclidean};
    /// # let (reader, rtxn): (Reader<Euclidean>, heed::RoTxn) = todo!();
    /// reader.nns(20).by_vector(&rtxn, &[1.25854, -0.75598, 0.58524]);
    /// ```
    pub fn by_vector(&self, rtxn: &RoTxn, vector: &'a [f32]) -> Result<Vec<(ItemId, f32)>> {
        if vector.len() != self.reader.dimensions() {
            return Err(Error::InvalidVecDimension {
                expected: self.reader.dimensions(),
                received: vector.len(),
            });
        }

        let vector = UnalignedVector::from_slice(vector);
        let item = Item { header: D::new_header(&vector), vector };

        let cancel_fn = || false;
        self.reader.nns_by_vec(rtxn, &item, self, cancel_fn).map(|res| res.into_inner())
    }

    /// Returns as many nearest neighbours to the query as possible before `cancel_fn` evaluates to
    /// true, and indicates whether or not search terminated early.
    ///
    /// See also [`Self::by_item_with_cancellation`].
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use hannoy::{Reader, distances::Euclidean};
    /// # let (reader, rtxn): (Reader<Euclidean>, heed::RoTxn) = todo!();
    /// use std::time::{Instant, Duration};
    ///
    /// let later = Instant::now().checked_add(Duration::from_secs(1)).unwrap();
    /// let cancel_fn = || Instant::now() > later;
    /// let (nns, did_cancel) = reader.nns(20).by_vector_with_cancellation(&rtxn, &[1.25854, -0.75598, 0.58524], cancel_fn)?;
    /// # Ok::<(), hannoy::Error>(())
    /// ```
    pub fn by_vector_with_cancellation(
        &self,
        rtxn: &RoTxn,
        vector: &'a [f32],
        cancel_fn: impl Fn() -> bool,
    ) -> Result<(Vec<(ItemId, f32)>, bool)> {
        if vector.len() != self.reader.dimensions() {
            return Err(Error::InvalidVecDimension {
                expected: self.reader.dimensions(),
                received: vector.len(),
            });
        }

        let vector = UnalignedVector::from_slice(vector);
        let item = Item { header: D::new_header(&vector), vector };

        let nns = self.reader.nns_by_vec(rtxn, &item, self, cancel_fn)?;
        match nns {
            Completion::Done(done) => Ok((done, false)),
            Completion::Cancelled(cancelled) => Ok((cancelled, true)),
        }
    }

    /// Specify a subset of candidates to inspect. Filters out everything else.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use hannoy::{Reader, distances::Euclidean};
    /// # let (reader, rtxn): (Reader<Euclidean>, heed::RoTxn) = todo!();
    /// let candidates = roaring::RoaringBitmap::from_iter([1, 3, 4, 5, 6, 7, 8, 9, 15, 16]);
    /// reader.nns(20).candidates(&candidates).by_item(&rtxn, 6);
    /// ```
    pub fn candidates(&mut self, candidates: &'a RoaringBitmap) -> &mut Self {
        self.candidates = Some(candidates);
        self
    }

    /// Specify a search buffer size from which the closest elements are returned. Increasing this
    /// value improves the search relevancy but increases latency as more neighbours need to be
    /// searched.
    /// In an ideal graph `ef`=`count` would suffice.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use hannoy::{Reader, distances::Euclidean};
    /// # let (reader, rtxn): (Reader<Euclidean>, heed::RoTxn) = todo!();
    /// reader.nns(20).ef_search(21).by_item(&rtxn, 6);
    /// ```
    pub fn ef_search(&mut self, ef: usize) -> &mut Self {
        self.ef = ef.max(self.count);
        self
    }
}

enum Completion<T> {
    Done(T),
    Cancelled(T),
}
impl<T> Completion<T> {
    pub fn into_inner(self) -> T {
        match self {
            Completion::Done(inner) => inner,
            Completion::Cancelled(inner) => inner,
        }
    }
    pub fn map<U>(self, op: impl FnOnce(T) -> U) -> Completion<U> {
        match self {
            Self::Done(inner) => Completion::Done(op(inner)),
            Self::Cancelled(inner) => Completion::Cancelled(op(inner)),
        }
    }
}

struct Visitor<'a> {
    pub eps: Vec<ItemId>,
    pub level: usize,
    pub ef: usize,
    pub candidates: Option<&'a RoaringBitmap>,
}
impl<'a> Visitor<'a> {
    pub fn new(
        eps: Vec<ItemId>,
        level: usize,
        ef: usize,
        candidates: Option<&'a RoaringBitmap>,
    ) -> Self {
        Self { eps, level, ef, candidates }
    }

    /// Iteratively traverse a given level of the HNSW graph, updating the search path history.
    /// Returns a Min-Max heap of size ef nearest neighbours to the query in that layer.
    #[allow(clippy::too_many_arguments)]
    pub fn visit<D: Distance>(
        &self,
        query: &Item<D>,
        reader: &Reader<D>,
        rtxn: &RoTxn,
        path: &mut RoaringBitmap,
        cancel_fn: &impl Fn() -> bool,
    ) -> Result<Completion<MinMaxHeap<ScoredLink>>> {
        use Completion::*;

        let mut search_queue = BinaryHeap::new();
        let mut res = MinMaxHeap::with_capacity(self.ef);

        // Register all entry points as visited and populate candidates
        for &ep in &self.eps[..] {
            let ve = get_item(reader.database, reader.index, rtxn, ep)?.unwrap();
            let dist = D::distance(query, &ve);

            search_queue.push((Reverse(OrderedFloat(dist)), ep));
            path.insert(ep);

            if self.candidates.is_none_or(|c| c.contains(ep)) {
                res.push((OrderedFloat(dist), ep));
            }
        }

        // Stop occurs either once we've done at least ef searches and notice no improvements, or
        // when we've exhausted the search queue.
        while let Some(&(Reverse(OrderedFloat(f)), _)) = search_queue.peek() {
            if cancel_fn() {
                return Ok(Cancelled(res));
            }
            let f_max = res.peek_max().map(|&(OrderedFloat(d), _)| d).unwrap_or(f32::MAX);
            if f > f_max {
                break;
            }
            let (_, c) = search_queue.pop().unwrap();

            let Links { links } = get_links(rtxn, reader.database, reader.index, c, self.level)?
                .expect("Links must exist");

            for point in links.iter() {
                if !path.insert(point) {
                    continue;
                }
                let dist = D::distance(
                    query,
                    &get_item(reader.database, reader.index, rtxn, point)?.unwrap(),
                );

                // The search queue can take points that aren't included in the (optional)
                // candidates bitmap, but the final result must *not* include them.
                if res.len() < self.ef || dist < f_max {
                    search_queue.push((Reverse(OrderedFloat(dist)), point));
                    if let Some(c) = self.candidates {
                        if !c.contains(point) {
                            continue;
                        }
                    }
                    if res.len() == self.ef {
                        let _ = res.push_pop_max((OrderedFloat(dist), point));
                    } else {
                        res.push((OrderedFloat(dist), point));
                    }
                }
            }
        }
        Ok(Done(res))
    }
}

/// A reader over the hannoy hnsw graph
#[derive(Debug)]
pub struct Reader<D: Distance> {
    pub(crate) database: Database<D>,
    pub(crate) index: u16,
    entry_points: Vec<ItemId>,
    max_level: usize,
    dimensions: usize,
    items: RoaringBitmap,
    version: Version,
    _marker: marker::PhantomData<D>,
}

impl<D: Distance> Reader<D> {
    /// Returns a reader over the database with the specified [`Distance`] type.
    pub fn open(rtxn: &RoTxn, index: u16, database: Database<D>) -> Result<Reader<D>> {
        let metadata_key = Key::metadata(index);

        let metadata = match database.remap_data_type::<MetadataCodec>().get(rtxn, &metadata_key)? {
            Some(metadata) => metadata,
            None => return Err(Error::MissingMetadata(index)),
        };
        let version =
            match database.remap_data_type::<VersionCodec>().get(rtxn, &Key::version(index))? {
                Some(version) => version,
                None => Version { major: 0, minor: 0, patch: 0 },
            };

        if D::name() != metadata.distance {
            return Err(Error::UnmatchingDistance {
                expected: metadata.distance.to_owned(),
                received: D::name(),
            });
        }

        // check if we need to rebuild
        if database
            .remap_types::<PrefixCodec, DecodeIgnore>()
            .prefix_iter(rtxn, &Prefix::updated(index))?
            .remap_key_type::<KeyCodec>()
            .next()
            .is_some()
        {
            return Err(Error::NeedBuild(index));
        }

        // Hint to the kernel that we'll probably need some vectors in RAM.
        Self::prefetch_graph(rtxn, &database, index, &metadata)?;

        Ok(Reader {
            database: database.remap_data_type(),
            index,
            entry_points: Vec::from_iter(metadata.entry_points.iter()),
            max_level: metadata.max_level as usize,
            dimensions: metadata.dimensions.try_into().unwrap(),
            items: metadata.items,
            version,
            _marker: marker::PhantomData,
        })
    }

    #[cfg(windows)]
    fn prefetch_graph(
        _rtxn: &RoTxn,
        _database: &Database<D>,
        _index: u16,
        _metadata: &Metadata,
    ) -> Result<()> {
        // madvise crate does not support windows.
        Ok(())
    }

    /// Instructs kernel to fetch nodes based on a fixed memory budget. It's OK for this operation
    /// to fail, it's not integral for search to work.
    #[cfg(not(windows))]
    fn prefetch_graph(
        rtxn: &RoTxn,
        database: &Database<D>,
        index: u16,
        metadata: &Metadata,
    ) -> Result<()> {
        use crate::unaligned_vector::UnalignedVectorCodec;

        use heed::types::Bytes;
        use madvise::AccessPattern;
        use std::collections::VecDeque;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use tracing::warn;

        let page_size = page_size::get();
        let mut available_memory: usize = std::env::var(READER_AVAILABLE_MEMORY)
            .ok()
            .and_then(|num| num.parse::<usize>().ok())
            .unwrap_or(0);

        if available_memory < page_size {
            return Ok(());
        }

        let largest_alloc = AtomicUsize::new(0);

        // adjusted length in memory of a vector
        let item_length = (metadata.dimensions as usize)
            .div_ceil(<D::VectorCodec as UnalignedVectorCodec>::word_size());

        let madvise_page = |item: &[u8]| -> Result<usize> {
            let start_ptr = item.as_ptr() as usize;
            let end_ptr = start_ptr + item_length;
            let start_page = start_ptr - (start_ptr % page_size);
            let end_page = end_ptr + ((end_ptr + page_size - 1) % page_size);
            let advised_size = end_page - start_page;

            unsafe {
                madvise::madvise(start_page as *const u8, advised_size, AccessPattern::WillNeed)?;
            }

            largest_alloc.fetch_max(advised_size, Ordering::Relaxed);
            Ok(advised_size)
        };

        // Load links and vectors for layers > 0.
        let mut added = RoaringBitmap::new();
        for lvl in (1..=metadata.max_level).rev() {
            for result in database.remap_data_type::<Bytes>().iter(rtxn)? {
                if available_memory < largest_alloc.load(Ordering::Relaxed) {
                    return Ok(());
                }
                let (key, item) = result?;
                if key.node.layer != lvl {
                    continue;
                }
                match madvise_page(item) {
                    Ok(usage) => available_memory -= usage,
                    Err(e) => {
                        warn!(e=?e);
                        return Ok(());
                    }
                }
                added.insert(key.node.item);
            }
        }

        // If we still have memory left over try fetching other nodes in layer zero.
        let mut queue = VecDeque::from_iter(added.iter());
        while let Some(item) = queue.pop_front() {
            if available_memory < largest_alloc.load(Ordering::Relaxed) {
                return Ok(());
            }
            if let Some(Node::Links(links)) = database.get(rtxn, &Key::links(index, item, 0))? {
                for l in links.iter() {
                    if !added.insert(l) {
                        continue;
                    }
                    if let Some(bytes) =
                        database.remap_data_type::<Bytes>().get(rtxn, &Key::item(index, l))?
                    {
                        match madvise_page(bytes) {
                            Ok(usage) => available_memory -= usage,
                            Err(e) => {
                                warn!(e=?e);
                                return Ok(());
                            }
                        }
                        queue.push_back(l);
                    }
                }
            }
        }

        Ok(())
    }

    /// Returns the number of dimensions in the index.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Returns the number of entry points to the hnsw index.
    pub fn n_entrypoints(&self) -> usize {
        self.entry_points.len()
    }

    /// Returns the number of vectors stored in the index.
    pub fn n_items(&self) -> u64 {
        self.items.len()
    }

    /// Returns all the item ids contained in this index.
    pub fn item_ids(&self) -> &RoaringBitmap {
        &self.items
    }

    /// Returns the index of this reader in the database.
    pub fn index(&self) -> u16 {
        self.index
    }

    /// Returns the version of the database.
    pub fn version(&self) -> Version {
        self.version
    }

    /// Returns the number of nodes in the index. Useful to run an exhaustive search.
    pub fn n_nodes(&self, rtxn: &RoTxn) -> Result<Option<NonZeroUsize>> {
        Ok(NonZeroUsize::new(self.database.len(rtxn)? as usize))
    }

    /// Returns the vector for item `i` that was previously added.
    pub fn item_vector(&self, rtxn: &RoTxn, item_id: ItemId) -> Result<Option<Vec<f32>>> {
        Ok(get_item(self.database, self.index, rtxn, item_id)?.map(|item| {
            let mut vec = item.vector.to_vec();
            vec.truncate(self.dimensions());
            vec
        }))
    }

    /// Returns `true` if the index is empty.
    pub fn is_empty(&self, rtxn: &RoTxn) -> Result<bool> {
        self.iter(rtxn).map(|mut iter| iter.next().is_none())
    }

    /// Returns `true` if the database contains the given item.
    pub fn contains_item(&self, rtxn: &RoTxn, item_id: ItemId) -> Result<bool> {
        self.database
            .remap_data_type::<DecodeIgnore>()
            .get(rtxn, &Key::item(self.index, item_id))
            .map(|opt| opt.is_some())
            .map_err(Into::into)
    }

    /// Returns an iterator over the items vector.
    pub fn iter<'t>(&self, rtxn: &'t RoTxn) -> Result<ItemIter<'t, D>> {
        ItemIter::new(self.database, self.index, self.dimensions, rtxn).map_err(Into::into)
    }

    /// Return a [`QueryBuilder`] that lets you configure and execute a search request.
    ///
    /// You must provide the number of items you want to receive.
    pub fn nns(&self, count: usize) -> QueryBuilder<'_, D> {
        QueryBuilder { reader: self, candidates: None, count, ef: DEFAULT_EF_SEARCH }
    }

    fn nns_by_vec(
        &self,
        rtxn: &RoTxn,
        query: &Item<D>,
        opt: &QueryBuilder<D>,
        cancel_fn: impl Fn() -> bool,
    ) -> Result<Completion<Vec<(ItemId, f32)>>> {
        use Completion::*;

        // If we will never find any candidates, return an empty vector
        if opt.candidates.is_some_and(|c| self.item_ids().is_disjoint(c)) {
            return Ok(Done(Vec::new()));
        }

        // If the number of candidates is less than a given threshold, perform linear search
        if let Some(candidates) = opt.candidates.filter(|c| c.len() < LINEAR_SEARCH_THRESHOLD) {
            return self.brute_force_search(query, rtxn, candidates, opt.count, cancel_fn);
        }

        // exhaustive search
        self.hnsw_search(query, rtxn, opt, cancel_fn)
    }

    /// Directly retrieves items in the candidate list and ranks them by distance to the query.
    fn brute_force_search(
        &self,
        query: &Item<D>,
        rtxn: &RoTxn,
        candidates: &RoaringBitmap,
        count: usize,
        cancel_fn: impl Fn() -> bool,
    ) -> Result<Completion<Vec<(ItemId, f32)>>> {
        use Completion::*;

        let mut item_distances = Vec::with_capacity(candidates.len() as usize);

        for item_id in candidates {
            if cancel_fn() {
                return Ok(Cancelled(item_distances));
            }

            let Some(vector) = self.item_vector(rtxn, item_id)? else { continue };
            let vector = UnalignedVector::from_vec(vector);
            let item = Item { header: D::new_header(&vector), vector };
            let distance = D::distance(&item, query);
            item_distances.push((item_id, distance));
        }
        item_distances.sort_by_key(|(_, dist)| OrderedFloat(*dist));
        item_distances.truncate(count);

        Ok(Done(item_distances))
    }

    /// Hnsw search according to arXiv:1603.09320.
    ///
    /// We perform greedy beam search from the top layer to the bottom, where the search frontier
    /// is controlled by `opt.ef`. Since the graph is not necessarily acyclic, search may become
    /// "trapped" in a local sub-graph with fewer elements than `opt.count` - to account for this
    /// we run an expensive exhaustive search at the end if fewer nns were returned.
    ///
    /// To break out of search early, users may wish to provide a `cancel_fn` which terminates the
    /// execution of the hnsw search and returns partial results so far.
    fn hnsw_search(
        &self,
        query: &Item<D>,
        rtxn: &RoTxn,
        opt: &QueryBuilder<D>,
        cancel_fn: impl Fn() -> bool,
    ) -> Result<Completion<Vec<(ItemId, f32)>>> {
        use Completion::*;

        let cancel_fn = &cancel_fn;
        let mut visitor = Visitor::new(self.entry_points.clone(), self.max_level, 1, None);

        let mut path = RoaringBitmap::new();
        for _ in (1..=self.max_level).rev() {
            let neighbours = visitor.visit(query, &self, rtxn, &mut path, &|| false)?.into_inner();
            let closest = neighbours.peek_min().map(|(_, n)| n).expect("No neighbor was found");

            visitor.eps = vec![*closest];
            visitor.level -= 1;
        }
        // clear visited set as we only care about level 0
        path.clear();
        debug_assert!(visitor.level == 0);

        visitor.ef = opt.ef.max(opt.count);
        visitor.candidates = opt.candidates;

        macro_rules! return_if_cancelled {
            ($completion: expr) => {
                match $completion {
                    Completion::Done(done) => done,
                    cancelled => {
                        return Ok(cancelled.map(|mut found| {
                            found
                                .drain_asc()
                                .map(|(OrderedFloat(f), i)| (i, f))
                                .take(opt.count)
                                .collect()
                        }))
                    }
                }
            };
        }

        let mut neighbours =
            return_if_cancelled!(visitor.visit(query, &self, rtxn, &mut path, cancel_fn)?);

        // If we still don't have enough nns (e.g. search encountered cyclic subgraphs) then do exhaustive
        // search over remaining unseen items.
        if neighbours.len() < opt.count {
            let mut cursor = self
                .database
                .remap_types::<PrefixCodec, DecodeIgnore>()
                .prefix_iter(rtxn, &Prefix::item(self.index))?
                .remap_key_type::<KeyCodec>();

            while let Some((key, _)) = cursor.next().transpose()? {
                let id = key.node.item;
                if path.contains(id) {
                    continue;
                }

                visitor.eps = vec![id];
                visitor.ef = opt.count - neighbours.len();

                let more_nns =
                    return_if_cancelled!(visitor.visit(query, &self, rtxn, &mut path, cancel_fn)?);

                neighbours.extend(more_nns.into_iter());
                if neighbours.len() >= opt.count {
                    break;
                }
            }
        }

        let found =
            neighbours.drain_asc().map(|(OrderedFloat(f), i)| (i, f)).take(opt.count).collect();
        Ok(Done(found))
    }

    /// Returns the nearest points to the item id, not including the point itself.
    ///
    /// Nearly identical behaviour to `Reader.nns_by_vec` except we only search layer 0 and use the
    /// `&[item]` instead of the hnsw entrypoints. Since search starts in the true neighbourhood of
    /// the item fewer comparisons are needed to retrieve the nearest neighbours, making it more
    /// efficient than simply calling `Reader.nns_by_vec` with the associated vector.
    fn nns_by_item(
        &self,
        rtxn: &RoTxn,
        item: ItemId,
        opt: &QueryBuilder<D>,
        cancel_fn: impl Fn() -> bool,
    ) -> Result<Option<Completion<Vec<(ItemId, f32)>>>> {
        use Completion::*;
        let cancel_fn = &cancel_fn;

        // If we will never find any candidates, return none
        if opt.candidates.is_some_and(|c| self.item_ids().is_disjoint(c)) {
            return Ok(None);
        }

        let Some(vector) = self.item_vector(rtxn, item)? else { return Ok(None) };
        let vector = UnalignedVector::from_vec(vector);
        let query = Item { header: D::new_header(&vector), vector };

        // If the number of candidates is less than a given threshold, perform linear search
        if let Some(candidates) = opt.candidates.filter(|c| c.len() < LINEAR_SEARCH_THRESHOLD) {
            let nns = self.brute_force_search(&query, rtxn, candidates, opt.count, cancel_fn)?;
            return Ok(Some(nns));
        }

        // Search over all items except `item`
        let ef = opt.ef.max(opt.count);
        let mut path = RoaringBitmap::new();
        let mut candidates = opt.candidates.unwrap_or_else(|| self.item_ids()).clone();
        candidates.remove(item);

        let mut visitor = Visitor::new(vec![item], 0, ef, Some(&candidates));

        macro_rules! return_if_cancelled {
            ($completion: expr) => {
                match $completion {
                    Completion::Done(done) => done,
                    cancelled => {
                        return Ok(Some(cancelled.map(|mut found| {
                            found
                                .drain_asc()
                                .map(|(OrderedFloat(f), i)| (i, f))
                                .take(opt.count)
                                .collect()
                        })))
                    }
                }
            };
        }
        let mut neighbours =
            return_if_cancelled!(visitor.visit(&query, &self, rtxn, &mut path, cancel_fn)?);

        // If we still don't have enough nns (e.g. search encountered cyclic subgraphs) then do exhaustive
        // search over remaining unseen items.
        if neighbours.len() < opt.count {
            let mut cursor = self
                .database
                .remap_types::<PrefixCodec, DecodeIgnore>()
                .prefix_iter(rtxn, &Prefix::item(self.index))?
                .remap_key_type::<KeyCodec>();

            while let Some((key, _)) = cursor.next().transpose()? {
                let id = key.node.item;
                if path.contains(id) {
                    continue;
                }

                // update walker
                visitor.eps = vec![id];
                visitor.ef = opt.count - neighbours.len();

                let more_nns =
                    return_if_cancelled!(visitor.visit(&query, &self, rtxn, &mut path, cancel_fn)?);
                neighbours.extend(more_nns.into_iter());
                if neighbours.len() >= opt.count {
                    break;
                }
            }
        }

        let found: Vec<_> =
            neighbours.drain_asc().map(|(OrderedFloat(f), i)| (i, f)).take(opt.count).collect();
        Ok(Some(Done(found)))
    }

    /// NOTE: a [`crate::Reader`] can't be opened unless updates are commited through a build !
    /// Verify that the whole reader is correctly formed:
    /// - All items are linked.
    /// - All links contain only items in the db (e.g. no previously deleted!).
    /// - All the entrypoints exist.
    ///
    /// This function should always be called in tests and on the latest version of the database which means
    /// we don't need to care about the version.
    #[cfg(any(test, feature = "assert-reader-validity"))]
    pub fn assert_validity(&self, rtxn: &RoTxn) -> Result<()> {
        // 1. Compare items in db with bitmap from metadata
        use crate::node::NodeCodec;
        let mut item_ids = RoaringBitmap::new();
        for result in self
            .database
            .remap_types::<PrefixCodec, DecodeIgnore>()
            .prefix_iter(rtxn, &Prefix::item(self.index))?
            .remap_key_type::<KeyCodec>()
        {
            let (i, _) = result?;
            item_ids.insert(i.node.unwrap_item());
        }
        assert_eq!(item_ids, self.items);

        // 2. Check links are valid
        let mut link_ids = RoaringBitmap::new();
        for result in self
            .database
            .remap_types::<PrefixCodec, NodeCodec<D>>()
            .prefix_iter(rtxn, &Prefix::links(self.index))?
            .remap_key_type::<KeyCodec>()
        {
            let (k, node) = result?;
            link_ids.insert(k.node.item);

            let Links { links } = match node {
                Node::Links(links) => links,
                Node::Item(_) => unreachable!("Node must not be an item"),
            };

            // this fails if links contains an item_id not in the db
            assert!(links.is_subset(&item_ids));
        }
        // each item should have one or more links
        assert_eq!(item_ids, link_ids);

        // 3. Check entry points
        for ep in self.entry_points.iter() {
            assert!(item_ids.contains(*ep));
        }

        Ok(())
    }
}

pub fn get_item<'a, D: Distance>(
    database: Database<D>,
    index: u16,
    rtxn: &'a RoTxn,
    item: ItemId,
) -> Result<Option<Item<'a, D>>> {
    match database.get(rtxn, &Key::item(index, item))? {
        Some(Node::Item(item)) => Ok(Some(item)),
        Some(Node::Links(_)) => Ok(None),
        None => Ok(None),
    }
}

pub fn get_links<'a, D: Distance>(
    rtxn: &'a RoTxn,
    database: Database<D>,
    index: u16,
    item_id: ItemId,
    level: usize,
) -> Result<Option<Links<'a>>> {
    match database.get(rtxn, &Key::links(index, item_id, level as u8))? {
        Some(Node::Links(links)) => Ok(Some(links)),
        Some(Node::Item(_)) => Ok(None),
        None => Ok(None),
    }
}
