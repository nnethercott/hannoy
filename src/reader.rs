use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};
use std::marker;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};

use heed::types::{Bytes, DecodeIgnore};
use heed::RoTxn;
use madvise::AccessPattern;
use min_max_heap::MinMaxHeap;
use roaring::RoaringBitmap;

use crate::distance::Distance;
use crate::hnsw::ScoredLink;
use crate::internals::KeyCodec;
use crate::item_iter::ItemIter;
use crate::metadata::Metadata;
use crate::node::{Item, ItemIds, Links};
use crate::ordered_float::OrderedFloat;
use crate::unaligned_vector::UnalignedVector;
use crate::version::{Version, VersionCodec};
use crate::{Database, Error, ItemId, Key, MetadataCodec, Node, Prefix, PrefixCodec, Result};

/// A good default value for the `ef` parameter.
const DEFAULT_EF_SEARCH: usize = 100;
const DEFAULT_AVAILABLE_MEMORY: usize = 100 * 1024 * 1024;

#[cfg(not(test))]
/// The threshold at which linear search is used instead of the HNSW algorithm.
const LINEAR_SEARCH_THRESHOLD: u64 = 1000;
#[cfg(test)]
/// Note that for tests purposes, we use set this threshold
/// to zero to make sure we test the HNSW algorithm.
const LINEAR_SEARCH_THRESHOLD: u64 = 0;

/// Options used to make a query against an hannoy [`Reader`].
pub struct QueryBuilder<'a, D: Distance> {
    reader: &'a Reader<'a, D>,
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
        match self.reader.item_vector(rtxn, item)? {
            Some(vector) => self.by_vector(rtxn, &vector).map(Some),
            None => Ok(None),
        }
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
        self.reader.nns_by_vec(rtxn, &item, self)
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

/// A reader over the hannoy hnsw graph
#[derive(Debug)]
pub struct Reader<'t, D: Distance> {
    database: Database<D>,
    index: u16,
    entry_points: ItemIds<'t>,
    max_level: usize,
    dimensions: usize,
    items: RoaringBitmap,
    version: Version,
    _marker: marker::PhantomData<D>,
}

impl<'t, D: Distance> Reader<'t, D> {
    /// Returns a reader over the database with the specified [`Distance`] type.
    pub fn open(rtxn: &'t RoTxn, index: u16, database: Database<D>) -> Result<Reader<'t, D>> {
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
            entry_points: metadata.entry_points,
            max_level: metadata.max_level as usize,
            dimensions: metadata.dimensions.try_into().unwrap(),
            items: metadata.items,
            version,
            _marker: marker::PhantomData,
        })
    }

    fn prefetch_graph(
        rtxn: &RoTxn,
        database: &Database<D>,
        index: u16,
        metadata: &Metadata,
    ) -> Result<()> {
        let page_size = page_size::get();
        let largest_alloc = AtomicUsize::new(0);

        let madvise_page = |item: &[u8]| -> usize {
            let start_ptr = item.as_ptr() as usize;
            let end_ptr = start_ptr + metadata.dimensions as usize;
            let start_page = start_ptr - (start_ptr % page_size);
            let end_page = end_ptr + ((end_ptr + page_size - 1) % page_size);
            let advised_size = end_page - start_page;

            unsafe {
                madvise::madvise(start_page as *const u8, advised_size, AccessPattern::WillNeed)
                    .expect("Advisory failed");
            }

            largest_alloc.fetch_max(advised_size, Ordering::Relaxed);
            advised_size
        };

        // Load links and vectors for layers > 0.
        let mut available_memory = DEFAULT_AVAILABLE_MEMORY;
        let mut added = RoaringBitmap::new();

        for lvl in (1..=metadata.max_level).rev() {
            for result in database.remap_data_type::<Bytes>().iter(&rtxn)? {
                if available_memory < largest_alloc.load(Ordering::Relaxed) {
                    return Ok(());
                }
                let (key, item) = result?;
                if key.node.layer != lvl {
                    continue;
                }
                available_memory -= madvise_page(item);
                added.insert(key.node.item);
            }
        }

        // If we still have memory left over fanout from neighbours of added in layer 0 until we
        // hit the mem threshold.
        let mut queue = VecDeque::from_iter(added.iter());
        while let Some(item) = queue.pop_front() {
            if available_memory < largest_alloc.load(Ordering::Relaxed) {
                return Ok(());
            }
            if let Some(Node::Links(links)) = database.get(&rtxn, &Key::links(index, item, 0))? {
                for l in links.iter() {
                    if !added.insert(l) {
                        continue;
                    }
                    if let Some(bytes) =
                        database.remap_data_type::<Bytes>().get(&rtxn, &Key::item(index, l))?
                    {
                        available_memory -= madvise_page(bytes);
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
    pub fn n_nodes(&self, rtxn: &'t RoTxn) -> Result<Option<NonZeroUsize>> {
        Ok(NonZeroUsize::new(self.database.len(rtxn)? as usize))
    }

    /// Returns the vector for item `i` that was previously added.
    pub fn item_vector(&self, rtxn: &'t RoTxn, item_id: ItemId) -> Result<Option<Vec<f32>>> {
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
    pub fn iter(&self, rtxn: &'t RoTxn) -> Result<ItemIter<'t, D>> {
        ItemIter::new(self.database, self.index, rtxn).map_err(Into::into)
    }

    /// Return a [`QueryBuilder`] that lets you configure and execute a search request.
    ///
    /// You must provide the number of items you want to receive.
    pub fn nns(&self, count: usize) -> QueryBuilder<D> {
        QueryBuilder { reader: self, candidates: None, count, ef: DEFAULT_EF_SEARCH }
    }

    /// Get a generic read node from the database using the version of the database found while creating the reader.
    /// Must be used every time we retrieve a node in this file.
    // FIXME: this is more or less a direct copy of the builder, except we use a single
    // RoTxn instead of a FrozzenReader
    fn explore_layer(
        &self,
        query: &Item<D>,
        eps: &[ItemId],
        level: usize,
        ef: usize,
        rtxn: &RoTxn,
    ) -> Result<MinMaxHeap<ScoredLink>> {
        let mut candidates = BinaryHeap::new();
        let mut res = MinMaxHeap::with_capacity(ef);
        let mut visited = RoaringBitmap::new();

        // Register all entry points as visited and populate candidates
        for &ep in eps {
            let ve = get_item(self.database, self.index, rtxn, ep)?.unwrap();
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
            let proximity = match get_links(rtxn, self.database, self.index, c, level)? {
                Some(Links { links }) => links.iter().collect::<Vec<ItemId>>(),
                None => unreachable!("Links must exist"),
            };
            for point in proximity {
                if !visited.insert(point) {
                    continue;
                }
                let dist =
                    D::distance(query, &get_item(self.database, self.index, rtxn, point)?.unwrap());

                if res.len() < ef || dist < f_max {
                    candidates.push((Reverse(OrderedFloat(dist)), point));

                    // optimized insert & removal maintaining original len
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

    fn nns_by_vec(
        &self,
        rtxn: &'t RoTxn,
        query: &Item<D>,
        opt: &QueryBuilder<D>,
    ) -> Result<Vec<(ItemId, f32)>> {
        // If we will never find any candidates, return an empty vector
        if opt.candidates.is_some_and(|c| self.item_ids().is_disjoint(c)) {
            return Ok(Vec::new());
        }

        // If the number of candidates is less than a given threshold, perform linear search
        if let Some(candidates) = opt.candidates.filter(|c| c.len() < LINEAR_SEARCH_THRESHOLD) {
            let mut item_distances = Vec::with_capacity(candidates.len() as usize);
            for item_id in candidates {
                let Some(vector) = self.item_vector(rtxn, item_id)? else { continue };
                let vector = UnalignedVector::from_vec(vector);
                let item = Item { header: D::new_header(&vector), vector };
                let distance = D::distance(&item, query);
                item_distances.push((item_id, distance));
            }
            item_distances.sort_by_key(|(_, dist)| OrderedFloat(*dist));
            item_distances.truncate(opt.count);
            return Ok(item_distances);
        }

        let mut eps = Vec::from_iter(self.entry_points.iter());

        // search layers L->1 with ef=1
        for lvl in (1..=self.max_level).rev() {
            let neighbours = self.explore_layer(query, &eps, lvl, 1, rtxn)?;
            let closest = neighbours.peek_min().map(|(_, n)| n).expect("No neighbor was found");
            eps = vec![*closest];
        }

        // search layer 0 with ef=max(ef, count)
        let mut neighbours = self.explore_layer(query, &eps, 0, opt.ef, rtxn)?;

        let mut nns = Vec::with_capacity(opt.count);
        while let Some((OrderedFloat(f), id)) = neighbours.pop_min() {
            if opt.candidates.is_none_or(|candidates| candidates.contains(id)) {
                nns.push((id, f));
            }
            if nns.len() == opt.count {
                break;
            }
        }

        Ok(nns)
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
            assert!(item_ids.contains(ep));
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
