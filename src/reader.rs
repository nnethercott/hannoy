use heed::types::DecodeIgnore;
use heed::RoTxn;
use min_max_heap::MinMaxHeap;
use roaring::RoaringBitmap;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::marker;
use std::num::NonZeroUsize;

use crate::distance::Distance;
use crate::hnsw::ScoredLink;
use crate::internals::KeyCodec;
use crate::item_iter::ItemIter;
use crate::node::{Item, ItemIds, Links};
use crate::ordered_float::OrderedFloat;
use crate::unaligned_vector::UnalignedVector;
use crate::version::{Version, VersionCodec};
use crate::{Database, Error, ItemId, Key, MetadataCodec, Node, Prefix, PrefixCodec, Result};

/// Options used to make a query against an arroy [`Reader`].
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
    /// # use arroy::{Reader, distances::Euclidean};
    /// # let (reader, rtxn): (Reader<Euclidean>, heed::RoTxn) = todo!();
    /// reader.nns(20).by_item(&rtxn, 5);
    /// ```
    pub fn by_item(&self, _rtxn: &RoTxn, _item: ItemId) -> Result<Option<Vec<(ItemId, f32)>>> {
        todo!()
    }

    /// Returns the closest items from the provided `vector`.
    ///
    /// See also [`Self::by_item`].
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use arroy::{Reader, distances::Euclidean};
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
    /// # use arroy::{Reader, distances::Euclidean};
    /// # let (reader, rtxn): (Reader<Euclidean>, heed::RoTxn) = todo!();
    /// let candidates = roaring::RoaringBitmap::from_iter([1, 3, 4, 5, 6, 7, 8, 9, 15, 16]);
    /// reader.nns(20).candidates(&candidates).by_item(&rtxn, 6);
    /// ```
    pub fn candidates(&mut self, candidates: &'a RoaringBitmap) -> &mut Self {
        self.candidates = Some(candidates);
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

    /// Returns the number of dimensions in the index.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Returns the number of trees in the index.
    pub fn n_trees(&self) -> usize {
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
        Ok(ItemIter {
            inner: self
                .database
                .remap_key_type::<PrefixCodec>()
                .prefix_iter(rtxn, &Prefix::item(self.index))?
                .remap_key_type::<KeyCodec>(),
        })
    }

    /// Return a [`QueryBuilder`] that lets you configure and execute a search request.
    ///
    /// You must provide the number of items you want to receive.
    pub fn nns(&self, count: usize, ef: usize) -> QueryBuilder<D> {
        QueryBuilder { reader: self, candidates: None, count, ef }
    }

    /// Get a generic read node from the database using the version of the database found while creating the reader.
    /// Must be used every time we retrieve a node in this file.
    fn database_get(&self, rtxn: &'t RoTxn, key: &Key) -> Result<Option<Node<D>>> {
        // NOTE: if we ever get more versions we'll have to update this like arroy
        Ok(self.database.get(rtxn, key)?)
    }

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
            visited.push(ep);
        }
        while let Some(&(Reverse(OrderedFloat(f)), c)) = candidates.peek() {
            let &(OrderedFloat(f_max), _) = res.peek_max().unwrap();
            if f > f_max {
                break;
            }
            let (_, c) = candidates.pop().unwrap(); // Now safe to pop

            // Get neighborhood of candidate either from self or LMDB
            let proximity = match get_links(rtxn, self.database, self.index, c, level)? {
                Some(Links { links }) => links.iter().collect::<Vec<ItemId>>(),
                None => unreachable!(),
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
                        let _ = res.push((OrderedFloat(dist), point));
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
        let mut eps = Vec::from_iter(self.entry_points.iter());

        // search layers L->1 with ef=1
        for lvl in (1..=self.max_level).rev() {
            let neighbours = self.explore_layer(&query, &eps, lvl, 1, rtxn)?;
            let closest = neighbours.peek_min().map(|(_, n)| n).expect("No neighbor was found");
            eps = vec![*closest];
        }

        // search layer 0 with ef=ef
        let mut neighbours = self.explore_layer(&query, &eps, 0, opt.ef, rtxn)?;

        let mut nns = Vec::with_capacity(opt.count);
        while let Some((OrderedFloat(f), id)) = neighbours.pop_min() {
            if opt.candidates.is_some_and(|candidates| candidates.contains(id)) {
                nns.push((id, f));
            }
            if nns.len() == opt.count {
                break;
            }
        }

        Ok(nns)
    }

    fn nns_by_item(
        &self,
        rtxn: &'t RoTxn,
        query: ItemId,
        opt: &QueryBuilder<D>,
    ) -> Result<Vec<(ItemId, f32)>> {
        todo!()
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
        Some(Node::Item(item)) => Ok(None),
        None => Ok(None),
    }
}
