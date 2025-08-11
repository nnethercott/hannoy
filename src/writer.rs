use std::any::TypeId;
use std::path::PathBuf;

use heed::types::{DecodeIgnore, Unit};
use heed::{PutFlags, RoTxn, RwTxn};
use rand::{Rng, SeedableRng};
use roaring::RoaringBitmap;
use steppe::default::DefaultProgress;

use crate::distance::Distance;
use crate::hnsw::HnswBuilder;
use crate::internals::KeyCodec;
use crate::item_iter::ItemIter;
use crate::node::{Item, ItemIds, Links, NodeCodec};
use crate::node_id::{NodeId, NodeMode};
use crate::parallel::{ImmutableItems, ImmutableLinks};
use crate::progress::HannoyBuild;
use crate::reader::get_item;
use crate::unaligned_vector::UnalignedVector;
use crate::version::{Version, VersionCodec};
use crate::{
    Database, Error, ItemId, Key, Metadata, MetadataCodec, Node, Prefix, PrefixCodec, Result,
    CANCELLATION_PROBING,
};

/// The options available when building the arroy database.
pub struct HannoyBuilder<'a, D: Distance, R: Rng + SeedableRng> {
    writer: &'a Writer<D>,
    rng: &'a mut R,
    inner: BuildOption<'a>,
}

/// The options available when building the arroy database.
pub(crate) struct BuildOption<'a> {
    pub(crate) ef_construction: usize,
    pub(crate) available_memory: Option<usize>,
    pub(crate) cancel: Box<dyn Fn() -> bool + 'a + Sync + Send>,
    pub(crate) progress: DefaultProgress,
}

impl Default for BuildOption<'_> {
    fn default() -> Self {
        Self {
            ef_construction: 100,
            available_memory: None,
            cancel: Box::new(|| false),
            progress: DefaultProgress::default(),
        }
    }
}

impl<'a, D: Distance, R: Rng + SeedableRng> HannoyBuilder<'a, D, R> {
    pub fn available_memory(&mut self, memory: usize) -> &mut Self {
        self.inner.available_memory = Some(memory);
        self
    }

    /// Provide a closure that can cancel the indexing process early if needed. There is no guarantee on when the process is going to cancel itself, but hannoy will try to stop as soon as possible once the closure returns true.
    ///
    /// Since the closure is not mutable and will be called from multiple threads at the same time it’s encouraged to make it quick to execute. A common way to use it is to fetch an AtomicBool inside it that can be set from another thread without lock.
    pub fn cancel(&mut self, cancel: impl Fn() -> bool + 'a + Sync + Send) -> &mut Self {
        self.inner.cancel = Box::new(cancel);
        self
    }

    pub fn ef_construction(&mut self, ef_construction: usize) -> &mut Self {
        self.inner.ef_construction = ef_construction;
        self
    }

    pub fn build<const M: usize, const M0: usize>(&mut self, wtxn: &mut RwTxn) -> Result<()> {
        self.writer.build::<R, M, M0>(wtxn, self.rng, &self.inner)
    }
}

/// A writer to store new items, remove existing ones,
/// and build the search index to query the nearest
/// neighbors to items or vectors.
#[derive(Debug)]
pub struct Writer<D: Distance> {
    database: Database<D>,
    index: u16,
    dimensions: usize,
    /// The folder in which tempfile will write its temporary files.
    tmpdir: Option<PathBuf>,
}

impl<D: Distance> Writer<D> {
    /// Creates a new writer from a database, index and dimensions.
    pub fn new(database: Database<D>, index: u16, dimensions: usize) -> Writer<D> {
        Writer { database, index, dimensions, tmpdir: None }
    }

    /// After opening an arroy database this function will prepare it for conversion,
    /// cleanup the arroy database and only keep the items/vectors entries.
    pub fn prepare_arroy_conversion(&self, wtxn: &mut RwTxn) -> Result<()> {
        let mut iter = self
            .database
            .remap_key_type::<PrefixCodec>()
            .prefix_iter_mut(wtxn, &Prefix::all(self.index))?
            .remap_key_type::<KeyCodec>();

        let mut new_items = RoaringBitmap::new();
        while let Some(result) = iter.next() {
            match result {
                Ok((
                    Key { index: _, node: NodeId { mode: NodeMode::Item, item, .. } },
                    Node::Item(Item { header: _, vector }),
                )) => {
                    // We only take care of the entries that can be decoded as Node Items (vectors) and
                    // mark them as newly inserted so the Writer::build method can compute the links for them.
                    new_items.insert(item);
                    if vector.len() != self.dimensions {
                        return Err(Error::InvalidVecDimension {
                            expected: self.dimensions,
                            received: vector.len(),
                        });
                    }
                }
                Ok((Key { .. }, _)) | Err(heed::Error::Decoding(_)) => unsafe {
                    // Every other entry that fails to decode can be considered as something
                    // else than an item, is useless for the conversion and is deleted.
                    // SAFETY: Safe because we don't keep any references to the entry
                    iter.del_current()?;
                },
                // If there is another error (lmdb...), it is returned.
                Err(e) => return Err(e.into()),
            }
        }

        drop(iter);

        // We mark all the items as updated so
        // the Writer::build method can handle them.
        for item in new_items {
            self.database.remap_data_type::<Unit>().put(
                wtxn,
                &Key::updated(self.index, item),
                &(),
            )?;
        }

        Ok(())
    }

    /// Returns a writer after having deleted the tree nodes and rewrote all the items
    /// for the new [`Distance`] format to be able to modify items safely.
    pub fn prepare_changing_distance<ND: Distance>(self, wtxn: &mut RwTxn) -> Result<Writer<ND>> {
        if TypeId::of::<ND>() != TypeId::of::<D>() {
            // If we are moving from a distance to the same but binary quantized
            // distance we do not need to clear links, otherwise we do.
            if ND::name()
                .strip_prefix("binary quantized ")
                .is_none_or(|raw_name| raw_name != D::name())
            {
                clear_links(wtxn, self.database, self.index)?;
            }

            let mut cursor = self
                .database
                .remap_key_type::<PrefixCodec>()
                .prefix_iter_mut(wtxn, &Prefix::item(self.index))?
                .remap_key_type::<KeyCodec>();

            while let Some((item_id, node)) = cursor.next().transpose()? {
                match node {
                    Node::Item(Item { header: _, vector }) => {
                        let vector = vector.to_vec();
                        let vector = UnalignedVector::from_vec(vector);
                        let new_leaf = Node::Item(Item { header: ND::new_header(&vector), vector });
                        unsafe {
                            // safety: We do not keep a reference to the current value, we own it.
                            cursor.put_current_with_options::<NodeCodec<ND>>(
                                PutFlags::empty(),
                                &item_id,
                                &new_leaf,
                            )?
                        };
                    }
                    Node::Links(_) => unreachable!("Node must not be a link"),
                }
            }
        }

        let Writer { database, index, dimensions, tmpdir } = self;
        Ok(Writer { database: database.remap_data_type(), index, dimensions, tmpdir })
    }

    pub fn set_tmpdir(&mut self, path: impl Into<PathBuf>) {
        self.tmpdir = Some(path.into());
    }

    /// Returns `true` if the index is empty.
    pub fn is_empty(&self, rtxn: &RoTxn) -> Result<bool> {
        self.iter(rtxn).map(|mut iter| iter.next().is_none())
    }

    /// Returns `true` if the index needs to be built before being able to read in it.
    pub fn need_build(&self, rtxn: &RoTxn) -> Result<bool> {
        Ok(self
            .database
            .remap_types::<PrefixCodec, DecodeIgnore>()
            .prefix_iter(rtxn, &Prefix::updated(self.index))?
            .remap_key_type::<KeyCodec>()
            .next()
            .is_some()
            || self
                .database
                .remap_data_type::<DecodeIgnore>()
                .get(rtxn, &Key::metadata(self.index))?
                .is_none())
    }

    /// Returns an `Option`al vector previous stored in this database.
    pub fn item_vector(&self, rtxn: &RoTxn, item: ItemId) -> Result<Option<Vec<f32>>> {
        Ok(get_item(self.database, self.index, rtxn, item)?.map(|leaf| {
            let mut vec = leaf.vector.to_vec();
            vec.truncate(self.dimensions);
            vec
        }))
    }

    /// Returns `true` if the database contains the given item.
    pub fn contains_item(&self, rtxn: &RoTxn, item: ItemId) -> Result<bool> {
        self.database
            .remap_data_type::<DecodeIgnore>()
            .get(rtxn, &Key::item(self.index, item))
            .map(|opt| opt.is_some())
            .map_err(Into::into)
    }

    /// Returns an iterator over the items vector.
    pub fn iter<'t>(&self, rtxn: &'t RoTxn) -> Result<ItemIter<'t, D>> {
        Ok(ItemIter {
            inner: self
                .database
                .remap_key_type::<PrefixCodec>()
                .prefix_iter(rtxn, &Prefix::item(self.index))?
                .remap_key_type::<KeyCodec>(),
        })
    }

    /// Add an item associated to a vector in the database.
    pub fn add_item(&self, wtxn: &mut RwTxn, item: ItemId, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dimensions {
            return Err(Error::InvalidVecDimension {
                expected: self.dimensions,
                received: vector.len(),
            });
        }

        let vector = UnalignedVector::from_slice(vector);
        let db_item = Item { header: D::new_header(&vector), vector };
        self.database.put(wtxn, &Key::item(self.index, item), &Node::Item(db_item))?;
        self.database.remap_data_type::<Unit>().put(wtxn, &Key::updated(self.index, item), &())?;

        Ok(())
    }

    /// Deletes an item stored in this database and returns `true` if it existed.
    pub fn del_item(&self, wtxn: &mut RwTxn, item: ItemId) -> Result<bool> {
        if self.database.delete(wtxn, &Key::item(self.index, item))? {
            self.database.remap_data_type::<Unit>().put(
                wtxn,
                &Key::updated(self.index, item),
                &(),
            )?;

            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Removes everything in the database, user items and internal graph links.
    pub fn clear(&self, wtxn: &mut RwTxn) -> Result<()> {
        let mut cursor = self
            .database
            .remap_key_type::<PrefixCodec>()
            .prefix_iter_mut(wtxn, &Prefix::all(self.index))?
            .remap_types::<DecodeIgnore, DecodeIgnore>();

        while let Some((_id, _node)) = cursor.next().transpose()? {
            // SAFETY: Safe because we don't keep any references to the entry
            unsafe { cursor.del_current() }?;
        }

        Ok(())
    }

    /// Returns an [`HannoyBuilder`] to configure the available options to build the database.
    pub fn builder<'a, R: Rng + SeedableRng>(&'a self, rng: &'a mut R) -> HannoyBuilder<'a, D, R> {
        HannoyBuilder { writer: self, rng, inner: BuildOption::default() }
    }

    fn build<R: Rng + SeedableRng, const M: usize, const M0: usize>(
        &self,
        wtxn: &mut RwTxn,
        rng: &mut R,
        options: &BuildOption,
    ) -> Result<()> {
        let item_indices = self.item_indices(wtxn, options)?;
        // updated items can be an update, an addition or a removed item
        let updated_items = self.reset_and_retrieve_updated_items(wtxn, options)?;

        let to_delete = updated_items.clone() - &item_indices;
        let to_insert = &item_indices & &updated_items;

        let metadata = self
            .database
            .remap_data_type::<MetadataCodec>()
            .get(wtxn, &Key::metadata(self.index))?;

        let (entry_points, max_level) = metadata.as_ref().map_or_else(
            || (Vec::new(), usize::MIN),
            |metadata| (metadata.entry_points.iter().collect(), metadata.max_level as usize),
        );

        // we should not keep a reference to the metadata since they're going to be moved by LMDB
        drop(metadata);

        let mut hnsw = HnswBuilder::<D, M, M0>::new(options)
            .with_entry_points(entry_points)
            .with_max_level(max_level);

        let stats =
            hnsw.build(to_insert, &to_delete, self.database, self.index, wtxn, rng, options)?;
        tracing::info!("{stats:?}");

        // Remove deleted links from lmdb AFTER build; in DiskANN we use a deleted item's
        // neighbours when filling in the "gaps" left in the graph from deletions. See
        // [`HnswBuilder::maybe_patch_old_links`] for more details.
        self.delete_links_from_db(to_delete, wtxn)?;

        tracing::debug!("write the metadata...");
        options.progress.update(HannoyBuild::WriteTheMetadata);

        let metadata = Metadata {
            dimensions: self.dimensions.try_into().unwrap(),
            items: item_indices,
            entry_points: ItemIds::from_slice(&hnsw.entry_points),
            max_level: hnsw.max_level as u8,
            distance: D::name(),
        };
        self.database.remap_data_type::<MetadataCodec>().put(
            wtxn,
            &Key::metadata(self.index),
            &metadata,
        )?;
        self.database.remap_data_type::<VersionCodec>().put(
            wtxn,
            &Key::version(self.index),
            &Version::current(),
        )?;

        Ok(())
    }

    fn reset_and_retrieve_updated_items(
        &self,
        wtxn: &mut RwTxn,
        options: &BuildOption,
    ) -> Result<RoaringBitmap, Error> {
        tracing::debug!("reset and retrieve the updated items...");
        options.progress.update(HannoyBuild::RetrieveTheUpdatedItems);

        let mut updated_items = RoaringBitmap::new();
        let mut updated_iter = self
            .database
            .remap_types::<PrefixCodec, DecodeIgnore>()
            .prefix_iter_mut(wtxn, &Prefix::updated(self.index))?
            .remap_key_type::<KeyCodec>();

        let mut index = 0;
        while let Some((key, _)) = updated_iter.next().transpose()? {
            if index % CANCELLATION_PROBING == 0 && (options.cancel)() {
                return Err(Error::BuildCancelled);
            }

            let inserted = updated_items.insert(key.node.item);
            debug_assert!(inserted, "The keys should be sorted by LMDB");
            // SAFETY: Safe because we don't hold any reference to the database currently
            unsafe { updated_iter.del_current()? };

            index += 1;
        }
        Ok(updated_items)
    }

    // Fetches the item's ids, not the tree nodes ones.
    fn item_indices(&self, wtxn: &mut RwTxn, options: &BuildOption) -> Result<RoaringBitmap> {
        tracing::debug!("started retrieving all the items ids...");
        options.progress.update(HannoyBuild::RetrievingTheItemsIds);

        let mut indices = RoaringBitmap::new();
        for (index, result) in self
            .database
            .remap_types::<PrefixCodec, DecodeIgnore>()
            .prefix_iter(wtxn, &Prefix::item(self.index))?
            .remap_key_type::<KeyCodec>()
            .enumerate()
        {
            if index % CANCELLATION_PROBING == 0 && (options.cancel)() {
                return Err(Error::BuildCancelled);
            }

            let (i, _) = result?;
            indices.insert(i.node.unwrap_item());
        }

        Ok(indices)
    }

    // Iterates over links in lmdb and deletes those in `to_delete`. There can be several links
    // with the same NodeId.item, each differing by their layer
    fn delete_links_from_db(&self, to_delete: RoaringBitmap, wtxn: &mut RwTxn) -> Result<()> {
        let mut cursor = self
            .database
            .remap_key_type::<PrefixCodec>()
            .prefix_iter_mut(wtxn, &Prefix::links(self.index))?
            .remap_types::<KeyCodec, DecodeIgnore>();

        while let Some((key, _)) = cursor.next().transpose()? {
            if to_delete.contains(key.node.item) {
                // SAFETY: Safe because we don't keep any references to the entry
                unsafe { cursor.del_current() }?;
            }
        }

        Ok(())
    }
}

#[derive(Clone)]
pub(crate) struct FrozenReader<'a, D: Distance> {
    pub index: u16,
    pub items: &'a ImmutableItems<'a, D>,
    pub links: &'a ImmutableLinks<'a, D>,
}

impl<'a, D: Distance> FrozenReader<'a, D> {
    pub fn get_item(&self, item_id: ItemId) -> Result<Item<'a, D>> {
        let key = Key::item(self.index, item_id);
        // key is a `Key::item` so returned result must be a Node::Item
        self.items.get(item_id)?.ok_or(Error::missing_key(key))
    }

    pub fn get_links(&self, item_id: ItemId, level: usize) -> Result<Links<'a>> {
        let key = Key::links(self.index, item_id, level as u8);
        // key is a `Key::item` so returned result must be a Node::Item
        self.links.get(item_id, level as u8)?.ok_or(Error::missing_key(key))
    }
}

/// Clears all the links. Starts from the last node and stops at the first leaf.
fn clear_links<D: Distance>(wtxn: &mut RwTxn, database: Database<D>, index: u16) -> Result<()> {
    database.delete(wtxn, &Key::metadata(index))?;
    let mut cursor = database
        .remap_types::<PrefixCodec, DecodeIgnore>()
        .prefix_iter_mut(wtxn, &Prefix::links(index))?
        .remap_key_type::<DecodeIgnore>();

    while let Some((_id, _node)) = cursor.next().transpose()? {
        // SAFETY: Safe because we don't keep any references to the entry
        unsafe { cursor.del_current()? };
    }

    Ok(())
}
