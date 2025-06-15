use std::path::PathBuf;

use heed::types::{DecodeIgnore, Unit};
use heed::{RoTxn, RwTxn};
use rand::distributions::Uniform;
use rand::{Rng, SeedableRng};
use roaring::RoaringBitmap;

use crate::distance::Distance;
use crate::internals::KeyCodec;
use crate::item_iter::ItemIter;
use crate::node::{ItemIds, Item};
use crate::parallel::{
    ConcurrentNodeIds, ImmutableItems, ImmutableNodes,
};
use crate::unaligned_vector::UnalignedVector;
use crate::version::{Version, VersionCodec};
use crate::{
    Database, Error, ItemId, Key, Metadata, MetadataCodec, DbItem, Prefix, PrefixCodec,
    Result,
};

/// The options available when building the arroy database.
pub struct HannoyBuilder<'a, D: Distance, R: Rng + SeedableRng> {
    writer: &'a Writer<D>,
    rng: &'a mut R,
    inner: BuildOption,
}

/// The options available when building the arroy database.
pub(crate) struct BuildOption {
    pub(crate) m: usize,
    pub(crate) m0: usize,
    pub(crate) ml: f32,
    pub(crate) ef_c: usize,
    pub(crate) available_memory: Option<usize>,
}

impl Default for BuildOption {
    fn default() -> Self {
        Self {
            m: 16,
            m0: 32,
            ml: 16f32.ln().powf(-1.0),
            ef_c: 6, // ad hoc
            available_memory: None,
        }
    }
}

impl<'a, D: Distance, R: Rng + SeedableRng> HannoyBuilder<'a, D, R> {
    pub fn available_memory(&mut self, memory: usize) -> &mut Self {
        self.inner.available_memory = Some(memory);
        self
    }

    pub fn build(&mut self, wtxn: &mut RwTxn) -> Result<()> {
        self.writer.build(wtxn, self.rng, &self.inner)
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
        let database: Database<D> = database.remap_data_type();
        Writer { database, index, dimensions, tmpdir: None }
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
        self.database.put(wtxn, &Key::item(self.index, item), &DbItem::Item(db_item))?;
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

    /// Removes everything in the database, user items and internal tree nodes.
    pub fn clear(&self, wtxn: &mut RwTxn) -> Result<()> {
        let mut cursor = self
            .database
            .remap_key_type::<PrefixCodec>()
            .prefix_iter_mut(wtxn, &Prefix::all(self.index))?
            .remap_types::<DecodeIgnore, DecodeIgnore>();

        while let Some((_id, _node)) = cursor.next().transpose()? {
            // safety: we don't have any reference to the database
            unsafe { cursor.del_current() }?;
        }

        Ok(())
    }

    fn used_nodes(&self, rtxn: &RoTxn, options: &BuildOption) -> Result<RoaringBitmap> {
        Ok(self
            .database
            .remap_key_type::<PrefixCodec>()
            .prefix_iter(rtxn, &Prefix::node(self.index))?
            .remap_types::<KeyCodec, DecodeIgnore>()
            .try_fold(RoaringBitmap::new(), |mut bitmap, used| -> Result<RoaringBitmap> {
                bitmap.insert(used?.0.node.item);
                Ok(bitmap)
            })
            .unwrap_or_default())
    }

    /// Returns an [`ArroyBuilder`] to configure the available options to build the database.
    pub fn builder<'a, R: Rng + SeedableRng>(&'a self, rng: &'a mut R) -> HannoyBuilder<'a, D, R> {
        HannoyBuilder { writer: self, rng, inner: BuildOption::default() }
    }

    fn build<R: Rng + SeedableRng>(
        &self,
        wtxn: &mut RwTxn,
        rng: &mut R,
        options: &BuildOption,
    ) -> Result<()> {
        let item_indices = self.item_indices(wtxn, options)?;
        let n_items = item_indices.len();
        // updated items can be an update, an addition or a removed item
        let updated_items = self.reset_and_retrieve_updated_items(wtxn, options)?;

        let _to_delete = updated_items.clone();
        let to_insert = &item_indices & &updated_items;

        let metadata = self
            .database
            .remap_data_type::<MetadataCodec>()
            .get(wtxn, &Key::metadata(self.index))?;
        let entry_points =
            metadata.as_ref().map_or_else(Vec::new, |metadata| metadata.entry_points.iter().collect());
        // we should not keep a reference to the metadata since they're going to be moved by LMDB
        drop(metadata);

        tracing::debug!("Getting a reference to your {n_items} items...");

        let used_node_ids = self.used_nodes(wtxn, options)?;
        let concurrent_node_ids = ConcurrentNodeIds::new(used_node_ids);

        self.index_hnsw(
            wtxn,
            rng,
            options,
            concurrent_node_ids,
            to_insert,
        )?;

        tracing::debug!("write the metadata...");
        let metadata = Metadata {
            dimensions: self.dimensions.try_into().unwrap(),
            items: item_indices,
            entry_points: ItemIds::from_slice(&entry_points),
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

    fn index_hnsw<R: Rng + SeedableRng>(
        &self,
        wtxn: &mut RwTxn,
        rng: &mut R,
        options: &BuildOption,
        concurrent_node_ids: ConcurrentNodeIds,
        to_insert: RoaringBitmap,
    ) -> Result<(), Error> {

        for item in to_insert{
            // sample from an exponential ~ exp(ml)
            let u = rng.sample(Uniform::new(0.0f32, 1.0));
            let l = (-1.0*u.ln()/options.ml).floor() as usize;

            // insert_item_in_index(item, l, wtxn, options);
        }

        Ok(())
    }


    fn reset_and_retrieve_updated_items(
        &self,
        wtxn: &mut RwTxn,
        options: &BuildOption,
    ) -> Result<RoaringBitmap, Error> {
        tracing::debug!("reset and retrieve the updated items...");
        let mut updated_items = RoaringBitmap::new();
        let mut updated_iter = self
            .database
            .remap_types::<PrefixCodec, DecodeIgnore>()
            .prefix_iter_mut(wtxn, &Prefix::updated(self.index))?
            .remap_key_type::<KeyCodec>();
        while let Some((key, _)) = updated_iter.next().transpose()? {
            let inserted = updated_items.push(key.node.item);
            debug_assert!(inserted, "The keys should be sorted by LMDB");
            // Safe because we don't hold any reference to the database currently
            unsafe {
                updated_iter.del_current()?;
            }
        }
        Ok(updated_items)
    }

    // Fetches the item's ids, not the tree nodes ones.
    fn item_indices(&self, wtxn: &mut RwTxn, options: &BuildOption) -> Result<RoaringBitmap> {
        tracing::debug!("started retrieving all the items ids...");

        let mut indices = RoaringBitmap::new();
        for result in self
            .database
            .remap_types::<PrefixCodec, DecodeIgnore>()
            .prefix_iter(wtxn, &Prefix::item(self.index))?
            .remap_key_type::<KeyCodec>()
        {
            let (i, _) = result?;
            indices.push(i.node.unwrap_item());
        }

        Ok(indices)
    }
}

#[derive(Clone)]
struct FrozzenReader<'a, D: Distance> {
    items: &'a ImmutableItems<'a, D>,
    nodes: &'a ImmutableNodes<'a, D>,
    concurrent_node_ids: &'a ConcurrentNodeIds,
}

fn clear_nodes<D: Distance>(
    wtxn: &mut RwTxn,
    database: Database<D>,
    index: u16,
) -> Result<()> {
    database.delete(wtxn, &Key::metadata(index))?;
    let mut cursor = database
        .remap_types::<PrefixCodec, DecodeIgnore>()
        .prefix_iter_mut(wtxn, &Prefix::node(index))?
        .remap_key_type::<DecodeIgnore>();
    while let Some((_id, _node)) = cursor.next().transpose()? {
        // safety: we keep no reference into the database between operations
        unsafe { cursor.del_current()? };
    }

    Ok(())
}
