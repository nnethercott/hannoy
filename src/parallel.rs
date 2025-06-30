use core::slice;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::marker;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

use heed::types::Bytes;
use heed::{BytesDecode, BytesEncode, RoTxn};
use memmap2::Mmap;
use nohash::{BuildNoHashHasher, IntMap, IntSet};
use roaring::RoaringBitmap;

use crate::internals::{HnswNodeCodec, KeyCodec, Node};
use crate::key::{Key, Prefix, PrefixCodec};
use crate::node::{DbItem, NodeCodec};
use crate::{Database, Distance, Error, ItemId, Result};

/// A structure to store the tree nodes out of the heed database.
pub struct TmpNodes<DE> {
    file: BufWriter<File>,
    ids: Vec<ItemId>,
    bounds: Vec<usize>,
    deleted: RoaringBitmap,
    remap_ids: IntMap<ItemId, ItemId>,
    _marker: marker::PhantomData<DE>,
}

impl<'a, DE: BytesEncode<'a>> TmpNodes<DE> {
    /// Creates an empty `TmpNodes`.
    pub fn new() -> heed::Result<TmpNodes<DE>> {
        Ok(TmpNodes {
            file: tempfile::tempfile().map(BufWriter::new)?,
            ids: Vec::new(),
            bounds: vec![0],
            deleted: RoaringBitmap::new(),
            remap_ids: IntMap::default(),
            _marker: marker::PhantomData,
        })
    }

    /// Creates an empty `TmpNodes` in the defined folder.
    pub fn new_in(path: &Path) -> heed::Result<TmpNodes<DE>> {
        Ok(TmpNodes {
            file: tempfile::tempfile_in(path).map(BufWriter::new)?,
            ids: Vec::new(),
            bounds: vec![0],
            deleted: RoaringBitmap::new(),
            remap_ids: IntMap::default(),
            _marker: marker::PhantomData,
        })
    }

    /// Add a new node in the file.
    /// Items do not need to be ordered.
    pub fn put(
        // TODO move that in the type
        &mut self,
        item: ItemId,
        data: &'a DE::EItem,
    ) -> heed::Result<()> {
        assert!(item != ItemId::MAX);
        let bytes = DE::bytes_encode(data).map_err(heed::Error::Encoding)?;
        self.file.write_all(&bytes)?;
        let last_bound = self.bounds.last().unwrap();
        self.bounds.push(last_bound + bytes.len());
        self.ids.push(item);

        // in the current algorithm, we should never insert a node that was deleted before
        debug_assert!(!self.deleted.contains(item));

        Ok(())
    }

    /// Remap the item id of an already inserted node to another node.
    ///
    /// Only applies to the nodes to insert. It won't interact with the to_delete nodes.
    pub fn remap(&mut self, current: ItemId, new: ItemId) {
        if current != new {
            self.remap_ids.insert(current, new);
        }
    }

    /// Delete the tmp_nodes and the node in the database.
    pub fn remove(&mut self, item: ItemId) {
        let deleted = self.deleted.insert(item);
        debug_assert!(deleted, "Removed the same item with id {item} twice");
    }

    /// Converts it into a readers to read the nodes.
    pub fn into_bytes_reader(self) -> Result<TmpNodesReader> {
        let file = self.file.into_inner().map_err(|iie| iie.into_error())?;
        // safety: No one should move our files around
        let mmap = unsafe { Mmap::map(&file)? };
        #[cfg(unix)]
        mmap.advise(memmap2::Advice::Sequential)?;
        Ok(TmpNodesReader {
            mmap,
            ids: self.ids,
            bounds: self.bounds,
            deleted: self.deleted,
            remap_ids: self.remap_ids,
        })
    }
}

/// A reader of nodes stored in a file.
pub struct TmpNodesReader {
    mmap: Mmap,
    ids: Vec<ItemId>,
    bounds: Vec<usize>,
    deleted: RoaringBitmap,
    remap_ids: IntMap<ItemId, ItemId>,
}

impl TmpNodesReader {
    pub fn to_delete(&self) -> impl Iterator<Item = ItemId> + '_ {
        self.deleted.iter()
    }

    /// Returns an forward iterator over the nodes.
    pub fn to_insert(&self) -> impl Iterator<Item = (ItemId, &[u8])> {
        self.ids
            .iter()
            .zip(self.bounds.windows(2))
            .filter(|(&id, _)| !self.deleted.contains(id))
            .map(|(id, bounds)| match self.remap_ids.get(id) {
                Some(new_id) => (new_id, bounds),
                None => (id, bounds),
            })
            .map(|(id, bounds)| {
                let [start, end] = [bounds[0], bounds[1]];
                (*id, &self.mmap[start..end])
            })
    }
}

/// A concurrent ID generate that will never return the same ID twice.
#[derive(Debug)]
pub struct ConcurrentNodeIds {
    /// The current tree node ID we should use if there is no other IDs available.
    current: AtomicU32,
    /// The total number of tree node IDs used.
    used: AtomicU64,

    /// A list of IDs to exhaust before picking IDs from `current`.
    available: RoaringBitmap,
    /// The current Nth ID to select in the bitmap.
    select_in_bitmap: AtomicU32,
    /// Tells if you should look in the roaring bitmap or if all the IDs are already exhausted.
    look_into_bitmap: AtomicBool,
}

impl ConcurrentNodeIds {
    /// Creates an ID generator returning unique IDs, avoiding the specified used IDs.
    pub fn new(used: RoaringBitmap) -> ConcurrentNodeIds {
        let last_id = used.max().map_or(0, |id| id + 1);
        let used_ids = used.len();
        let available = RoaringBitmap::from_sorted_iter(0..last_id).unwrap() - used;

        ConcurrentNodeIds {
            current: AtomicU32::new(last_id),
            used: AtomicU64::new(used_ids),
            select_in_bitmap: AtomicU32::new(0),
            look_into_bitmap: AtomicBool::new(!available.is_empty()),
            available,
        }
    }

    /// Returns a new unique ID and increase the count of IDs used.
    pub fn next(&self) -> Result<u32> {
        if self.used.fetch_add(1, Ordering::Relaxed) > u32::MAX as u64 {
            Err(Error::DatabaseFull)
        } else if self.look_into_bitmap.load(Ordering::Relaxed) {
            let current = self.select_in_bitmap.fetch_add(1, Ordering::Relaxed);
            match self.available.select(current) {
                Some(id) => Ok(id),
                None => {
                    self.look_into_bitmap.store(false, Ordering::Relaxed);
                    Ok(self.current.fetch_add(1, Ordering::Relaxed))
                }
            }
        } else {
            Ok(self.current.fetch_add(1, Ordering::Relaxed))
        }
    }
}

/// A struture used to keep a list of the leaf nodes in the tree.
///
/// It is safe to share between threads as the pointer are pointing
/// in the mmapped file and the transaction is kept here and therefore
/// no longer touches the database.
pub struct ImmutableItems<'t, D> {
    items: IntMap<ItemId, *const u8>,
    constant_length: Option<usize>,
    _marker: marker::PhantomData<(&'t (), D)>,
}

impl<'t, D: Distance> ImmutableItems<'t, D> {
    /// Creates the structure by fetching all the leaf pointers
    /// and keeping the transaction making the pointers valid.
    /// Do not take more items than memory allows.
    /// Remove from the list of candidates all the items that were selected and return them.
    pub fn new(
        rtxn: &'t RoTxn,
        database: Database<D>,
        index: u16,
        candidates: &mut RoaringBitmap,
        memory: usize,
    ) -> heed::Result<(Self, RoaringBitmap)> {
        let page_size = page_size::get();
        let nb_page_allowed = (memory as f64 / page_size as f64).floor() as usize;

        let mut items = IntMap::with_capacity_and_hasher(
            nb_page_allowed.min(candidates.len() as usize), // We cannot approximate the capacity better because we don't know yet the size of an item
            BuildNoHashHasher::default(),
        );
        let mut pages_used = IntSet::with_capacity_and_hasher(
            nb_page_allowed.min(candidates.len() as usize),
            BuildNoHashHasher::default(),
        );
        let mut selected_items = RoaringBitmap::new();
        let mut constant_length = None;

        while let Some(item_id) = candidates.select(0) {
            let bytes =
                database.remap_data_type::<Bytes>().get(rtxn, &Key::item(index, item_id))?.unwrap();
            assert_eq!(*constant_length.get_or_insert(bytes.len()), bytes.len());

            let ptr = bytes.as_ptr();
            let addr = ptr as usize;
            let start = addr / page_size;
            let end = (addr + bytes.len()) / page_size;

            pages_used.insert(start);
            if start != end {
                pages_used.insert(end);
            }

            if pages_used.len() >= nb_page_allowed && items.len() >= 200 {
                break;
            }

            // Safe because the items comes from another roaring bitmap
            selected_items.push(item_id);
            candidates.remove_smallest(1);
            items.insert(item_id, ptr);
        }

        Ok((
            ImmutableItems { items, constant_length, _marker: marker::PhantomData },
            selected_items,
        ))
    }

    /// Returns the leafs identified by the given ID.
    pub fn get(&self, item_id: ItemId) -> heed::Result<Option<Node<'t, D>>> {
        let len = match self.constant_length {
            Some(len) => len,
            None => return Ok(None),
        };
        let ptr = match self.items.get(&item_id) {
            Some(ptr) => *ptr,
            None => return Ok(None),
        };

        // safety:
        // - ptr: The pointer comes from LMDB. Since the database cannot be written to, it is still valid.
        // - len: All the items share the same dimensions and are the same size
        let bytes = unsafe { slice::from_raw_parts(ptr, len) };
        NodeCodec::bytes_decode(bytes).map_err(heed::Error::Decoding).map(|node| Some(node))
    }
}

unsafe impl<D> Sync for ImmutableItems<'_, D> {}

/// A subset of leafs that are accessible for read.
pub struct ImmutableSubsetItems<'t, D> {
    subset: &'t RoaringBitmap,
    items: &'t ImmutableItems<'t, D>,
}

impl<'t, D: Distance> ImmutableSubsetItems<'t, D> {
    /// Creates a subset view of the available leafs.
    pub fn from_item_ids(leafs: &'t ImmutableItems<'t, D>, subset: &'t RoaringBitmap) -> Self {
        ImmutableSubsetItems { subset, items: leafs }
    }

    /// Returns the nodes identified by the given ID in the subset.
    pub fn get(&self, item_id: ItemId) -> heed::Result<Option<Node<'t, D>>> {
        if self.subset.contains(item_id) {
            self.items.get(item_id)
        } else {
            Ok(None)
        }
    }

    pub fn len(&self) -> u64 {
        self.subset.len()
    }
}

/// A struture used to keep a list of all the tree nodes in the tree.
///
/// It is safe to share between threads as the pointer are pointing
/// in the mmapped file and the transaction is kept here and therefore
/// no longer touches the database.
pub struct ImmutableNodes<'t, D> {
    nodes: IntMap<ItemId, (usize, *const u8)>,
    _marker: marker::PhantomData<(&'t (), D)>,
}

impl<'t, D: Distance> ImmutableNodes<'t, D> {
    /// Creates the structure by fetching all the root pointers
    /// and keeping the transaction making the pointers valid.
    pub fn new(
        rtxn: &'t RoTxn,
        database: Database<D>,
        index: u16,
        nb_trees: u64,
    ) -> heed::Result<Self> {
        let mut nodes =
            IntMap::with_capacity_and_hasher(nb_trees as usize, BuildNoHashHasher::default());

        let iter = database
            .remap_types::<PrefixCodec, Bytes>()
            .prefix_iter(rtxn, &Prefix::node(index))?
            .remap_key_type::<KeyCodec>();

        for result in iter {
            let (key, bytes) = result?;
            let tree_id = key.node.unwrap_node();
            nodes.insert(tree_id, (bytes.len(), bytes.as_ptr()));
        }

        Ok(ImmutableNodes { nodes, _marker: marker::PhantomData })
    }

    pub fn empty() -> Self {
        Self { nodes: IntMap::default(), _marker: marker::PhantomData }
    }

    /// Returns the tree node identified by the given ID.
    pub fn get(&self, item_id: ItemId) -> heed::Result<Option<DbItem<'t, D>>> {
        let (ptr, len) = match self.nodes.get(&item_id) {
            Some((len, ptr)) => (*ptr, *len),
            None => return Ok(None),
        };

        // safety:
        // - ptr: The pointer comes from LMDB. Since the database cannot be written to, it is still valid.
        // - len: The len cannot change either
        let bytes = unsafe { slice::from_raw_parts(ptr, len) };
        HnswNodeCodec::bytes_decode(bytes).map_err(heed::Error::Decoding).map(Some)
    }
}

unsafe impl<D> Sync for ImmutableNodes<'_, D> {}
