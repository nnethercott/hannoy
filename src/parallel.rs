use core::slice;
use std::borrow::Cow;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::marker;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

use hashbrown::HashMap;
use heed::types::Bytes;
use heed::{BytesDecode, BytesEncode, RoTxn};
use memmap2::Mmap;
use nohash::IntMap;
use roaring::RoaringBitmap;

use crate::internals::{Item, KeyCodec};
use crate::key::{Key, Prefix, PrefixCodec};
use crate::node::{Links, Node, NodeCodec};
use crate::{Database, Distance, Error, ItemId, LayerId, Result};

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
    items: HashMap<ItemId, *const u8>,
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
        items: &RoaringBitmap,
        index: u16,
    ) -> heed::Result<Self> {
        let mut map = HashMap::with_capacity(items.len() as usize);
        let mut constant_length = None;

        for item_id in items {
            let bytes =
                database.remap_data_type::<Bytes>().get(rtxn, &Key::item(index, item_id))?.unwrap();
            assert_eq!(*constant_length.get_or_insert(bytes.len()), bytes.len());

            let ptr = bytes.as_ptr();
            map.insert(item_id, ptr);
        }

        Ok(ImmutableItems { items: map, constant_length, _marker: marker::PhantomData })
    }

    /// Returns the leafs identified by the given ID.
    pub fn get(&self, item_id: ItemId) -> heed::Result<Option<Item<'t, D>>> {
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
        NodeCodec::bytes_decode(bytes).map_err(heed::Error::Decoding).map(|node| node.item())
    }
}

unsafe impl<D> Sync for ImmutableItems<'_, D> {}

/// A struture used to keep a list of all the links.
/// It is safe to share between threads as the pointer are pointing
/// in the mmapped file and the transaction is kept here and therefore
/// no longer touches the database.
pub struct ImmutableLinks<'t, D> {
    links: HashMap<(u32, u8), (usize, *const u8)>,
    _marker: marker::PhantomData<(&'t (), D)>,
}

impl<'t, D: Distance> ImmutableLinks<'t, D> {
    /// Creates the structure by fetching all the root pointers
    /// and keeping the transaction making the pointers valid.
    pub fn new(
        rtxn: &'t RoTxn,
        database: Database<D>,
        index: u16,
        nb_links: u64,
    ) -> heed::Result<Self> {
        let mut links = HashMap::with_capacity(nb_links as usize);

        let iter = database
            .remap_types::<PrefixCodec, Bytes>()
            .prefix_iter(rtxn, &Prefix::links(index))?
            .remap_key_type::<KeyCodec>();

        for result in iter {
            let (key, bytes) = result?;
            let links_id = key.node.unwrap_node();
            links.insert(links_id, (bytes.len(), bytes.as_ptr()));
        }

        Ok(ImmutableLinks { links, _marker: marker::PhantomData })
    }

    pub fn empty() -> Self {
        Self { links: HashMap::default(), _marker: marker::PhantomData }
    }

    /// Returns the tree node identified by the given ID.
    pub fn get(&self, item_id: ItemId, level: LayerId) -> heed::Result<Option<Links<'t>>> {
        let key = (item_id, level);
        let (ptr, len) = match self.links.get(&key) {
            Some((len, ptr)) => (*ptr, *len),
            None => return Ok(None),
        };

        // safety:
        // - ptr: The pointer comes from LMDB. Since the database cannot be written to, it is still valid.
        // - len: The len cannot change either
        let bytes = unsafe { slice::from_raw_parts(ptr, len) };
        NodeCodec::bytes_decode(bytes)
            .map_err(heed::Error::Decoding)
            .map(|node: Node<'t, D>| node.links())
    }

    pub fn iter(&self) -> impl Iterator<Item = ((ItemId, u8), Cow<'_, RoaringBitmap>)> {
        self.links.keys().map(|&k| {
            let (item_id, level) = k;

            let links = match self.get(item_id, level) {
                Ok(Some(Links { links })) => links,
                _ => panic!("fix me later"),
            };
            (k, links)
        })
    }
}

unsafe impl<D> Sync for ImmutableLinks<'_, D> {}
