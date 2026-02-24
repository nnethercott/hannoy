use std::borrow::Cow;

use heed::{RoTxn, RwTxn, WithoutTls};
use roaring::RoaringBitmap;

use crate::key::Key;
use crate::node::{Item, Links, Node};
use crate::node_id::NodeId;
use crate::{Database, Distance, Error, ItemId, Result};

pub(crate) struct FrozenReader<'t, D> {
    rtxns: thread_local::ThreadLocal<RoTxn<'t, WithoutTls>>,
    index: u16,
    database: Database<D>,
}

impl<'t, D: Distance> FrozenReader<'t, D> {
    pub fn new(wtxn: &'t mut RwTxn<'_>, index: u16, database: Database<D>) -> Result<Self> {
        let num_threads = rayon::current_num_threads();
        let (sender, receiver) = crossbeam_channel::bounded(num_threads);

        // Sequentially generate read transactions from the writer transaction
        for _ in 0..num_threads {
            let rtxn = wtxn.nested_read_txn()?;
            sender.try_send(rtxn).unwrap();
        }

        // To clarify that we are done sending read transactions
        drop(sender);

        // Store the read transactions in the thread local for later use
        let rtxns = thread_local::ThreadLocal::new();
        rayon::broadcast(|_| {
            let _ = rtxns.get_or(|| receiver.try_recv().unwrap());
        });

        Ok(Self { rtxns, index, database })
    }

    pub fn item<'a>(&'a self, item_id: ItemId) -> Result<Item<'a, D>> {
        let rtxn = self.rtxns.get().expect("missing nested read txn from the thread local");
        let key = Key::item(self.index, item_id);
        // key is a `Key::item` so returned result must be a Node::Item
        self.database.get(rtxn, &key)?.and_then(|node| node.item()).ok_or(Error::missing_key(key))
    }

    pub fn links<'a>(&'a self, item_id: ItemId, level: usize) -> Result<Links<'a>> {
        let rtxn = self.rtxns.get().expect("missing nested read txn from the thread local");
        let key = Key::links(self.index, item_id, level as u8);
        // key is a `Key::item` so returned result must be a Node::Item
        self.database.get(rtxn, &key)?.and_then(|node| node.links()).ok_or(Error::missing_key(key))
    }

    /// `Iter`s only over links in a given level
    pub fn iter_layer_links(
        &self,
        layer: u8,
    ) -> heed::Result<impl Iterator<Item = heed::Result<((ItemId, u8), Cow<'_, RoaringBitmap>)>>>
    {
        let rtxn = self.rtxns.get().expect("missing nested read txn from the thread local");
        Ok(self.database.lazily_decode_data().iter(rtxn)?.filter_map(move |result| {
            let (key, value) = match result {
                Ok(value) => value,
                Err(e) => return Some(Err(e)),
            };

            let Key { node: NodeId { item: item_id, layer: level, .. }, .. } = key;

            if level != layer {
                return None;
            }

            match value.decode() {
                Ok(Node::Links(Links { links })) => Some(Ok(((item_id, level), links))),
                Ok(Node::Item(_)) => {
                    unreachable!("link at level {level} with item_id {item_id} not found")
                }
                Err(e) => Some(Err(heed::Error::Decoding(e))),
            }
        }))
    }

    pub fn iter_links(
        &self,
    ) -> heed::Result<impl Iterator<Item = heed::Result<((ItemId, u8), Cow<'_, RoaringBitmap>)>>>
    {
        let rtxn = self.rtxns.get().expect("missing nested read txn from the thread local");
        Ok(self.database.lazily_decode_data().iter(rtxn)?.filter_map(move |result| {
            let (key, value) = match result {
                Ok(value) => value,
                Err(e) => return Some(Err(e)),
            };

            let Key { node: NodeId { item: item_id, layer: level, .. }, .. } = key;

            match value.decode() {
                Ok(Node::Links(Links { links })) => Some(Ok(((item_id, level), links))),
                Ok(Node::Item(_)) => {
                    unreachable!("link at level {level} with item_id {item_id} not found")
                }
                Err(e) => Some(Err(heed::Error::Decoding(e))),
            }
        }))
    }
}
