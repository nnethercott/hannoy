use heed::RoTxn;

use crate::distance::Distance;
use crate::internals::KeyCodec;
use crate::key::{Prefix, PrefixCodec};
use crate::node::Item;
use crate::{Database, ItemId, Node, NodeCodec, Result};

// used by the reader
pub struct ItemIter<'t, D: Distance> {
    pub inner: heed::RoPrefix<'t, KeyCodec, NodeCodec<D>>,
}

impl<'t, D: Distance> ItemIter<'t, D> {
    pub fn new(database: Database<D>, index: u16, rtxn: &'t RoTxn) -> heed::Result<Self> {
        Ok(ItemIter {
            inner: database
                .remap_key_type::<PrefixCodec>()
                .prefix_iter(rtxn, &Prefix::item(index))?
                .remap_key_type::<KeyCodec>(),
        })
    }
}

impl<D: Distance> Iterator for ItemIter<'_, D> {
    type Item = Result<(ItemId, Vec<f32>)>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.inner.next() {
            Some(Ok((key, node))) => match node {
                Node::Item(Item { header: _, vector }) => {
                    Some(Ok((key.node.item, vector.to_vec())))
                }
                Node::Links(_) => unreachable!("Node must not be a link"),
            },
            Some(Err(e)) => Some(Err(e.into())),
            None => None,
        }
    }
}
