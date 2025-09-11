use heed::RoTxn;

use crate::distance::Distance;
use crate::internals::KeyCodec;
use crate::key::{Prefix, PrefixCodec};
use crate::node::Item;
use crate::{Database, ItemId, Node, NodeCodec, Result};

// used by the reader
pub struct ItemIter<'t, D: Distance> {
    pub inner: heed::RoPrefix<'t, KeyCodec, NodeCodec<D>>,
    dimensions: usize,
}

impl<'t, D: Distance> ItemIter<'t, D> {
    pub fn new(
        database: Database<D>,
        index: u16,
        dimensions: usize,
        rtxn: &'t RoTxn,
    ) -> heed::Result<Self> {
        Ok(ItemIter {
            inner: database
                .remap_key_type::<PrefixCodec>()
                .prefix_iter(rtxn, &Prefix::item(index))?
                .remap_key_type::<KeyCodec>(),
            dimensions,
        })
    }
}

impl<D: Distance> Iterator for ItemIter<'_, D> {
    type Item = Result<(ItemId, Vec<f32>)>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.inner.next() {
            Some(Ok((key, node))) => match node {
                Node::Item(Item { header: _, vector }) => {
                    let mut vector = vector.to_vec();
                    if vector.len() != self.dimensions {
                        // quantized codecs pad to 8-bytes so we truncate to recover len
                        vector.truncate(self.dimensions);
                    }
                    Some(Ok((key.node.item, vector)))
                }
                Node::Links(_) => unreachable!("Node must not be a link"),
            },
            Some(Err(e)) => Some(Err(e.into())),
            None => None,
        }
    }
}
