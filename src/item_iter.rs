use crate::distance::Distance;
use crate::internals::KeyCodec;
use crate::node::Item;
use crate::{Node, NodeCodec, ItemId, Result};

// used by the reader
pub struct ItemIter<'t, D: Distance> {
    pub(crate) inner: heed::RoPrefix<'t, KeyCodec, NodeCodec<D>>,
}

impl<D: Distance> Iterator for ItemIter<'_, D> {
    type Item = Result<(ItemId, Vec<f32>)>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.inner.next() {
            #[allow(unreachable_patterns)]
            Some(Ok((key, node))) => match node {
                Node::Item(Item { header: _, vector }) => {
                    Some(Ok((key.node.item, vector.to_vec())))
                }
                _ => unreachable!(),
            },
            Some(Err(e)) => Some(Err(e.into())),
            None => None,
        }
    }
}
