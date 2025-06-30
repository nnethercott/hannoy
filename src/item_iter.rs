use crate::distance::Distance;
use crate::internals::KeyCodec;
use crate::node::HnswNode;
use crate::{DbItem, HnswNodeCodec, ItemId, Result};

// used by the reader
pub struct ItemIter<'t, D: Distance> {
    pub(crate) inner: heed::RoPrefix<'t, KeyCodec, HnswNodeCodec<D>>,
}

impl<D: Distance> Iterator for ItemIter<'_, D> {
    type Item = Result<(ItemId, Vec<f32>)>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.inner.next() {
            #[allow(unreachable_patterns)]
            Some(Ok((key, node))) => match node {
                DbItem::Item(HnswNode { header: _, vector, links: _ }) => {
                    Some(Ok((key.node.item, vector.to_vec())))
                }
                _ => unreachable!(),
            },
            Some(Err(e)) => Some(Err(e.into())),
            None => None,
        }
    }
}
