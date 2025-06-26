use std::borrow::Cow;
use std::fmt;
use std::mem::{size_of, transmute};
use std::num::NonZeroU32;

use bytemuck::{bytes_of, cast_slice, pod_read_unaligned};
use byteorder::{BigEndian, ByteOrder, LittleEndian, NativeEndian};
use heed::{BoxedError, BytesDecode, BytesEncode};
use roaring::RoaringBitmap;

use crate::distance::Distance;
use crate::unaligned_vector::UnalignedVector;
use crate::ItemId;

#[derive(Clone, Debug)]
pub enum DbItem<'a, D: Distance> {
    // FIXME: Items need to have the links as well since we need vectors during search
    // however : we can prefix search and just deserialize bitset before we get to level 0 ?
    Item(Item<'a, D>),
    Node(GraphNode<'a>),
}

const NODE_TAG: u8 = 0;
const LINKS_TAG: u8 = 1;

impl<'a, D: Distance> DbItem<'a, D> {
    pub fn item(self) -> Option<Item<'a, D>> {
        if let DbItem::Item(item) = self {
            Some(item)
        } else {
            None
        }
    }
}

/// A leaf node which corresponds to the vector inputed
/// by the user and the distance header.
pub struct Item<'a, D: Distance> {
    /// edges from the node
    pub links: Cow<'a, RoaringBitmap>,
    /// edge to next layer
    pub next: Option<ItemId>,
    /// The header of this leaf.
    pub header: D::Header,
    /// The vector of this leaf.
    pub vector: Cow<'a, UnalignedVector<D::VectorCodec>>,
}

impl<D: Distance> fmt::Debug for Item<'_, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Leaf").field("header", &self.header).field("vector", &self.vector).finish()
    }
}

impl<D: Distance> Clone for Item<'_, D> {
    fn clone(&self) -> Self {
        Self {
            links: Cow::Owned(RoaringBitmap::new()),
            next: None,
            header: self.header,
            vector: self.vector.clone(),
        }
    }
}

impl<D: Distance> Item<'_, D> {
    /// Converts the leaf into an owned version of itself by cloning
    /// the internal vector. Doing so will make it mutable.
    pub fn into_owned(self) -> Item<'static, D> {
        Item {
            links: Cow::Owned(self.links.into_owned()),
            next: self.next,
            header: self.header,
            vector: Cow::Owned(self.vector.into_owned()),
        }
    }
}

#[derive(Clone)]
pub struct GraphNode<'a> {
    // A descendants node can only contains references to the leaf nodes.
    // We can get and store their ids directly without the `Mode`.
    // pub level: u16,
    pub links: Cow<'a, RoaringBitmap>,
}

impl fmt::Debug for GraphNode<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let links = self.links.iter().collect::<Vec<_>>();
        f.debug_struct("Links").field("links", &links).finish()
    }
}

#[derive(Clone)]
pub struct ItemIds<'a> {
    bytes: &'a [u8],
}

impl<'a> ItemIds<'a> {
    pub fn from_slice(slice: &[u32]) -> ItemIds<'_> {
        ItemIds::from_bytes(cast_slice(slice))
    }

    pub fn from_bytes(bytes: &[u8]) -> ItemIds<'_> {
        ItemIds { bytes }
    }

    pub fn raw_bytes(&self) -> &[u8] {
        self.bytes
    }

    pub fn len(&self) -> usize {
        self.bytes.len() / size_of::<ItemId>()
    }

    pub fn iter(&self) -> impl Iterator<Item = ItemId> + 'a {
        self.bytes.chunks_exact(size_of::<ItemId>()).map(NativeEndian::read_u32)
    }
}

impl fmt::Debug for ItemIds<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut list = f.debug_list();
        self.iter().for_each(|integer| {
            list.entry(&integer);
        });
        list.finish()
    }
}

/// The codec used internally to encode and decode nodes.
pub struct NodeCodec<D>(D);

impl<'a, D: Distance> BytesEncode<'a> for NodeCodec<D> {
    type EItem = DbItem<'a, D>;

    fn bytes_encode(item: &Self::EItem) -> Result<Cow<'a, [u8]>, BoxedError> {
        let mut bytes = Vec::new();
        match item {
            DbItem::Item(Item { links, next, header, vector }) => {
                bytes.push(NODE_TAG);
                let links_size = links.serialized_size() as u16;
                bytes.extend_from_slice(&links_size.to_be_bytes());
                links.serialize_into(&mut bytes)?;

                if let Some(item_id) = next {
                    bytes.push(1);
                    bytes.extend_from_slice(&item_id.to_be_bytes());
                } else {
                    bytes.push(0);
                }

                bytes.extend_from_slice(bytes_of(header));
                bytes.extend_from_slice(vector.as_bytes());
            }
            DbItem::Node(GraphNode { links }) => {
                bytes.push(LINKS_TAG);
                links.serialize_into(&mut bytes)?;
            }
        }
        Ok(Cow::Owned(bytes))
    }
}

impl<'a, D: Distance> BytesDecode<'a> for NodeCodec<D> {
    type DItem = DbItem<'a, D>;

    fn bytes_decode(bytes: &'a [u8]) -> Result<Self::DItem, BoxedError> {
        match bytes {
            [NODE_TAG, bytes @ ..] => {
                let links_size = BigEndian::read_u16(bytes);
                let bytes = &bytes[std::mem::size_of_val(&links_size) as usize..];
                let links: Cow<'_, RoaringBitmap> = Cow::Owned(
                    RoaringBitmap::deserialize_from(&bytes[..links_size as usize]).unwrap(),
                );
                let bytes = &bytes[links_size as usize..];

                let (next, bytes) = match bytes[0] {
                    1 => (
                        Some(BigEndian::read_u32(&bytes[1..])),
                        &bytes[1 + std::mem::size_of::<ItemId>()..],
                    ),
                    0 => (None, &bytes[1..]),
                    _ => unreachable!(),
                };

                let (header_bytes, remaining) = bytes.split_at(size_of::<D::Header>());
                let header: D::Header = pod_read_unaligned(header_bytes);
                let vector = UnalignedVector::<D::VectorCodec>::from_bytes(remaining).unwrap();

                Ok(DbItem::Item(Item { header, vector, links, next }))
            }
            [LINKS_TAG, bytes @ ..] => Ok(DbItem::Node(GraphNode {
                links: Cow::Owned(RoaringBitmap::deserialize_from(bytes)?),
            })),
            unknown => panic!(
                "Did not recognize node tag type: {unknown:?} while decoding a node from v0.7.0"
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{DbItem, Item, NodeCodec};
    use crate::{distance::Cosine, internals::UnalignedVector, Distance};
    use heed::{BytesDecode, BytesEncode};
    use roaring::RoaringBitmap;
    use std::borrow::Cow;

    #[test]
    fn check_bytes_encode_decode() {
        type D = Cosine;

        let vector = UnalignedVector::from_vec(vec![1.0f32, 2.0f32]);
        let header = D::new_header(&vector);
        let item = Item { vector, header, next: None, links: Cow::Owned(RoaringBitmap::new()) };
        let db_item = DbItem::Item(item);

        let bytes = NodeCodec::<D>::bytes_encode(&db_item);
        assert!(bytes.is_ok());
        let bytes = bytes.unwrap();
        dbg!("{}, {}", std::mem::size_of_val(&db_item), bytes.len());
        // dbg!("{:?}", &bytes);

        let db_item2 = NodeCodec::<D>::bytes_decode(bytes.as_ref());
        assert!(db_item2.is_ok());
        let db_item2 = db_item2.unwrap();

        dbg!("{:?}", &db_item2);
        dbg!("{:?}", &db_item);
    }
}
