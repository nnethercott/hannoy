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
    Item(HnswNode<'a, D>),
    // keeping this in an enum just in case
}

const NODE_TAG: u8 = 0;
const LINKS_TAG: u8 = 1;

impl<'a, D: Distance> DbItem<'a, D> {
    pub fn item(self) -> Option<HnswNode<'a, D>> {
        if let DbItem::Item(item) = self {
            Some(item)
        } else {
            None
        }
    }
}

/// A leaf node which corresponds to the vector inputed
/// by the user and the distance header.
///
/// NOTE: this is nice cause greedy search during retrieval goes like 
/// while let Some(next) = ep.next.take(){
///     todo!()
/// }
pub struct HnswNode<'a, D: Distance> {
    /// edges from the node
    pub links: Cow<'a, RoaringBitmap>,
    /// edge to next layer
    pub next: Option<ItemId>,
    /// The header of this leaf.
    pub header: D::Header,
    /// The vector of this leaf.
    pub vector: Cow<'a, UnalignedVector<D::VectorCodec>>,
}

impl<D: Distance> fmt::Debug for HnswNode<'_, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let links = self.links.iter().collect::<Vec<_>>();
        f.debug_struct("Leaf")
            .field("header", &self.header)
            .field("vector", &self.vector)
            .field("links", &links)
            .field("next", &self.next)
            .finish()
    }
}

impl<D: Distance> Clone for HnswNode<'_, D> {
    fn clone(&self) -> Self {
        Self {
            links: Cow::Owned(RoaringBitmap::new()),
            next: None,
            header: self.header,
            vector: self.vector.clone(),
        }
    }
}

impl<D: Distance> HnswNode<'_, D> {
    /// Converts the leaf into an owned version of itself by cloning
    /// the internal vector. Doing so will make it mutable.
    pub fn into_owned(self) -> HnswNode<'static, D> {
        HnswNode {
            links: Cow::Owned(self.links.into_owned()),
            next: self.next,
            header: self.header,
            vector: Cow::Owned(self.vector.into_owned()),
        }
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
pub struct HnswNodeCodec<D>(D);

impl<'a, D: Distance> BytesEncode<'a> for HnswNodeCodec<D> {
    type EItem = DbItem<'a, D>;

    fn bytes_encode(item: &Self::EItem) -> Result<Cow<'a, [u8]>, BoxedError> {
        let mut bytes = Vec::new();
        match item {
            DbItem::Item(HnswNode { links, next, header, vector }) => {
                bytes.push(NODE_TAG);
                bytes.extend_from_slice(bytes_of(header));
                let vbytes = vector.as_bytes();
                let len = vbytes.len() as u16;
                bytes.extend_from_slice(&len.to_be_bytes());
                bytes.extend(vbytes);

                if let Some(item_id) = next {
                    bytes.push(1);
                    bytes.extend_from_slice(&item_id.to_be_bytes());
                } else {
                    bytes.push(0);
                }
                links.serialize_into(&mut bytes)?;
            }
        }
        Ok(Cow::Owned(bytes))
    }
}

impl<'a, D: Distance> BytesDecode<'a> for HnswNodeCodec<D> {
    type DItem = DbItem<'a, D>;

    fn bytes_decode(bytes: &'a [u8]) -> Result<Self::DItem, BoxedError> {
        match bytes {
            [NODE_TAG, bytes @ ..] => {
                let (header_bytes, bytes) = bytes.split_at(size_of::<D::Header>());
                let header: D::Header = pod_read_unaligned(header_bytes);
                let len = BigEndian::read_u16(bytes) as usize;
                let bytes = &bytes[std::mem::size_of::<u16>()..];
                let vector = UnalignedVector::<D::VectorCodec>::from_bytes(&bytes[..len]).unwrap();

                let bytes = &bytes[len..];
                let (next, bytes) = match bytes[0] {
                    1 => (
                        Some(BigEndian::read_u32(&bytes[1..])),
                        &bytes[1 + std::mem::size_of::<ItemId>()..],
                    ),
                    0 => (None, &bytes[1..]),
                    _ => unreachable!(),
                };
                let links: Cow<'_, RoaringBitmap> =
                    Cow::Owned(RoaringBitmap::deserialize_from(bytes).unwrap());

                Ok(DbItem::Item(HnswNode { header, vector, links, next }))
            }
            unknown => panic!("Did not recognize node tag type: {unknown:?}"),
        }
    }
}

pub struct Node<'a, D: Distance> {
    /// The header of this leaf.
    pub header: D::Header,
    /// The vector of this leaf.
    pub vector: Cow<'a, UnalignedVector<D::VectorCodec>>,
}

impl<D: Distance> fmt::Debug for Node<'_, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Leaf").field("header", &self.header).field("vector", &self.vector).finish()
    }
}
pub struct NodeCodec<D>(D);

impl<'a, D: Distance> BytesDecode<'a> for NodeCodec<D> {
    type DItem = Node<'a, D>;

    fn bytes_decode(bytes: &'a [u8]) -> Result<Self::DItem, BoxedError> {
        match bytes {
            [NODE_TAG, bytes @ ..] => {
                let (header_bytes, bytes) = bytes.split_at(size_of::<D::Header>());
                let header: D::Header = pod_read_unaligned(header_bytes);
                let len = BigEndian::read_u16(bytes) as usize;
                let bytes = &bytes[std::mem::size_of::<u16>()..];
                let vector = UnalignedVector::<D::VectorCodec>::from_bytes(&bytes[..len]).unwrap();
                Ok(Node { header, vector })
            }
            unknown => panic!("Did not recognize node tag type: {unknown:?}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{DbItem, HnswNode, HnswNodeCodec};
    use crate::{distance::Cosine, internals::UnalignedVector, node::NodeCodec, Distance};
    use heed::{BytesDecode, BytesEncode};
    use roaring::RoaringBitmap;
    use std::borrow::Cow;

    #[test]
    fn check_bytes_encode_decode() {
        type D = Cosine;

        let vector = UnalignedVector::from_vec(vec![1.0f32, 2.0f32]);
        let header = D::new_header(&vector);
        let item = HnswNode { vector, header, next: None, links: Cow::Owned(RoaringBitmap::new()) };
        let db_item = DbItem::Item(item);

        let bytes = HnswNodeCodec::<D>::bytes_encode(&db_item);
        assert!(bytes.is_ok());
        let bytes = bytes.unwrap();
        dbg!("{}, {}", std::mem::size_of_val(&db_item), bytes.len());
        // dbg!("{:?}", &bytes);

        let db_item2 = HnswNodeCodec::<D>::bytes_decode(bytes.as_ref());
        assert!(db_item2.is_ok());
        let db_item2 = db_item2.unwrap();

        dbg!("{:?}", &db_item2);
        dbg!("{:?}", &db_item);
    }

    #[test]
    fn test_prefix_codec() {
        type D = Cosine;

        let vector = UnalignedVector::from_vec(vec![1.0f32, 2.0f32]);
        let header = D::new_header(&vector);
        let item = HnswNode { vector, header, next: None, links: Cow::Owned(RoaringBitmap::new()) };
        let db_item = DbItem::Item(item.clone());

        let bytes = HnswNodeCodec::<D>::bytes_encode(&db_item);
        assert!(bytes.is_ok());
        let bytes = bytes.unwrap();

        let new_item = NodeCodec::<D>::bytes_decode(bytes.as_ref());
        assert!(new_item.is_ok());
        let new_item = new_item.unwrap();

        assert!(matches!(new_item.vector, Cow::Borrowed(_)));
        assert_eq!(new_item.vector.as_bytes(), item.vector.as_bytes());
    }
}
