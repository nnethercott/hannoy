use std::borrow::Cow;
use std::fmt;
use std::mem::size_of;

use bytemuck::{bytes_of, cast_slice, pod_read_unaligned};
use byteorder::{ByteOrder, NativeEndian};
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

const LEAF_TAG: u8 = 0;
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
    // NOTE: put the graph info first so we can maybe do greedy search

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
        Self { header: self.header, vector: self.vector.clone() }
    }
}

impl<D: Distance> Item<'_, D> {
    /// Converts the leaf into an owned version of itself by cloning
    /// the internal vector. Doing so will make it mutable.
    pub fn into_owned(self) -> Item<'static, D> {
        Item { header: self.header, vector: Cow::Owned(self.vector.into_owned()) }
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
            DbItem::Item(Item { header, vector }) => {
                bytes.push(LEAF_TAG);
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
            [LEAF_TAG, bytes @ ..] => {
                let (header_bytes, remaining) = bytes.split_at(size_of::<D::Header>());
                let header = pod_read_unaligned(header_bytes);
                let vector = UnalignedVector::<D::VectorCodec>::from_bytes(remaining)?;

                Ok(DbItem::Item(Item { header, vector }))
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
