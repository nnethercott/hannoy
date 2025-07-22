mod distance;
mod error;
mod hnsw;
mod item_iter;
mod key;
mod metadata;
mod node;
mod node_id;
mod parallel;
mod reader;
mod roaring;
mod spaces;
mod stats;
mod version;
mod writer;

mod ordered_float;
mod unaligned_vector;

pub use distance::Distance;
pub use error::Error;
pub use reader::{QueryBuilder, Reader};
pub use writer::{HannoyBuilder, Writer};

use key::{Key, Prefix, PrefixCodec};
use metadata::{Metadata, MetadataCodec};
use node::{Node, NodeCodec};
use node_id::{NodeId, NodeMode};

/// The set of types used by the [`Distance`] trait.
pub mod internals {

    pub use crate::distance::{
        NodeHeaderBinaryQuantizedCosine, NodeHeaderCosine, NodeHeaderEuclidean,
    };
    pub use crate::key::KeyCodec;
    pub use crate::node::{Item, NodeCodec};
    pub use crate::unaligned_vector::{SizeMismatch, UnalignedVector, UnalignedVectorCodec};
}

/// The set of distances implementing the [`Distance`] and supported by arroy.
pub mod distances {
    pub use crate::distance::{
        BinaryQuantizedCosine, BinaryQuantizedEuclidean, BinaryQuantizedManhattan, Cosine,
        Euclidean, Hamming, Manhattan,
    };
}

/// A custom Result type that is returning an arroy error by default.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// The database required by arroy for reading or writing operations.
pub type Database<D> = heed::Database<internals::KeyCodec, NodeCodec<D>>;

/// An identifier for the items stored in the database.
pub type ItemId = u32;
pub type LayerId = u8;
