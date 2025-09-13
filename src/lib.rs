//! Hannoy is a key-value backed [HNSW][1] implementation based on [arroy][2].
//!
//! Many popular HNSW libraries are built in memory, meaning you need enough RAM to store all the vectors you're indexing. Instead, `hannoy` uses
//! [LMDB][3] — a memory-mapped KV store — as a storage backend.
//!
//! This is more well-suited for machines running multiple programs, or cases where the
//! dataset you're indexing won't fit in memory. LMDB also supports non-blocking concurrent reads by design, meaning its safe to query the index in
//! multi-threaded environments.
//!
//! [1]: https://www.pinecone.io/learn/series/faiss/hnsw/
//! [2]: https://github.com/meilisearch/arroy
//! [3]: https://en.wikipedia.org/wiki/Lightning_Memory-Mapped_Database
//!
//! # Examples
//!
//! Open an LMDB database, store some vectors in it and query the nearest item from some query vector. This is the most
//! trivial way to use hannoy and it's fairly easy. Just do not forget to [`HannoyBuilder::build<M0,M>`] and [`heed::RwTxn::commit`]
//! when you are done inserting your items.
//!
//! ```rust
//! use hannoy::{distances::Cosine, Database, Reader, Result, Writer};
//! use heed::EnvOpenOptions;
//! use rand::{rngs::StdRng, SeedableRng};
//!
//! fn main() -> Result<()> {
//!     const DIM: usize = 3;
//!     let vecs: Vec<[f32; DIM]> = vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
//!
//!     let env = unsafe {
//!         EnvOpenOptions::new()
//!             .map_size(1024 * 1024 * 1024 * 1) // 1GiB
//!             .open("./")
//!     }
//!     .unwrap();
//!
//!     let mut wtxn = env.write_txn().unwrap();
//!     let db: Database<Cosine> = env.create_database(&mut wtxn, None)?;
//!     let writer: Writer<Cosine> = Writer::new(db, 0, DIM);
//!
//!     // insert into lmdb
//!     writer.add_item(&mut wtxn, 0, &vecs[0])?;
//!     writer.add_item(&mut wtxn, 1, &vecs[1])?;
//!     writer.add_item(&mut wtxn, 2, &vecs[2])?;
//!
//!     // ...and build hnsw
//!     let mut rng = StdRng::seed_from_u64(42);
//!
//!     let mut builder = writer.builder(&mut rng);
//!     builder.ef_construction(100).build::<16,32>(&mut wtxn)?;
//!     wtxn.commit()?;
//!
//!     // search hnsw using a new lmdb read transaction
//!     let rtxn = env.read_txn()?;
//!     let reader = Reader::<Cosine>::open(&rtxn, 0, db)?;
//!
//!     let query = vec![0.0, 1.0, 0.0];
//!     let nns = reader.nns(1).ef_search(10).by_vector(&rtxn, &query)?;
//!
//!     dbg!("{:?}", &nns);
//!     Ok(())
//! }
//! ```
#![warn(missing_docs)]
#![doc(
    html_favicon_url = "https://raw.githubusercontent.com/nnethercott/hannoy/main/assets/hanoi_new.png?raw=true"
)]
#![doc(
    html_logo_url = "https://raw.githubusercontent.com/nnethercott/hannoy/main/assets/hanoi_new.png?raw=true"
)]
#![warn(clippy::todo)]

mod distance;
mod error;
mod hnsw;
mod item_iter;
mod key;
mod metadata;
mod node;
mod node_id;
mod parallel;
mod progress;
mod reader;
mod roaring;
mod spaces;
mod stats;
mod version;
mod writer;

#[cfg(test)]
mod tests;

mod ordered_float;
mod unaligned_vector;

#[cfg(feature = "python")]
pub mod python;

pub use distance::Distance;
pub use error::Error;
use key::{Key, Prefix, PrefixCodec};
use metadata::{Metadata, MetadataCodec};
use node::{Node, NodeCodec};
use node_id::{NodeId, NodeMode};
pub use reader::{QueryBuilder, Reader};
pub use writer::{HannoyBuilder, Writer};

/// The set of types used by the [`Distance`] trait.
pub mod internals {
    pub use crate::distance::{
        NodeHeaderBinaryQuantizedCosine, NodeHeaderCosine, NodeHeaderEuclidean,
    };
    pub use crate::key::KeyCodec;
    pub use crate::node::{Item, NodeCodec};
    pub use crate::unaligned_vector::{SizeMismatch, UnalignedVector, UnalignedVectorCodec};
}

/// The set of distances implementing the [`Distance`] and supported by hannoy.
pub mod distances {
    pub use crate::distance::{
        BinaryQuantizedCosine, BinaryQuantizedEuclidean, BinaryQuantizedManhattan, Cosine,
        Euclidean, Hamming, Manhattan,
    };
}

/// A custom Result type that is returning an hannoy error by default.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// The database required by hannoy for reading or writing operations.
pub type Database<D> = heed::Database<internals::KeyCodec, NodeCodec<D>>;

/// An identifier for the items stored in the database.
pub type ItemId = u32;
/// An indentifier for the links of the hnsw. We can guarantee mathematically there will always be
/// less than 256 layers.
pub type LayerId = u8;

/// The number of iterations to process before checking if the indexing process should be cancelled.
const CANCELLATION_PROBING: usize = 10000;
