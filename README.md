<p align="center"><img width="280px" title="this is a cowboy bebop ref" src="assets/hanoi_new.png"></a>
<h1 align="center">hannoy 🗼</h1>

[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Crates.io](https://img.shields.io/crates/v/hannoy)](https://crates.io/crates/hannoy)
[![dependency status](https://deps.rs/repo/github/nnethercott/hannoy/status.svg)](https://deps.rs/repo/github/nnethercott/hannoy)
[![Build](https://github.com/nnethercott/hannoy/actions/workflows/rust.yml/badge.svg?event=pull_request)](https://github.com/nnethercott/hannoy/actions/workflows/rust.yml)
[![CodSpeed Badge](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://codspeed.io/nnethercott/hannoy)

hannoy is a key-value backed [HNSW](https://www.pinecone.io/learn/series/faiss/hnsw/) implementation based on [arroy](https://github.com/meilisearch/arroy).

## Motivation
Many popular HNSW libraries are built in memory, meaning you need enough RAM to store all the vectors you're indexing. Instead, `hannoy` uses [LMDB](https://en.wikipedia.org/wiki/Lightning_Memory-Mapped_Database) — a memory-mapped KV store — as a storage backend. This is more well-suited for machines running multiple programs, or cases where the dataset you're indexing won't fit in memory. LMDB also supports non-blocking concurrent reads by design, meaning its safe to query the index in multi-threaded environments.

## Features
- Supported metrics: [euclidean](https://en.wikipedia.org/wiki/Euclidean_distance#:~:text=In%20mathematics%2C%20the%20Euclidean%20distance,occasionally%20called%20the%20Pythagorean%20distance.), [cosine](https://en.wikipedia.org/wiki/Cosine_similarity#Cosine_distance), [manhattan](https://en.wikipedia.org/wiki/Taxicab_geometry), [hamming](https://en.wikipedia.org/wiki/Hamming_distance), as well as quantized counterparts.
- Multithreaded builds using rayon
- Build index on disk to enable indexing big datasets that won't fit into memory using LMDB
- [Compressed bitmaps](https://github.com/RoaringBitmap/roaring-rs) to store graph edges with minimal overhead, adding overhead of only ~200 bytes per vector
- Dynamic document insertions and deletions

## Missing Features
- Python support
- GPU-accelerated indexing

## Usage
Here's a quick demo:

```rust
use hannoy::{distances::Cosine, Database, Reader, Result, Writer};
use heed::EnvOpenOptions;
use rand::{rngs::StdRng, SeedableRng};

fn main() -> Result<()> {
    const DIM: usize = 3;
    let vecs: Vec<[f32; DIM]> = vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    let env = unsafe {
        EnvOpenOptions::new()
            .map_size(1024 * 1024 * 1024 * 1) // 1GiB
            .open("./")
    }
    .unwrap();

    let mut wtxn = env.write_txn().unwrap();
    let db: Database<Cosine> = env.create_database(&mut wtxn, None)?;
    let writer: Writer<Cosine> = Writer::new(db, 0, DIM);

    // insert into lmdb
    writer.add_item(&mut wtxn, 0, &vecs[0])?;
    writer.add_item(&mut wtxn, 1, &vecs[1])?;
    writer.add_item(&mut wtxn, 2, &vecs[2])?;

    // ...and build hnsw
    let mut rng = StdRng::seed_from_u64(42);

    let mut builder = writer.builder(&mut rng);
    builder.ef_construction(100).build::<16,32>(&mut wtxn)?;
    wtxn.commit()?;

    // search hnsw using a new lmdb read transaction
    let rtxn = env.read_txn()?;
    let reader = Reader::<Cosine>::open(&rtxn, 0, db)?;

    let query = vec![0.0, 1.0, 0.0];
    let nns = reader.nns(1).ef_search(10).by_vector(&rtxn, &query)?;

    dbg!("{:?}", &nns);
    Ok(())
}
```

## Tips and tricks
### Reducing cold start latencies
Search in an hnsw always traverses from the top to bottom layers of the graph, so we know a priori some vectors will be needed. We can hint to the kernel that these vectors (and their neighbours) should be loaded into RAM using [`madvise`](https://man7.org/linux/man-pages/man2/madvise.2.html) to speed up search.

Doing so can reduce cold-start latencies by several milliseconds, and is configured through the `HANNOY_READER_PREFETCH_MEM` environment variable.

E.g. prefetching 10MiB of vectors into RAM.
```bash
export HANNOY_READER_PREFETCH_MEM=10485760
```
