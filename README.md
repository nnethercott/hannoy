# hannoy
hannoy is an [LMDB](https://en.wikipedia.org/wiki/Lightning_Memory-Mapped_Database)-based [HNSW](https://www.pinecone.io/learn/series/faiss/hnsw/) implementation based on [arroy](https://github.com/meilisearch/arroy).

Some links:
- [paper](https://arxiv.org/abs/1603.09320)
- [faiss hnsw.cpp](https://github.com/facebookresearch/faiss/blob/main/faiss/impl/HNSW.cpp)
- [hnsw.rs](https://github.com/rust-cv/hnsw)

# Notes: 
- downgraded smallvec to 0.14.0 to integrate with benchmark

## roadmap

- [x] fix hardcode of M0 for M in build/get_neighbours
- [x] add hnsw entrypoints to db `Node::Metadata`
- [ ] update edge bitmap of re-indexed nodes
- [ ] handle re-indexing case where new node may be on higher level
- [x] parallelize indexing
- [x] implement heuristic edge selection (mandatory; improves perfs non trivially -> Sparse Neighborhood Graph condition)
- [ ] use [papaya](https://github.com/ibraheemdev/papaya) for NodeStates? (n_reads >> n_writes). `papaya::HashMap::<NoHash>`
- [ ] add explanations to readme (KV rationale, pic of hnsw, etc.)
- [ ] LRU cache for recently accessed vectors ? -> effectively solved with frozzen reader
- [x] remove hardcode on lmdb_index=0 in builder
- [ ] either make Writer<R,D,M,M0>, remove SmallVec, or make Writer<R,D,M> (M0=2*M)
- [x] make hannoy work on [vector-relevancy-benchmark](https://github.com/meilisearch/vector-store-relevancy-benchmark)
- [ ] see if we can memoize <p,q> in a cache during search heuristic
- [ ] check to make sure each node only has at most M links (data races in parallel build might violate this)
  - add a metrics struct to the build to track number of link add ops

## ideas for improvement
- use a centroid as graph entry point
- only parallelize last layer build 

## Comments:
- `Reader::by_item` **much** faster in hnsw since we have a direct bitmap of neighbours
- For a dataset with n=10e^6 elements the link bitmaps are always uncompressed (M is generally < 128 << 4096) & there's at most 10e^6 / 2^16 = 16 buckets. In the worst case the links are in unique buckets => M*4 bytes per node. If instead all links are in the same bucket this number is (2 + 2 * M). Typical M is M=16 so we can expect the size of the links to be between 64 and 128 bytes per layer (not including overhead of bitmap structure) since M0 is generally 2 * M.
  - Each node is expected to be on 2-3 layers  & we store per link a key with size 8 bytes => we use between 2*(64+8) = 144 and 2*(128+8) = 272 bytes. This is within the range outlined in the paper (60-450 bytes per node) but is most likely slightly more efficient due to bitmaps
  - With the number above the links take O(10^2) bytes per node => for 10e^6 elements we use 100's of MiB to store the links
