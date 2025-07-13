# hannoy
hannoy is an [LMDB](https://en.wikipedia.org/wiki/Lightning_Memory-Mapped_Database)-based [HNSW](https://www.pinecone.io/learn/series/faiss/hnsw/) implementation based on [arroy](https://github.com/meilisearch/arroy).

Some links:
- [paper](https://arxiv.org/abs/1603.09320)
- [faiss hnsw.cpp](https://github.com/facebookresearch/faiss/blob/main/faiss/impl/HNSW.cpp)
- [hnsw.rs](https://github.com/rust-cv/hnsw)

# Notes:
- downgraded smallvec to 0.14.0 to integrate with benchmark
- single threaded builds result in best graph quality (high recall, low search time).
- target: <100ms query latency on 10-100M vectors. (ideally <50ms)

## roadmap
- [x] fix hardcode of M0 for M in build/get_neighbours
- [x] add hnsw entrypoints to db `Node::Metadata`
- [ ] update edge bitmap of re-indexed nodes
- [ ] handle re-indexing case where new node may be on higher level
- [x] parallelize indexing
- [x] implement heuristic edge selection (mandatory; improves perfs non trivially -> Sparse Neighborhood Graph condition)
- [x] use [papaya](https://github.com/ibraheemdev/papaya) for NodeStates? (n_reads >> n_writes). `papaya::HashMap::<NoHash>`
- [ ] add explanations to readme (KV rationale, pic of hnsw, etc.)
- [ ] LRU cache for recently accessed vectors ? -> effectively solved with frozzen reader
- [x] remove hardcode on lmdb_index=0 in builder
- [ ] either make Writer<R,D,M,M0>, remove SmallVec, or make Writer<R,D,M> (M0=2*M)
- [x] make hannoy work on [vector-relevancy-benchmark](https://github.com/meilisearch/vector-store-relevancy-benchmark)
- [ ] see if we can memoize <p,q> in a cache during search heuristic
- [x] check to make sure each node only has at most M links (data races in parallel build might violate this), using `tinyvec` enforces this
- [x] add a metrics struct to the build to track number of link add ops
- [ ] add tests back to hannoy
- [ ] add other distances to hannoy
- [ ] check to see if filtered search is feasible
- [ ] add search stats ? figure out how many "jumps" are being done to see if ideas below can help
- [ ] add tracing

## ideas for improvement
- keep a counter of most frequently accessed nodes during build and make those entry points (e.g. use centroid-like)
- merge upper layers of graph if they only have one element
- product quantization `UnalignedVectorCodec`
- cache layers 1->L in RAM (speeds up M*(L-1) reads) using a hash table storing raw byte offsets and lengths
- *threadpool for `Reader` to parallelize searching neighbours

## Comments:
- `Reader::by_item` **much** faster in hnsw since we have a direct bitmap of neighbours
- For a dataset with n=10e^6 elements the link bitmaps are always uncompressed (M is generally < 128 << 4096) & there's at most 10e^6 / 2^16 = 16 buckets. In the worst case the links are in unique buckets => M*4 bytes per node. If instead all links are in the same bucket this number is (2 + 2 * M). Typical M is M=16 so we can expect the size of the links to be between 64 and 128 bytes per layer (not including overhead of bitmap structure) since M0 is generally 2 * M.
  - Each node is expected to be on 2-3 layers  & we store per link a key with size 8 bytes => we use between 2*(64+8) = 144 and 2*(128+8) = 272 bytes. This is within the range outlined in the paper (60-450 bytes per node) but is most likely slightly more efficient due to bitmaps
  - With the number above the links take O(10^2) bytes per node => for 10e^6 elements we use 100's of MiB to store the links
- if the hashmap of pointers to our lmdb can't fit in RAM we need some way of incrementally building. This is challenging since outgoing connections from a node in the insert bitmap may not fit in the frozzenreader
  - For 10 million vectors though the overhead of the frozzenreader map is O(100) MB so running this on a dedicated machine _should_ be fine ...
