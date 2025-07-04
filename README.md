# hannoy
hannoy is an [LMDB](https://en.wikipedia.org/wiki/Lightning_Memory-Mapped_Database)-based [HNSW](https://www.pinecone.io/learn/series/faiss/hnsw/) implementation based on [arroy](https://github.com/meilisearch/arroy).

Some links:
- [paper](https://arxiv.org/abs/1603.09320)
- [faiss hnsw.cpp](https://github.com/facebookresearch/faiss/blob/main/faiss/impl/HNSW.cpp)
- [hnsw.rs](https://github.com/rust-cv/hnsw)

## roadmap

- [x] fix hardcode of M0 for M in build/get_neighbours
- [x] add hnsw entrypoints to db `Node::Metadata`
- [ ] update edge bitmap of re-indexed nodes
- [ ] handle re-indexing case where new node may be on higher level
- [ ] parallelize indexing
- [ ] implement heuristic edge selection
- [ ] use [papaya](https://github.com/ibraheemdev/papaya) for NodeStates? (n_reads >> n_writes). `papaya::HashMap::<NoHash>`
- [ ] add explanations to readme (KV rationale, pic of hnsw, etc.)
- [ ] LRU cache for recently accessed vectors ? -> effectively solved with frozzen reader
- [x] remove hardcode on lmdb_index=0 in builder
- [ ] either make Writer<R,D,M,M0> or remove SmallVec
- [ ] make hannoy work on [vector-relevancy-benchmark](https://github.com/meilisearch/vector-store-relevancy-benchmark)

## Notes:
- `Reader::by_item` **much** faster in hnsw since we have a direct bitmap of neighbours
