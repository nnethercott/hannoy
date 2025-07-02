# hannoy
hannoy is an [LMDB](https://en.wikipedia.org/wiki/Lightning_Memory-Mapped_Database)-based [HNSW](https://www.pinecone.io/learn/series/faiss/hnsw/) implementation based on [arroy](https://github.com/meilisearch/arroy).

Some links:
- [paper](https://arxiv.org/abs/1603.09320)
- [faiss hnsw.cpp](https://github.com/facebookresearch/faiss/blob/main/faiss/impl/HNSW.cpp)
- [hnsw.rs](https://github.com/rust-cv/hnsw)

## roadmap

- [ ] fix hardcode of M0 for M in build/get_neighbours
- [ ] add hnsw entrypoints to db `Node::Metadata`
- [ ] update edge bitmap of re-indexed nodes
- [ ] handle re-indexing case where new node may be on higher level
- [ ] parallelize indexing
- [ ] add f32::epsilon to assign_probas lambda so HnswBuilder<D,1,2> can work
- [ ] implement heuristic edge selection
- [ ] use [papaya](https://github.com/ibraheemdev/papaya) for NodeStates? (n_reads >> n_writes). `papaya::HashMap::<NoHash>`
- [ ] add explanations to readme (KV rationale, pic of hnsw, etc.)
- [ ] LRU cache for recently accessed vectors ?

## Notes:
- `Reader::by_item` **much** faster in hnsw since we have a direct bitmap of neighbours
