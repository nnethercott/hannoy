# hannoy
hnsw + lmdb. annoy -> arroy -> hannoy

Some links:
- [paper](https://arxiv.org/abs/1603.09320)
- [faiss hnsw.cpp](https://github.com/facebookresearch/faiss/blob/main/faiss/impl/HNSW.cpp)
- [hnsw.rs](https://github.com/rust-cv/hnsw)

## Some notes and ideas:
- db schema for arroy may not be appropriate for an hnsw approach. Search works in arroy by retrieving splitting planes and computing the margin while in hnsw we need follow graph edges. If we store edges separately we'll need to make 2 requests to retrieve a) the edges/links and b) for each link a vector
  - probably need something like
    ```rust
    struct Item{
      links: <RoaringBitMap as heed:BytesEncode>,
      next: u32, // <- id of closest doc in layer below
      header: NodeHeader,
      vector: UnalignedVector,
      maybe_padding: todo!(),
    }
    ```
  - for greedy search we can defined a Prefix which deserializes only the roaring bitmap of links for each item
  - simpler with one `Node` variant than before (`Item`, and `Tree`)

- building should be cheaper than arroy since no new vectors are generated; we just need to keep track of graph edges. Also no duplicated trees should cut cost down a ton
