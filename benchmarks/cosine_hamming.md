# cosine vs hamming vs bq-cosine

Results below ran on a machine with 8xIntel(R) Core(TM) i7-6900K CPUs @ 3.20GHz using [this branch](https://github.com/meilisearch/vector-store-relevancy-benchmark/compare/main...nnethercott:vector-store-relevancy-benchmark:arroy-hannoy) in [this repo](https://github.com/meilisearch/vector-store-relevancy-benchmark).

### Datacomp Small (768 dimensions)
- hannoy: `M=24`, `ef_construction=512`, `ef_search=200`
- distance: cosine

| # of Vectors | Metric    | Build Time | DB Size    | Search Latency | Recall@1 | Recall@5 | Recall@10 | Recall@50 | Recall@100 |
|--------------|-----------|------------|------------|----------------|----------|----------|-----------|-----------|------------|
| 10K          | Cosine    | 1.16s      | 40.31 MiB | 9.53ms         |  0.91     | 0.95     | 0.95      | 0.97      | 0.97       |
|              | Hamming   | 1.13s      | 2.87 MiB   | 13.19ms        | 0.96     | 0.99     | 0.98      | 0.98      | 0.98       |
|              | BQ Cosine | 1.87s      | 2.80 MiB   | 15.78ms        | 0.95     | 0.59     | 0.55      | 0.50      | 0.48       |
| 50K          | Cosine    | 12.27s     | 201.39 MiB| 13.49ms        |  0.91     | 0.93     | 0.93      | 0.95      | 0.91       |
|              | Hamming   | 9.10s      | 14.54 MiB  | 17.57ms        | 0.93     | 0.95     | 0.94      | 0.95      | 0.94       |
|              | BQ Cosine | 17.29s     | 14.24 MiB  | 21.36ms        | 0.93     | 0.58     | 0.54      | 0.47      | 0.45       |
| 100K         | Cosine    | 31.51s     | 404.24 MiB| 15.73ms        |  0.92     | 0.92     | 0.93      | 0.94      | 0.92       |
|              | Hamming   | 22.77s     | 30.31 MiB  | 19.56ms        | 0.95     | 0.94     | 0.94      | 0.94      | 0.93       |
|              | BQ Cosine | 43.77s     | 29.63 MiB  | 22.91ms        | 0.95     | 0.53     | 0.51      | 0.48      | 0.45       |
| 500K         | Cosine    | 226.31s    | 2.00 GiB  | 24.23ms        |  0.86     | 0.90     | 0.90      | 0.91      | 0.89       |
|              | Hamming   | 186.54s    | 183.99 MiB | 29.10ms        | 0.91     | 0.89     | 0.89      | 0.90      | 0.89       |
|              | BQ Cosine | 301.97s    | 180.21 MiB | 32.94ms        | 0.88     | 0.53     | 0.50      | 0.48      | 0.47       |
| 1M           | Cosine    | 506.41s    | 4.03 GiB  | 29.89ms        |  0.95     | 0.93     | 0.94      | 0.94      | 0.94       |
|              | Hamming   | 418.03s    | 433.24 MiB | 32.90ms        | 0.96     | 0.92     | 0.92      | 0.93      | 0.92       |
|              | BQ Cosine | 648.22s    | 425.88 MiB | 36.67ms        | 0.96     | 0.55     | 0.52      | 0.49      | 0.50       |


### Wikipedia 22-12 Simple Embeddings (768 dimensions)
- hannoy: `M=16`, `ef_construction=48`, `ef_search=5*nns.min(100)`
- distance: cosine

| # of Vectors | Metric    | Build Time | DB Size    | Search Latency | Recall@1 | Recall@5 | Recall@10 | Recall@50 | Recall@100 |
|--------------|-----------|------------|------------|----------------|----------|----------|-----------|-----------|------------|
| 10K          | Cosine    | 259.23ms   | 40.17 MiB | 6.95ms         |  0.98     | 0.99     | 0.99      | 1.00      | 1.00       |
|              | Hamming   | 166.95ms   | 2.30 MiB   | 6.39ms         | 0.98     | 0.97     | 0.97      | 0.98      | 0.98       |
|              | BQ Cosine | 290.26ms   | 2.17 MiB   | 7.35ms         | 0.98     | 0.73     | 0.71      | 0.66      | 0.62       |
| 50K          | Cosine    | 1.95s      | 200.81 MiB| 11.58ms        |  0.90     | 0.97     | 0.98      | 0.99      | 0.99       |
|              | Hamming   | 1.06s      | 11.48 MiB  | 8.95ms         | 0.91     | 0.95     | 0.96      | 0.96      | 0.97       |
|              | BQ Cosine | 1.82s      | 11.06 MiB  | 10.50ms        | 0.93     | 0.72     | 0.69      | 0.60      | 0.57       |
| 100K         | Cosine    | 4.91s      | 402.59 MiB| 13.31ms        |  0.89     | 0.97     | 0.97      | 0.98      | 0.99       |
|              | Hamming   | 2.45s      | 24.10 MiB  | 10.08ms        | 0.87     | 0.97     | 0.95      | 0.96      | 0.96       |
|              | BQ Cosine | 4.17s      | 23.36 MiB  | 11.87ms        | 0.92     | 0.66     | 0.65      | 0.58      | 0.56       |
| 485K         | Cosine    | 36.10s     | 1.92 GiB  | 20.00ms        |  0.77     | 0.86     | 0.89      | 0.94      | 0.96       |
|              | Hamming   | 18.43s     | 139.33 MiB | 16.12ms        | 0.79     | 0.86     | 0.86      | 0.90      | 0.91       |
|              | BQ Cosine | 28.73s     | 135.14 MiB | 18.48ms        | 0.73     | 0.58     | 0.54      | 0.49      | 0.48       |

### DB Pedia OpenAI text-embedding-ada-002 (1536 dimensions)
- hannoy: `M=16`, `ef_construction=33`, `ef_search=5*nns.min(100)`
- distance: cosine

| # of Vectors | Metric    | Build Time | DB Size    | Search Latency | Recall@1 | Recall@5 | Recall@10 | Recall@50 | Recall@100 |
|--------------|-----------|------------|------------|----------------|----------|----------|-----------|-----------|------------|
| 10K          | Cosine    | 474.26ms   | 79.49 MiB | 9.53ms         |  0.95     | 0.95     | 0.95      | 0.98      | 0.98       |
|              | Hamming   | 191.43ms   | 3.45 MiB   | 7.09ms         | 0.97     | 0.96     | 0.97      | 0.98      | 0.98       |
|              | BQ Cosine | 410.61ms   | 3.45 MiB   | 9.79ms         | 1.00     | 0.74     | 0.72      | 0.71      | 0.70       |
| 50K          | Cosine    | 3.61s      | 397.35 MiB| 12.53ms        |  0.93     | 0.92     | 0.93      | 0.95      | 0.96       |
|              | Hamming   | 1.01s      | 17.19 MiB  | 8.91ms         | 0.95     | 0.92     | 0.92      | 0.96      | 0.96       |
|              | BQ Cosine | 2.27s      | 17.11 MiB  | 12.41ms        | 0.92     | 0.74     | 0.73      | 0.70      | 0.70       |
| 100K         | Cosine    | 12.28s     | 796.44 MiB| 24.51ms        |  0.97     | 0.95     | 0.96      | 0.98      | 0.98       |
|              | Hamming   | 2.31s      | 35.38 MiB  | 10.85ms        | 0.86     | 0.89     | 0.91      | 0.94      | 0.95       |
|              | BQ Cosine | 4.76s      | 35.41 MiB  | 14.30ms        | 0.89     | 0.72     | 0.72      | 0.71      | 0.70       |
| 500K         | Cosine    | 72.18s     | 3.92 GiB  | 29.87ms        |  0.93     | 0.92     | 0.94      | 0.96      | 0.97       |
|              | Hamming   | 16.01s     | 204.70 MiB | 14.65ms        | 0.80     | 0.87     | 0.89      | 0.92      | 0.93       |
|              | BQ Cosine | 27.86s     | 204.45 MiB | 17.96ms        | 0.89     | 0.73     | 0.70      | 0.71      | 0.70       |
| 1M           | Cosine    | 152.81s    | 7.87 GiB  | 30.54ms        |  0.91     | 0.90     | 0.91      | 0.95      | 0.97       |
|              | Hamming   | 36.71s     | 445.76 MiB | 15.59ms        | 0.78     | 0.83     | 0.83      | 0.92      | 0.93       |
|              | BQ Cosine | 58.98s     | 445.71 MiB | 18.66ms        | 0.75     | 0.70     | 0.69      | 0.68      | 0.69       |

### DB Pedia OpenAI text-embedding-3-large (3072 dimensions)
- hannoy: `M=16`, `ef_construction=33`, `ef_search=5*nns.min(100)`
- distance: cosine

| # of Vectors | Metric    | Build Time | DB Size    | Search Latency | Recall@1 | Recall@5 | Recall@10 | Recall@50 | Recall@100 |
|--------------|-----------|------------|------------|----------------|----------|----------|-----------|-----------|------------|
| 10K          | Cosine    | 1.49s      | 157.71 MiB| 27.67ms        |  1.00     | 0.99     | 0.99      | 0.99      | 1.00       |
|              | Hamming   | 202.53ms   | 6.03 MiB   | 7.62ms         | 0.99     | 0.96     | 0.97      | 0.98      | 0.99       |
|              | BQ Cosine | 636.52ms   | 5.50 MiB   | 13.03ms        | 1.00     | 0.82     | 0.80      | 0.77      | 0.76       |
| 50K          | Cosine    | 10.76s     | 788.34 MiB| 38.16ms        |  0.98     | 0.95     | 0.95      | 0.98      | 0.99       |
|              | Hamming   | 1.18s      | 30.09 MiB  | 10.52ms        | 0.92     | 0.92     | 0.93      | 0.96      | 0.97       |
|              | BQ Cosine | 3.47s      | 27.37 MiB  | 16.46ms        | 0.92     | 0.81     | 0.78      | 0.77      | 0.77       |
| 100K         | Cosine    | 23.21s     | 1.54 GiB  | 41.27ms        |  0.99     | 0.94     | 0.94      | 0.97      | 0.98       |
|              | Hamming   | 2.67s      | 61.22 MiB  | 12.47ms        | 0.90     | 0.90     | 0.92      | 0.95      | 0.96       |
|              | BQ Cosine | 7.29s      | 55.78 MiB  | 17.87ms        | 0.91     | 0.78     | 0.76      | 0.77      | 0.77       |
| 500K         | Cosine    | 124.33s    | 7.73 GiB  | 45.73ms        |  0.94     | 0.92     | 0.94      | 0.96      | 0.97       |
|              | Hamming   | 18.63s     | 332.71 MiB | 15.13ms        | 0.88     | 0.92     | 0.89      | 0.92      | 0.93       |
|              | BQ Cosine | 40.51s     | 305.28 MiB | 20.65ms        | 0.83     | 0.75     | 0.75      | 0.77      | 0.77       |
| 1M           | Cosine    | 253.80s    | 15.50 GiB | 45.17ms        |  0.87     | 0.93     | 0.94      | 0.95      | 0.96       |
|              | Hamming   | 42.05s     | 699.48 MiB | 15.97ms        | 0.76     | 0.83     | 0.85      | 0.91      | 0.93       |
|              | BQ Cosine | 85.06s     | 646.07 MiB | 20.87ms        | 0.78     | 0.77     | 0.74      | 0.76      | 0.76       |

## hamming but with more compute

### DB Pedia OpenAI text-embedding-ada-002 (1536 dimensions)
- `M=16`, `ef_construction=64`
- `ef_search=100`

| # of Vectors | Metric  | Build Time | DB Size    | Search Latency | Recall\@1 | Recall\@5 | Recall\@10 | Recall\@50 | Recall\@100 |
| ------------ | ------- | ---------- | ---------- | -------------- | --------- | --------- | ---------- | ---------- | ----------- |
| 10K          | Hamming | 345.10ms   | 3.59 MiB   | 6.27ms         | 1.00      | 0.99      | 0.99       | 0.96       | 0.93        |
| 50K          | Hamming | 1.96s      | 17.94 MiB  | 7.66ms         | 1.00      | 0.98      | 0.98       | 0.94       | 0.89        |
| 100K         | Hamming | 4.35s      | 36.98 MiB  | 8.80ms         | 1.00      | 0.97      | 0.96       | 0.93       | 0.87        |
| 500K         | Hamming | 30.22s     | 215.59 MiB | 12.50ms        | 1.00      | 0.97      | 0.96       | 0.91       | 0.86        |
| 999K         | Hamming | 67.32s     | 481.26 MiB | 13.30ms        | 1.00      | 0.95      | 0.94       | 0.91       | 0.87        |

### DB Pedia OpenAI text-embedding-3-large (3072 dimensions)
- `M=16`, `ef_construction=64`
- `ef_search=100`

| # of Vectors | Metric  | Build Time | DB Size    | Search Latency | Recall\@1 | Recall\@5 | Recall\@10 | Recall\@50 | Recall\@100 |
| ------------ | ------- | ---------- | ---------- | -------------- | --------- | --------- | ---------- | ---------- | ----------- |
| 10K          | Hamming | 378.69ms   | 6.20 MiB   | 6.61ms         | 1.00      | 0.99      | 0.99       | 0.97       | 0.94        |
| 50K          | Hamming | 2.22s      | 30.74 MiB  | 8.64ms         | 1.00      | 0.98      | 0.98       | 0.94       | 0.91        |
| 100K         | Hamming | 5.13s      | 62.50 MiB  | 10.46ms        | 1.00      | 0.97      | 0.97       | 0.94       | 0.89        |
| 500K         | Hamming | 34.58s     | 343.96 MiB | 13.01ms        | 1.00      | 0.97      | 0.96       | 0.92       | 0.88        |
| 999K         | Hamming | 76.56s     | 736.45 MiB | 13.69ms        | 1.00      | 0.97      | 0.96       | 0.92       | 0.88        |
