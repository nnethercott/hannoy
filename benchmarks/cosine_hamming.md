# consine vs hamming vs bq-cosine

Results below ran on a machine with 8xIntel(R) Core(TM) i7-6900K CPUs @ 3.20GHz using [this branch](https://github.com/meilisearch/vector-store-relevancy-benchmark/compare/main...nnethercott:vector-store-relevancy-benchmark:arroy-hannoy) in [this repo](https://github.com/meilisearch/vector-store-relevancy-benchmark).

## Datacomp Small (768 dimensions)
- hannoy: `M=24`, `ef_construction=512`, `ef_search=200`
- distance: cosine

| # of Vectors | Method    | Recall@1 | Recall@5 | Recall@10 | Recall@50 | Recall@100 | Build Time | DB Size    | Search Latency |
|--------------|-----------|----------|----------|-----------|-----------|------------|------------|------------|----------------|
| 10K          | Cosine    | 0.91     | 0.95     | 0.95      | 0.97      | 0.97       | 1.16s      | 40.31 MiB | 9.53ms         |
|              | Hamming   | 0.96     | 0.99     | 0.98      | 0.98      | 0.98       | 1.13s      | 2.87 MiB   | 13.19ms        |
|              | BQ Cosine | 0.95     | 0.59     | 0.55      | 0.50      | 0.48       | 1.87s      | 2.80 MiB   | 15.78ms        |
| 50K          | Cosine | 0.91     | 0.93     | 0.93      | 0.95      | 0.91       | 12.27s     | 201.39 MiB| 13.49ms        |
|              | Hamming   | 0.93     | 0.95     | 0.94      | 0.95      | 0.94       | 9.10s      | 14.54 MiB  | 17.57ms        |
|              | BQ Cosine | 0.93     | 0.58     | 0.54      | 0.47      | 0.45       | 17.29s     | 14.24 MiB  | 21.36ms        |
| 100K         | Cosine | 0.92     | 0.92     | 0.93      | 0.94      | 0.92       | 31.51s     | 404.24 MiB| 15.73ms        |
|              | Hamming   | 0.95     | 0.94     | 0.94      | 0.94      | 0.93       | 22.77s     | 30.31 MiB  | 19.56ms        |
|              | BQ Cosine | 0.95     | 0.53     | 0.51      | 0.48      | 0.45       | 43.77s     | 29.63 MiB  | 22.91ms        |
| 500K         | Cosine | 0.86     | 0.90     | 0.90      | 0.91      | 0.89       | 226.31s    | 2.00 GiB  | 24.23ms        |
|              | Hamming   | 0.91     | 0.89     | 0.89      | 0.90      | 0.89       | 186.54s    | 183.99 MiB | 29.10ms        |
|              | BQ Cosine | 0.88     | 0.53     | 0.50      | 0.48      | 0.47       | 301.97s    | 180.21 MiB | 32.94ms        |
| 1M           | Cosine | 0.95     | 0.93     | 0.94      | 0.94      | 0.94       | 506.41s    | 4.03 GiB  | 29.89ms        |
|              | Hamming   | 0.96     | 0.92     | 0.92      | 0.93      | 0.92       | 418.03s    | 433.24 MiB | 32.90ms        |
|              | BQ Cosine | 0.96     | 0.55     | 0.52      | 0.49      | 0.50       | 648.22s    | 425.88 MiB | 36.67ms        |


## Wikipedia 22-12 Simple Embeddings (768 dimensions)
- hannoy: `M=16`, `ef_construction=48`, `ef_search=5*nns.min(100)`
- distance: cosine

| # of Vectors | Method    | Recall@1 | Recall@5 | Recall@10 | Recall@50 | Recall@100 | Build Time | DB Size    | Search Latency |
|--------------|-----------|----------|----------|-----------|-----------|------------|------------|------------|----------------|
| 10K          | Cosine | 0.98     | 0.99     | 0.99      | 1.00      | 1.00       | 259.23ms   | 40.17 MiB | 6.95ms         |
|          | Hamming   | 0.98     | 0.97     | 0.97      | 0.98      | 0.98       | 166.95ms   | 2.30 MiB   | 6.39ms         |
|          | BQ Cosine | 0.98     | 0.73     | 0.71      | 0.66      | 0.62       | 290.26ms   | 2.17 MiB   | 7.35ms         |
| 50K          | Cosine | 0.90     | 0.97     | 0.98      | 0.99      | 0.99       | 1.95s      | 200.81 MiB| 11.58ms        |
|          | Hamming   | 0.91     | 0.95     | 0.96      | 0.96      | 0.97       | 1.06s      | 11.48 MiB  | 8.95ms         |
|          | BQ Cosine | 0.93     | 0.72     | 0.69      | 0.60      | 0.57       | 1.82s      | 11.06 MiB  | 10.50ms        |
| 100K         | Cosine | 0.89     | 0.97     | 0.97      | 0.98      | 0.99       | 4.91s      | 402.59 MiB| 13.31ms        |
|          | Hamming   | 0.87     | 0.97     | 0.95      | 0.96      | 0.96       | 2.45s      | 24.10 MiB  | 10.08ms        |
|          | BQ Cosine | 0.92     | 0.66     | 0.65      | 0.58      | 0.56       | 4.17s      | 23.36 MiB  | 11.87ms        |
| 485K         | Cosine | 0.77     | 0.86     | 0.89      | 0.94      | 0.96       | 36.10s     | 1.92 GiB  | 20.00ms        |
|              | Hamming   | 0.79     | 0.86     | 0.86      | 0.90      | 0.91       | 18.43s     | 139.33 MiB | 16.12ms        |
|          | BQ Cosine | 0.73     | 0.58     | 0.54      | 0.49      | 0.48       | 28.73s     | 135.14 MiB | 18.48ms        |

## DB Pedia OpenAI text-embedding-ada-002 (1536 dimensions)
- hannoy: `M=16`, `ef_construction=33`, `ef_search=5*nns.min(100)`
- distance: cosine

| # of Vectors | Method    | Recall@1 | Recall@5 | Recall@10 | Recall@50 | Recall@100 | Build Time | DB Size    | Search Latency |
|--------------|-----------|----------|----------|-----------|-----------|------------|------------|------------|----------------|
| 10K          | Cosine | 0.95     | 0.95     | 0.95      | 0.98      | 0.98       | 474.26ms   | 79.49 MiB | 9.53ms         |
|          | Hamming   | 0.97     | 0.96     | 0.97      | 0.98      | 0.98       | 191.43ms   | 3.45 MiB   | 7.09ms         |
|          | BQ Cosine | 1.00     | 0.74     | 0.72      | 0.71      | 0.70       | 410.61ms   | 3.45 MiB   | 9.79ms         |
| 50K          | Cosine | 0.93     | 0.92     | 0.93      | 0.95      | 0.96       | 3.61s      | 397.35 MiB| 12.53ms        |
|          | Hamming   | 0.95     | 0.92     | 0.92      | 0.96      | 0.96       | 1.01s      | 17.19 MiB  | 8.91ms         |
|          | BQ Cosine | 0.92     | 0.74     | 0.73      | 0.70      | 0.70       | 2.27s      | 17.11 MiB  | 12.41ms        |
| 100K         | Cosine | 0.97     | 0.95     | 0.96      | 0.98      | 0.98       | 12.28s     | 796.44 MiB| 24.51ms        |
|          | Hamming   | 0.86     | 0.89     | 0.91      | 0.94      | 0.95       | 2.31s      | 35.38 MiB  | 10.85ms        |
|          | BQ Cosine | 0.89     | 0.72     | 0.72      | 0.71      | 0.70       | 4.76s      | 35.41 MiB  | 14.30ms        |
| 500K         | Cosine | 0.93     | 0.92     | 0.94      | 0.96      | 0.97       | 72.18s     | 3.92 GiB  | 29.87ms        |
|          | Hamming   | 0.80     | 0.87     | 0.89      | 0.92      | 0.93       | 16.01s     | 204.70 MiB | 14.65ms        |
|          | BQ Cosine | 0.89     | 0.73     | 0.70      | 0.71      | 0.70       | 27.86s     | 204.45 MiB | 17.96ms        |
| 1M           | Cosine | 0.91     | 0.90     | 0.91      | 0.95      | 0.97       | 152.81s    | 7.87 GiB  | 30.54ms        |
|              | Hamming   | 0.78     | 0.83     | 0.83      | 0.92      | 0.93       | 36.71s     | 445.76 MiB | 15.59ms        |
|          | BQ Cosine | 0.75     | 0.70     | 0.69      | 0.68      | 0.69       | 58.98s     | 445.71 MiB | 18.66ms        |

## DB Pedia OpenAI text-embedding-3-large (3072 dimensions)
- hannoy: `M=16`, `ef_construction=33`, `ef_search=5*nns.min(100)`
- distance: cosine

| # of Vectors | Method    | Recall@1 | Recall@5 | Recall@10 | Recall@50 | Recall@100 | Build Time | DB Size    | Search Latency |
|--------------|-----------|----------|----------|-----------|-----------|------------|------------|------------|----------------|
| 10K          | Cosine | 1.00     | 0.99     | 0.99      | 0.99      | 1.00       | 1.49s      | 157.71 MiB| 27.67ms        |
|          | Hamming   | 0.99     | 0.96     | 0.97      | 0.98      | 0.99       | 202.53ms   | 6.03 MiB   | 7.62ms         |
|          | BQ Cosine | 1.00     | 0.82     | 0.80      | 0.77      | 0.76       | 636.52ms   | 5.50 MiB   | 13.03ms        |
| 50K          | Cosine | 0.98     | 0.95     | 0.95      | 0.98      | 0.99       | 10.76s     | 788.34 MiB| 38.16ms        |
|          | Hamming   | 0.92     | 0.92     | 0.93      | 0.96      | 0.97       | 1.18s      | 30.09 MiB  | 10.52ms        |
|          | BQ Cosine | 0.92     | 0.81     | 0.78      | 0.77      | 0.77       | 3.47s      | 27.37 MiB  | 16.46ms        |
| 100K         | Cosine | 0.99     | 0.94     | 0.94      | 0.97      | 0.98       | 23.21s     | 1.54 GiB  | 41.27ms        |
|          | Hamming   | 0.90     | 0.90     | 0.92      | 0.95      | 0.96       | 2.67s      | 61.22 MiB  | 12.47ms        |
|          | BQ Cosine | 0.91     | 0.78     | 0.76      | 0.77      | 0.77       | 7.29s      | 55.78 MiB  | 17.87ms        |
| 500K         | Cosine | 0.94     | 0.92     | 0.94      | 0.96      | 0.97       | 124.33s    | 7.73 GiB  | 45.73ms        |
|          | Hamming   | 0.88     | 0.92     | 0.89      | 0.92      | 0.93       | 18.63s     | 332.71 MiB | 15.13ms        |
|          | BQ Cosine | 0.83     | 0.75     | 0.75      | 0.77      | 0.77       | 40.51s     | 305.28 MiB | 20.65ms        |
| 1M           | Cosine | 0.87     | 0.93     | 0.94      | 0.95      | 0.96       | 253.80s    | 15.50 GiB | 45.17ms        |
|              | Hamming   | 0.76     | 0.83     | 0.85      | 0.91      | 0.93       | 42.05s     | 699.48 MiB | 15.97ms        |
|          | BQ Cosine | 0.78     | 0.77     | 0.74      | 0.76      | 0.76       | 85.06s     | 646.07 MiB | 20.87ms        |
