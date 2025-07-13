# benchmarks
Results below ran on a machine with 16xIntel(R) Core(TM) i7-6900K CPUs @ 3.20GHz using [this branch](https://github.com/meilisearch/vector-store-relevancy-benchmark/compare/main...nnethercott:vector-store-relevancy-benchmark:arroy-hannoy) in [this repo](https://github.com/meilisearch/vector-store-relevancy-benchmark).

*note: latencies seem *off by a factor of 10* vs those in [kero's blog post](https://blog.kerollmops.com/meilisearch-vs-qdrant-tradeoffs-strengths-and-weaknesses) ...

## datacomp-small-768
- hannoy: `M=24`, `ef_construction=512`, `ef_search=200`
- distance: cosine

| # of Vectors | Build Time | DB Size     | Recall@1 | Recall@5 | Recall@10 | Recall@50 | Recall@100 | Search Latency (ms)                             |
|--------------|------------|-------------|----------|----------|-----------|-----------|------------|-------------------------------------------------|
| 10K (arroy)  | 1.18 s     | 71.30 MiB   | 0.95     | 0.84     | 0.84      | 0.91      | 0.95       | 16.75 ms (100%) / 11.25 ms (10%)               |
| 10K (hannoy) | 1.16 s     | 40.31 MiB   | 0.91     | 0.95     | 0.95      | 0.97      | 0.97       | 9.53 ms                                        |
| 50K (arroy)  | 25.61 s    | 451.89 MiB  | 0.93     | 0.78     | 0.80      | 0.88      | 0.88       | 43.81 ms (100%) / 27.92 ms (10%)               |
| 50K (hannoy) | 12.27 s    | 201.39 MiB  | 0.91     | 0.93     | 0.93      | 0.95      | 0.91       | 13.49 ms                                       |
| 100K (arroy) | 78.31 s    | 1.02 GiB    | 0.95     | 0.75     | 0.77      | 0.85      | 0.89       | 65.95 ms (100%) / 46.91 ms (10%)               |
| 100K (hannoy)| 31.51 s    | 404.24 MiB  | 0.92     | 0.92     | 0.93      | 0.94      | 0.92       | 15.73 ms                                       |
| 500K (arroy) | 860.64 s   | 6.87 GiB    | 0.88     | 0.75     | 0.77      | 0.85      | 0.87       | 142.34 ms (100%) / 114.12 ms (10%)             |
| 500K (hannoy)| 226.31 s   | 2.00 GiB    | 0.86     | 0.90     | 0.90      | 0.91      | 0.89       | 24.23 ms                                       |
| 1M (arroy)   | 2386.92 s  | 16.19 GiB   | 0.96     | 0.80     | 0.83      | 0.87      | 0.90       | 190.84 ms (100%) / 160.12 ms (10%)             |
| 1M (hannoy)  | 506.41 s   | 4.03 GiB    | 0.95     | 0.93     | 0.94      | 0.94      | 0.94       | 29.89 ms                                       |

## wikipedia-22-12-simple-768
- hannoy: `M=16`, `ef_construction=48`, `ef_search=5*nns.min(100)`
- distance: cosine

| # of Vectors | Build Time | DB Size     | Recall@1 | Recall@5 | Recall@10 | Recall@50 | Recall@100 | Search Latency (ms)                             |
|--------------|------------|-------------|----------|----------|-----------|-----------|------------|-------------------------------------------------|
| 10K (arroy)  | 723.92 ms  | 69.18 MiB   | 1.00     | 0.96     | 0.98      | 0.99      | 1.00       | 15.82 ms (100%) / 9.43 ms (10%)                |
| 10K (hannoy) | 259.23 ms  | 40.17 MiB   | 0.98     | 0.99     | 0.99      | 1.00      | 1.00       | 6.95 ms                                        |
| 50K (arroy)  | 15.19 s    | 445.64 MiB  | 1.00     | 0.90     | 0.92      | 0.97      | 0.99       | 48.11 ms (100%) / 26.36 ms (10%)               |
| 50K (hannoy) | 1.95 s     | 200.81 MiB  | 0.90     | 0.97     | 0.98      | 0.99      | 0.99       | 11.58 ms                                       |
| 100K (arroy) | 46.77 s    | 1008.54 MiB | 1.00     | 0.89     | 0.92      | 0.97      | 0.98       | 72.95 ms (100%) / 44.91 ms (10%)               |
| 100K (hannoy)| 4.91 s     | 402.59 MiB  | 0.89     | 0.97     | 0.97      | 0.98      | 0.99       | 13.31 ms                                       |
| 485K (arroy) | 483.82 s   | 6.64 GiB    | 1.00     | 0.81     | 0.86      | 0.96      | 0.97       | 152.87 ms (100%) / 117.14 ms (10%)             |
| 485K (hannoy)| 36.10 s    | 1.92 GiB    | 0.77     | 0.86     | 0.89      | 0.94      | 0.96       | 20.00 ms                                       |

## db-pedia-ada002-1536
- hannoy: `M=16`, `ef_construction=33`, `ef_search=5*nns.min(100)`
- distance: cosine

| # of Vectors | Build Time | DB Size     | Recall@1 | Recall@5 | Recall@10 | Recall@50 | Recall@100 | Search Latency (ms)                             |
|--------------|------------|-------------|----------|----------|-----------|-----------|------------|-------------------------------------------------|
| 10K (arroy)  | 494.72 ms  | 93.94 MiB   | 1.00     | 0.80     | 0.86      | 0.98      | 0.99       | 22.15 ms (100%) / 8.93 ms (10%)                |
| 10K (hannoy) | 474.26 ms  | 79.49 MiB   | 0.95     | 0.95     | 0.95      | 0.98      | 0.98       | 9.53 ms                                        |
| 50K (arroy)  | 9.86 s     | 527.14 MiB  | 1.00     | 0.68     | 0.75      | 0.93      | 0.97       | 56.16 ms (100%) / 28.80 ms (10%)               |
| 50K (hannoy) | 3.61 s     | 397.35 MiB  | 0.93     | 0.92     | 0.93      | 0.95      | 0.96       | 12.53 ms                                       |
| 100K (arroy) | 29.73 s    | 1.10 GiB    | 1.00     | 0.69     | 0.74      | 0.91      | 0.96       | 89.72 ms (100%) / 51.55 ms (10%)               |
| 100K (hannoy)| 12.28 s    | 796.44 MiB  | 0.97     | 0.95     | 0.96      | 0.98      | 0.98       | 24.51 ms                                       |
| 500K (arroy) | 343.20 s   | 6.55 GiB    | 1.00     | 0.70     | 0.74      | 0.91      | 0.95       | 176.91 ms (100%) / 138.49 ms (10%)             |
| 500K (hannoy)| 72.18 s    | 3.92 GiB    | 0.93     | 0.92     | 0.94      | 0.96      | 0.97       | 29.87 ms                                       |
| 1M (arroy)   | 955.92 s   | 14.45 GiB   | 1.00     | 0.75     | 0.77      | 0.92      | 0.95       | 227.89 ms (100%) / 191.47 ms (10%)             |
| 1M (hannoy)  | 152.81 s   | 7.87 GiB    | 0.91     | 0.90     | 0.91      | 0.95      | 0.97       | 30.54 ms                                       |

## db-pedia3-large-3072
- hannoy: `M=16`, `ef_construction=33`, `ef_search=5*nns.min(100)`
- distance: cosine

| # of Vectors | Build Time | DB Size     | Recall@1 | Recall@5 | Recall@10 | Recall@50 | Recall@100 | Search Latency (ms)                             |
|--------------|------------|-------------|----------|----------|-----------|-----------|------------|-------------------------------------------------|
| 10K (arroy)  | 694.63 ms  | 172.35 MiB  | 1.00     | 0.87     | 0.86      | 0.98      | 1.00       | 46.01 ms (100%) / 14.77 ms (10%)               |
| 10K (hannoy) | 1.49 s     | 157.71 MiB  | 1.00     | 0.99     | 0.99      | 0.99      | 1.00       | 27.67 ms                                       |
| 50K (arroy)  | 17.55 s    | 934.51 MiB  | 1.00     | 0.68     | 0.75      | 0.94      | 0.97       | 138.75 ms (100%) / 56.77 ms (10%)              |
| 50K (hannoy) | 10.76 s    | 788.34 MiB  | 0.98     | 0.95     | 0.95      | 0.98      | 0.99       | 38.16 ms                                       |
| 100K (arroy) | 54.16 s    | 1.90 GiB    | 1.00     | 0.64     | 0.72      | 0.91      | 0.96       | 190.94 ms (100%) / 96.88 ms (10%)              |
| 100K (hannoy)| 23.21 s    | 1.54 GiB    | 0.99     | 0.94     | 0.94      | 0.97      | 0.98       | 41.27 ms                                       |
| 500K (arroy) | 612.11 s   | 10.75 GiB   | 1.00     | 0.67     | 0.71      | 0.90      | 0.95       | 356.43 ms (100%) / 256.94 ms (10%)             |
| 500K (hannoy)| 124.33 s   | 7.73 GiB    | 0.94     | 0.92     | 0.94      | 0.96      | 0.97       | 45.73 ms                                       |
| 1M (arroy)   | 1695.77 s  | 23.02 GiB   | 1.00     | 0.71     | 0.72      | 0.90      | 0.95       | 444.07 ms (100%) / 356.85 ms (10%)             |
| 1M (hannoy)  | 253.80 s   | 15.50 GiB   | 0.87     | 0.93     | 0.94      | 0.95      | 0.96       | 45.17 ms                                       |


## raw data
<detail>
<summary>Open here</summary>

```bash
db pedia OpenAI text-embedding ada  002 - 999999 vectors of 1536 dimensions
10000 vectors are used for this measure and 18446744073709551615B of memory
Recall tested is:   [   1,    5,   10,   50,  100]
Starting indexing process
Database size: 93.94 MiB
Total time to index: 1.18s (550.82ms)
  => Vectors:        10000
  => Insertions:   56.09ms
  => Builds:      494.72ms
  => Trees:            153
  => Db size:    93.94 MiB
[arroy]  Cosine x1: [1.00, 0.80, 0.86, 0.98, 0.99], searched for: 22.15ms, searched in 100.00%
[arroy]  Cosine x1: [1.00, 0.95, 0.99, 1.00, 1.00], searched for: 8.93ms, searched in 10.00%

Recall tested is:   [   1,    5,   10,   50,  100]
Starting indexing process
Database size: 79.49 MiB
Total time to index: 1.03s (495.30ms)
  => Vectors:        10000
  => Insertions:   21.04ms
  => Builds:      474.26ms
  => Db size:    79.49 MiB
[hannoy]  Cosine [0.95, 0.95, 0.95, 0.98, 0.98], searched for: 9.61ms, searched in 100.00%
[hannoy]  Cosine [0.95, 0.95, 0.95, 0.98, 0.98], searched for: 9.53ms, searched in 10.00%

db pedia OpenAI text-embedding ada  002 - 999999 vectors of 1536 dimensions
50000 vectors are used for this measure and 18446744073709551615B of memory
Recall tested is:   [   1,    5,   10,   50,  100]
Starting indexing process
Database size: 527.14 MiB
Total time to index: 12.71s (10.12s)
  => Vectors:         50000
  => Insertions:   256.68ms
  => Builds:          9.86s
  => Trees:             247
  => Db size:    527.14 MiB
[arroy]  Cosine x1: [1.00, 0.68, 0.75, 0.93, 0.97], searched for: 56.16ms, searched in 100.00%
[arroy]  Cosine x1: [1.00, 0.84, 0.92, 0.99, 1.00], searched for: 28.80ms, searched in 10.00%

Recall tested is:   [   1,    5,   10,   50,  100]
Starting indexing process
Database size: 397.35 MiB
Total time to index: 5.81s (3.72s)
  => Vectors:         50000
  => Insertions:   108.01ms
  => Builds:          3.61s
  => Db size:    397.35 MiB
[hannoy]  Cosine [0.93, 0.92, 0.93, 0.95, 0.96], searched for: 13.15ms, searched in 100.00%
[hannoy]  Cosine [0.93, 0.92, 0.93, 0.95, 0.96], searched for: 12.53ms, searched in 10.00%

db pedia OpenAI text-embedding ada  002 - 999999 vectors of 1536 dimensions
100000 vectors are used for this measure and 18446744073709551615B of memory
Recall tested is:   [   1,    5,   10,   50,  100]
db pedia OpenAI text-embedding ada  002 - 999999 vectors of 1536 dimensions
10000 vectors are used for this measure and 18446744073709551615B of memory
Recall tested is:   [   1,    5,   10,   50,  100]
Starting indexing process
Database size: 93.89 MiB
Total time to index: 1.18s (511.79ms)
  => Vectors:        10000
  => Insertions:   48.94ms
  => Builds:      462.85ms
  => Trees:            153
  => Db size:    93.89 MiB
[arroy]  Cosine x1: [1.00, 0.80, 0.86, 0.98, 0.99], searched for: 21.37ms, searched in 100.00%
[arroy]  Cosine x1: [1.00, 0.95, 0.99, 1.00, 1.00], searched for: 8.78ms, searched in 10.00%

Recall tested is:   [   1,    5,   10,   50,  100]
Starting indexing process
Database size: 79.54 MiB
Total time to index: 1.15s (727.26ms)
  => Vectors:        10000
  => Insertions:   26.13ms
  => Builds:      701.13ms
  => Db size:    79.54 MiB
[hannoy]  Cosine [1.00, 0.99, 0.99, 0.99, 1.00], searched for: 15.43ms, searched in 100.00%
[hannoy]  Cosine [1.00, 0.99, 0.99, 0.99, 1.00], searched for: 14.91ms, searched in 10.00%

db pedia OpenAI text-embedding ada  002 - 999999 vectors of 1536 dimensions
50000 vectors are used for this measure and 18446744073709551615B of memory
Recall tested is:   [   1,    5,   10,   50,  100]
Starting indexing process
Database size: 527.16 MiB
Total time to index: 12.85s (9.96s)
  => Vectors:         50000
  => Insertions:   237.47ms
  => Builds:          9.73s
  => Trees:             247
  => Db size:    527.16 MiB
[arroy]  Cosine x1: [1.00, 0.68, 0.75, 0.93, 0.97], searched for: 55.65ms, searched in 100.00%
[arroy]  Cosine x1: [1.00, 0.84, 0.92, 0.99, 1.00], searched for: 29.99ms, searched in 10.00%

Recall tested is:   [   1,    5,   10,   50,  100]
Starting indexing process
Database size: 397.6 MiB
Total time to index: 7.89s (5.61s)
  => Vectors:         50000
  => Insertions:   111.55ms
  => Builds:          5.49s
  => Db size:    397.60 MiB
[hannoy]  Cosine [0.97, 0.97, 0.96, 0.98, 0.99], searched for: 23.02ms, searched in 100.00%
[hannoy]  Cosine [0.97, 0.97, 0.96, 0.98, 0.99], searched for: 22.36ms, searched in 10.00%

db pedia OpenAI text-embedding ada  002 - 999999 vectors of 1536 dimensions
100000 vectors are used for this measure and 18446744073709551615B of memory
Recall tested is:   [   1,    5,   10,   50,  100]
Starting indexing process
Database size: 1.1 GiB
Total time to index: 35.15s (30.21s)
  => Vectors:      100000
  => Insertions: 486.32ms
  => Builds:       29.73s
  => Trees:           305
  => Db size:    1.10 GiB
[arroy]  Cosine x1: [1.00, 0.69, 0.74, 0.91, 0.96], searched for: 89.72ms, searched in 100.00%
[arroy]  Cosine x1: [1.00, 0.83, 0.89, 0.99, 0.99], searched for: 51.55ms, searched in 10.00%

Recall tested is:   [   1,    5,   10,   50,  100]
Starting indexing process
Database size: 796.44 MiB
Total time to index: 16.42s (12.50s)
  => Vectors:        100000
  => Insertions:   222.59ms
  => Builds:         12.28s
  => Db size:    796.44 MiB
[hannoy]  Cosine [0.97, 0.95, 0.96, 0.98, 0.98], searched for: 25.33ms, searched in 100.00%
[hannoy]  Cosine [0.97, 0.95, 0.96, 0.98, 0.98], searched for: 24.51ms, searched in 10.00%

db pedia OpenAI text-embedding ada  002 - 999999 vectors of 1536 dimensions
500000 vectors are used for this measure and 18446744073709551615B of memory
Recall tested is:   [   1,    5,   10,   50,  100]
Starting indexing process
Database size: 6.55 GiB
Total time to index: 362.30s (346.54s)
  => Vectors:      500000
  => Insertions:    3.33s
  => Builds:      343.20s
  => Trees:           494
  => Db size:    6.55 GiB
[arroy]  Cosine x1: [1.00, 0.70, 0.74, 0.91, 0.95], searched for: 176.91ms, searched in 100.00%
[arroy]  Cosine x1: [1.00, 0.78, 0.83, 0.96, 0.98], searched for: 138.49ms, searched in 10.00%

Recall tested is:   [   1,    5,   10,   50,  100]
Starting indexing process
Database size: 3.92 GiB
Total time to index: 79.28s (75.11s)
  => Vectors:      500000
  => Insertions:    2.93s
  => Builds:       72.18s
  => Db size:    3.92 GiB
[hannoy]  Cosine [0.93, 0.92, 0.94, 0.96, 0.97], searched for: 31.10ms, searched in 100.00%
[hannoy]  Cosine [0.93, 0.92, 0.94, 0.96, 0.97], searched for: 29.87ms, searched in 10.00%

db pedia OpenAI text-embedding ada  002 - 999999 vectors of 1536 dimensions
999999 vectors are used for this measure and 18446744073709551615B of memory
Recall tested is:   [   1,    5,   10,   50,  100]
Starting indexing process
Database size: 14.45 GiB
Total time to index: 1018.19s (963.07s)
  => Vectors:       999999
  => Insertions:     7.15s
  => Builds:       955.92s
  => Trees:            609
  => Db size:    14.45 GiB
[arroy]  Cosine x1: [1.00, 0.75, 0.77, 0.92, 0.95], searched for: 227.89ms, searched in 100.00%
[arroy]  Cosine x1: [1.00, 0.81, 0.83, 0.96, 0.98], searched for: 191.47ms, searched in 10.00%

Recall tested is:   [   1,    5,   10,   50,  100]
Starting indexing process
Database size: 7.87 GiB
Total time to index: 171.26s (159.72s)
  => Vectors:      999999
  => Insertions:    6.91s
  => Builds:      152.81s
  => Db size:    7.87 GiB
[hannoy]  Cosine [0.91, 0.90, 0.91, 0.95, 0.97], searched for: 31.19ms, searched in 100.00%
[hannoy]  Cosine [0.91, 0.90, 0.91, 0.95, 0.97], searched for: 30.54ms, searched in 10.00%

db pedia OpenAI text-embedding 3 large - 999999 vectors of 3072 dimensions
10000 vectors are used for this measure and 18446744073709551615B of memory
Recall tested is:   [   1,    5,   10,   50,  100]
Starting indexing process
Database size: 172.35 MiB
Total time to index: 2.16s (781.72ms)
  => Vectors:         10000
  => Insertions:    87.09ms
  => Builds:       694.63ms
  => Trees:             180
  => Db size:    172.35 MiB
[arroy]  Cosine x1: [1.00, 0.87, 0.86, 0.98, 1.00], searched for: 46.01ms, searched in 100.00%
[arroy]  Cosine x1: [1.00, 0.98, 1.00, 1.00, 1.00], searched for: 14.77ms, searched in 10.00%

Recall tested is:   [   1,    5,   10,   50,  100]
Starting indexing process
Database size: 157.71 MiB
Total time to index: 2.80s (1.52s)
  => Vectors:         10000
  => Insertions:    34.67ms
  => Builds:          1.49s
  => Db size:    157.71 MiB
[hannoy]  Cosine [1.00, 0.99, 0.99, 0.99, 1.00], searched for: 28.09ms, searched in 100.00%
[hannoy]  Cosine [1.00, 0.99, 0.99, 0.99, 1.00], searched for: 27.67ms, searched in 10.00%

db pedia OpenAI text-embedding 3 large - 999999 vectors of 3072 dimensions
50000 vectors are used for this measure and 18446744073709551615B of memory
Recall tested is:   [   1,    5,   10,   50,  100]
Starting indexing process
Database size: 934.51 MiB
Total time to index: 28.47s (18.00s)
  => Vectors:         50000
  => Insertions:   454.40ms
  => Builds:         17.55s
  => Trees:             293
  => Db size:    934.51 MiB
[arroy]  Cosine x1: [1.00, 0.68, 0.75, 0.94, 0.97], searched for: 138.75ms, searched in 100.00%
[arroy]  Cosine x1: [1.00, 0.85, 0.93, 1.00, 1.00], searched for: 56.77ms, searched in 10.00%

Recall tested is:   [   1,    5,   10,   50,  100]
Starting indexing process
Database size: 788.34 MiB
Total time to index: 17.12s (10.93s)
  => Vectors:         50000
  => Insertions:   171.25ms
  => Builds:         10.76s
  => Db size:    788.34 MiB
[hannoy]  Cosine [0.98, 0.95, 0.95, 0.98, 0.99], searched for: 39.41ms, searched in 100.00%
[hannoy]  Cosine [0.98, 0.95, 0.95, 0.98, 0.99], searched for: 38.16ms, searched in 10.00%

db pedia OpenAI text-embedding 3 large - 999999 vectors of 3072 dimensions
100000 vectors are used for this measure and 18446744073709551615B of memory
Recall tested is:   [   1,    5,   10,   50,  100]
Starting indexing process
Database size: 1.9 GiB
Total time to index: 66.08s (55.13s)
  => Vectors:      100000
  => Insertions: 963.71ms
  => Builds:       54.16s
  => Trees:           360
  => Db size:    1.90 GiB
[arroy]  Cosine x1: [1.00, 0.64, 0.72, 0.91, 0.96], searched for: 190.94ms, searched in 100.00%
[arroy]  Cosine x1: [1.00, 0.81, 0.89, 0.98, 1.00], searched for: 96.88ms, searched in 10.00%

Recall tested is:   [   1,    5,   10,   50,  100]
Starting indexing process
Database size: 1.54 GiB
Total time to index: 33.14s (23.57s)
  => Vectors:      100000
  => Insertions: 351.00ms
  => Builds:       23.21s
  => Db size:    1.54 GiB
[hannoy]  Cosine [0.99, 0.94, 0.94, 0.97, 0.98], searched for: 42.15ms, searched in 100.00%
[hannoy]  Cosine [0.99, 0.94, 0.94, 0.97, 0.98], searched for: 41.27ms, searched in 10.00%

db pedia OpenAI text-embedding 3 large - 999999 vectors of 3072 dimensions
500000 vectors are used for this measure and 18446744073709551615B of memory
Recall tested is:   [   1,    5,   10,   50,  100]
Starting indexing process
Database size: 10.75 GiB
Total time to index: 659.43s (618.05s)
  => Vectors:       500000
  => Insertions:     5.94s
  => Builds:       612.11s
  => Trees:            585
  => Db size:    10.75 GiB
[arroy]  Cosine x1: [1.00, 0.67, 0.71, 0.90, 0.95], searched for: 356.43ms, searched in 100.00%
[arroy]  Cosine x1: [1.00, 0.76, 0.83, 0.97, 0.99], searched for: 256.94ms, searched in 10.00%

Recall tested is:   [   1,    5,   10,   50,  100]
Starting indexing process
Database size: 7.73 GiB
Total time to index: 144.97s (129.62s)
  => Vectors:      500000
  => Insertions:    5.29s
  => Builds:      124.33s
  => Db size:    7.73 GiB
[hannoy]  Cosine [0.94, 0.92, 0.94, 0.96, 0.97], searched for: 46.71ms, searched in 100.00%
[hannoy]  Cosine [0.94, 0.92, 0.94, 0.96, 0.97], searched for: 45.73ms, searched in 10.00%

db pedia OpenAI text-embedding 3 large - 999999 vectors of 3072 dimensions
999999 vectors are used for this measure and 18446744073709551615B of memory
Recall tested is:   [   1,    5,   10,   50,  100]
Starting indexing process
Database size: 23.02 GiB
Total time to index: 1828.56s (1722.63s)
  => Vectors:       999999
  => Insertions:    26.86s
  => Builds:      1695.77s
  => Trees:            720
  => Db size:    23.02 GiB
[arroy]  Cosine x1: [1.00, 0.71, 0.72, 0.90, 0.95], searched for: 444.07ms, searched in 100.00%
[arroy]  Cosine x1: [1.00, 0.78, 0.81, 0.96, 0.98], searched for: 356.85ms, searched in 10.00%

Recall tested is:   [   1,    5,   10,   50,  100]
Starting indexing process
Database size: 15.5 GiB
Total time to index: 298.90s (280.14s)
  => Vectors:       999999
  => Insertions:    26.34s
  => Builds:       253.80s
  => Db size:    15.50 GiB
[hannoy]  Cosine [0.87, 0.93, 0.94, 0.95, 0.96], searched for: 46.09ms, searched in 100.00%
[hannoy]  Cosine [0.87, 0.93, 0.94, 0.95, 0.96], searched for: 45.17ms, searched in 10.00%

Datacomp small - 12799999 vectors of 768 dimensions
10000 vectors are used for this measure and 18446744073709551615B of memory
Recall tested is:   [   1,    5,   10,   50,  100]
Starting indexing process
Database size: 71.3 MiB
Total time to index: 2.01s (1.22s)
  => Vectors:        10000
  => Insertions:   35.42ms
  => Builds:         1.18s
  => Trees:            237
  => Db size:    71.30 MiB
[arroy]  Cosine x1: [0.95, 0.84, 0.84, 0.91, 0.95], searched for: 16.75ms, searched in 100.00%
[arroy]  Cosine x1: [0.95, 0.92, 0.94, 0.98, 0.99], searched for: 11.25ms, searched in 10.00%

Recall tested is:   [   1,    5,   10,   50,  100]
Starting indexing process
Database size: 40.31 MiB
Total time to index: 1.61s (1.18s)
  => Vectors:        10000
  => Insertions:   19.70ms
  => Builds:         1.16s
  => Db size:    40.31 MiB
[hannoy]  Cosine [0.91, 0.95, 0.95, 0.97, 0.97], searched for: 9.53ms, searched in 100.00%
[hannoy]  Cosine [0.91, 0.95, 0.95, 0.97, 0.97], searched for: 8.50ms, searched in 10.00%

Datacomp small - 12799999 vectors of 768 dimensions
50000 vectors are used for this measure and 18446744073709551615B of memory
Recall tested is:   [   1,    5,   10,   50,  100]
Starting indexing process
Database size: 451.89 MiB
Total time to index: 30.65s (25.78s)
  => Vectors:         50000
  => Insertions:   170.47ms
  => Builds:         25.61s
  => Trees:             384
  => Db size:    451.89 MiB
[arroy]  Cosine x1: [0.93, 0.78, 0.80, 0.88, 0.88], searched for: 43.81ms, searched in 100.00%
[arroy]  Cosine x1: [0.93, 0.87, 0.90, 0.95, 0.95], searched for: 27.92ms, searched in 10.00%

Recall tested is:   [   1,    5,   10,   50,  100]
Starting indexing process
Database size: 201.39 MiB
Total time to index: 14.69s (12.35s)
  => Vectors:         50000
  => Insertions:    76.62ms
  => Builds:         12.27s
  => Db size:    201.39 MiB
[hannoy]  Cosine [0.91, 0.93, 0.93, 0.95, 0.91], searched for: 14.19ms, searched in 100.00%
[hannoy]  Cosine [0.91, 0.93, 0.93, 0.95, 0.91], searched for: 13.49ms, searched in 10.00%

Datacomp small - 12799999 vectors of 768 dimensions
100000 vectors are used for this measure and 18446744073709551615B of memory
Recall tested is:   [   1,    5,   10,   50,  100]
Starting indexing process
Database size: 1017.45 MiB
Total time to index: 91.51s (78.61s)
  => Vectors:         100000
  => Insertions:    300.84ms
  => Builds:          78.31s
  => Trees:              473
  => Db size:    1017.45 MiB
[arroy]  Cosine x1: [0.95, 0.75, 0.77, 0.85, 0.89], searched for: 65.95ms, searched in 100.00%
[arroy]  Cosine x1: [0.95, 0.84, 0.88, 0.93, 0.95], searched for: 46.91ms, searched in 10.00%

Recall tested is:   [   1,    5,   10,   50,  100]
Starting indexing process
Database size: 404.24 MiB
Total time to index: 36.79s (31.67s)
  => Vectors:        100000
  => Insertions:   157.80ms
  => Builds:         31.51s
  => Db size:    404.24 MiB
[hannoy]  Cosine [0.92, 0.92, 0.93, 0.94, 0.92], searched for: 16.65ms, searched in 100.00%
[hannoy]  Cosine [0.92, 0.92, 0.93, 0.94, 0.92], searched for: 15.73ms, searched in 10.00%

Datacomp small - 12799999 vectors of 768 dimensions
500000 vectors are used for this measure and 18446744073709551615B of memory
Recall tested is:   [   1,    5,   10,   50,  100]
Starting indexing process
Database size: 6.87 GiB
Total time to index: 918.06s (862.51s)
  => Vectors:      500000
  => Insertions:    1.87s
  => Builds:      860.64s
  => Trees:           768
  => Db size:    6.87 GiB
[arroy]  Cosine x1: [0.88, 0.75, 0.77, 0.85, 0.87], searched for: 142.34ms, searched in 100.00%
[arroy]  Cosine x1: [0.88, 0.80, 0.84, 0.90, 0.92], searched for: 114.12ms, searched in 10.00%

Recall tested is:   [   1,    5,   10,   50,  100]
Starting indexing process
Database size: 2 GiB
Total time to index: 234.52s (228.07s)
  => Vectors:      500000
  => Insertions:    1.76s
  => Builds:      226.31s
  => Db size:    2.00 GiB
[hannoy]  Cosine [0.86, 0.90, 0.90, 0.91, 0.89], searched for: 24.97ms, searched in 100.00%
[hannoy]  Cosine [0.86, 0.90, 0.90, 0.91, 0.89], searched for: 24.23ms, searched in 10.00%

Datacomp small - 12799999 vectors of 768 dimensions
1000000 vectors are used for this measure and 18446744073709551615B of memory
Recall tested is:   [   1,    5,   10,   50,  100]
Starting indexing process
Database size: 16.19 GiB
Total time to index: 2553.24s (2391.13s)
  => Vectors:      1000000
  => Insertions:     4.20s
  => Builds:      2386.92s
  => Trees:            946
  => Db size:    16.19 GiB
[arroy]  Cosine x1: [0.96, 0.80, 0.83, 0.87, 0.90], searched for: 190.84ms, searched in 100.00%
[arroy]  Cosine x1: [0.96, 0.87, 0.87, 0.91, 0.94], searched for: 160.12ms, searched in 10.00%

Recall tested is:   [   1,    5,   10,   50,  100]
Starting indexing process
Database size: 4.03 GiB
Total time to index: 517.65s (510.44s)
  => Vectors:     1000000
  => Insertions:    4.03s
  => Builds:      506.41s
  => Db size:    4.03 GiB
[hannoy]  Cosine [0.95, 0.93, 0.94, 0.94, 0.94], searched for: 30.48ms, searched in 100.00%
[hannoy]  Cosine [0.95, 0.93, 0.94, 0.94, 0.94], searched for: 29.89ms, searched in 10.00%

wikipedia 22 12 simple embeddings - 485858 vectors of 768 dimensions
10000 vectors are used for this measure and 18446744073709551615B of memory
Recall tested is:   [   1,    5,   10,   50,  100]
Starting indexing process
Database size: 69.18 MiB
Total time to index: 1.52s (757.38ms)
  => Vectors:        10000
  => Insertions:   33.46ms
  => Builds:      723.92ms
  => Trees:            237
  => Db size:    69.18 MiB
[arroy]  Cosine x1: [1.00, 0.96, 0.98, 0.99, 1.00], searched for: 15.82ms, searched in 100.00%
[arroy]  Cosine x1: [1.00, 1.00, 1.00, 1.00, 1.00], searched for: 9.43ms, searched in 10.00%

Recall tested is:   [   1,    5,   10,   50,  100]
Starting indexing process
Database size: 40.17 MiB
Total time to index: 677.04ms (277.96ms)
  => Vectors:        10000
  => Insertions:   18.72ms
  => Builds:      259.23ms
  => Db size:    40.17 MiB
[hannoy]  Cosine [0.98, 0.99, 0.99, 1.00, 1.00], searched for: 6.95ms, searched in 100.00%
[hannoy]  Cosine [0.98, 0.99, 0.99, 1.00, 1.00], searched for: 6.62ms, searched in 10.00%

wikipedia 22 12 simple embeddings - 485858 vectors of 768 dimensions
50000 vectors are used for this measure and 18446744073709551615B of memory
Recall tested is:   [   1,    5,   10,   50,  100]
Starting indexing process
Database size: 445.64 MiB
Total time to index: 20.12s (15.33s)
  => Vectors:         50000
  => Insertions:   142.84ms
  => Builds:         15.19s
  => Trees:             384
  => Db size:    445.64 MiB
[arroy]  Cosine x1: [1.00, 0.90, 0.92, 0.97, 0.99], searched for: 48.11ms, searched in 100.00%
[arroy]  Cosine x1: [1.00, 0.98, 0.99, 1.00, 1.00], searched for: 26.36ms, searched in 10.00%

Recall tested is:   [   1,    5,   10,   50,  100]
Starting indexing process
Database size: 200.81 MiB
Total time to index: 4.17s (2.03s)
  => Vectors:         50000
  => Insertions:    76.93ms
  => Builds:          1.95s
  => Db size:    200.81 MiB
[hannoy]  Cosine [0.90, 0.97, 0.98, 0.99, 0.99], searched for: 12.11ms, searched in 100.00%
[hannoy]  Cosine [0.90, 0.97, 0.98, 0.99, 0.99], searched for: 11.58ms, searched in 10.00%

wikipedia 22 12 simple embeddings - 485858 vectors of 768 dimensions
100000 vectors are used for this measure and 18446744073709551615B of memory
Recall tested is:   [   1,    5,   10,   50,  100]
Starting indexing process
Database size: 1008.54 MiB
Total time to index: 56.56s (47.08s)
  => Vectors:         100000
  => Insertions:    300.38ms
  => Builds:          46.77s
  => Trees:              473
  => Db size:    1008.54 MiB
[arroy]  Cosine x1: [1.00, 0.89, 0.92, 0.97, 0.98], searched for: 72.95ms, searched in 100.00%
[arroy]  Cosine x1: [1.00, 0.97, 0.98, 1.00, 1.00], searched for: 44.91ms, searched in 10.00%

Recall tested is:   [   1,    5,   10,   50,  100]
Starting indexing process
Database size: 402.59 MiB
Total time to index: 9.49s (5.07s)
  => Vectors:        100000
  => Insertions:   155.34ms
  => Builds:          4.91s
  => Db size:    402.59 MiB
[hannoy]  Cosine [0.89, 0.97, 0.97, 0.98, 0.99], searched for: 14.19ms, searched in 100.00%
[hannoy]  Cosine [0.89, 0.97, 0.97, 0.98, 0.99], searched for: 13.31ms, searched in 10.00%

wikipedia 22 12 simple embeddings - 485858 vectors of 768 dimensions
485858 vectors are used for this measure and 18446744073709551615B of memory
Recall tested is:   [   1,    5,   10,   50,  100]
Starting indexing process
Database size: 6.64 GiB
Total time to index: 571.09s (485.68s)
  => Vectors:      485858
  => Insertions:    1.86s
  => Builds:      483.82s
  => Trees:           762
  => Db size:    6.64 GiB
[arroy]  Cosine x1: [1.00, 0.81, 0.86, 0.96, 0.97], searched for: 152.87ms, searched in 100.00%
[arroy]  Cosine x1: [1.00, 0.89, 0.94, 0.99, 1.00], searched for: 117.14ms, searched in 10.00%

Recall tested is:   [   1,    5,   10,   50,  100]
Starting indexing process
Database size: 1.92 GiB
Total time to index: 43.71s (37.75s)
  => Vectors:      485858
  => Insertions:    1.65s
  => Builds:       36.10s
  => Db size:    1.92 GiB
[hannoy]  Cosine [0.77, 0.86, 0.89, 0.94, 0.96], searched for: 20.79ms, searched in 100.00%
[hannoy]  Cosine [0.77, 0.86, 0.89, 0.94, 0.96], searched for: 20.00ms, searched in 10.00%
```

</detail>
