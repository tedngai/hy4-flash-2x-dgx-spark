# Measured inference matrix

Effective prefill includes queueing and mixed decode work until the last request reaches its first token. Total decode is the sum of per-request token rates inside the common decode interval. Shared decode/request is the median rate in that same interval; per-request decode is the median of each request's own decode window and is defined even when no common interval exists. TTFT is the median time to first token.

Synthetic repeated notes with unique prefixes; 512-token forced output budget; prefix reuse excluded by a nonce prefix. This is not a quality test.

| Input tokens/request | C | Effective prefill tok/s | Total decode tok/s | Shared decode/request tok/s | Per-request decode tok/s | TTFT s | Shared decode s | Peak overlap | Success |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1 | 955.7 | 35.2 | 35.2 | 36.2 | 0.5 | 14.5 | 1 | 1/1 |
| 512 | 2 | 987.0 | 75.2 | 37.6 | 38.5 | 1.0 | 12.2 | 2 | 2/2 |
| 512 | 4 | 1241.1 | 117.1 | 28.8 | 29.8 | 1.6 | 16.1 | 4 | 4/4 |
| 2,048 | 1 | 1272.8 | 40.6 | 40.6 | 41.8 | 1.6 | 12.6 | 1 | 1/1 |
| 2,048 | 2 | 1530.0 | 85.1 | 42.5 | 43.5 | 2.7 | 11.4 | 2 | 2/2 |
| 2,048 | 4 | 1335.8 | 114.6 | 27.8 | 28.8 | 4.3 | 16.1 | 4 | 4/4 |
| 8,192 | 1 | 1164.0 | 47.6 | 47.6 | 47.5 | 7.0 | 10.7 | 1 | 1/1 |
| 8,192 | 2 | 1227.0 | 71.5 | 35.8 | 37.2 | 9.8 | 12.7 | 2 | 2/2 |
| 8,192 | 4 | 1186.5 | 89.4 | 24.2 | 23.2 | 19.6 | 18.4 | 4 | 4/4 |
| 32,768 | 1 | 1476.4 | 50.6 | 50.6 | 50.8 | 22.2 | 10.1 | 1 | 1/1 |
| 32,768 | 2 | 1486.6 | 93.2 | 46.6 | 47.4 | 33.1 | 9.9 | 2 | 2/2 |
| 32,768 | 4 | 1471.6 | 136.4 | 33.8 | 35.7 | 57.4 | 10.8 | 4 | 4/4 |
| 131,072 | 1 | 1409.3 | 52.9 | 52.9 | 52.9 | 93.0 | 9.7 | 1 | 1/1 |
| 131,072 | 2 | 1293.7 | 82.3 | 41.1 | 25.7 | 151.2 | 8.2 | 2 | 2/2 |
| 131,072 | 4 | 1155.6 | — | — | 8.9 | 290.1 | 0.0 | 3 | 4/4 |
| 200,000 | 1 | 1162.8 | 47.0 | 47.0 | 45.5 | 172.0 | 10.9 | 1 | 1/1 |
| 200,000 | 2 | 1357.0 | 81.7 | 40.9 | 25.7 | 218.4 | 6.2 | 2 | 2/2 |
| 200,000 | 4 | 1310.4 | — | — | 3.2 | 380.3 | 0.0 | 3 | 4/4 |
| 250,000 | 1 | 1378.5 | 54.6 | 54.6 | 54.7 | 181.4 | 9.4 | 1 | 1/1 |
| 250,000 | 2 | 1309.6 | 72.9 | 36.5 | 28.9 | 283.1 | 3.4 | 2 | 2/2 |
| 250,000 | 4 | 1276.7 | — | — | 3.0 | 483.9 | 0.0 | 2 | 4/4 |
