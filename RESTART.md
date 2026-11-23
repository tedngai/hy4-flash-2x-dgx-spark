# Restart and recovery

## Hosts

| Role | Management address | QSFP address |
|---|---|---|
| Head | this Spark (`head-node`) | `198.51.100.14` |
| Worker | `192.0.2.11` (`worker-node`) | `198.51.100.15` |

The launcher discovers the QSFP interfaces itself; only `WORKER` must be set.

## Normal restart

```bash
cd ~/data/hy4-flash-2x-dgx-spark
WORKER=192.0.2.11 ./stop.sh
WORKER=192.0.2.11 ./start.sh
```

`start.sh` returns as soon as both containers are created. The API is ready
when:

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/v1/models
```

returns `200`.

With the compile/autotune cache warm, a restart took about **2–3 minutes**.
The very first boot after a cache-clearing rebuild took about **25 minutes**
because FlashInfer and CuTeDSL compiled and autotuned SM120 kernels. Do not
kill it early.

## LAN access

The default binding is `127.0.0.1` (loopback only). To expose the API on the
management LAN:

```bash
WORKER=192.0.2.11 ./stop.sh
HOST=192.0.2.10 WORKER=192.0.2.11 ./start.sh
# or HOST=0.0.0.0 to bind every interface
```

The recipe has no authentication. Use a trusted network, a firewall, or a
reverse proxy. The bind address does not change throughput.

## 500K / 3-slot profile

The default is the 262,144-token / 4-slot profile. To run the 500,000-token /
3-slot profile:

```bash
WORKER=192.0.2.11 ./stop.sh
RECIPE=llm/hy4-flash-nvfp4-vllm-tp2-500k \
  HOST=0.0.0.0 WORKER=192.0.2.11 ./start.sh
```

It uses batch 2048 and auto KV. At the measured start: 1,522,579 KV tokens
(3.05× at 500K), a 490K C1 request at 488.8 s / 1,002 tok/s prefill, and three
short concurrent requests in 0.90 s. The profile has its own `cache/` directory;
if it is cold, the first boot can spend ~25 minutes in FlashInfer/CuTeDSL
autotune.

## First run or rebuild

```bash
WORKER=192.0.2.11 ./start.sh --setup
```

`--setup`:

1. builds `dgx-spark-recipes/vllm-hy4-flash:77766db` from
   `images/vllm-hy4-flash/` (the public `vllm/vllm-openai@sha256:5e2c1bad…`
   image plus six Python files);
2. copies the 26.5 GB image to the worker over the QSFP link if it is missing;
3. downloads `atakans/Hy4-flash-NVFP4@30f201ac…` (154 GiB) on the head and on
   the worker.

The checkpoint is 154 GiB. The model cache is shared between the two Sparks by
an `rsync` over the QSFP link in this installation; `--setup` uses `hf download`
on each host.

## What to check after boot

```bash
docker ps --format '{{.Names}}\t{{.Status}}'
docker logs --tail 40 hy4-flash-nvfp4-vllm-tp2
curl -s http://127.0.0.1:8000/v1/models
curl -s http://127.0.0.1:8000/metrics | grep '^vllm:cache_config_info'
free -m
```

Expected published config:

- `hy4-flash` served on `127.0.0.1:8000`
- TP2 + expert parallel across the two Sparks
- 262,144-token served window
- DFly, 5 draft tokens
- 1,501,761 KV tokens (18.26 GiB), 5.73× at 262K
- 4 admission slots

The worker container has the same name. Check it with:

```bash
ssh tngai@192.0.2.11 "docker ps --format '{{.Names}}\t{{.Status}}'"
ssh tngai@192.0.2.11 "docker logs --tail 20 hy4-flash-nvfp4-vllm-tp2"
```

## Common failures

| Symptom | Cause / fix |
|---|---|
| `a two-Spark recipe needs WORKER=...` | Export `WORKER=<ssh host of the second Spark>`. |
| `cannot ssh to the worker` | Key login is not set up, or the management address changed. Test `ssh tngai@192.0.2.11 true`. |
| `no QSFP link up on one of the Sparks` | The 200G link is down. Check `ibdev2netdev` on both hosts. |
| `no image … run with --setup` | The image is missing on one rank. Run `./start.sh --setup`; it copies the image to the worker. |
| First boot appears hung in `shm_broadcast` | Normal while FlashInfer/CuTeDSL compile. Wait; do not restart. |
| Port 8000 already answers | Another server is running. `./stop.sh` or `FORCE=1 ./start.sh`. |
| Health never reaches 200 | Read both logs. A worker OOM or a missing checkpoint on the worker is the usual cause. |

## Memory

At the published config each Spark has roughly 3–7 GiB of host memory
available (`free -m`), depending on page cache. The 250,000-token matrix cell
completed at this headroom. If you add another process on either Spark, expect
prefill OOM before decode OOM.

## Stop

```bash
WORKER=192.0.2.11 ./stop.sh
```

This removes the container on both ranks. The image, checkpoint and compile
cache remain.
