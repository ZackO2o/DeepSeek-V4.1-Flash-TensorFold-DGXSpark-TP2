<h1 align="center">DeepSeek-V4.1-Flash on DGX Sparks with TensorFold</h1>

<p align="center">
  <sub>by <a href="https://github.com/ZackO2o">ZackO2o</a></sub>
</p>

Serve **DeepSeek-V4.1-Flash** from two NVIDIA DGX Sparks (GB10, 128 GB of unified memory each, linked by their
ConnectX-7 ports) through an OpenAI-compatible API, with **4 concurrent requests**, a window of **400,000 tokens**
per request, and a shared FP8 KV pool of **1,638,400 tokens**. It runs
[TensorFold](https://github.com/ashhart/TensorFold) v0.6.0 (pinned submodule, plus the two-Spark engine stack from
the [2x DGX Spark DeepSeek-V4.1 recipe](https://github.com/jayleaton/deepseek-v41-tensorfold-spark), one rank on
each Spark): EXL3 expert and dense kernels tuned for GB10, the CSA2 attention path with FP8 KV rows and the
lightning indexer, Single-Pass mHC, Engram rows read from local NVMe, **exact** DSpark speculative decoding, CED
bounded-replay prefill, several requests over one shared cache pool, native image input, structured output and tool
calls, `/tokenize` and `/metrics`.

- Checkpoint: [`dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw`](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw),
  the uncensored variant of the 2.9 bpw EXL3 pack (39 safetensors shards, ~197 GB, same layout as
  [Mia-AiLab's pack](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw))
- API model id: `DeepSeek-V4.1-Flash-TF`
- Window: **400,000 tokens** a request, **4 requests at once**, sharing an FP8 KV pool of **1,638,400 tokens** (4.1x a full window)
- Engram n-gram rows kept on each rank's local NVMe
- `/tokenize`, `/detokenize`, Prometheus `/metrics`, tool calls, and `reasoning_effort` `low` / `high` / `max`
- **Thinking defaults to off** (see [Thinking and sampling](#thinking-and-sampling)); a request turns it back on
- A prepared per-rank layout: **ready in 53 s**, restarts included

## Performance

Two DGX Sparks at the deployed configuration (4 streams, 400,000-token window, FP8 KV cache, DSpark speculative
decoding, thinking off), measured **through the HTTP API from the client side**, greedy, with prefix caching cold on
every single-stream cell. Nothing else on the GPUs.

**Decode** (single stream, first token to last token; and the same at four requests at once)

| Workload | 1 request | 4 requests, aggregate | 4 requests, per request |
| --- | ---: | ---: | ---: |
| Code (`LRUCache`, 384-token reply) | **81.8 tok/s** | **150.5 tok/s** | 38.8 tok/s |
| Prose (400-word essay, 384-token reply) | 42.5 tok/s | 72.9 tok/s | 18.4 tok/s |
| Structured (count 1-200, 384-token reply) | 101.5 tok/s | — | — |
| Code, same prompt again (warm prefix) | 82.6 tok/s | — | — |

TTFT on these single-stream cells is **299-433 ms**. Four requests at once are identical to the same requests served
one at a time: the four slots finish within 0.2 s of each other (6.6 / 6.6 / 6.6 / 6.8 s for a 256-token reply), so
the shared pool does not serialise them.

**Prefill** (cold prompts, unique prefixes, a needle planted mid-document)

| Prompt | Prefill | Time to first token | Needle |
| ---: | ---: | ---: | --- |
| 214,319 tokens | 1,921 tok/s | 111.6 s | found |
| 257,310 tokens | 1,878 tok/s | 137.0 s | found |
| 291,719 tokens | 1,667 tok/s | 175.0 s | found |
| 325,889 tokens | 1,899 tok/s | 171.6 s | found |
| 368,640 tokens | 1,859 tok/s | 198.3 s | found |

**Longer windows** (same pair, other profiles; see [Window and slots](#window-and-slots))

| Window | Prompt | Prefill | Needle |
| ---: | ---: | ---: | --- |
| 500,000 x 3 | 454,053 tokens | 1,738 tok/s | found |
| 600,000 x 1 | 599,310 tokens | 1,624 tok/s | found |
| 1,000,000 x 1 | 855,600 tokens | 1,431 tok/s | found |

**Context** (how a request that does not fit is refused, and what the real limit is)

```json
{"error": {"message": "This model's maximum context length is 400000 tokens. However, you requested
1000004 tokens (5 in the messages, 999999 in the completion). Please reduce the length of the messages
or completion."}}
```

## Requirements

- **Two DGX Sparks** (or two GB10 systems with 128 GB unified memory), with nothing else large on their GPUs: the
  server holds roughly **101 GiB** of weights and leaves single-digit GiB available. Stop other GPU work.
- **A direct ConnectX-7 link:** a QSFP cable between the CX7 ports and an IPv4 address on each end in one private
  subnet (`ping` must work), with a RoCE v2 GID. One cabled CX7 port of a Spark reaches the GB10 over two PCIe Gen5
  x4 links, so it appears as two netdevs and two RoCE devices; address both twins, in the link's subnet or each in
  its own — both are then used, and the ranks' exchanges ride both rails.
- **Key-based ssh** from the first Spark (the head, which runs the launcher and the API) to the second (the worker):
  `ssh-copy-id user@<worker>`; check with `ssh -o BatchMode=yes user@<worker> true`.
- Docker with the NVIDIA container runtime, and `rsync`, on both Sparks.
- **Disk, on each Spark:** the checkpoint (~197 GB) and the Engram shards prepared locally, plus the image under
  Docker's root. Prepared per-rank folders are what make the 53 s restarts possible.
- A Hugging Face token only if you also want to pull weights yourself: this deployment reads the local cache.

## Quick start

On the head:

```bash
git clone https://github.com/jayleaton/deepseek-v41-tensorfold-spark.git
cd deepseek-v41-tensorfold-spark
cp config/prod.env.example config/prod.env      # then set your paths in it
scripts/serve.sh build && scripts/serve.sh preflight && scripts/serve.sh start
```

The deployed window is set in `config/prod.env` (see [Configuration](#configuration)):

```ini
CONTEXT=400000
PARALLEL=4
TF_DSV41_POOL_TOKENS=1638400
```

Any OpenAI client works with `base_url = "http://<head-address>:8300/v1"` and the model
`DeepSeek-V4.1-Flash-TF`:

```bash
curl -s http://<head-address>:8300/v1/models
curl -s http://<head-address>:8300/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "DeepSeek-V4.1-Flash-TF",
  "messages": [{"role": "user", "content": "Write a Python fibonacci function."}],
  "max_tokens": 2000
}'
curl -s http://<head-address>:8300/health        # inflight, streams, pool figures
```

`/health` on the deployed pair, idle:

```json
{"ok": true, "inflight": 0, "requests_running": 0, "streams": {"decoding": 0, "prefilling": 0, "max": 4},
 "context_length": 400000, "backend": "tensorfold"}
```

**Thinking off by default.** The server ships with thinking off, so the reply is in `content`. A request can ask it
to think (see [Thinking and sampling](#thinking-and-sampling)) — give it enough `max_tokens` when it does.

## Window and slots

The engine solves memory for `CONTEXT x PARALLEL` at boot and refuses to start when the remaining headroom drops
under a hard floor (`TF_DSV41_FLOOR_HARD_GIB`, 4.0 GiB). The refusal names the shortfall, so the ladder below is
what one pair of Sparks can actually do — **window and slots trade against each other**:

| Window per request | Requests at once | Boot | Note |
| ---: | ---: | --- | --- |
| 300,000 | 4 | 55 s | |
| **400,000** | **4** | **53 s** | **deployed** |
| 500,000 | 3 | 52 s | |
| 500,000 | 4 | refused | `4 x 500000 on rank 0 leaves 3.90 GiB, under the 4.0 GiB hard floor; fit: 3 stream(s)` |
| 600,000 | 1 | 47 s | |
| 1,000,000 | 1 | 47 s | |
| 1,000,000 | 4 | refused | `4 x 1000000 on rank 0 leaves 0.56 GiB, under the 4.0 GiB hard floor; fit: 1 stream(s)` |

On one pair, **500K is a 3-slot profile and 1M is a single-slot profile**; there is no configuration that gives
both a 1M window and four concurrent slots. More simultaneous long requests need more Sparks, not more tuning.

### Profile switching

Keep one environment file per profile and swap it. Editing three settings is enough:

```ini
CONTEXT=512000        # a longer window...
PARALLEL=3            # ...one fewer slot
TF_DSV41_POOL_TOKENS=1638400
```

Restart and read the boot line: `ready after Ns` plus `request slots: N` means it took; `start failed` plus a
`fit:` line means it does not fit on your pair. Because the refusal states the exact shortfall, a wrapper can report
"does not fit" instead of leaving a dead service behind. Profiles verified on this pair:
`300K x 4`, `400K x 4`, `500K x 3`, `512K x 3`, `600K x 1`, `1M x 1`.

## KV pool and memory

With `PARALLEL` above 1, all requests draw their per-token caches from **one shared pool**, sized in tokens by
`TF_DSV41_POOL_TOKENS`:

| | Deployed |
| --- | ---: |
| Requests at once (`PARALLEL`) | 4 |
| Window per request (`CONTEXT`) | 400,000 tokens |
| KV precision | FP8 |
| **Shared pool** | **1,638,400 tokens** (4.1x a full window) |
| Rank 0's weight footprint | ~101 GiB |
| `MemAvailable` at serving | single-digit GiB, both ranks |
| Boot to `/v1/models` | 53 s |

Any one request can grow to the full window, and the four together share the pool. **The pool is the binding knob,
not `max_model_len`** — a request larger than the pool is refused with a message about *pool pages*:

```
CONTEXT=1000000 PARALLEL=1 POOL_TOKENS=614400                # sized for 4 x 300K
{"error": {"message": "the request needs 3343 KV pool pages, the pool has 2400"}}
CONTEXT=1000000 PARALLEL=1 POOL_TOKENS=1228800               # doubled
→ 855,600-token request served in 598 s, prefill 1,431 tok/s, needle found
```

Rule of thumb that has held here: **pool >= slots x window + reply budget**. When a long request is refused at a
length that looks comfortably inside `CONTEXT`, read the pool number before touching the window.

## Configuration

Every setting lives in `config/prod.env` (copied from `config/prod.env.example`). The deployed values:

| Variable | Value | Meaning |
| --- | ---: | --- |
| `CONTEXT` | `400000` | prompt + reply window per request |
| `PARALLEL` | `4` | requests decoded together |
| `KV_DTYPE` | `fp8` | KV cache precision |
| `MAX_TOKENS` | `32768` | reply budget of a request that sets no `max_tokens` |
| `TF_DSV41_POOL_TOKENS` | `1638400` | shared pool, in tokens |
| `TF_DSV41_FLOOR_GIB` / `_HARD_GIB` | `5` / `4` | soft warning / hard boot refusal, GiB |
| `TF_DSV41_PREFILL` | `replay` | bounded-replay prefill (CED), the default |
| `TF_DSV41_PREFILL_CHUNK` / `_ROWS` | `2048` | prefill rows per chunk |
| `TF_DSV41_PREFILL_ADAPT_GIB` | `4.5` | adaptive row count target |
| `TF_DSV41_PREFILL_KERNELS` | `fast` | prompt-row kernels (`exact` for bit-identical gates) |
| `TF_DSV41_PREFILL_ROUTER` / `TF_DSV41_ROUTER` | `gemv` | router path |
| `TF_DSV41_SPEC_DRAFT` | `1` | exact DSpark speculative decoding |
| `TF_DSV41_GRAPH_MODE` | `rows` | CUDA graph capture mode |
| `TF_DSV41_MHC_CUDA` / `_ATTN_CUDA` / `_DENSE_V3` | `1` | the three kernel rewrites of the upstream tuning |
| `TF_DSV41_L2PF` / `_MB` / `_PACE_GBPS` | `1` / `12` / `150` | L2 prefetch of the next kernels' weights |
| `TF_DSV41_CALIB` | `real` | calibration from a previous run |
| `TF_DSV41_PLAN_LINK` | `nccl` | cross-rank plan transport |
| `TF_DSV41_THINKING` | `0` | thinking off by default |
| `TF_DSV41_DEFAULT_EFFORT` | `high` | effort used when a request turns thinking on |

Two warnings:

- **`TF_DSV41_PLAN_LINK=rdma` does not boot in the release we run.** `config/prod.env.example` sets `rdma`, but the
  shipped validator accepts only `nccl` or `tcp` and aborts the start:
  `tensorfold: TF_DSV41_PLAN_LINK='rdma': expected nccl or tcp` then `[dsv41-tf] start failed`. We run `nccl`, and
  measured `nccl` ahead of `tcp` on the same profile (4-way code aggregate 148.8 vs 135.7 tok/s). Reported upstream.
- **`TF_DSV41_CALIB=real` expects a calibration directory** from a previous run; on a cold box it silently costs a
  slow first boot.

## Thinking and sampling

| Request | Result |
| --- | --- |
| Prose, `max_tokens=1200`, no thinking flag | `content=""` (empty), reasoning channel 5,918-13,755 chars, `finish_reason=length` |
| Same, `effort` `low` / `high` / `max` x `max_tokens` 1200 / 3000 — six combinations | **all six** `content=""`, `finish_reason=length` |
| Same, thinking disabled | `content` 3,000+ chars (~740 words), `finish_reason=stop` |

A client that sends no thinking flag therefore sees an empty reply and retries, paying another prefill each time.
**We run the server with thinking off by default** (`TF_DSV41_THINKING=0`) and let a request turn it on:

```json
{"chat_template_kwargs": {"enable_thinking": true}}          // think for this request
{"reasoning_effort": "low" | "high" | "max"}                 // effort when thinking
```

Turning thinking off did not change the checks we run: arithmetic, a word-problem trap, a small equation, a JSON
schema case and a tool call all returned the expected answers, and the reply lands in `content` instead of
`reasoning_content`. Side effect on this workload: single-stream code decode **55.6 -> 81.8 tok/s**, since the
thinking budget had been consuming the same reply cap. **Multi-step math and hard debugging are where an explicit
thinking budget still wins** — turn it on per request for those.

Sampling: `temperature`, `top_p`, `top_k`, `min_p` and `seed` are per request; `temperature: 0` decodes greedily,
which is how every number above was measured. Without a `seed` the sampler's key comes from the prompt, so the same
request gives the same reply.

## How these numbers were taken

- One pair, idle — nothing else on the GPUs. A shared GPU invalidates all of it.
- Greedy (`temperature 0`), a fixed `max_tokens` per cell, **prefix caching cold on single-stream cells** (unique
  prompts); the warm-prefix cell is labelled as such.
- Token counts come from the response's `usage` block, never from counting streamed deltas: under speculative
  decoding one streamed chunk can carry several tokens.
- Decode rates are **first token to last token** (the upstream engine's own definition); TTFT is reported
  separately, and prefill is prompt tokens / wall clock for a cold prompt.
- Aggregate cells use a **shared clock across the concurrent requests**, the same `max_tokens` on all of them, and
  the same prompt on every slot (a one-line suffix each) so no workload skews the window.
- Boot time is wall clock from container start to a successful `/v1/models` on the local port.
- Quality: a needle planted mid-document at every long-context length above, arithmetic, a trap word problem, an
  equation, a JSON schema case and a tool call.

Known limits, stated rather than hidden:

- **A comparison against an upstream table is not apples to apples unless you match the paths.** The upstream
  recipe reports its single-stream cells from an *in-engine* benchmark (`m2bench`, no HTTP server in the loop);
  everything here is what a client sees over HTTP. On the same prompts we measure 90-106% of its cells
  (code 82.6 vs 78.15 at a warm prefix; structured 102.4 vs 122.91), and the residual is presumably the HTTP hop
  plus the shared pool's scheduling. Say which path a number came from or the comparison misleads.
- The gap on the structured/counting workload is the widest of the set and we have no complete explanation for it;
  the `PLAN_LINK` branch missing from the release (above) is our leading candidate.
- 1M with concurrent slots is impossible on one pair (see [Window and slots](#window-and-slots)).
- The 400K x 4 profile has been through multi-hour use, but this repository does not claim a formal soak test.

## Repository layout

```
README.md          this file
README.zh-CN.md    the same, in Chinese
REDACTION-MAP.md   what was scanned before publishing, and what the numbers are allowed to say
CREDITS.md         who and what this builds on
NOTICE.md          third-party notices that go with the MIT license
LICENSE            MIT (this repository's own material)
tools/bench.py     the client-side benchmark: single stream, N-way aggregate, long-context needle
```

## License

MIT for this repository's own text, configuration and measurement scripts, see [`LICENSE`](LICENSE).
Model weights, the engine and the upstream recipe keep their own licenses, and nothing of theirs is redistributed
here; see [`NOTICE.md`](NOTICE.md) and [`CREDITS.md`](CREDITS.md).
