---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash
tags:
  - deepseek
  - deepseek-v4.1-flash
  - tensorfold
  - dgx-spark
  - gb10
  - sm121
  - tensor-parallel
  - exl3
  - fp8-kv
  - engram
  - speculative-decoding
  - dspark
  - 512k-context
language:
  - en
  - zh
pipeline_tag: text-generation
---

# DeepSeek-V4.1-Flash on 2× NVIDIA DGX Spark (GB10), TensorFold TP=2
### 512K context · 3 concurrent slots · FP8 KV · exact DSpark speculative decoding · Engram read from local NVMe

**中文版见 [README.zh-CN.md](README.zh-CN.md).**

This repository publishes our **as-deployed** configuration and every number we measured for serving
DeepSeek-V4.1-Flash on **two GB10 systems ("DGX Spark", SM 12.1, 128 GB unified memory each)** with the
[TensorFold](https://github.com/ashhart/TensorFold) engine, tensor-parallel across the single CX7 QSFP link.

It is a field notebook, not a paper. The value is in the parts that were not obvious: which
**context × slot combinations physically fit** on this pair, what the **KV pool** — not `max_model_len` —
does to long requests, and how a **client-visible** measurement compares with an engine-internal one.

Everything below was measured on our own pair, at the client, with prefix caching off unless noted.

| | |
|---|---|
| **Base model** | DeepSeek-V4.1-Flash (2.9 bpw EXL3 pack, 39 safetensors shards, ~197 GB) |
| **Hardware** | 2 × GB10 / SM 12.1, 128 GB unified memory, 2 × 200 Gb/s CX7 (RoCE v2) |
| **Engine** | TensorFold `v0.6.0` (pinned submodule) + the recipe's patch set, image built for GB10 |
| **Parallelism** | TP = 2, one QSFP link, RoCE |
| **KV cache** | FP8, shared pool sized in tokens |
| **Deployed context** | **512,000 tokens** (`CONTEXT=512000`), **3 request slots** |
| **KV pool (deployed)** | **1,638,400 tokens** = 3.2× a 512K request |
| **Decode (512K profile)** | code **82.2** / prose **42.1** / structured **103.1** tok/s single stream; 3-way same-workload aggregate **104.0** |
| **Prefill** | **1,810 tok/s** at a genuine 419,879-token prompt (232 s), needle hit |
| **Quality gates** | needle hit at 214K → 599K → 855K; tool calls, structured JSON, chat all correct |
| **Boot** | 47-55 s from cold to serving, per profile |
| **License** | MIT for this repository's own material; the weights keep their own license |

---

## 1. The three things that actually matter

### 1.1 Context and slots trade against each other — the fit check is a hard floor

The engine solves memory for `CONTEXT × PARALLEL` at boot and refuses to start when the remaining
headroom drops under a hard floor (`TF_DSV41_FLOOR_HARD_GIB`, default **4.0 GiB**). The error text names the
number, so the ladder is quick to walk — but it means **"longer context" and "more concurrency" are a trade,
not two independent knobs**:

| profile | outcome |
|---|---|
| `CONTEXT=300000 PARALLEL=4` | boots, `request slots: 4`, ready 55 s |
| `CONTEXT=400000 PARALLEL=4` | boots, ready 55 s |
| `CONTEXT=500000 PARALLEL=4` | **refused** — `4 x 500000 on rank 0 leaves 3.90 GiB, under the 4.0 GiB hard floor; fit: 3 stream(s)` |
| **`CONTEXT=512000 PARALLEL=3`** | **boots, `request slots: 3`, ready 53 s ← deployed** |
| `CONTEXT=600000 PARALLEL=1` | boots, ready 47 s |
| `CONTEXT=1000000 PARALLEL=1` | boots, ready 47-57 s |
| `CONTEXT=1000000 PARALLEL=4` | **refused** — `4 x 1000000 on rank 0 leaves 0.56 GiB, under the 4.0 GiB hard floor; fit: 1 stream(s)` |

So on one pair: **512K is a 3-slot profile**, and anything at 4 slots has to stay at or below 400K.
There is no configuration on this hardware that gives both 1M *and* four concurrent slots — the
2-GPU pair simply runs out of unified memory.

### 1.2 The KV **pool** is what refuses long requests, not `max_model_len`

`TF_DSV41_POOL_TOKENS` is a total token budget for the shared pool, independent of the slot count.
Long requests fail against the *pool*, and the error says "pool pages", which reads like a context error
at first glance:

```
$ CONTEXT=1000000 PARALLEL=1 POOL_TOKENS=614400        # 2400 pages
{"error": {"message": "the request needs 3343 KV pool pages, the pool has 2400"}}
$ CONTEXT=1000000 PARALLEL=1 POOL_TOKENS=1228800       # pool doubled
→ 855,600-token request served in 598 s, prefill 1,431 tok/s, needle hit
```

Rule of thumb that has held on this pair: **pool ≥ slots × context + reply budget**. Deployed profile uses
`1,638,400` for 3 × 512K. When a long request 400s at a length that looks comfortably inside
`max_model_len`, read the pool number before touching the context.

### 1.3 Client-visible vs engine-internal throughput

The upstream recipe reports single-stream numbers from an **in-engine benchmark** (`m2bench`) — no HTTP
server in the loop. Serving over HTTP adds the stack, so the same workload measures lower at the client.
Both are legitimate; mixing them produces the wrong conclusion ("we are half as fast as upstream").
Our tables below are **client-side, over HTTP**, and upstream's are labelled as theirs.

When comparing, also pin down whether the cell is *first-token-to-last-token* (decode-only) or
prompt-inclusive end-to-end. A 384-token reply with a 300 ms TTFT reads ~6 % lower if you count the
prompt, and much lower on a long prompt.

---

## 2. Measured: the deployed 512K × 3 profile

Prompt texts are the upstream benchmark's own (`CODE` = an O(1) LRU cache class; `PROSE` = a 400-word
essay on lighthouses; `STRUCTURED` = "count 1 to 200"), 384-token replies, greedy, thinking off.

### 2.1 Single stream

| workload | T = 0 | T = 0.7 (warm prefix) |
|---|---:|---:|
| `CODE` | **78.3** tok/s | **82.2** tok/s |
| `PROSE` | 40.5 tok/s | 42.1 tok/s |
| `STRUCTURED` | 98.6 tok/s | 103.1 tok/s |

TTFT in these runs was 0.26-0.35 s, so the numbers are decode-dominated, not prefill-dominated.

### 2.2 Concurrency (same-workload, so no workload skew)

| | aggregate |
|---|---:|
| 3 × `CODE`, identical prompt + one-line suffix each | **104.0** tok/s (per-slot wall 3.6 / 3.7 / 3.7 s — no serialisation) |
| 3 × mixed (`CODE` / `PROSE` / `STRUCTURED`) | 75.4 tok/s |

The same-workload cell is the honest one for "does the pool serialise?" — all three slots finish within
0.1 s of each other. A mixed cell is dominated by whichever prompt runs longest, and a
prompt that terminates early drags the measurement window out; publish both or the number misleads.

### 2.3 Long context

A genuine long request with a needle planted mid-document:

| prompt tokens | result | wall | prefill | needle |
|---:|---|---:|---:|---|
| **419,879** (deployed 512K profile) | served | 232.0 s | **1,810 tok/s** | **hit** |
| 214,319 | served | 111.6 s | 1,921 tok/s | hit |
| 291,719 | served | 175.0 s | 1,667 tok/s | hit |
| 411,362 | served | 232.0 s | 1,774 tok/s | hit |
| 582,210 (600K profile) | served | 354.6 s | 1,642 tok/s | hit |
| 599,310 (600K profile) | served | 369.0 s | 1,624 tok/s | hit |
| 855,600 (1M profile) | served | 597.8 s | 1,431 tok/s | hit |

Prefill degrades gently — roughly 1.9K tok/s at 200K down to 1.4K tok/s at the 1M end — and retrieval held
at every length. **Plan the client timeout around prefill**: a 512K prompt is ~4 minutes before the first
token, a 1M prompt ~10 minutes. Stream if the client UI needs to show progress.

### 2.4 Boot

| profile | time to serving |
|---|---:|
| 512K × 3 (deployed) | **53 s** |
| 400K × 4 | 55 s |
| 600K × 1 / 1M × 1 | 47 s |

A prepared per-rank layout and cached compiled kernels are what make sub-minute restarts possible; a cold
build is a different story.

---

## 3. Configuration (deployed)

Selected settings from the production environment file — the values are the interesting part, the file is
otherwise paths and paths:

```ini
CONTEXT=512000
PARALLEL=3
KV_DTYPE=fp8
MAX_TOKENS=32768

# KV pool: total tokens, independent of slots
TF_DSV41_POOL_TOKENS=1638400

# memory floors: soft refusal / hard boot refusal (GiB)
TF_DSV41_FLOOR_GIB=5
TF_DSV41_FLOOR_HARD_GIB=4

# prefill strategy: bounded replay with adaptive rows
TF_DSV41_PREFILL=replay
TF_DSV41_PREFILL_CHUNK=2048
TF_DSV41_PREFILL_ROWS=2048
TF_DSV41_PREFILL_ADAPT_GIB=4.5
TF_DSV41_PREFILL_KERNELS=fast
TF_DSV41_PREFILL_GATHER=bf16
TF_DSV41_PREFILL_ROUTER=gemv

# decode path
TF_DSV41_SPEC_DRAFT=1            # exact speculative decoding
TF_DSV41_GRAPH_MODE=rows
TF_DSV41_MHC_CUDA=1
TF_DSV41_ATTN_CUDA=1
TF_DSV41_DENSE_V3=1
TF_DSV41_DENSE=0
TF_DSV41_ROUTER=gemv
TF_DSV41_L2PF=1
TF_DSV41_L2PF_MB=12
TF_DSV41_L2PF_PACE_GBPS=150
TF_DSV41_CALIB=real

# cross-rank plan transport
TF_DSV41_PLAN_LINK=nccl

# serving default: thinking off (see §4)
TF_DSV41_THINKING=0
TF_DSV41_DEFAULT_EFFORT=high
```

Two of these deserve a warning:

* **`TF_DSV41_PLAN_LINK=rdma` does not boot in the shipped image.** The upstream example sets it, but the
  validator in the release we run accepts only `nccl` or `tcp` and aborts the start. We run `nccl`, and
  measured `nccl` clearly ahead of `tcp` (4-way code aggregate 148.8 vs 135.7 tok/s in an A/B on the same
  profile). Reported upstream.
* **Calibration and prefix tiers assume a prepared layout.** `TF_DSV41_CALIB=real` expects a calibration
  directory from a previous run; on a cold box it silently costs a slow first boot.

---

## 4. Thinking default: `content` is empty unless the client opts out

This one cost us a debugging round, so it is written down. With the server default effort at `high`, a
prose/short-answer request with a bounded `max_tokens` spends the entire budget inside the reasoning
channel and returns **an empty `content` string**:

| request | result |
|---|---|
| `PROSE`, `max_tokens=1200`, no thinking kwargs | `content=""`, reasoning channel 5,918-13,755 chars, `finish_reason=length` |
| same, `effort = low / high / max` × `max_tokens = 1200 / 3000` — six combinations | **all six** `content=""`, `finish_reason=length` |
| same, thinking explicitly disabled | `content` 3,000+ chars / ~740 words, `finish_reason=stop` |

A client that omits the thinking flag therefore sees an empty reply and retries — paying another prefill
each time. We run the server with thinking **off by default** and let a request turn it back on; a
side effect on this workload was single-stream code decode 55.6 → 77.6 tok/s, because the thinking budget
had been consuming the same reply cap.

Accuracy on the checks we run did not move with thinking off: arithmetic, a word-problem trap, a small
equation, and the tool-call path all returned the expected answers. Multi-step math and hard debugging are
where an explicit thinking budget is still the better choice — turn it on per request for those.

---

## 5. Operations

* **Boot supervision**: run the server as a **systemd user unit** with lingering enabled, plus a
  short-interval watchdog unit that re-checks health and restarts on failure. A prepared per-rank layout
  means restarts land in well under a minute, so an unattended box recovers on its own.
* **Profile switching**: keep one environment file per profile (300K×4, 400K×4, 512K×3, 600K×1, 1M×1) and
  switch by swapping the file and restarting — a small wrapper that edits three settings
  (`CONTEXT`, `PARALLEL`, `POOL_TOKENS`), archives the result, restarts and reports whether the boot
  succeeded is enough. Because a refused boot prints the exact shortfall, the wrapper can report
  "cannot fit" instead of leaving a silent dead service.
* **Memory is the binding constraint, always.** With weights resident, available memory at serving time is
  single-digit GiB. Watch it on both ranks; a long request admitted while another is decoding is what
  pushes it to the floor.
* **Client timeouts must exceed prefill.** Server-side, a long request holds its slot for minutes; if the
  client side cuts at 60 s, the visible behaviour is a timeout even though the engine is working.

---

## 6. How these numbers were taken

* One pair, idle — nothing else on the GPUs. A shared GPU invalidates everything below.
* `greedy` (`temperature 0`), `max_tokens` fixed per cell, **prefix caching off on single-stream cells**
  (unique prompts) and explicitly noted where a warm-prefix cell is shown.
* Token counts come from the response's `usage` block, never from counting streamed deltas — under
  speculative decoding a single streamed chunk can carry several tokens.
* Decode rates are **first-token-to-last-token**; the prompt/TTFT span is reported separately as prefill.
* Boot timings are wall clock from process start to a successful `/v1/models` on the local port.
* Quality checks are the ones quoted in the text: a needle planted mid-document at each long-context
  length, arithmetic, a trap word problem, an equation, a JSON schema case and a tool call.

Things we know are **not** settled by this repository:

* The gap to the upstream engine-internal table is not fully explained; the largest residual is on the
  structured/counting workload, and the missing cross-rank transport branch (§3) is our leading candidate.
* Concurrency at 1M on one pair is impossible (§1.1). More simultaneous 1M sessions need more hardware,
  not more tuning.
* The 512K × 3 profile was measured after a fresh boot; we have not yet run a multi-hour soak on it.

---

## 7. Credits

This deployment stands on other people's work, and the interesting ideas are theirs:

* **TensorFold** (Ash Hart) — the engine: EXL3 kernels, the server, the drafting path this recipe builds on.
* **The 2× DGX Spark DeepSeek-V4.1-Flash recipe** (jayleaton) — the two-Spark engine stack, the DeepSeek-V4.1
  family implementation, prepared per-rank folders, the benchmark definitions we re-used verbatim, and the
  public issue discussion that fixed the packaging and memory-leak problems we hit.
* **MiaAI-Lab** — the 2.9 bpw EXL3 pack and the vLLM kit that serves as the baseline in the upstream tables.
* **ExLlamaV3** (turboderp and contributors) — the EXL3 format.
* **DeepSeek** — the model, its technical report, and the DSpark / Engram lines of work the drafting and
  n-gram paths implement.
* The benchmark prompts in §2 are quoted from the upstream recipe's own workload definitions so the
  comparisons are like-for-like.

**What this repository adds**: the 512K × 3 profile and its measurements, the slot-scaling fit ladder on
this exact pair, the pool-vs-context failure mode with reproductions, and the client-side
comparison methodology.

## 8. License

MIT for this repository's own text, configuration and measurement scripts. Model weights and upstream
components keep their respective licenses; see [NOTICE.md](NOTICE.md) and [CREDITS.md](CREDITS.md).
