# Changelog

## 1.0.0 — 2026-10-06

First published revision of this deployment note.

**Deployed configuration**

- DeepSeek-V4.1-Flash (2.9 bpw EXL3 pack, 39 shards) on **2 x DGX Spark (GB10)**, TP=2 over the single CX7 link,
  TensorFold v0.6.0 with the upstream two-Spark DeepSeek-V4.1 engine stack.
- Window **400,000 tokens x 4 concurrent requests**, shared FP8 KV pool of **1,638,400 tokens**.
- **Thinking off by default** at the server, per-request opt-in.
- Ready in **53 s** from a prepared per-rank layout.

**Measured in this revision**

- Single stream: code **81.8** / prose **42.5** / structured **101.5** tok/s, TTFT 299-433 ms.
- Four requests at once, same workload: code **150.5** tok/s aggregate (per request 38.8), prose 72.9.
- Cold prefill 214K-369K prompt tokens: **1,667-1,921 tok/s**, needle found at every length.
- Longer windows measured on the same pair: 500K x 3 (454,053-token prompt, 1,738 tok/s), 600K x 1 (599,310,
  1,624 tok/s), 1M x 1 (855,600, 1,431 tok/s).
- Boot ladder with the engine's own refusal text for the combinations that do **not** fit
  (500K x 4 and 1M x 4).
- KV-pool-versus-context failure mode reproduced, with the error text and the fix (raise the pool, not `CONTEXT`).
- Thinking-on/thinking-off comparison: six effort x budget combinations returning an empty `content`, and the
  thinking-off cell returning a full reply.

**Known issues carried into this revision**

- `TF_DSV41_PLAN_LINK=rdma`, which the upstream example sets, does not boot in the release used here
  (the validator accepts only `nccl` or `tcp`). Running `nccl`; reported upstream.
- 1M with concurrent slots is not bootable on one pair — a hardware limit, not a tuning gap.
- The structured/counting workload is the widest single-stream gap against the upstream in-engine table and has no
  complete explanation yet.
