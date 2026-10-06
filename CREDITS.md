# CREDITS

This deployment is built on other people's work. The engine, the model, the quantisation format and the
cross-project recipe are theirs; what is ours is the measured profile and the notes in the README.

## Direct foundations

* **TensorFold** — https://github.com/ashhart/TensorFold
  The engine that serves this model: EXL3 kernels, the server, the drafting path. Pinned as a submodule
  in the upstream recipe we deploy.

* **jayleaton / deepseek-v41-tensorfold-spark** — https://github.com/jayleaton/deepseek-v41-tensorfold-spark
  The two-Spark stack this deployment follows: the engine family implementation for DeepSeek-V4.1, the
  prepared per-rank layout that makes sub-minute restarts possible, the benchmark workload definitions we
  re-used verbatim, and the public issue discussion that fixed the packaging gaps and a host-memory growth
  problem we hit on our own pair. Our deployment would have taken considerably longer without it.

* **MiaAI-Lab** — https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks
  The 2x DGX Spark vLLM kit that serves as the upstream baseline, and the publisher of the 2.9 bpw EXL3
  pack family our checkpoint belongs to.

* **ExLlamaV3 (turboderp-org)** — https://github.com/turboderp-org/exllamav3
  The EXL3 quantisation format and its reference kernels.

* **DeepSeek** — the model, its technical report, and the DSpark / Engram lines of work that the
  speculative-decoding and n-gram paths in this stack implement.

## Community references consulted

* The DGX Spark / GB10 community working on unified-memory serving of large MoE models — the reasoning
  about what fits in 128 GB of unified memory per node comes largely from public field reports, not from
  our own theory.

## What this repository adds

1. The **512K x 3 slot** profile and its measurements (single stream, 3-way aggregate, genuine 419,879-token
   request with retrieval).
2. The **slot-scaling fit ladder** on one pair: which context/slot combinations boot and which are refused,
   with the exact refusal text.
3. The **pool-versus-context** failure mode, with reproductions: the KV pool, not `max_model_len`, is what
   refuses long requests.
4. The **client-side comparison methodology** against an engine-internal benchmark, and why mixing the two
   overstates the gap.

Attribution request: if you build on this notes-style repository, please credit the upstream projects above
as well — the ideas are theirs.
