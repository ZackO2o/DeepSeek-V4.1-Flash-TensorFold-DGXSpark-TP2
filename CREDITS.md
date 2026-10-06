# Credits

This repository is a measured deployment note. The engine, the model, the quantisation format and the
cross-project recipe were built by others; what is ours is the profile, the numbers, and the notes about what does
not fit.

## Model

- **DeepSeek-V4.1-Flash** by DeepSeek — the model, its technical report, and the DSpark speculative decoding and
  Engram n-gram lines of work that this stack implements. Its license, on the model card, governs any use of the
  weights. The weights are not part of this repository.
- **The 2.9 bpw EXL3 pack family** — the checkpoint deployed here is the uncensored variant,
  [`dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw`](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw);
  the pack layout is
  [Mia-AiLab's](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw). Quantisation family:
  **exllamav3** by turboderp and contributors.

## Engine and recipe

- **[TensorFold](https://github.com/ashhart/TensorFold)** by Ash Hart — the engine that serves this model: the EXL3
  kernels, the server, and the drafting path everything here builds on. Pinned as a submodule in the recipe below.
- **[jayleaton/deepseek-v41-tensorfold-spark](https://github.com/jayleaton/deepseek-v41-tensorfold-spark)** — the
  two-Spark DeepSeek-V4.1 engine stack this deployment follows, including the exact speculative-decoding path,
  bounded-replay prefill, the prepared per-rank layout that makes sub-minute restarts possible, and the benchmark
  workload definitions whose prompt texts are quoted in the Performance section so the comparison is like-for-like.
  The repository's public issue discussion also fixed the packaging gaps and a host-memory growth problem that this
  pair hit. This deployment would have taken considerably longer without it.
- **[MiaAI-Lab](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks)** — the 2x DGX Spark vLLM kit
  published as the baseline for the upstream tables, and the publisher of the 2.9 bpw EXL3 pack family.
- The GB10 / DGX Spark community working on unified-memory serving of large MoE models — the reasoning about what
  fits in 128 GB of unified memory per node comes largely from public field reports rather than our own theory.

## What this repository adds

1. The deployed **400,000 x 4** profile with client-side measurements: single stream, four-way aggregate, TTFT.
2. The **window x slot fit ladder** on one pair, including the engine's refusal text for the combinations that do
   not fit — and the observation that 500K caps at three slots.
3. The **pool-versus-context** failure mode with reproductions: the KV pool, not `max_model_len`, refuses long
   requests.
4. The **thinking-default** finding: with the server's default effort, a bounded reply returns an empty `content`;
   documented with the six-combination evidence and the per-request opt-in that restores it.
5. A note on **client-side versus engine-internal measurement**, which is the difference that makes two published
   sets of numbers look contradictory when they are not.

If you build on this note, please credit the upstream projects above as well — the ideas are theirs.
