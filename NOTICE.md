# NOTICE

This repository contains **deployment notes and measurements** for serving a DeepSeek-V4.1-Flash
checkpoint with the TensorFold engine on two NVIDIA DGX Spark (GB10) systems. It does not redistribute
model weights, engine source, or container images.

## Upstream components referenced

| Component | Owner | License (as published) |
|---|---|---|
| TensorFold (engine) | Ash Hart / ashhart | see the upstream repository |
| 2x DGX Spark DeepSeek-V4.1-Flash recipe | jayleaton | Apache-2.0 (upstream project code) |
| ExLlamaV3 (EXL3 format & kernels) | turboderp-org | see the upstream repository |
| 2.9 bpw EXL3 weight pack | MiaAI-Lab | see the model card |
| DeepSeek-V4.1-Flash (base model) | DeepSeek | see the model card |

The benchmark prompt texts quoted in README sections 2.1-2.2 are reproduced from the upstream recipe's
own workload definitions so that the comparison is like-for-like; they are short programmatic prompts,
not creative works.

## Environment strings quoted

Section 1.1 and 1.2 quote **error messages emitted by the upstream engine** (the memory-floor refusal and
the KV-pool page refusal). They are reproduced verbatim because the exact wording is what makes those
failure modes recognisable.

## What this repository's own material is licensed under

MIT — see [LICENSE](LICENSE). That covers the text, the annotated configuration excerpts, and the
measurement methodology described here. It does not cover any upstream component.
