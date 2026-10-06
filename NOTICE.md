# Notices

DeepSeek-V4.1-Flash on DGX Sparks with TensorFold — deployment notes and measurements
Copyright 2026 ZackO2o

This repository's own text, configuration excerpts and measurement scripts are licensed under the MIT License
(see `LICENSE`). It does **not** redistribute model weights, engine source, container images, or third-party code.

## Third-party work referenced

| Component | Project | License |
| --- | --- | --- |
| TensorFold (engine) | https://github.com/ashhart/TensorFold | Apache-2.0 from 0.6.0 (MIT before it); see the upstream repository |
| 2x DGX Spark DeepSeek-V4.1 recipe | https://github.com/jayleaton/deepseek-v41-tensorfold-spark | Apache-2.0 (upstream project code) |
| DeepSeek-V4.1-Flash (base model) | https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash | see the model card |
| 2.9 bpw EXL3 pack family | https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw | see the model card |
| exllamav3 (EXL3 format and kernels) | https://github.com/turboderp-org/exllamav3 | MIT |

## Quoted material

- The benchmark prompt texts in the Performance section are short programmatic prompts reproduced from the
  upstream recipe's own workload definitions, so that the numbers are comparable like-for-like.
- Sections "Window and slots" and "KV pool and memory" quote **error messages emitted by the upstream engine**
  (the memory-floor refusal and the KV-pool page refusal). They are reproduced verbatim because the exact wording
  is what makes those failure modes recognisable.
- Environment variable names and their meanings follow the upstream recipe's configuration file.

## No warranty

These are field notes from one deployment. They are not a specification, they are not certified, and numbers will
differ on other hardware, other clock settings, and other software revisions.
