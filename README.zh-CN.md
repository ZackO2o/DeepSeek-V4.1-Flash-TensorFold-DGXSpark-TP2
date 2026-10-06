<h1 align="center">DeepSeek-V4.1-Flash 在 DGX Sparks 上，使用 TensorFold</h1>

<p align="center">
  <sub>作者 <a href="https://github.com/ZackO2o">ZackO2o</a></sub>
</p>

在两台 NVIDIA DGX Spark（GB10，各 128 GB 统一内存，通过各自的 ConnectX-7 端口直连）上，通过 OpenAI 兼容 API
服务 **DeepSeek-V4.1-Flash**，支持 **4 路并发请求**、每请求 **400,000 token** 窗口，以及 **1,638,400 token** 的
共享 FP8 KV 池。引擎为 [TensorFold](https://github.com/ashhart/TensorFold) v0.6.0（以子模块固定，另加
[2× DGX Spark DeepSeek-V4.1 配方](https://github.com/jayleaton/deepseek-v41-tensorfold-spark) 的双 Spark 引擎栈，
每台 Spark 一个 rank）：为 GB10 调优的 EXL3 专家与稠密内核、带 FP8 KV 行与 lightning indexer 的 CSA2 注意力路径、
Single-Pass mHC、从本地 NVMe 读取的 Engram 行、**精确** DSpark 投机解码、CED 有界重放预填充、
多请求共用一个缓存池、原生图像输入、结构化输出与工具调用、`/tokenize` 与 `/metrics`。

- 权重：[`dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw`](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw)，
  即 2.9 bpw EXL3 包的无审查版（39 个 safetensors 分片，约 197 GB，与
  [Mia-AiLab 的包](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw) 同构）
- API 模型名：`DeepSeek-V4.1-Flash-TF`
- 窗口：每请求 **400,000 token**，**4 路并发**，共用 **1,638,400 token** 的 FP8 KV 池（= 4.1 倍完整窗口）
- Engram n-gram 行保存在各 rank 的本地 NVMe 上
- `/tokenize`、`/detokenize`、Prometheus `/metrics`、工具调用，以及 `reasoning_effort` 的 `low` / `high` / `max`
- **默认关思考**（见 [思考与采样](#思考与采样)），请求可按需打开
- 预先准备好的分 rank 目录：**53 秒即可服务**，重启同样

## 性能

两台 DGX Spark，部署配置（4 流、400,000 token 窗口、FP8 KV、DSpark 投机解码、关思考），
**从客户端侧经 HTTP API 实测**，greedy，单流单元的前缀缓存为冷。GPU 上没有其它负载。

**解码**（单流为"首 token → 末 token"；以及 4 路并发时同样的负载）

| 负载 | 单请求 | 4 路聚合 | 4 路每路 |
| --- | ---: | ---: | ---: |
| Code（`LRUCache`，384 token 回复）| **81.8 tok/s** | **150.5 tok/s** | 38.8 tok/s |
| Prose（400 词散文，384 token 回复）| 42.5 tok/s | 72.9 tok/s | 18.4 tok/s |
| Structured（数 1-200，384 token 回复）| 101.5 tok/s | — | — |
| Code，同提示词重发（前缀命中）| 82.6 tok/s | — | — |

这些单流单元的 TTFT 为 **299-433 ms**。4 路并发与逐个单独服务的回复一致：四个槽在 0.2 秒内先后收尾
（256 token 回复各 6.6 / 6.6 / 6.6 / 6.8 秒），说明共享池没有把它们串行化。

**预填充**（冷提示词、前缀各不相同、needle 埋在中段）

| Prompt | 预填充 | 首 token 时间 | Needle |
| ---: | ---: | ---: | --- |
| 214,319 tokens | 1,921 tok/s | 111.6 s | 命中 |
| 257,310 tokens | 1,878 tok/s | 137.0 s | 命中 |
| 291,719 tokens | 1,667 tok/s | 175.0 s | 命中 |
| 325,889 tokens | 1,899 tok/s | 171.6 s | 命中 |
| 368,640 tokens | 1,859 tok/s | 198.3 s | 命中 |

**更长窗口**（同一对机器，其它档位；见 [窗口与槽数](#窗口与槽数)）

| 窗口 | Prompt | 预填充 | Needle |
| ---: | ---: | ---: | --- |
| 500,000 × 3 | 454,053 tokens | 1,738 tok/s | 命中 |
| 600,000 × 1 | 599,310 tokens | 1,624 tok/s | 命中 |
| 1,000,000 × 1 | 855,600 tokens | 1,431 tok/s | 命中 |

**上下文**（装不下的请求如何被拒，以及真实上限）

```json
{"error": {"message": "This model's maximum context length is 400000 tokens. However, you requested
1000004 tokens (5 in the messages, 999999 in the completion). Please reduce the length of the messages
or completion."}}
```

## 环境要求

- **两台 DGX Spark**（或两台 128 GB 统一内存的 GB10 机器），GPU 上没有其它大负载：服务常驻约 **101 GiB** 权重，
  服务态可用内存只剩个位 GiB。请停掉其它 GPU 工作。
- **一条直连 ConnectX-7 链路**：两台 Spark 的 CX7 口之间一根 QSFP 线，两端各配一个同一私网子网的 IPv4 地址
  （`ping` 必须通），并有 RoCE v2 GID。一个已接线的 CX7 口在 Spark 内通过两条 PCIe Gen5 x4 链路到达 GB10，
  因此会呈现为两个 netdev 与两个 RoCE 设备；把双胞胎口都配上地址（同一子网或各自子网均可）——两条 rail 都会被用上，
  rank 间的交换走两条 rail。
- **从第一台 Spark（head，跑启动脚本与 API）到第二台（worker）的密钥登录**：`ssh-copy-id user@<worker>`，
  用 `ssh -o BatchMode=yes user@<worker> true` 验证。
- 两台 Spark 上都有 Docker（含 NVIDIA container runtime）与 `rsync`。
- **磁盘（每台）**：在本机准备好权重（约 197 GB）与 Engram 分片，另加 Docker root 下的镜像。预先准备好的分 rank
  目录是 53 秒重启的前提。
- 只有当你想自己拉权重时才需要 Hugging Face token：本部署读本地缓存。

## 快速开始

在 head 上：

```bash
git clone https://github.com/jayleaton/deepseek-v41-tensorfold-spark.git
cd deepseek-v41-tensorfold-spark
cp config/prod.env.example config/prod.env      # 再把里面的路径改成你的
scripts/serve.sh build && scripts/serve.sh preflight && scripts/serve.sh start
```

部署的窗口在 `config/prod.env` 里（见 [配置](#配置)）：

```ini
CONTEXT=400000
PARALLEL=4
TF_DSV41_POOL_TOKENS=1638400
```

任何 OpenAI 客户端都可以用 `base_url = "http://<head-address>:8300/v1"` 与模型名
`DeepSeek-V4.1-Flash-TF`：

```bash
curl -s http://<head-address>:8300/v1/models
curl -s http://<head-address>:8300/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "DeepSeek-V4.1-Flash-TF",
  "messages": [{"role": "user", "content": "Write a Python fibonacci function."}],
  "max_tokens": 2000
}'
curl -s http://<head-address>:8300/health        # inflight、streams、池子数字
```

本对机器空闲时的 `/health`：

```json
{"ok": true, "inflight": 0, "requests_running": 0, "streams": {"decoding": 0, "prefilling": 0, "max": 4},
 "context_length": 400000, "backend": "tensorfold"}
```

**默认关思考。** 服务端默认不思考，回复直接落在 `content` 里。请求可以要求它思考
（见 [思考与采样](#思考与采样)）——那时记得给足 `max_tokens`。

## 窗口与槽数

引擎在起服时按 `CONTEXT x PARALLEL` 求解内存，余量低于硬地板（`TF_DSV41_FLOOR_HARD_GIB`，4.0 GiB）就拒绝起服，
并报出差额。因此下面这张阶梯就是一对 Spark 实际能做的全部 —— **窗口与槽数此消彼长**：

| 每请求窗口 | 并发数 | 起服 | 说明 |
| ---: | ---: | --- | --- |
| 300,000 | 4 | 55 s | |
| **400,000** | **4** | **53 s** | **现役** |
| 500,000 | 3 | 52 s | |
| 500,000 | 4 | 拒起 | `4 x 500000 on rank 0 leaves 3.90 GiB, under the 4.0 GiB hard floor; fit: 3 stream(s)` |
| 600,000 | 1 | 47 s | |
| 1,000,000 | 1 | 47 s | |
| 1,000,000 | 4 | 拒起 | `4 x 1000000 on rank 0 leaves 0.56 GiB, under the 4.0 GiB hard floor; fit: 1 stream(s)` |

也就是说，单对机器上 **500K 只能是 3 槽档、1M 只能是单槽档**；不存在"既 1M 又 4 路并发"的配置。
要更多并发长请求，需要加机器，而不是继续调参。

### 档位切换

每个档位存一份环境文件，换文件 + 重启即可。改三项就够：

```ini
CONTEXT=512000        # 更长窗口……
PARALLEL=3            # ……少一个槽
TF_DSV41_POOL_TOKENS=1638400
```

重启后看起服行：出现 `ready after Ns` 加 `request slots: N` 就是成功；出现 `start failed` 加 `fit:` 行
就是这对机器装不下。因为拒起会给出确切差额，包装脚本可以报"装不下"，而不是留下一个悄悄死掉的服务。
本对机器已验证过的档位：`300K × 4`、`400K × 4`、`500K × 3`、`512K × 3`、`600K × 1`、`1M × 1`。

## KV 池与内存

`PARALLEL` 大于 1 时，所有请求的逐 token 缓存都从**同一个共享池**里取，按 token 计量（`TF_DSV41_POOL_TOKENS`）：

| | 现役 |
| --- | ---: |
| 并发请求数（`PARALLEL`）| 4 |
| 每请求窗口（`CONTEXT`）| 400,000 tokens |
| KV 精度 | FP8 |
| **共享池** | **1,638,400 tokens**（4.1 倍完整窗口）|
| rank 0 的权重占用 | 约 101 GiB |
| 服务态 `MemAvailable` | 两台均为个位 GiB |
| 起到 `/v1/models` | 53 s |

任何单条请求都能长到完整窗口，四条一起共享池子。**真正卡人的是池，不是 `max_model_len`** ——
比池子还大的请求会被拒，而报错文案说的是 *pool pages*：

```
CONTEXT=1000000 PARALLEL=1 POOL_TOKENS=614400                # 按 4 × 300K 配的池
{"error": {"message": "the request needs 3343 KV pool pages, the pool has 2400"}}
CONTEXT=1000000 PARALLEL=1 POOL_TOKENS=1228800               # 池翻倍
→ 855,600 token 请求成功，598 s，预填充 1,431 tok/s，needle 命中
```

这里一直成立的经验公式：**池 ≥ 槽数 × 窗口 + 回复预算**。当长请求在"看起来远小于 `CONTEXT`"的长度上被拒时，
先看池的数字，别先动窗口。

## 配置

所有设置都在 `config/prod.env`（从 `config/prod.env.example` 复制而来）。现役取值：

| 变量 | 值 | 含义 |
| --- | ---: | --- |
| `CONTEXT` | `400000` | 每请求的 prompt + 回复窗口 |
| `PARALLEL` | `4` | 同时解码的请求数 |
| `KV_DTYPE` | `fp8` | KV 缓存精度 |
| `MAX_TOKENS` | `32768` | 未指定 `max_tokens` 的请求的回复预算 |
| `TF_DSV41_POOL_TOKENS` | `1638400` | 共享池（token） |
| `TF_DSV41_FLOOR_GIB` / `_HARD_GIB` | `5` / `4` | 软告警 / 起服硬拒（GiB）|
| `TF_DSV41_PREFILL` | `replay` | 有界重放预填充（CED），默认 |
| `TF_DSV41_PREFILL_CHUNK` / `_ROWS` | `2048` | 每块预填充行数 |
| `TF_DSV41_PREFILL_ADAPT_GIB` | `4.5` | 自适应行数目标 |
| `TF_DSV41_PREFILL_KERNELS` | `fast` | prompt 行内核（比特一致对比时用 `exact`）|
| `TF_DSV41_PREFILL_ROUTER` / `TF_DSV41_ROUTER` | `gemv` | 路由路径 |
| `TF_DSV41_SPEC_DRAFT` | `1` | 精确 DSpark 投机解码 |
| `TF_DSV41_GRAPH_MODE` | `rows` | CUDA graph 捕获模式 |
| `TF_DSV41_MHC_CUDA` / `_ATTN_CUDA` / `_DENSE_V3` | `1` | 上游调优的三处内核重写 |
| `TF_DSV41_L2PF` / `_MB` / `_PACE_GBPS` | `1` / `12` / `150` | 下一层内核权重的 L2 预取 |
| `TF_DSV41_CALIB` | `real` | 沿用上一轮运行的校准 |
| `TF_DSV41_PLAN_LINK` | `nccl` | 跨 rank 通信计划 |
| `TF_DSV41_THINKING` | `0` | 默认关思考 |
| `TF_DSV41_DEFAULT_EFFORT` | `high` | 请求打开思考时使用的 effort |

两条警告：

- **`TF_DSV41_PLAN_LINK=rdma` 在我们所用的发布版里起不来。** `config/prod.env.example` 写的是 `rdma`，
  但随镜像发布的校验器只接受 `nccl` 或 `tcp`，会直接中止起服：
  `tensorfold: TF_DSV41_PLAN_LINK='rdma': expected nccl or tcp` 然后 `[dsv41-tf] start failed`。
  我们跑 `nccl`，并在同一档位实测 `nccl` 优于 `tcp`（4 路 code 聚合 148.8 vs 135.7 tok/s）。已向上游反馈。
- **`TF_DSV41_CALIB=real` 需要上一轮运行留下的校准目录**；在全新机器上它会静默地让首次起服变慢。

## 思考与采样

| 请求 | 结果 |
| --- | --- |
| Prose，`max_tokens=1200`，不传思考参数 | `content=""`（空），推理通道 5,918-13,755 字符，`finish_reason=length` |
| 同上，`effort` `low` / `high` / `max` × `max_tokens` 1200 / 3000 共六组 | **六组全部** `content=""`，`finish_reason=length` |
| 同上，关闭思考 | `content` 3,000+ 字符（约 740 词），`finish_reason=stop` |

因此不传思考开关的客户端会看到空回复并重试，每次都再付一遍预填充。**我们把服务端默认改成关思考**
（`TF_DSV41_THINKING=0`），由请求按需打开：

```json
{"chat_template_kwargs": {"enable_thinking": true}}          // 本条请求思考
{"reasoning_effort": "low" | "high" | "max"}                 // 思考时的 effort
```

关思考没有改变我们跑的检查：算术、文字陷阱题、小型方程、JSON schema 用例与工具调用都返回预期答案，
且回复落在 `content` 而不是 `reasoning_content`。在这类负载上的副作用是单流 code 解码
**55.6 → 81.8 tok/s**，因为原来的思考预算吃掉了同一个回复上限。**多步数学与复杂调试仍然是显式思考预算更强** ——
那类请求按需打开即可。

采样：`temperature`、`top_p`、`top_k`、`min_p`、`seed` 均为按请求；`temperature: 0` 为贪心解码，
上文所有数字都是这样测的。不传 `seed` 时采样键来自提示词，因此同一个请求会得到同一个回复。

## 这些数字是怎么测的

- 单对机器、空闲 —— GPU 上没有别的东西。共用 GPU 会让以上全部失效。
- 贪心（`temperature 0`），每个单元固定 `max_tokens`，**单流单元前缀缓存为冷**（提示词各不相同）；
  热前缀单元会明确标注。
- token 数取自响应的 `usage` 字段，**不靠数流式增量** —— 投机解码下单个流式块可能携带多个 token。
- 解码速率是**首 token → 末 token**（上游引擎自己的定义）；TTFT 单独报告；预填充 = 冷提示词的 prompt token / 墙钟。
- 聚合单元**用跨并发请求的同一时钟**，所有请求同样的 `max_tokens`，且每个槽用同一提示词（各自加一行后缀），
  避免任何负载偏斜测量窗口。
- 起服时间 = 从容器启动到本地端口 `/v1/models` 成功的墙钟时间。
- 质量检查：上表每个长档位的中段 needle、算术、文字陷阱题、方程、JSON schema 用例与工具调用。

明确写出的已知边界（而非藏起来）：

- **不与上游表格逐项对比就不是同口径。** 上游配方的单流单元来自**引擎内部**基准（`m2bench`，中间没有 HTTP 服务器）；
  本文全部是客户端经 HTTP 看到的数字。同一提示词下我们测到其单元的 90-106%
  （code 热前缀 82.6 vs 78.15；structured 102.4 vs 122.91），残差大概来自 HTTP 一跳与共享池调度。
  报数字时必须说清是哪条路径，否则对比会误导。
- structured/counting 负载的差距是差距最大的一项，我们还没有完整解释；发行版里缺的 `PLAN_LINK` 分支（见上）
  是最可能的原因。
- 单对机器上"1M + 并发槽"在物理上不可能（见 [窗口与槽数](#窗口与槽数)）。
- 400K × 4 档已有多小时实际使用，但本仓库不宣称做过正式的 soak 测试。

## 仓库结构

```
README.md          本文件（英文）
README.zh-CN.md    中文版
REDACTION-MAP.md   发布前的脱敏扫描范围，以及数字被允许说明到什么程度
CREDITS.md         建立在其之上的工作与致谢
NOTICE.md          随 MIT 许可一起的第三方声明
LICENSE            MIT（本仓库自有材料）
tools/bench.py     客户端基准：单流、N 路聚合、长上下文 needle
```

## 许可

本仓库自有的文本、配置与测量脚本采用 MIT，见 [`LICENSE`](LICENSE)。
模型权重、引擎与上游配方沿用各自许可，本仓库不重新分发其中任何部分；
详见 [`NOTICE.md`](NOTICE.md) 与 [`CREDITS.md`](CREDITS.md)。
