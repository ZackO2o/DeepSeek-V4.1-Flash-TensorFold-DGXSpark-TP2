---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash
tags:
  - deepseek
  - deepseek-v4.1-flash
  - tensorfold
  - dgx-spark
  - gb10
  - exl3
  - 512k-context
language:
  - zh
pipeline_tag: text-generation
---

# DeepSeek-V4.1-Flash 在 2× NVIDIA DGX Spark (GB10) 上，TensorFold TP=2
### 512K 上下文 · 3 并发槽 · FP8 KV · 精确 DSpark 投机解码 · Engram 走本地 NVMe

**English version: [README.md](README.md)**

本仓库公开我们在**两台 GB10（"DGX Spark"，SM 12.1，各 128 GB 统一内存）**上用
[TensorFold](https://github.com/ashhart/TensorFold) 引擎、经单条 CX7 QSFP 链路做张量并行（TP=2）
**实际部署** DeepSeek-V4.1-Flash 的配置，以及我们测到的全部数字。

这不是论文，是现场记录。价值在那些不显然的地方：这对机器上**哪些"上下文 × 槽数"组合物理上能装下**、
真正卡住长请求的是 **KV 池**而不是 `max_model_len`、以及**客户端口径**与**引擎内部口径**的差别有多大。

以下所有数字均在我们自己的机器上、从客户端侧实测；除注明外单流单元的**前缀缓存为冷**。

| | |
|---|---|
| **基座模型** | DeepSeek-V4.1-Flash（2.9 bpw EXL3 包，39 个 safetensors 分片，约 197 GB）|
| **硬件** | 2 × GB10 / SM 12.1，128 GB 统一内存，2 × 200 Gb/s CX7（RoCE v2）|
| **引擎** | TensorFold `v0.6.0`（pin 住的子模块）+ 配方补丁集，镜像为 GB10 构建 |
| **并行** | TP = 2，单条 QSFP，RoCE |
| **KV 缓存** | FP8，共享池，按 token 计量 |
| **部署档位** | **512,000 tokens**（`CONTEXT=512000`），**3 个请求槽** |
| **KV 池（部署值）** | **1,638,400 tokens** = 512K 请求的 3.2 倍 |
| **解码（512K 档）** | code **82.2** / prose **42.1** / structured **103.1** tok/s（单流）；3 路同负载聚合 **104.0** |
| **预填充** | 真实 **419,879** token 请求 **1,810 tok/s**（232 s），needle 命中 |
| **质量闸门** | 214K / 292K / 411K / 582K / 599K / 855K 全长度 needle 命中；工具调用、结构化 JSON、对话均正确 |
| **起服** | 各档位 47-55 s 冷起到可用 |
| **许可** | 本仓库自有材料 MIT；权重沿用其自身许可 |

---

## 1. 三件真正要紧的事

### 1.1 上下文与槽数此消彼长 —— 起服时的内存地板是硬约束

引擎在起服时按 `CONTEXT × PARALLEL` 预解内存，余量低于硬地板（`TF_DSV41_FLOOR_HARD_GIB`，默认 **4.0 GiB**）
就直接拒绝起服并报出差额。因此**"更长上下文"和"更多并发"是取舍，不是两个独立旋钮**：

| 档位 | 结果 |
|---|---|
| `CONTEXT=300000 PARALLEL=4` | 起服 ✅ `request slots: 4`，55 s |
| `CONTEXT=400000 PARALLEL=4` | 起服 ✅ 55 s |
| `CONTEXT=500000 PARALLEL=4` | **拒起** —— `4 x 500000 on rank 0 leaves 3.90 GiB, under the 4.0 GiB hard floor; fit: 3 stream(s)` |
| **`CONTEXT=512000 PARALLEL=3`** | **起服 ✅ `request slots: 3`，53 s ← 现役档** |
| `CONTEXT=600000 PARALLEL=1` | 起服 ✅ 47 s |
| `CONTEXT=1000000 PARALLEL=1` | 起服 ✅ 47-57 s |
| `CONTEXT=1000000 PARALLEL=4` | **拒起** —— `4 x 1000000 on rank 0 leaves 0.56 GiB, under the 4.0 GiB hard floor; fit: 1 stream(s)` |

也就是说：单对 DGX Spark 上 **512K 只能是 3 槽档**，想上 4 槽就必须回到 400K 及以下。
**这对机器上不存在"既是 1M 又开 4 槽"的配置** —— 统一内存就是不够。

### 1.2 卡住长请求的是 KV **池**，不是 `max_model_len`

`TF_DSV41_POOL_TOKENS` 是共享池的**总 token 预算**，与槽数独立。长请求是撞**池**失败的，
而报错文案里是"pool pages"，乍看像是上下文超限：

```
$ CONTEXT=1000000 PARALLEL=1 POOL_TOKENS=614400        # 2400 页
{"error": {"message": "the request needs 3343 KV pool pages, the pool has 2400"}}
$ CONTEXT=1000000 PARALLEL=1 POOL_TOKENS=1228800       # 池翻倍
→ 855,600 token 请求成功，598 s，prefill 1,431 tok/s，needle 命中
```

我们这对机器上一直成立的经验公式：**池 ≥ 槽数 × 上下文 + 回复预算**。现役档用
`1,638,400` 支撑 3 × 512K。当长请求在"看起来远小于 `max_model_len`"的长度上被 400 时，
**先看池的数字，别先动上下文**。

### 1.3 客户端口径 vs 引擎内部口径

上游配方的单流数字来自**引擎内的基准程序**（`m2bench`），中间没有 HTTP 服务器；走 HTTP 对外服务时，
同一负载在客户端测会更低。两个口径都对，**混用就会得出错误结论**（"我们比上游慢一半"）。
下表是我们的**客户端侧 HTTP** 数字，上游的表格在文中会标明是他们的。

对比时还要分清这个单元是*首 token→末 token*（纯解码）还是**含 prompt 的端到端**：384 token 回复、
TTFT 300 ms 时，含 prompt 会让读数低约 6%，prompt 越长差得越多。

---

## 2. 实测：现役 512K × 3 档

提示词用上游基准自带的原文（`CODE` = O(1) 的 LRU 缓存类；`PROSE` = 400 词灯塔散文；
`STRUCTURED` = "从 1 数到 200"），384 token 回复，greedy，关思考。

### 2.1 单流

| 负载 | T = 0 | T = 0.7（前缀命中）|
|---|---:|---:|
| `CODE` | **78.3** tok/s | **82.2** tok/s |
| `PROSE` | 40.5 tok/s | 42.1 tok/s |
| `STRUCTURED` | 98.6 tok/s | 103.1 tok/s |

这些运行的 TTFT 为 0.26-0.35 s，所以数字由解码主导，不是预填充主导。

### 2.2 并发（同负载，避免负载偏斜）

| | 聚合 |
|---|---:|
| 3 × `CODE`（同提示词 + 每路一行后缀）| **104.0** tok/s（各路墙钟 3.6 / 3.7 / 3.7 s —— 无串行化）|
| 3 × 混合（`CODE` / `PROSE` / `STRUCTURED`）| 75.4 tok/s |

"池子会不会串行化"要用**同负载**单元回答：三路在 0.1 s 内同时收尾。混合负载单元由最长的那个提示词主导，
而**提前结束的提示词会把测量窗口拖长** —— 两个口径都要给，否则数字会误导。

### 2.3 长上下文

真实长请求，needle 埋在文档中段：

| prompt tokens | 结果 | 墙钟 | 预填充 | needle |
|---:|---|---:|---:|---|
| **419,879**（现役 512K 档）| 成功 | 232.0 s | **1,810 tok/s** | **命中** |
| 214,319 | 成功 | 111.6 s | 1,921 tok/s | 命中 |
| 291,719 | 成功 | 175.0 s | 1,667 tok/s | 命中 |
| 411,362 | 成功 | 232.0 s | 1,774 tok/s | 命中 |
| 582,210（600K 档）| 成功 | 354.6 s | 1,642 tok/s | 命中 |
| 599,310（600K 档）| 成功 | 369.0 s | 1,624 tok/s | 命中 |
| 855,600（1M 档）| 成功 | 597.8 s | 1,431 tok/s | 命中 |

预填充衰减很平缓 —— 200K 附近约 1.9K tok/s，1M 端仍 1.4K tok/s —— 各长度检索全部命中。
**客户端超时必须按预填充来定**：512K 提示词约 4 分钟才出首 token，1M 约 10 分钟；
需要界面显示进度就用流式。

### 2.4 起服

| 档位 | 到可用耗时 |
|---|---:|
| 512K × 3（现役）| **53 s** |
| 400K × 4 | 55 s |
| 600K × 1 / 1M × 1 | 47 s |

能做到亚分钟重启，靠的是**预先准备好的分 rank 目录**与已编译内核缓存；从零构建是另一回事。

---

## 3. 配置（现役）

下面是生产环境文件里**有意义**的那部分 —— 其余都是路径：

```ini
CONTEXT=512000
PARALLEL=3
KV_DTYPE=fp8
MAX_TOKENS=32768

# KV 池：总 token 预算，与槽数独立
TF_DSV41_POOL_TOKENS=1638400

# 内存地板：软拒绝 / 起服硬拒绝（GiB）
TF_DSV41_FLOOR_GIB=5
TF_DSV41_FLOOR_HARD_GIB=4

# 预填充：有界重放 + 自适应行数
TF_DSV41_PREFILL=replay
TF_DSV41_PREFILL_CHUNK=2048
TF_DSV41_PREFILL_ROWS=2048
TF_DSV41_PREFILL_ADAPT_GIB=4.5
TF_DSV41_PREFILL_KERNELS=fast
TF_DSV41_PREFILL_GATHER=bf16
TF_DSV41_PREFILL_ROUTER=gemv

# 解码路径
TF_DSV41_SPEC_DRAFT=1            # 精确投机解码
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

# 跨 rank 通信计划
TF_DSV41_PLAN_LINK=nccl

# 服务默认：关思考（见 §4）
TF_DSV41_THINKING=0
TF_DSV41_DEFAULT_EFFORT=high
```

其中两条要特别提醒：

* **`TF_DSV41_PLAN_LINK=rdma` 在发布镜像里起不来。** 上游示例写的是 `rdma`，但我们所用版本的校验器只接受
  `nccl` 或 `tcp`，传 `rdma` 会直接中止起服。我们跑 `nccl`，并在同一档位做过 A/B：
  `nccl` 明显优于 `tcp`（4 路 code 聚合 148.8 vs 135.7 tok/s）。已向上游反馈。
* **校准与前缀分层依赖预先准备的目录。** `TF_DSV41_CALIB=real` 需要上一次运行留下的校准目录；
  在全新机器上它会静默地让首次起服变慢。

---

## 4. 思考默认档：客户端不显式关闭，`content` 就是空的

这一条让我们多花了一轮排查，所以写下来。服务端默认 effort 为 `high` 时，散文/短答类请求配上有限的
`max_tokens`，会把预算全部花在推理通道里，**`content` 返回空字符串**：

| 请求 | 结果 |
|---|---|
| `PROSE`，`max_tokens=1200`，不传思考参数 | `content=""`，推理通道 5,918-13,755 字符，`finish_reason=length` |
| 同上，`effort = low / high / max` × `max_tokens = 1200 / 3000` 共六组 | **六组全部** `content=""`，`finish_reason=length` |
| 同上，显式关闭思考 | `content` 3,000+ 字符 / 约 740 词，`finish_reason=stop` |

也就是说，**不传思考开关的客户端会看到空回复并重试**，每次都再付一遍预填充。我们把服务端默认改成
**关思考**，由请求按需打开；在这类负载上的副作用是单流 code 解码 55.6 → 77.6 tok/s，
因为原来的思考预算吃掉了同一个回复上限。

关思考后我们跑的质量检查没有退化：算术、文字陷阱题、小型方程、工具调用路径都给出预期答案。
**多步数学与复杂调试**仍然更适合显式打开思考预算 —— 那类请求按需开启即可。

---

## 5. 运维

* **起服托管**：用 **systemd 用户单元**跑服务（开 linger），再加一个短周期看门狗单元做健康检查与失败重启。
  有预先准备的分 rank 目录，重启在 1 分钟内完成，无人值守也能自愈。
* **档位切换**：每档一个环境文件（300K×4、400K×4、512K×3、600K×1、1M×1），换文件 + 重启即可。
  写个小包装脚本改三项（`CONTEXT`、`PARALLEL`、`POOL_TOKENS`）、归档、重启并**判定是否起服成功**就够；
  因为拒起会打印确切的差额，脚本可以报"装不下"，而不是留下一个悄悄死掉的服务。
* **内存永远是约束**：权重常驻后，服务态可用内存只剩个位 GiB。两个 rank 都要盯 ——
  在一个请求解码时再放进一条长请求，就会顶到地板。
* **客户端超时必须大于预填充时长**：服务端一条长请求会占住槽位数分钟；客户端 60 s 就切断的话，
  表面现象是超时，其实引擎还在算。

---

## 6. 数字是怎么测的

* 单对机器、空闲 —— GPU 上没有别的东西。共用 GPU 会让以下全部失效。
* `greedy`（`temperature 0`），每个单元固定 `max_tokens`，单流单元**前缀缓存为冷**（提示词各不相同），
  显示热前缀单元的会明确标注。
* token 数取自响应的 `usage` 字段，**不靠数流式增量** —— 投机解码下单个流式块可能携带多个 token。
* 解码速率是**首 token → 末 token**；prompt/TTFT 那一段单独作为预填充报告。
* 起服耗时是从进程启动到本地端口 `/v1/models` 成功的墙钟时间。
* 质量检查即正文所述：各长上下文档位的中段 needle、算术、文字陷阱题、方程、JSON schema 用例与工具调用。

本仓库**没有解决的问题**（明确写出，而非藏起来）：

* 与上游引擎内表格的差距尚未完全解释；最大残差出现在 structured/counting 负载上，
  而我们怀疑的主因正是 §3 里缺失的那条跨 rank 传输分支。
* 单对机器上 1M 并发在物理上不可能（见 §1.1）。要更多并发的 1M 会话需要加硬件，不是继续调参。
* 512K × 3 档是刚起服后测的，我们还没有对它做数小时级 soak。

---

## 7. 致谢

这个部署站在别人的工作上，有意思的想法都是他们的：

* **TensorFold**（Ash Hart）—— 引擎：EXL3 内核、服务端、本配方所基于的投机路径。
* **2× DGX Spark DeepSeek-V4.1-Flash 配方**（jayleaton）—— 双 Spark 引擎栈、DeepSeek-V4.1 家族实现、
  预准备分 rank 目录、我们原样复用的基准定义，以及修掉我们踩到的打包与内存泄漏问题的那串公开 issue 讨论。
* **MiaAI-Lab** —— 2.9 bpw EXL3 包，以及上游表格里作为基线的 vLLM kit。
* **ExLlamaV3**（turboderp 及贡献者）—— EXL3 格式。
* **DeepSeek** —— 模型、技术报告，以及投机与 n-gram 路径所实现的 DSpark / Engram 工作。
* §2 的基准提示词引自上游配方自己的工作负载定义，以保证同口径对比。

**本仓库新增的部分**：512K × 3 档位及其测量、这对机器上的槽数伸缩阶梯、池子与上下文的失败模式及复现、
以及客户端侧对比方法学。

## 8. 许可

本仓库自有的文本、配置与测量脚本采用 MIT。模型权重与上游组件沿用各自许可，
详见 [NOTICE.md](NOTICE.md) 与 [CREDITS.md](CREDITS.md)。
