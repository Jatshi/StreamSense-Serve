# StreamSense-Serve 2.0：vLLM/SGLang、量化、压测与数据飞轮学习手册

版本：2.0  
目标：真正理解并能独立重建服务层，而不是只会运行一条启动命令  
验证硬件：单张 RTX 4090 24GB  
基准模型：固定 revision 的 `Qwen/Qwen2.5-VL-3B-Instruct`

---

## 0. 项目在三仓作品集里的位置

### 0.1 一句话定义

StreamSense-Serve 是面向流式音视频理解和可验证 Agent 的服务与数据飞轮：它用自适应路由决定何时调用重型 VLM，通过统一 profile 启动 vLLM/SGLang，记录真实 TTFT/TPOT/吞吐/显存和量化质量回归，保存可追踪事件，并把明确同意且有许可证的纠错反馈导出为 SFT/DPO 样本。

### 0.2 它不重复训练仓库

三仓职责：

```text
Audio-LLM
  负责模型如何通过 SFT/DPO/GRPO 学会

EvidenceAgent-MM
  负责回答为什么有证据、何时追问或拒答

StreamSense-Serve
  负责模型怎样启动、被路由、被压测、被切换，以及错误怎样回流
```

服务层不应重新实现 GRPO；训练仓库也不应混入生产模型注册、鉴权和负载测试。

### 0.3 学完后的验收题

你应该能回答：

- 为什么 vLLM 和 SGLang 要用同一套 profile contract？
- TTFT、TPOT、端到端延迟和吞吐分别说明什么？
- 为什么 streaming delta 不能冒充 tokenizer token？
- FP8 为什么可能省显存，又为什么必须做质量回归？
- 如何防止后端启动参数被 shell injection？
- 模型切换为什么必须校验 revision 和 validated 状态？
- 反馈为什么要同时检查 consent 与 source license？
- content hash 去重解决了什么，不能解决什么？
- 为什么矩阵压测要在模型常驻后跑多个并发，而不是每组重启？
- 为什么一帧 VLM 演示不能代表生产多模态准确率？

---

## 1. 一张图理解 2.0

```text
audio/video upload
  ↓
lightweight analyzers
  ├─ energy / ASR
  ├─ frame change
  └─ uncertainty / conflict / risk
  ↓ RuleRouter
  ├─ lightweight path
  └─ VLM escalation
       ↓ BackendProfile
       ├─ vLLM BF16
       ├─ vLLM FP8
       └─ SGLang BF16
  ↓
typed observations → events → SQLite
  ↓
EvidenceAgent three-state query
  ↓
feedback
  ├─ consent=false → audit only
  ├─ no license → fail closed
  └─ correction + consent + license
       ↓
SFT / DPO / EvidenceAgent bridge / raw audit
       ↓
manifest + SHA-256
```

服务的第二条证据链：

```text
pinned profile
  ↓ safe launcher
healthy resident backend
  ↓ warmup
quality fixture
  ↓ concurrency 1/4/8/16/32
TTFT / TPOT / latency / req/s / output tokens/s / VRAM
  ↓
BF16 vs FP8 vs SGLang matrix
```

---

## 2. Backend profile 是唯一启动事实源

### 2.1 为什么不用散落的环境变量

如果 model、revision、dtype、port、quantization 分散在 shell、Docker 和 Python 中，会出现：

- README 写一个模型，服务实际加载另一个；
- 量化基准与 BF16 不是同 revision；
- 健康检查返回 alias，却无法追踪真实 checkpoint；
- benchmark 无法复现；
- 热切换时端口和模型名错配。

`configs/backends.json` 把这些字段绑定成一个原子 profile。

### 2.2 `EngineConfig`

核心字段：

```text
model_path
served_model_name
revision
dtype
quantization
tensor_parallel_size
max_model_len
gpu_memory_utilization
trust_remote_code
extra_args
```

`quantization` 只接受：

```text
none, auto, awq, gptq, bitsandbytes, fp8
```

这不是说每种组合都已经实测，只是 schema 能表达。只有真实启动和报告通过后才可以声称支持某组合。

### 2.3 本轮三个实测 profile

| profile | runtime | precision | port |
|---|---|---|---:|
| `vllm-qwen25-vl-3b` | vLLM | BF16 基线权重 | 8001 |
| `vllm-qwen25-vl-3b-fp8` | vLLM | 动态 FP8 | 8002 |
| `sglang-qwen25-vl-3b` | SGLang | BF16 基线权重 | 30000 |

三者固定同一 upstream revision：

```text
66285546d2b821cf421d4f5eb2576359d3770cd3
```

### 2.4 safe `extra_args`

每个额外参数必须：

- 以 `--` 开头；
- 不包含换行；
- 不包含 `; & | \` $`。

launcher 使用 `subprocess.run(list)` 而不是 `shell=True`。这两层一起避免 profile 被当成 shell 程序执行。

### 2.4.1 为什么 SGLang 不能只写 `>=0.5`

真实 AutoDL 预检暴露了运行时与驱动的三层兼容关系：

```text
NVIDIA driver
  ↔ PyTorch wheel 的 CUDA build
  ↔ SGLang / sglang-kernel / FlashInfer ABI
```

SGLang 0.5.11 起默认依赖线升级到 Torch 2.11/CUDA 13。在 560 系列驱动上，pip 可以成功安装全部包，但 `torch.cuda.is_available()` 会返回 false；“安装成功”因此不能算服务可用。最终矩阵使用独立的 SGLang 0.5.10 环境，固定 Torch 2.9.1+cu128，并在模型加载前执行真实 CUDA 矩阵运算。

这也解释了为什么 vLLM 和 SGLang 必须分 venv：两个 serving runtime 的 kernel wheel 和 Torch 约束可能不同，共用环境会让后安装者悄悄替换前一个 runtime 的底层依赖。

FlashInfer 的 JIT-cache wheel 约 1.2 GiB，AutoDL 直连 GitHub 可能超时。脚本因此允许通过
`STREAMSENSE_FLASHINFER_JIT_CACHE_SPEC=/绝对路径/flashinfer_jit_cache.whl`
指定已完成校验的本地 wheel；变量未设置时仍使用固定版本
`flashinfer_jit_cache==0.6.7.post2`。解析器以阿里云 PyPI 镜像为主索引，并把元数据和 wheel
持久化到 `/root/autodl-tmp/.cache/uv`，中断后无需重新扫描全部海外索引。这些优化只改变传输与缓存路径，不改变依赖版本、训练内容或评测口径。

全矩阵第一次启动 SGLang 时还暴露了一个更隐蔽的依赖边界：包导入、CUDA
矩阵预检和模型权重加载都能通过，但 JIT 扩展真正编译时才调用外部 `ninja`，
环境中没有该可执行文件，于是以 `FileNotFoundError: ninja` 退出。处理过程没有
改请求数或跳过 SGLang：

1. 保存失败的 `backend.log`；
2. 确认进程退出且 `nvidia-smi` 中没有残留 CUDA PID；
3. 在 SGLang venv 固定安装 `ninja==1.13.0`；
4. 发现“直接调用 `/venv/bin/python`”不会像 `activate` 那样修改 `PATH`，而且
   对该符号链接调用 `resolve()` 会错误回到 base Python 的目录；
5. launcher 因此把未解析符号链接的 `/venv/bin` 放在子进程 `PATH` 首位，并把
   ninja 版本写回 bootstrap，避免“手工修好但无法复现”；
6. 使用原定 64 请求和 1/4/8/16/32 并发重跑完整 SGLang profile。

这个案例说明 preflight 只能覆盖已知契约。JIT 依赖直到真实模型路径被执行才会暴露，
因此失败日志、环境修复和无缩减重跑都是实验结果的一部分。

### 2.5 vLLM 与 SGLang 参数映射

同一语义在两个 runtime 中名字不同：

| 语义 | vLLM | SGLang |
|---|---|---|
| model | `--model` | `--model-path` |
| served name | `--served-model-name` | 同名 |
| tensor parallel | `--tensor-parallel-size` | `--tp-size` |
| context | `--max-model-len` | `--context-length` |
| memory fraction | `--gpu-memory-utilization` | `--mem-fraction-static` |

`build_command` 是适配层；上层 benchmark 不应关心这些差异。

---

## 3. 后端启动与健康检查

### 3.1 启动状态机

```text
validate config
  ↓
render argv list
  ↓
spawn backend process
  ↓
poll PID + /health
  ├─ PID exits → fail and preserve log
  ├─ timeout → fail
  └─ healthy → warmup and benchmark
```

只检查端口打开不够。进程可能是旧服务，或模型仍未 ready。健康检查必须对 profile 的 base URL 发请求，并记录实际模型标识。

### 3.2 为什么需要 20 分钟启动上限

冷启动可能包含：

- 权重读取；
- safetensors mmap；
- CUDA context；
- kernel/JIT compilation；
- KV cache 预分配；
- JSON grammar 或 prefix cache warmup。

上限过短会把正常冷启动误判失败；没有上限则进程可能永远等待。脚本同时轮询 PID，后端提前崩溃会立即报告，而不是浪费完整超时时间。

### 3.3 退出清理

benchmark 每个 profile 结束时：

1. `kill` backend PID；
2. `wait` 回收子进程；
3. 查询 `nvidia-smi --query-compute-apps`；
4. 确认 CUDA context 消失；
5. 再启动下一 precision/runtime。

launcher 的`--execute`路径使用`os.execv`把包装进程替换成真正的vLLM/SGLang进程，
因此脚本记录和终止的PID就是模型服务器PID，而不是一个可能提前死亡并留下孤儿子进程的
Python wrapper。清理等待结束后若仍检测到任意CUDA PID，矩阵直接失败，不会继续下一组。
否则上一后端残留显存会让下一组 OOM，结果也不再公平。

---

## 4. vLLM、SGLang 与模型服务基本原理

### 4.1 为什么不用原生 `generate()` 直接开 API

原生 batch 推理通常缺少：

- continuous batching；
- paged KV cache；
- request scheduler；
- streaming OpenAI API；
- prefix caching；
- 并发下的显存管理；
- 标准 metrics 和健康检查。

vLLM/SGLang 的核心价值不是“代码少”，而是在模型常驻时动态组合请求，提高并发吞吐并降低 KV cache 碎片。

### 4.2 continuous batching

传统静态 batch 等所有序列完成：

```text
request A: ───────────── done
request B: ───── done → 空等
request C: ───────── done → 空等
```

continuous batching 在 B 完成后立刻插入新请求 D。调度单位接近 token step，而不是固定整批。

### 4.3 KV cache

自回归生成时，历史 key/value 不应每个 token 重算。KV cache 大致随以下量增长：

\[
M_{KV}
\propto
B\times L\times N_{layers}\times H_{kv}\times d\times bytes
\]

因此并发、上下文长度和 precision 会共同影响显存。`gpu_memory_utilization=0.88` 是容量/稳定性的取舍，不代表应该无条件设到 0.98。

### 4.4 prefix caching

如果多请求共享 system prompt 或证据前缀，可以复用 prefix KV。它提升取决于前缀重复率；固定 benchmark prompt 会有利于 cache，所以报告必须写明已启用，不能外推到完全不同的生产请求。

---

## 5. FP8 量化：必须同时测速度和质量

### 5.1 为什么会省显存

FP16/BF16 每权重约 2 bytes，FP8 约 1 byte。理想化权重内存：

\[
M_{FP8}\approx\frac{1}{2}M_{FP16}
\]

真实总显存还包括 KV cache、activation、CUDA graph、workspace 和 runtime allocator，所以 resident VRAM 不会严格减半。

### 5.2 动态 FP8 的代价

量化把连续值映射到较少离散值：

\[
q=\operatorname{clip}
\left(
\operatorname{round}(x/s),
q_{min},q_{max}
\right)
\]

再近似恢复：

\[
\hat{x}=s q
\]

scale 选择和 outlier 会带来误差。对于严格 JSON、citation ID、时间戳和数字，少量 logits 变化也可能改变最终 token，因此性能提升不能脱离质量回归。

### 5.3 本项目的质量 fixture

固定 12 个 case，覆盖：

- 中英文 exact instruction；
- answered/abstained/needs_clarification；
- 严格 JSON；
- citation ID；
- 时间戳；
- 算术 sanity；
- 无证据身份推断拒绝；
- 缺失视觉模态；
- 置信度数值。

每个 case 保存：

```text
prompt
raw output
expected substrings
forbidden substrings
JSON field checks
latency
usage
pass/fail
```

这仍然只是 contract-quality regression，不是通用多模态 benchmark。它回答“量化后是否破坏本服务最关键的格式与证据行为”，不是“模型总体能力下降多少”。

---

## 6. 压测指标必须知道分母

### 6.1 TTFT

\[
\operatorname{TTFT}
=t_{\text{first non-empty content}}-t_{\text{request start}}
\]

它包含排队、prefill 和服务开销。交互式应用通常对 TTFT 很敏感。

### 6.2 TPOT

近似定义：

\[
\operatorname{TPOT}
=\frac{t_{\text{finish}}-t_{\text{first token}}}
{N_{\text{output token}}-1}
\]

旧版脚本只能观测 SSE content delta；一个 delta 不等于一个 tokenizer token。2.0 请求：

```json
"stream_options": {"include_usage": true}
```

只有后端对每个成功请求都返回 `completion_tokens` 时，才报告真实 output tokens/s；否则该字段为 `null`，保留 delta/s 并明确语义。

### 6.3 吞吐

请求吞吐：

\[
QPS=\frac{N_{\text{success}}}{t_{\text{wall}}}
\]

输出吞吐：

\[
TPS=\frac{\sum N_{\text{completion token}}}{t_{\text{wall}}}
\]

只给 QPS 会受 max_tokens 和回答长度影响，跨实验难以比较；只给 TPS 又不能直接说明用户并发体验，所以二者都保留。

### 6.4 延迟分位数

排序后第 \(p\) 分位：

\[
P_p=x_{\lceil pN\rceil}
\]

报告 p50、p95、p99 和 max。均值会掩盖队尾延迟。

### 6.5 为什么跑 1/4/8/16/32

- concurrency 1：无排队基础性能；
- 4/8：轻中等负载；
- 16：高并发；
- 32：观察 scheduler、KV cache 和尾延迟拐点。

每组 64 个请求，模型在同一 profile 内保持常驻。若每个并发都重启，冷启动和 kernel warmup 会污染结果。

### 6.6 GPU 采样

每 0.5 秒读取：

```text
memory.used
memory.total
utilization.gpu
```

报告 peak memory，但也保存全部 samples。只在请求结束后读一次显存会错过峰值。

---

## 7. 自适应路由：为什么不是所有帧都进 VLM

### 7.1 四类升级条件

`RuleRouter` 检查：

- risk score；
- uncertainty；
- cross-modal conflict；
- needs visual grounding。

若没有触发项，还按 exploration rate 抽样升级，避免永远看不到规则误判的困难样本。

### 7.2 决策公式

概念上：

\[
E=
\mathbb{1}[risk\ge\tau_r]
\lor
\mathbb{1}[u\ge\tau_u]
\lor
\mathbb{1}[conflict\ge\tau_c]
\lor visual
\lor exploration
\]

若 \(E=1\)，走 `vlm_escalated`；否则走 `lightweight`。

### 7.3 reasons 为什么必须返回

只返回 route 无法回答：

- 为什么成本突然上升？
- 哪个阈值触发过多？
- 是不确定性还是视觉需求？
- exploration 样本能否进入无偏评测？

reasons 既是可观测性字段，也是后续训练路由器的标签来源。

### 7.4 路由 benchmark 的边界

手工 fixture 上节省 45% GPU 调用，只能证明给定阈值对该 fixture 的行为。若阈值已根据 fixture 调过，就不是无偏 test。正式发布必须说明这一点。

---

## 8. EvidenceAgent 兼容层

### 8.1 三态保持

服务接收统一 request/evidence，映射：

```text
answered
needs_clarification
abstained
```

生成后端不能把 `abstained` 改写成一个看起来更有帮助的猜测，也不能删除 evidence ID。

### 8.2 引用为什么由结构层控制

语言模型擅长措辞，不适合决定“哪些 ID 真实存在”。正确顺序：

```text
retrieve evidence
  ↓
deterministic citation set
  ↓
generate wording
  ↓
schema validate
  ↓
verify citations are subset of allowed IDs
```

如果生成模型输出陌生引用，应失败或回退，而不是自动把它加入数据库。

### 8.3 API 路由

2.0 核心端点：

```text
GET  /v2/inference/health
POST /v2/evidence-agent/query
POST /v2/feedback
POST /v2/feedback/export
GET  /v2/models
POST /v2/models/activate
POST /v2/models/rollback
```

旧 `/v1` API 保持兼容，2.0 功能不应破坏已发布客户端。

---

## 9. 反馈数据飞轮

### 9.1 为什么 feedback 不能直接进训练

反馈可能：

- 没有用户授权；
- 包含隐私；
- 来源许可证不允许训练；
- 重复提交；
- 修正为空；
- 恶意篡改 request/response；
- 只有低评分但没有可学习目标。

因此保存和训练导出是两个独立动作。

### 9.2 content hash 去重

对规范化提交计算 SHA-256，并在 SQLite 中唯一约束。它能防重复点击和重试导致的数据膨胀。

它不能自动发现语义近似但措辞不同的重复样本；更高级的 near-duplicate 需要 embedding/MinHash，并且仍要保留审计关系。

### 9.3 fail-closed 导出

只有同时满足：

```text
consent_for_training == true
source_license 非空
存在 corrected content
```

才进入训练候选。历史记录即使绕过新校验，导出时仍再次检查。

### 9.4 四类输出

```text
sft_candidates.jsonl
dpo_candidates.jsonl
evidenceagent_bridge.jsonl
consented_feedback_raw.jsonl
```

- SFT：纠正答案作为 assistant；
- DPO：纠正答案 chosen，原错误答案 rejected；
- bridge：保留结构化 request/target，供 EvidenceAgent 使用；
- raw：完整审计记录。

### 9.5 manifest

导出清单记录：

- eligible record 数；
- source content hashes；
- source manifest SHA-256；
- 每个输出文件 SHA-256；
- 每类 examples 数；
- 创建时间。

这样下一轮训练可以回答“这个 adapter 究竟由哪批反馈产生”。

---

## 10. 模型注册、切换与回滚

### 10.1 manifest 是唯一事实源

每个 artifact 包含：

```text
model_id
revision
status
backend/profile
artifact location
validation evidence
```

只有 `status=validated` 的模型可激活。

### 10.2 revision guard

激活请求必须携带 `expected_revision`。如果调用方看到的是旧 manifest，而服务已更新，revision 不匹配会失败，避免 TOCTOU 式误切换。

### 10.3 原子状态写入

先写临时文件再 replace：

```text
active_model_id
previous_model_ids
updated_at
reason
```

`previous_model_ids` 最多保留 20 个，支持明确回滚。

### 10.4 当前实现的边界

注册状态原子更新不自动等于“模型进程零停机热加载”。真正热切换还需要：

1. 启动 candidate；
2. 健康与质量检查；
3. 原子切流；
4. drain 旧请求；
5. 关闭旧进程；
6. 失败时保留旧服务。

本仓库只对实际实现的状态与检查做声明。

---

## 11. 安全与隐私

### 11.1 默认 localhost

后端和 API 默认绑定 `127.0.0.1`。公开暴露前必须加入反向代理、TLS、认证、速率限制和上传大小限制。

### 11.2 token 不进入日志

API key 只从环境变量读取，header 在请求时构造。benchmark 输出不得保存 token 或完整 Authorization。

### 11.3 上传边界

必须检查：

- 文件大小；
- MIME 与实际解码；
- 路径 traversal；
- 临时文件清理；
- FFmpeg 超时；
- 并发占用；
- 是否得到媒体处理授权。

### 11.4 不是身份识别系统

anonymous speaker turn 不能映射真实身份。任何要求“仅凭声音猜真实姓名”的请求应拒绝。

---

## 12. 从零重建：十二个 commit

1. `chore: create package fastapi and tests`
2. `feat: define event observation and evidence schemas`
3. `feat: add sqlite event store`
4. `feat: implement adaptive rule router`
5. `feat: add audio and video lightweight analyzers`
6. `feat: add openai-compatible vlm enhancer`
7. `feat: build media pipeline and evidence agent`
8. `feat: define backend profiles and safe launcher`
9. `feat: add streaming load metrics and gpu sampling`
10. `feat: add consent-gated feedback flywheel`
11. `feat: add revision-guarded model registry`
12. `feat: add bf16 fp8 sglang matrix and release docs`

每个 commit 都应有对应测试。先写 schema/纯函数/假的 HTTP server，再接真实 4090。

---

## 13. 七天学习实验

### Day 1：Backend contract

- 手写 `EngineConfig`；
- 构造非法 quantization；
- 尝试在 extra args 中放 `;`，确认失败；
- 输出 vLLM/SGLang argv。

### Day 2：SSE 和指标

- 手写 SSE parser；
- 用 fake server 逐块返回 content；
- 计算 TTFT/TPOT；
- 证明 delta count 不等于 token count。

### Day 3：真实 vLLM

- 启动固定 revision；
- 记录 cold/warm startup；
- concurrency 1/4/8；
- 查看 KV cache 与显存。

### Day 4：SGLang

- 用相同 prompt/长度跑；
- 对比健康检查、输出兼容性；
- 分析吞吐和尾延迟差异；
- 不只报“哪个更快”，还解释负载条件。

### Day 5：FP8

- 跑固定质量 fixture；
- 跑并发矩阵；
- 对比 resident/peak VRAM；
- 计算相对质量损失：

\[
\Delta Q=Q_{FP8}-Q_{BF16}
\]

### Day 6：反馈飞轮

- 创建同意、不同意、无 license、重复和篡改样本；
- 确认 fail closed；
- 检查四类导出和 SHA；
- 把 bridge 交给 EvidenceAgent validator。

### Day 7：模型切换和答辩

- 激活 validated model；
- 用错误 revision 触发失败；
- rollback；
- 画端到端架构；
- 回答后面的面试题。

---

## 14. 高频故障

### 14.1 后端进程在但 health 不通

看 `backend.log`：

- 权重 revision；
- CUDA OOM；
- port 占用；
- kernel compile；
- tokenizer/processor 版本；
- model architecture 是否受 runtime 支持。

不要只增加 sleep。

### 14.2 vLLM 安装很慢

训练 GPU 若仍满载，网络安装可并行。大 wheel 可从可信镜像手动下载并校验 SHA-256，再本地 pip install；不能跳过 hash 或使用来历不明的 wheel。

### 14.3 FP8 启动失败

依次检查：

- GPU 架构是否支持；
- runtime 版本；
- 模型层是否支持动态 FP8；
- dtype/quantization 组合；
- 是否需要额外 quantization package；
- 后端日志中第一个 root cause。

失败时不能悄悄改成 BF16 并仍把结果标成 FP8。

### 14.4 `reported_output_tokens` 为 null

说明至少一个响应没有 server usage。保留 delta metrics，但不能把它标为 tokens/s。确认 runtime 是否支持 streaming usage。

### 14.5 高并发 error rate 上升

检查：

- timeout；
- scheduler queue；
- KV cache capacity；
- max model length；
- client connection pool；
- 429/500/503 分类；
- GPU OOM；
- 是否在同机混跑训练。

### 14.6 质量 fixture 失败

性能 benchmark 仍可继续并保存失败，但发布结论必须包含 pass rate 和 raw output。量化若更快但破坏引用/JSON，不应默认上线。

---

## 15. 面试问题

### Q1：为什么同时做 vLLM 和 SGLang？

它们都提供高性能 OpenAI-compatible serving，但调度、kernel、结构化生成和生态取舍不同。使用统一 profile 和同一模型/请求矩阵，可以展示后端抽象和有控制变量的工程评测，而不是绑定单一框架。

### Q2：TTFT 和 TPOT 哪个更重要？

交互式问答更敏感于 TTFT，长文本吞吐更敏感于 TPOT。高并发时还必须看 p95/p99，因为平均值会掩盖排队。

### Q3：为什么 QPS 不能直接跨报告比较？

回答长度、max tokens、prompt 长度、cache、并发和 error rate 都会影响 QPS。必须同时给 token throughput 和配置。

### Q4：量化为什么要测 citation ID？

引用和时间戳是本系统的硬合同。量化造成的少量 logits 扰动若改变一个字符，语义看似接近但引用已经不可回放。

### Q5：prefix caching 会不会让 benchmark 虚高？

固定前缀会受益，因此报告必须说明启用。它是合法优化，但不能外推到无共享前缀的工作负载。

### Q6：为什么反馈导出需要 license？

用户同意只能说明隐私/授权意愿，不自动获得源内容的训练再利用权。consent 与 license 是两个不同条件。

### Q7：模型 manifest 和 Docker tag 有什么区别？

Docker tag 可变且只描述镜像；manifest 记录模型 revision、状态、后端、验证证据和 artifact。切换依据应是不可歧义的 manifest 条目。

### Q8：当前热切换的最大边界？

注册状态可原子切换和回滚，但真正的零停机模型进程切流还需要双实例、健康 gate 和 connection draining。

### Q9：路由节省 45% 能否写生产收益？

不能。它来自手工 fixture 且阈值已被观察，只能写工程 behavior/cost check。生产收益需要冻结阈值后的真实留出流量。

### Q10：如何进一步升级？

加入真实多模态 workload、tokenizer-exact benchmark、Prometheus/Grafana、请求分布回放、KV cache 命中率、CUDA graph 状态、双实例 canary、真实用户反馈闭环和数据漂移监控。

---

## 16. 简历表述边界

真实矩阵完成后可以写：

> 构建统一 vLLM/SGLang/OpenAI-compatible 模型服务层，固定 Qwen2.5-VL-3B revision，在 RTX 4090 上完成 BF16、动态 FP8 与 SGLang 的并发 1–32 基准；同时采集 TTFT/TPOT、服务端 completion-token 吞吐、尾延迟和显存，并以固定证据合同用例约束量化质量回归。

还可以写：

> 设计 consent/license 双闸门反馈数据飞轮，使用 SQLite 内容哈希去重并原子导出 SFT、DPO、EvidenceAgent bridge 与原始审计记录，为每次导出生成 source/output SHA-256 manifest。

不能写：

- “生产 QPS”；
- “FP8 零精度损失”，除非固定测试确实为零且明确测试范围；
- “零停机热更新”，除非真实双实例切流已验证；
- “多 GPU 推理”，本轮 TP=1；
- “通用多模态准确率”，固定 12 case 只验证服务合同。

---

## 17. 本轮真实矩阵

三组 profile 均固定 Qwen2.5-VL-3B revision
`66285546d2b821cf421d4f5eb2576359d3770cd3`、8,192 context、64-token
上限和同一请求合同。15 个单元全部完成 64/64 请求，合计 960/960 成功、零错误。
下面不是挑最快的一行，而是完整矩阵：

| profile | 并发 | req/s | completion tok/s | TTFT p50 / p95 ms | TPOT p50 ms | latency p95 ms | 峰值显存 MiB | 质量 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vLLM BF16 | 1 | 24.828 | 74.484 | 23.508 / 27.303 | 16.160 | 42.866 | 21,677 | 8/12 |
| vLLM BF16 | 4 | 64.740 | 194.220 | 37.029 / 64.162 | 19.624 | 82.742 | 21,677 | 8/12 |
| vLLM BF16 | 8 | 97.814 | 293.443 | 57.635 / 77.260 | 21.816 | 103.694 | 21,677 | 8/12 |
| vLLM BF16 | 16 | 152.576 | 457.729 | 67.286 / 107.737 | 21.785 | 135.869 | 21,677 | 8/12 |
| vLLM BF16 | 32 | 176.806 | 530.417 | 124.098 / 161.209 | 32.309 | 194.168 | 21,677 | 8/12 |
| vLLM dynamic FP8 | 1 | 26.291 | 78.872 | 24.370 / 31.005 | 12.429 | 43.627 | 21,683 | 7/12 |
| vLLM dynamic FP8 | 4 | 75.932 | 227.797 | 36.372 / 69.039 | 13.543 | 80.634 | 21,683 | 7/12 |
| vLLM dynamic FP8 | 8 | 100.158 | 300.473 | 55.418 / 82.578 | 17.083 | 103.898 | 21,683 | 7/12 |
| vLLM dynamic FP8 | 16 | 131.703 | 395.108 | 86.911 / 111.597 | 23.343 | 135.004 | 21,683 | 7/12 |
| vLLM dynamic FP8 | 32 | 189.843 | 569.530 | 123.866 / 158.425 | 24.236 | 193.499 | 21,683 | 7/12 |
| SGLang BF16 | 1 | 24.294 | 72.883 | 27.598 / 28.420 | 13.301 | 42.017 | 22,563 | 7/12 |
| SGLang BF16 | 4 | 40.424 | 121.273 | 51.770 / 412.955 | 18.275 | 428.161 | 22,583 | 7/12 |
| SGLang BF16 | 8 | 78.434 | 235.301 | 76.192 / 86.275 | 20.826 | 103.897 | 22,583 | 7/12 |
| SGLang BF16 | 16 | 123.563 | 370.688 | 91.752 / 110.130 | 24.072 | 131.649 | 22,583 | 7/12 |
| SGLang BF16 | 32 | 144.422 | 433.267 | 131.241 / 160.483 | 68.406 | 218.769 | 22,623 | 7/12 |

如何解读：

1. 动态 FP8 在并发 32 的短输出 workload 上最快，比 BF16 的请求吞吐高约
   7.4%，但这不是所有 prompt 长度和模态组合下都成立；
2. 两个 vLLM profile 的服务预留显存几乎相同，说明本轮动态 FP8 不是“显存减半”
   方案，不能只依据权重量化名称推断端到端显存；
3. BF16 质量 fixture 是 8/12，FP8 与 SGLang 是 7/12，因此不能写
   “FP8 零精度损失”；
4. SGLang 并发 4 出现 TTFT p95 412.955 ms 的尾延迟离群点。该行保留在结果中，
   没有为得到更好数字而删除或重跑；
5. 64 请求负载的输出都严格匹配 `benchmark-ok`，而 12-case fixture 专门测
   JSON、引用、时间戳、拒答等合同。前者证明服务可靠，后者才提供有限的行为差异；
6. 完整机器可读证据在 `docs/benchmark_matrix_4090.json`，原始交付包还包含
   15 份并发报告、3 份 quality、3 份 environment 和全部 backend log。

面试时可说：我做的是控制变量的 serving 工程矩阵，结论只对这张卡、这个 revision、
这些版本和这个 workload 有效；它证明 vLLM/SGLang 抽象、量化对照、压测语义、
失败恢复和生命周期治理，不冒充生产容量或通用模型准确率。

---

## 18. 2.0 真实实施日志与踩坑复盘

### 18.1 正确实施顺序

项目不是先装两个推理框架再“看看谁快”。真正顺序是：冻结请求与输出合同 → 写 backend
profile → 用假 OpenAI server 测 launcher/stream parser → 固定模型 revision 与 workload →
分别安装并启动真实后端 → 每个 profile 先过 health 和 12-case quality gate → 再跑五档
并发 → 确认 CUDA PID 清零后切下一个 profile → 汇总完整矩阵。

这个顺序避免两个常见错误：把后端启动失败误判成模型质量问题，以及让上一个服务残留
显存污染下一个 profile。

### 18.2 坑一：`sglang[all]` 不等于系统里有 Ninja

第一次 SGLang 启动失败发生在 JIT 编译。Python extra 名称看起来像“全依赖”，但实际
kernel 构建调用的是 PATH 中的外部 `ninja` 可执行文件。排查依据是 backend log 中第一
个 root cause，而不是进程是否仍存在。最终 bootstrap 显式固定 `ninja==1.13.0`，并在
preflight 同时检查 Python package 和 `ninja --version`。

### 18.3 坑二：调用 venv Python 不会自动继承 venv PATH

第二次启动时 Ninja 已安装在 venv，但 launcher 用绝对路径调用 `venv/bin/python`；
子进程 PATH 仍来自父 shell，JIT 找不到 `venv/bin/ninja`。这说明“解释器来自 venv”与
“shell 可执行搜索路径包含 venv”是两条独立状态。修复是在 launcher 构造子进程环境时
把解释器所在 `bin` 置于 PATH 首位。

### 18.4 坑三：`Path.resolve()` 把修复又破坏了

第三次失败更隐蔽：为得到解释器目录，代码先对 `venv/bin/python` 调 `resolve()`，它
沿符号链接回到 base Python，于是加入 PATH 的仍是错误目录。最终使用未解析的可执行
路径父目录，并写回归测试构造“venv python 是 symlink”的场景。这个案例适合面试，
因为它展示了修复本身也需要可证伪测试。

### 18.5 坑四：动态 FP8 更快但没有省服务预留显存

仅看权重位宽会预期显存大幅下降；实测 vLLM BF16/FP8 峰值分别 21,677/21,683 MiB。
原因是服务总显存还包括 KV cache、workspace、CUDA graph 和按比例预留，动态量化也
不等价于完整静态 FP8 checkpoint。最终 README 同时报吞吐、质量和显存：FP8 c32
吞吐高约 7.4%，但质量 7/12 对 BF16 8/12，不能只发布“最快”一列。

### 18.6 坑五：SGLang 并发 4 尾延迟离群

SGLang c4 的 TTFT p95 为 412.955 ms，明显差于相邻并发。没有删行或无限重跑直到
数字好看，而是保存 raw report 并明确它可能来自 warmup/JIT/scheduler 抖动。严谨做法
是增加重复轮次和置信区间；当前发布只称“固定短测中的观测值”。

### 18.7 坑六：streaming delta 不是 token

有些 OpenAI-compatible 后端在 SSE chunk 中不返回 usage。若把每个 delta、字符或词
直接叫 token，吞吐不可比较。实现保留 client-observed output units；只有响应明确提供
completion tokens 时才报告 token/s。发布矩阵固定后端和请求合同，仍注明 tokenizer
与 usage 来源。

### 18.8 坑七：模型切换不能只改一个配置指针

若先写 active model 再启动候选，失败会让流量指向不可用服务。2.0 的顺序是注册候选
revision → 启动 → health → quality contract → 原子写 activation state，并保留 previous
以便 rollback。当前只有单实例治理，缺少双实例 connection draining，所以不能写
“零停机热更新”。

### 18.9 从失败到完整 15 格矩阵

三次失败都保留 log；修复后也没有减少请求数。每个 profile 的验收步骤是：

1. 记录模型/后端/torch/CUDA/Ninja 精确版本；
2. health 返回预期模型 ID；
3. 跑 12-case JSON/引用/时间戳/拒答 fixture；
4. 依次跑并发 1/4/8/16/32，每格 64 请求；
5. 保存 SSE raw metrics 和 NVML peak；
6. 终止进程并确认没有 CUDA PID；
7. 最后一次性生成 15 行矩阵，禁止人工挑行。

### 18.10 可直接用于面试的 STAR 案例

| Situation | Task | Action | Result / Learning |
| --- | --- | --- | --- |
| SGLang 三次启动失败 | 保持原 benchmark 范围完成对照 | 顺着首个 root cause 查 Ninja、PATH、symlink，逐次加回归测试 | 固定环境后 320/320 SGLang 请求成功；环境是服务合同的一部分 |
| FP8 未省总显存 | 解释量化是否值得上线 | 同时测吞吐、quality、NVML，不用位宽推断 | c32 更快但少过 1 case；不默认替换 BF16 |
| c4 出现 413 ms TTFT p95 | 决定是否重跑/删除 | 保留 raw 行，限定结论并提出重复实验 | 结果可审计；避免 cherry-pick |
| 后端 usage 缺失 | 统一吞吐语义 | token 指标 fail closed，另报 output units | 不再把字符或 SSE chunk 冒充 token |

### 18.11 亲手复现练习

- 写一个 20 行假 SSE server，故意把 JSON 跨两个 chunk，验证 parser；
- 手算四个请求的 TTFT p50/p95，并说明插值法；
- 删除 `source_license`，确认 feedback export fail closed；
- 把 venv Python 改成 symlink，复现 PATH 回归测试；
- 在固定 c8 下只改变 `max_tokens`，观察 RPS 为什么不可直接比较；
- 设计双实例 blue/green 切换状态机，补上 drain 与 rollback 条件。

完成这些练习后，你应该能解释服务框架背后的调度、测量与生命周期，而不是只会说
“我用过 vLLM”。
