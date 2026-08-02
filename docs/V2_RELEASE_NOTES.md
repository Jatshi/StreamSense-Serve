# StreamSense-Serve 2.0：新增内容、实测矩阵与工程边界

发布日期：2026-08-02

版本：`v2.0.0`
验证硬件：单张 NVIDIA RTX 4090 24 GiB

## 1. 为什么服务层值得单独做一个项目

模型训练完成不等于系统可用。真实服务需要决定何时调用昂贵 VLM、如何流式返回、怎样
量化 TTFT/TPOT、如何在模型切换失败时回滚，以及用户反馈能否合法进入下一轮训练。
StreamSense-Serve 2.0 把这些“最后一公里”问题做成可测试、可观测、可恢复的服务层。

## 2. 2.0 全量新增内容

### 2.1 统一 backend profile

- 用版本化 `EngineConfig` 描述 vLLM、SGLang 和通用 OpenAI-compatible 后端；
- profile 固定模型 revision、dtype/quantization、context、显存比例、timeout、health path；
- `extra_args` 采用 allowlist，避免配置注入任意启动参数；
- launcher 把命令、环境和后端日志都写入运行目录。

### 2.2 流式推理与正确的性能语义

- 新增 SSE streaming client；
- 同时采集 TTFT、TPOT、端到端 latency、request throughput、服务端 completion-token
  throughput、错误率和 NVML 显存；
- 如果响应没有 usage，只报告 output units，不把字符/增量冒充 token；
- 并发固定为 1/4/8/16/32，每格 64 请求，失败请求不会从分母中删除。

### 2.3 自适应路由

规则路由结合风险、不确定性、跨模态冲突、视觉依赖和固定 seed exploration。返回
`reasons` 使每次升级可审计。20-case fixture 保留全部 10 个 oracle-positive，只升级
11/20，相对 always-escalate 减少 45% VLM 调用；这是手工 fixture 的行为检查，不是
生产节省率。

### 2.4 EvidenceAgent 兼容层

- 新增 `/v2/evidence-agent` 请求/响应映射；
- 保持 answered / needs_clarification / abstained 三态；
- 服务层校验 citation ID 与 evidence store，而不是相信模型自由文本；
- 现有 `/v1` API 保持兼容。

### 2.5 反馈数据飞轮

- `/v2/feedback` 需要独立 bearer token；
- 内容哈希去重；
- `consent_for_training=true` 与 `source_license` 双闸门；
- 原子导出 raw audit、SFT、DPO、EvidenceAgent bridge 四类文件；
- manifest 保存输入/输出数量和 SHA-256，篡改可检测。

### 2.6 模型注册、切换和回滚

- manifest 固定 revision 和 validated 状态；
- 激活前先启动候选、检查真实 health、执行合同 gate；
- activation state 原子写入并保留上一版本；
- 失败时回滚；当前是单实例生命周期治理，不声称双实例零停机切流。

### 2.7 安全、可观测性和交付

- admin、feedback、inference 使用不同 token；
- 默认绑定 localhost，SQLite 参数化查询，上传大小与媒体类型受限；
- Prometheus metrics、可选 OpenTelemetry、Docker 和 CI；
- 新增 AutoDL bootstrap/preflight/run、15 格 benchmark、质量 fixture、学习手册与 GIF。

## 3. RTX 4090 完整控制变量矩阵

固定 Qwen2.5-VL-3B-Instruct revision
`66285546d2b821cf421d4f5eb2576359d3770cd3`、8,192 context、64-token 输出上限。
15 格共 960/960 成功、零错误。

| Profile | Quality | Peak VRAM | RPS @ c1 | RPS @ c32 | TTFT p50 @ c32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| vLLM 0.15.1 BF16 | 8/12 | 21,677 MiB | 24.828 | 176.806 | 124.098 ms |
| vLLM 0.15.1 dynamic FP8 | 7/12 | 21,683 MiB | 26.291 | 189.843 | 123.866 ms |
| SGLang 0.5.10 BF16 | 7/12 | 22,623 MiB | 24.294 | 144.422 | 131.241 ms |

FP8 在此短输出 workload 的并发 32 最快，但没有降低服务预留显存，质量 fixture 还少
通过一项。因此不能写“FP8 显存减半”或“零精度损失”。SGLang 并发 4 的 TTFT p95
`412.955 ms` 离群点也保留，没有删除不利结果。

## 4. 最有价值的失败：SGLang 连续三次启动失败

1. `sglang[all]` 没有安装 JIT 实际调用的外部 Ninja；
2. 安装 Ninja 后，直接调用 venv Python 并不会自动把 venv `bin` 放进子进程 PATH；
3. 首次 launcher 修复用了 `Path.resolve()`，把 venv Python 符号链接解析回 base Python，
   又丢失正确 PATH。

最终 bootstrap 固定 `ninja==1.13.0`，launcher 使用未 resolve 的解释器父目录置于 PATH
首位，并加入回归测试。成功后仍跑原定 5 档并发 × 64 请求，没有缩减矩阵。完整定位
方法和面试表达见 [学习手册](streamsense_v2_from_scratch_zh.md#18-20-真实实施日志与踩坑复盘)。

## 5. 代码地图

| 路径 | 2.0 职责 |
| --- | --- |
| `src/streamsense/backends.py` | backend profile 合同与校验 |
| `src/streamsense/backend_launcher.py` | 进程、环境、health 与清理 |
| `src/streamsense/evidence_agent.py` | 三态和引用兼容层 |
| `src/streamsense/feedback.py` | consent/license/去重/导出 |
| `src/streamsense/model_registry.py` | revision guard、激活、回滚 |
| `scripts/vlm_stream_benchmark.py` | SSE 指标与并发压测 |
| `docs/benchmark_matrix_4090.json` | 15 行机器可读发布事实 |

## 6. 面试时最准确的一句话

> 我构建了统一 vLLM/SGLang 的可验证多模态服务层，在单张 4090 上固定模型 revision
> 完成 BF16/动态 FP8、并发 1–32 的 960 请求矩阵，同时实现证据约束、自适应路由、
> consent/license 数据飞轮和 revision-guarded 回滚；结论限定在固定 workload，不冒充
> 生产容量或多 GPU 推理。
