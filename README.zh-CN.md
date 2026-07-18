# StreamSense-Serve

一个“证据优先”的音视频事件推理服务：先用轻量分析器产生带时间戳的候选事件，再依据风险、
不确定性、跨模态冲突和视觉依赖决定是否升级到本地视觉语言模型。任何非拒答结论都必须关联
可回放的音频区间、转写或视频帧。

[English](README.md)

## 已实现能力

- 16-bit PCM WAV 能量活动检测与 faster-whisper 时间戳转写。
- 视频抽帧、场景变化检测及证据帧持久化。
- 可解释的规则路由、本地 OpenAI 兼容 VLM 升级，以及失败后的人工复核降级。
- SQLite 事件库、证据约束问答和证据不足时的明确拒答。
- FastAPI、Prometheus、可选 OpenTelemetry、Docker、CLI 与可视化时间线。
- RTX 4090 上可复现的 ASR、噪声鲁棒性、路由成本、API 负载和 VLM 延迟原始结果。

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,media]"
make test
make smoke
streamsense serve --host 127.0.0.1 --port 8000
```

若需 GPU ASR，安装 `.[asr]` 并设置 `STREAMSENSE_ASR_MODEL=small`。若需 VLM，另行启动
OpenAI 兼容的本地服务，并设置 `STREAMSENSE_VLM_BASE_URL` 与
`STREAMSENSE_VLM_MODEL`。开发服务默认仅监听本机；对网络开放前必须增加 TLS、认证和限流。

## 如何解读结果

README 中的性能数字均是固定样本和单机环境下的工程检查，并不代表真实业务数据上的总体准确率
或生产容量。模型版本见 `models.lock`，原始结果见 `benchmarks/results/`，复现记录见
`run_manifest.md`。数据边界、模型限制和失败案例分别见 `docs/DATASET_CARD.md`、
`docs/MODEL_CARD.md` 和 `docs/FAILURE_CASES.md`。

在固定的合成视频上，本地 Qwen2.5-VL-3B-Instruct 完成“变化检测→升级→结构化描述→证据落库”
用时 4.168 秒；JSON 语法预热后的流式中位 TTFT 为 54.8 ms，中位总延迟为 313.6 ms，常驻
显存 17,855 MiB。它只是一帧可复现的工程检查，并非视觉准确率或生产容量结论。

## 安全与许可

项目不做人脸识别、身份追踪或意图推断，也不能用于自动化医疗、安全、就业、执法或监控决策。
代码采用 Apache-2.0；第三方模型和数据仍受各自条款约束。当前固定的
Qwen2.5-VL-3B-Instruct 权重采用 Qwen Research License，未经另行许可仅限非商业用途；详见
`THIRD_PARTY_NOTICES.md`。
