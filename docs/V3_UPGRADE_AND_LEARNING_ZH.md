# StreamSense-Serve 3.0：SLO 推理系统学习与工程复盘

## 1. 为什么从“能推理”升级到“SLO 系统”

模型服务在单次 Demo 中返回正确答案，并不等于它能上线。真实请求会同时带来截止时间、隐私等级、模态能力和质量要求；后端会排队、超时、OOM 或返回坏 JSON。StreamSense-Serve 3.0 的问题定义是：在请求进入 GPU 前判断哪个健康后端能满足约束；如果没有后端满足，明确拒绝并说明原因，而不是悄悄路由到一个不合格模型。

这使项目覆盖了招聘 JD 常见的推理加速、vLLM、服务治理、可观测性、故障恢复和容量规划，而不需要把方向转向机器人控制。

## 2. 3.0 新增模块

- typed request/backend/SLO policy：截止时间、质量、成本、隐私、能力和健康状态都有显式类型。
- admission control：先判断可行性，再在可行后端中排序。
- SLO-aware routing：综合队列、预测延迟、质量、安全和成本。
- fault schedule：可重复注入 timeout、OOM、process exit、malformed response 和 exporter failure。
- allow-list telemetry：只允许批准的 OpenTelemetry GenAI 属性离开进程。
- 四目标 Pareto frontier：延迟、成本、质量与安全不压成一个武断总分。
- 真实 vLLM smoke 和 100 请求并发压测入口，所有参数与结果写入 manifest。

## 3. Admission control 的实现思路

对请求 `r` 和后端 `b`，先做硬约束过滤：

1. `b` 是否 healthy；
2. 是否支持请求模态和所需工具；
3. 隐私域是否兼容；
4. `queue_wait + predicted_latency` 是否小于 deadline；
5. 预测质量是否高于最低阈值。

任何硬约束失败都返回结构化 reason。通过后才计算软排序分数，例如延迟余量、成本、质量裕量和当前队列。硬约束与软偏好不能混在同一个加权和里，否则很便宜但违反隐私的后端可能被错误选中。

## 4. P50/P95/P99 与吞吐怎样正确测

- warmup 请求不能混入正式分位数；
- 每个请求记录 enqueue、dispatch、first token、last token；
- TTFT 与端到端 latency 分开报告；
- 并发数、输入 token、输出 token 和 sampling 参数必须固定；
- P99 需要足够样本，4 请求 smoke 只能验证链路，不能宣称尾延迟结论；
- throughput 提高不代表单请求体验提高，必须同时看 queue time 和 tail latency。

3.0 的 4 请求 smoke 结果是 4/4 成功、0 error、exact/contains rate 1.0，仅用于证明真实模型和 API 链路工作。随后完成 100 请求、并发 4 的真实 vLLM 压测：100/100 成功，0 error，46.64 requests/s；chat latency P50/P95/P99 为 54.75/68.49/793.54ms，TTFT P50/P95/P99 为 33.48/44.59/771.34ms，峰值显存 21,511MiB。P99 的约 0.79 秒长尾是真实测量，不能用 P95 掩盖。

## 5. vLLM、数据并行和张量并行

张量并行（TP）把一层矩阵计算拆到多卡，适合单卡放不下模型，但每层都产生通信。数据并行（DP）复制模型，每个副本处理不同请求，适合提高吞吐，但每张卡都要放一份权重。两者不能用“都叫并行”混为一谈。

单张 4090 可以验证 vLLM 的 paged KV cache、continuous batching、OpenAI-compatible API、调度与压测代码，但不能制造真实的多卡扩展效率。DeepSpeed ZeRO-3 在单卡上可以验证分片/卸载代码路径，却不会产生并行加速；world size=1 时很多框架会退化成 NO_SHARD。

## 6. 4090 真实验证与版本坑

实际加载 `Qwen/Qwen2.5-VL-3B-Instruct`，模型权重分片约 3.98GB 和 3.53GB。首次下载约 422.7 秒；因此后续运行使用持久化 Hugging Face cache，避免反复烧付费时间。smoke 峰值显存约 21,511 MiB。

最初固定的 vLLM 0.26 与环境中的 Torch 2.11/CUDA 13 组合不可用。最终验证组合是 Torch 2.9.0+cu128、vLLM 0.12.0、NumPy<2、OpenCV 4.11.0.86。这里的经验是：推理框架与 PyTorch/CUDA 的二进制 ABI 必须作为整体锁定，不能只固定 Python 包名。

## 7. 故障注入为什么要确定性

如果测试依赖“恰好发生一次 OOM”，CI 会不稳定且无法复盘。3.0 使用 schedule：第几个请求触发何种故障、持续多久、期望路由器怎样响应都写进配置。这样可以测试：

- timeout 是否触发有界重试；
- OOM 后后端是否熔断并恢复；
- malformed response 是否被 schema gate 拒绝；
- telemetry exporter 故障是否影响主请求；
- 所有后端都不满足时是否明确拒绝。

## 8. 可观测性与隐私

“打更多日志”不是可观测性。请求 trace 至少要关联 admission、queue、backend、TTFT、completion 和 error；但 prompt、音视频内容、用户身份和工具输出不能默认上报。项目采用 allow-list，而不是先全量记录再做黑名单删除，因为新字段加入时黑名单很容易漏掉敏感信息。

## 9. 面试问答

**为什么不能总是 fallback？** fallback 可能违反隐私、能力或 deadline。显式拒绝比返回一个不满足合同的答案更可控。

**怎样估算容量？** 先用固定输入/输出分布测单副本 service time，再结合目标并发和 utilization 估算队列；最终以压测的 P95/P99 和错误率校准，而不是只用平均 tokens/s。

**为什么选择 Pareto frontier？** 成本、延迟、质量和安全没有普适权重。Pareto 集先去掉被全面支配的方案，再由业务策略在剩余点中选择。

**单卡结果能写“分布式推理”吗？** 不能。可以写“实现并验证了 DP/TP 配置与启动契约”，但只有真实多卡运行后才能写吞吐扩展比或通信效率。

## 10. 亲手练习

1. 手写一个只有两个后端的 admission 函数，并为每个拒绝 reason 写测试。
2. 构造 1000 个延迟样本，手算并核对 P95/P99。
3. 将 timeout 注入到固定请求 ID，验证重跑得到相同故障序列。
4. 增加一个敏感 telemetry 字段，证明它不会通过 allow-list。
5. 分别画 latency-cost 和 quality-safety Pareto 图，解释为什么某些点被支配。

掌握这些内容后，应能从代码、指标和系统权衡三个层面讲清楚一个生产推理服务，而不是只会回答“我用 vLLM 启了个接口”。
