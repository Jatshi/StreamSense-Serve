# StreamSense-Serve 3.0 development line

The v3 line turns the verified v2 backend matrix into an SLO-controlled serving
system. A request is admitted only when a healthy backend can meet deadline,
quality, capability, and privacy constraints.

## Implemented locally

- typed request/backend/SLO policy contracts;
- deterministic admission and backend selection using latency, quality, cost, queue,
  privacy, health, and capability signals;
- explicit rejection reasons rather than silent fallback;
- strict allow-list redaction for OpenTelemetry GenAI attributes;
- deterministic timeout/OOM/process-exit/malformed-response/exporter-failure schedules;
- four-objective latency/cost/quality/safety Pareto frontier calculation;
- pinned dual-GPU experiment matrix in `configs/v3_dual_gpu_matrix.json`.

## Required dual-GPU acceptance

Run vLLM DP=2 and TP=2 separately on the same model/revision, request trace, and load
shape. Then run SGLang TP=2. Every point must include warmup, queue distribution,
P50/P95/P99, throughput, failure recovery, quality, safety, and cost assumptions.

vLLM data parallelism replicates model weights across DP ranks and needs load balancing;
it is distinct from tensor parallelism, which partitions model computation. The official
current reference is https://docs.vllm.ai/en/latest/serving/data_parallel_deployment.html.

No dual-GPU number is currently claimed. The local tests validate routing and fault
contracts only.
