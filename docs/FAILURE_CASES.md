# Failure cases and mitigations

| Failure | Observed behavior | Mitigation / current status |
|---|---|---|
| Severe additive noise | The single-sample ASR WER rose from 0 at 10 dB to 0.091 at 0 dB and 0.318 at -5 dB | Preserve confidence and timestamps; review low-confidence speech |
| VLM server absent or response invalid | Optional escalation can fail independently of lightweight evidence | Persist original evidence as `human_review`; never label the request as successfully escalated |
| CUDA 13 wheel on CUDA 12.8 driver | vLLM 0.25.1 pulled PyTorch cu130 and failed the GPU self-test | Isolate vLLM 0.15.1 with PyTorch cu128; verify a real CUDA matrix operation before model load |
| SGLang 0.5.11+ on a 560-series driver | SGLang's default line moved to Torch 2.11/CUDA 13; installation succeeded but `torch.cuda.is_available()` was false | Use an isolated SGLang 0.5.10 + Torch 2.9.1/cu128 environment and require a CUDA self-test marker before the benchmark watcher proceeds |
| Large FlashInfer wheel times out through AutoDL's direct GitHub route | The 1.2 GiB release asset repeatedly stalls or retries while the paid GPU remains active | Fetch it once with resumable segmented transfer, then set `STREAMSENSE_FLASHINFER_JIT_CACHE_SPEC` to the verified local wheel; retain the exact version pin |
| Model-server launcher is killed but its child keeps the CUDA context | A wrapper process that waits on `subprocess.run` can die without forwarding the signal to vLLM/SGLang | Replace the launcher process with the backend via `os.execv`; after termination, fail the matrix if any CUDA PID remains |
| Hugging Face Xet authentication on a public model | Initial download path returned an authorization error | Use the normal HTTP download path with `HF_HUB_DISABLE_XET=1`; pin the resolved revision |
| Missing evidence | Retrieval has no event supporting a question | Return an explicit abstention with no invented citation |
| Unsupported or corrupt media | Decoder/analyzer cannot safely process input | Reject at the API boundary with 415/422 and delete the incomplete upload |
| Uploaded evidence outside the media root | A stored path could otherwise expose arbitrary files | Evidence endpoint resolves paths and serves only files under the configured media root |

The engineering fixtures do not cover camera shake, glare, multi-speaker diarization, multilingual
speech, adversarial prompts, or long-running RTSP/WebRTC reconnection. These remain explicit v0.1
gaps rather than implied capabilities.
