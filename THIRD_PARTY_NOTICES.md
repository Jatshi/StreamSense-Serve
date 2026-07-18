# Third-party notices

StreamSense-Serve source code is licensed separately from optional models, datasets, and runtime
dependencies. The repository does not redistribute model weights.

| Component | Upstream terms | Notes |
|---|---|---|
| Qwen/Qwen2.5-VL-3B-Instruct | [Qwen Research License](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/LICENSE) | Non-commercial use unless a separate commercial license is obtained from Alibaba Cloud |
| Systran/faster-whisper-small | [MIT model card](https://huggingface.co/Systran/faster-whisper-small) | Optional ASR weights, downloaded by the runtime |
| faster-whisper | [MIT](https://github.com/SYSTRAN/faster-whisper/blob/master/LICENSE) | Optional ASR runtime |
| vLLM | [Apache-2.0](https://github.com/vllm-project/vllm/blob/main/LICENSE) | Optional OpenAI-compatible model server |
| OpenCV | [Apache-2.0](https://github.com/opencv/opencv/blob/4.x/LICENSE) | Optional video decoder and fixture generator |

Python packages installed from `pyproject.toml` retain their own licenses and copyright notices.
Container builders and downstream distributors should generate and review a complete dependency
bill of materials for the exact resolved environment. The `jfk.wav` engineering sample is fetched
from `whisper.cpp` rather than redistributed; users must confirm the upstream media terms for their
use case.

