# Development environment

## Remote server

- gpu: remote
- provider: AutoDL
- GPU: NVIDIA GeForce RTX 4090 24 GB
- base directory: configured outside Git as `STREAMSENSE_REMOTE_BASE`
- code directory: `${STREAMSENSE_REMOTE_BASE}/repo`
- data directory: `${STREAMSENSE_REMOTE_BASE}/data`
- cache directory: `${STREAMSENSE_REMOTE_BASE}/cache`
- runs directory: `${STREAMSENSE_REMOTE_BASE}/runs`
- code_sync: git
- wandb: false

SSH endpoint and credentials are intentionally kept outside this repository.
