# GEMM Results

Ad-hoc raw run CSV files belong under `raw/` and are ignored by default. Only reviewed canonical release datasets are durable, authoritative data; add and commit those datasets explicitly (using `git add -f` when they remain under `raw/`) after confirming that they include the environment and commit metadata needed to reproduce and interpret every measurement:

| Field | Required content |
| --- | --- |
| `hardware` | GPU model and relevant device configuration |
| `toolchain` | CUDA, compiler, and build-tool versions |
| `git_commit` | Full Git commit SHA for the measured code |
| `shape` | GEMM dimensions and batch information, if applicable |
| `kernel` | Kernel or baseline identifier |
| `selected_path` | Runtime-selected implementation path |
| `latency` | Measured latency with units |
| `gflops` | Computed GFLOPS |
| `correctness` | Validation result and tolerance |
| `timestamp` | Measurement timestamp with timezone |

Generated Markdown reports belong under `generated/` and are ignored by default.