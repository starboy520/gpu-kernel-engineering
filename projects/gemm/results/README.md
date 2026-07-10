# GEMM 实验结果

临时跑出来的 CSV 放在 `raw/`，默认不提交。只有经过复查、能够对应到具体代码版本和实验环境的数据，才作为正式结果加入仓库；如果文件仍放在 `raw/`，需要使用 `git add -f` 显式提交。

正式数据至少包含以下字段：

| 字段 | 内容 |
| --- | --- |
| `hardware` | GPU 型号和相关设备配置 |
| `toolchain` | CUDA、编译器和构建工具版本 |
| `git_commit` | 被测试代码的完整 Git commit SHA |
| `shape` | GEMM 的 M、N、K |
| `kernel` | kernel 或基线名称 |
| `selected_path` | 运行时实际选择的实现路径 |
| `latency` | 延迟及单位 |
| `gflops` | 根据 shape 和延迟计算的 GFLOPS |
| `correctness` | 对拍结果和误差阈值 |
| `timestamp` | 带时区的测试时间 |

自动生成的 Markdown 表格放在 `generated/`，默认不提交。