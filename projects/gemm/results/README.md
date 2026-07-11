# GEMM 实验结果

正式 benchmark 统一由 `projects/gemm/scripts/benchmark.sh` 产出。不要手工维护 `raw/` 里的 CSV，也不要把终端里零散打印的延迟当作正式结果。

当前约定：

- canonical CSV：`projects/gemm/results/raw/a100-fp32.csv`
- generated Markdown：`projects/gemm/results/generated/a100-fp32.md`
- smoke CSV：`projects/gemm/results/raw/smoke.csv`
- smoke Markdown：`projects/gemm/results/generated/smoke.md`

`benchmark.sh` 会把 runner 的原始重复结果先写到临时文件，按每个 `kernel + shape` 选出中位数延迟后，再生成 canonical CSV；因此 `raw/a100-fp32.csv` 中每行都应该对应一条最终保留的中位数记录，而不是单次试跑。

只要 shape、warmup、iterations 或 repeats 偏离正式协议，且没有显式指定输出路径，脚本就会自动写入 smoke 文件，不会覆盖 canonical CSV。

canonical CSV 的表头固定为：

```text
timestamp,git_commit,gpu,cuda,nvcc,kernel,path,m,n,k,warmup,iterations,latency_ms,gflops,passed,max_abs,max_rel,reference
```

其中：

- `path` 是运行时实际选择的实现路径，例如 `fast-float4` 或 `fallback-register`
- `reference` 是正确性对拍来源，例如 `cpu` 或 `cublas-pedantic-fp32`
- `passed`、`max_abs`、`max_rel` 来自同一次 benchmark 前置校验，便于后续回溯

自动生成的 Markdown 只做展示和阶段对比，不替代原始 CSV。引用数据、复现问题、追查 fallback 时，先看 canonical CSV。