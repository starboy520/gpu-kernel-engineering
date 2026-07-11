# FP32 GEMM 优化阶梯

**状态：** 正在重写

## 范围

在 NVIDIA A100（`sm_80`）上实现 row-major 的 FP32 GEMM：$C=A\times B$。

当前只做 CUDA Core 版本，不包含 Tensor Core、转置输入和 batched GEMM。

## 实现顺序

1. [Naive](docs/naive.md)
2. [Shared memory tiling](docs/shared-tiled.md)
3. [2D register tiling](docs/register-tiled.md)
4. [`float4` 向量化加载](docs/vectorized.md)
5. [`cp.async` 双缓冲](docs/async-pipeline.md)
6. [cuBLAS pedantic FP32 基线](docs/cublas-baseline.md)

所有版本共用同一套输入、CPU 对拍和计时框架。性能数据会在完整正确性测试和 sanitizer 通过后重新采集，不沿用学习仓库里的旧结果。

## 验收流程

每完成一个 kernel，都按照 [CUDA GEMM Kernel 验收手册](docs/kernel-verification-guide.md) 逐步执行编译、对拍、sanitizer、回归测试和 benchmark。

### 自动化命令

以下命令都从仓库根目录执行。脚本会根据自身位置定位仓库，不依赖当前工作目录；未指定 runner 时默认使用 `build/projects/gemm/gemm_runner`。

```bash
projects/gemm/scripts/validate.sh
projects/gemm/scripts/sanitize.sh quick
projects/gemm/scripts/sanitize.sh full
projects/gemm/scripts/profile.sh vectorized
projects/gemm/scripts/extract_sass.sh vectorized
```

`validate.sh` 先运行完整 CTest，再读取 `projects/gemm/tests/correctness_cases.csv`。`kernel=all` 会展开为 `naive`、`shared`、`register`、`vectorized`、`async-16b` 五个作者实现，不包含 cuBLAS；具名行只运行指定实现。每次 runner 调用都使用 `--mode validate --warmup 1 --iterations 1`，并检查退出码、`status=PASS`，以及具名用例要求的精确 `path=` token。当前 CSV 共执行 59 次 runner 对拍。

`sanitize.sh quick` 执行 7 条 memcheck：五个作者实现各一个代表 shape，并额外覆盖 vectorized 与 async-16b 的 N 非整倍数 fallback。`sanitize.sh full` 包含 quick 的全部命令，再对 shared、register、vectorized、async-16b 执行 racecheck、synccheck、initcheck，共 19 条命令。full 覆盖更完整，但运行时间会明显更长。

`benchmark.sh` 是唯一认可的正式 benchmark 入口，负责按固定协议批量执行 `naive`、`shared`、`register`、`vectorized`、`async-16b`、`cublas-fp32`，对每个 shape 重复多次后只保留中位数延迟，并自动渲染 Markdown 汇总。默认协议如下：

- shape：`512x512x512`、`1024x1024x1024`、`2048x2048x2048`
- warmup：`10`
- iterations：`50`
- repeats：`3`
- seed：固定为 `1234`

官方 benchmark 从仓库根目录执行：

```bash
projects/gemm/scripts/benchmark.sh
```

脚本默认拒绝在 dirty working tree 上执行任何 benchmark；如确实需要覆盖，显式设置 `GEMM_ALLOW_DIRTY=1`：

```bash
GEMM_ALLOW_DIRTY=1 projects/gemm/scripts/benchmark.sh
```

默认 runner 路径为 `build/projects/gemm/gemm_runner`，也可以传入自定义 runner：

```bash
projects/gemm/scripts/benchmark.sh /path/to/gemm_runner
```

烟雾模式通过环境变量覆盖 shape / warmup / iterations / repeats，不改变脚本本身：

```bash
GEMM_SHAPES='512x512x512' \
GEMM_WARMUP=2 \
GEMM_ITERATIONS=3 \
GEMM_REPEATS=1 \
projects/gemm/scripts/benchmark.sh
```

非正式协议默认写入 `projects/gemm/results/raw/smoke.csv` 和 `projects/gemm/results/generated/smoke.md`，不会覆盖 canonical 正式结果。需要自定义路径时可设置 `GEMM_OUTPUT_CSV` 和 `GEMM_OUTPUT_MD`。

正式结果会写入 `projects/gemm/results/raw/a100-fp32.csv`，随后自动生成 `projects/gemm/results/generated/a100-fp32.md`。对外引用性能数字时，以这条脚本产物为准，不手工拼接或摘抄临时日志。

### Profiler 与 SASS 复现

`profile.sh` 默认使用 `2048×2048×2048`、validate 模式、`warmup=0`，并通过 demangled kernel regex 只采集一次目标 kernel launch。完整报告和可读文本分别写入 `projects/gemm/results/profiles/<kernel>-<shape>.ncu-rep` 与同名 `.txt`：

```bash
projects/gemm/scripts/profile.sh shared
projects/gemm/scripts/profile.sh register 2048 2048 2048
projects/gemm/scripts/profile.sh vectorized 2048 2048 2048
projects/gemm/scripts/profile.sh async-16b 2048 2048 2048
```

需要覆盖默认 runner 时设置 `GEMM_RUNNER`：

```bash
GEMM_RUNNER=/path/to/gemm_runner \
	projects/gemm/scripts/profile.sh async-16b 2048 2048 2048
```

`extract_sass.sh` 从完整 `cuobjdump --dump-sass` 输出中切出目标函数，保存完整函数 SASS，并生成可提交的小型 opcode 计数与宽加载/异步指令片段：

```bash
projects/gemm/scripts/extract_sass.sh vectorized
projects/gemm/scripts/extract_sass.sh async-16b
projects/gemm/scripts/extract_sass.sh vectorized /path/to/gemm_runner
```

完整 report、文本和 `.sass` 都是 gitignored 诊断产物；紧凑证据位于 `projects/gemm/results/evidence/`。采集协议、指标解释和 Async 负结果见 [GEMM 性能实验方法](docs/methodology.md)。

两个脚本都接受自定义 runner 路径：

```bash
projects/gemm/scripts/validate.sh /path/to/gemm_runner
projects/gemm/scripts/sanitize.sh quick /path/to/gemm_runner
```

为了在无 GPU 环境中测试 CSV 解析，可用 `GEMM_CASES_FILE` 临时覆盖用例文件；该文件仍必须使用表头 `kernel,m,n,k,expected_path`，且脚本仍会先执行 CTest。正常验收不要设置此变量。

```bash
GEMM_CASES_FILE=/tmp/correctness_cases.csv \
	projects/gemm/scripts/validate.sh /tmp/fake-gemm-runner
```