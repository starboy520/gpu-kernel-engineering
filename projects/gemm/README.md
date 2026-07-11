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
```

`validate.sh` 先运行完整 CTest，再读取 `projects/gemm/tests/correctness_cases.csv`。`kernel=all` 会展开为 `naive`、`shared`、`register`、`vectorized`、`async-16b` 五个作者实现，不包含 cuBLAS；具名行只运行指定实现。每次 runner 调用都使用 `--mode validate --warmup 1 --iterations 1`，并检查退出码、`status=PASS`，以及具名用例要求的精确 `path=` token。当前 CSV 共执行 59 次 runner 对拍。

`sanitize.sh quick` 执行 7 条 memcheck：五个作者实现各一个代表 shape，并额外覆盖 vectorized 与 async-16b 的 N 非整倍数 fallback。`sanitize.sh full` 包含 quick 的全部命令，再对 shared、register、vectorized、async-16b 执行 racecheck、synccheck、initcheck，共 19 条命令。full 覆盖更完整，但运行时间会明显更长。

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