# Tensor Core GEMM（G0～G5）

独立于 FP32 CUDA Core GEMM 的 A100 Tensor Core 学习项目。

## 代码归属

- 核心 WMMA、`mma.sync`、`ldmatrix` 和 Warp Tile Kernel 由学习者亲自实现；
- AI coding agent 提供 CMake、FP16-quantized CPU reference、runner、测试、sanitizer 和 SASS 框架；
- 当前 `wmma_single.cu` 只有 G1 TODO，不包含核心答案。

## 数值合同

```text
GPU：A100 sm_80
A/B：FP16 row-major
Multiply：FP16
Accumulator：FP32
Output：FP32 row-major
Reference：读取实际 FP16 值后转 FP32 累加
```

该结果不与 pedantic FP32 GEMM 混在同一性能表中。

## 当前进度

```text
G0 数值语义与 MMA Shape       [教材与 Roadmap 就绪]
G1 WMMA 单 Warp 单 Tile        [脚手架完成，Kernel 待实现]
G2 单条 mma.sync               [未开始]
G3 ldmatrix 映射实验           [未开始]
G4 最小手写 MMA GEMM           [未开始]
G5 Warp Tile                   [未开始]
```

作品集 Roadmap：[../gemm/TENSOR_CORE_ROADMAP.md](../gemm/TENSOR_CORE_ROADMAP.md)。完整教材位于同级 `cuda_study` 仓库的 `docs/courses/gemm/G0_G5_A100_TensorCore_GEMM完整学习实操.md`。

## G1 固定实验

```text
一个 Warp / Block
一个 16×16×16 WMMA Tile
A/B FP16 row-major
C FP32 row-major
不处理 tail
不循环多个 Tile
```

输入模式：

- `ones`：所有输出应为 16；
- `identity`：A 为 Identity，输出应等于量化后的 B；
- `random`：固定 seed，与 CPU reference 对拍。

## 构建目标

```text
gemm_tensorcore_common
gemm_tensorcore_wmma_single
gemm_tensorcore_runner
gemm_tensorcore_reference_tests
```

项目通过 VS Code CMake Tools 构建。

## 当前运行方式

Reference tests 应当通过：

```text
gemm_tensorcore_reference_tests
```

G1 Kernel 尚未实现，因此以下命令当前应稳定 FAIL，输出仍为 NaN：

```bash
build/projects/gemm_tensorcore/gemm_tensorcore_runner --input ones
projects/gemm_tensorcore/scripts/test_wmma_single.sh
```

完成 Kernel 后依次运行：

```bash
projects/gemm_tensorcore/scripts/test_wmma_single.sh
projects/gemm_tensorcore/scripts/sanitize.sh
projects/gemm_tensorcore/scripts/extract_sass.sh
```

G1 完成门槛：

- ones/identity/random 全部 PASS；
- 四类 Compute Sanitizer 通过；
- SASS 出现 `HMMA`；
- 学习者能解释 fragment、layout 和 leading dimension；
- 不把 WMMA 直接当作最终手写 MMA 结论。
