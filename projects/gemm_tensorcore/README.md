# Tensor Core GEMM（G0～G5）

独立于 FP32 CUDA Core GEMM 的 A100 Tensor Core 学习项目。

## 代码归属

- 核心 WMMA、`mma.sync`、`ldmatrix` 和 Warp Tile Kernel 由学习者亲自实现；
- AI coding agent 提供 CMake、FP16-quantized CPU reference、runner、测试、sanitizer 和 SASS 框架；
- `wmma_single.cu` 与 `wmma_direct.cu` 的核心 WMMA 逻辑由学习者完成。

## 数值合同

```text
GPU：A100 sm_80
A：FP16 row-major
B：G1 row-major；G1.5 Direct col-major
Multiply：FP16
Accumulator：FP32
Output：FP32 row-major
Reference：读取实际 FP16 值后转 FP32 累加
```

该结果不与 pedantic FP32 GEMM 混在同一性能表中。

## 当前进度

```text
G0 数值语义与 MMA Shape       [教材与 Roadmap 就绪]
G1 WMMA 单 Warp 单 Tile        [完成：correctness/sanitizer/HMMA]
G1.5 Direct WMMA Tiled GEMM    [完成：padding/col-major B/K loop]
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

## G1.5 Direct WMMA

G1.5 在进入手写 PTX 前，用高层 WMMA 黑盒复习完整 GEMM tiling：

```text
一个 Warp / Block
一个 Warp 计算一个 16×16 C Tile
A FP16 row-major
B FP16 col-major
C FP32 row-major
K 每轮前进 16，Accumulator 跨轮累加
Host 将任意正整数 M/N/K padding 到 16 的倍数
不使用 Shared Memory、异步搬运或 tail 分支
```

Direct runner 对拍整个 padded C，因此同时验证有效输出和补零区域。当前 8 组 shape × 3 种输入共 24 次执行全部通过；`17×19×23` 与 `31×33×47` 覆盖非整除 padding。

## 构建目标

```text
gemm_tensorcore_common
gemm_tensorcore_wmma_single
gemm_tensorcore_runner
gemm_tensorcore_wmma_direct
gemm_tensorcore_direct_runner
gemm_tensorcore_reference_tests
```

项目通过 VS Code CMake Tools 构建。

## 当前运行方式

Reference tests 应当通过：

```text
gemm_tensorcore_reference_tests
```

G1 验收：

```bash
projects/gemm_tensorcore/scripts/test_wmma_single.sh
projects/gemm_tensorcore/scripts/sanitize.sh
projects/gemm_tensorcore/scripts/extract_sass.sh
```

G1.5 Direct 验收：

```bash
projects/gemm_tensorcore/scripts/test_wmma_direct.sh
projects/gemm_tensorcore/scripts/sanitize_wmma_direct.sh
projects/gemm_tensorcore/scripts/extract_wmma_direct_sass.sh
```

G1 完成门槛：

- ones/identity/random 全部 PASS；
- 四类 Compute Sanitizer 通过；
- SASS 出现 `HMMA`；
- 学习者能解释 fragment、layout 和 leading dimension；
- 不把 WMMA 直接当作最终手写 MMA 结论。

G1.5 只验证 Direct `Global → fragment → HMMA → Global` 路径，不发布性能结论。Shared Memory 双缓冲与 `cuda::memcpy_async` 若继续实验，必须作为独立版本与 Direct baseline 对照。
