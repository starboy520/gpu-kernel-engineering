# Tensor Core GEMM Roadmap（G0～G5）

> 当前 FP32 CUDA Core GEMM 保持冻结。Tensor Core 使用独立 FP16 输入/FP32 累加项目和结果表。
> 完整学习实操位于同级 `cuda_study`：`docs/courses/gemm/G0_G5_A100_TensorCore_GEMM完整学习实操.md`。

## 当前范围

```text
G0 数值语义与 MMA Shape       [教材就绪，开始学习]
G1 WMMA 单 Warp 单 Tile        [框架就绪，Kernel 待实现]
G2 单条 mma.sync               [未开始]
G3 ldmatrix 映射实验           [未开始]
G4 最小手写 MMA GEMM           [未开始]
G5 Warp Tile                   [未开始]
G6～G8                         [冻结，G5 后决定]
```

## 阶段表

| 阶段 | 当日学习 | 当日实践 | 验收证据 |
| --- | --- | --- | --- |
| G0 | FP16×FP16、FP32 accumulate、`m16n8k16`、Warp collective | Shape 图、FLOP/寄存器账本、三组手算 | 闭卷自检 |
| G1 | WMMA fragment 生命周期 | 单 Warp `16×16×16` | ones/identity/random、memcheck、`HMMA` |
| G2 | PTX MMA lane/register mapping | 单条 `mma.sync.m16n8k16`、raw accumulator dump | logical rebuild、`HMMA`、无映射黑箱 |
| G3 | Shared→register 分发 | `ldmatrix.x1/x2/x4/trans` Lane dump | mapping 表、sanitizer、`LDSM` |
| G4 | `ldmatrix + mma.sync` K 循环 | 单 Warp `16×8×K`，K=16/32/48 | correctness、sanitizer、`LDSM/HMMA` |
| G5 | 多 MMA accumulator 与 fragment 复用 | `16×32`、`32×32` Warp Tile | registers、spill、occupancy、SASS、候选 Tile |

## 固定数值合同

```text
GPU：A100 sm_80
A/B：FP16 storage
Multiply：FP16
Accumulator：FP32
Output：FP32
Reference：实际 FP16 输入转 FP32 后 CPU 累加
```

当前 pedantic FP32 GEMM 不作为数值相同的实现参与排名；FP16 Tensor Core 结果独立成表。

## 推荐项目边界

G1 项目已经创建：

```text
projects/gemm_tensorcore/
├── CMakeLists.txt
├── README.md
├── include/gemm_tensorcore/
├── kernels/
├── runner/
├── tests/
├── scripts/
└── results/
```

建议逐阶段保留独立 Kernel：

```text
wmma_single.cu
mma_single.cu
ldmatrix_dump.cu
mma_gemm_warp.cu
mma_warp_tile.cu
```

不覆盖上一阶段，便于对比 correctness、资源与 SASS。

## 每阶段固定流程

```text
概念与官方 ISA
→ 纸面 Shape/ownership
→ 最小输入 expected
→ 学习者实现核心 Kernel
→ CPU reference 对拍
→ sanitizer
→ SASS
→ 资源与 smoke benchmark
→ worklog
→ 下一阶段唯一变量
```

## 暂不进入 G6～G8

以下内容在 G5 后再决定：

- Multi-Warp Block Tile；
- `cp.async` multi-stage；
- Shared Memory swizzle；
- M/N/K tail 与 fallback；
- FP16 cuBLAS Tensor Core canonical benchmark。

若主要目标是 Tensor Core Attention，G5 后可以转入 Attention M4/M5/M6，再按需要返回 GEMM G6～G8。

## G0 启动条件

开始 G0 前只需确认：

- [ ] FP32 GEMM v1 不再改动；
- [ ] 接受 Tensor Core 使用独立 dtype/reference/结果表；
- [ ] 第一条路线固定 FP16 input + FP32 accumulator；
- [ ] G0 只做手算和设计，不创建空 Kernel；
- [x] `gemm_tensorcore` 脚手架已创建，核心 Kernel 保留 TODO。

一句话记忆：

> G0～G5 每天产出一个可验证实验，不先学完整套理论，也不提前进入生产级流水线。
