# `float4` Vectorized Load GEMM

## 这一版改了什么

计算部分沿用 2D Register Tiling：一个 block 计算 `64×64` 的 C tile，一个 thread 负责 `8×4` 个输出。变化只发生在 global memory 到 shared memory 的搬运阶段。

Register 版本按单个 `float` 搬运 A/B tile；这一版把连续 4 个 FP32 元素组成一个 `float4`，尝试让编译器生成 128-bit global load。读取到寄存器后，再拆成 4 个标量写入带 padding 的 shared memory。

```text
global memory：float4 load（16 bytes）
register：      value.x / y / z / w
shared memory：4 次 scalar store
```

shared memory 不直接强转成 `float4*`，因为当前二维数组每行带 `+1` padding，行起点不保证 16-byte 对齐。

## Vector 任务映射

参数仍为：

```text
BM=64, BN=64, BK=16
TM=8,  TN=4
block=16×8=128 threads
```

A tile 和 B tile 都有：

$$
64\times16=1024\text{ floats}
$$

换算为：

$$
1024/4=256\text{ 个 float4}
$$

一个 block 有 128 threads，因此每个 thread 分别搬 2 个 A vectors 和 2 个 B vectors。

A vector 的 tile 坐标：

```cpp
int tile_row = vector_index * 4 / BK;
int tile_col = vector_index * 4 % BK;
```

B vector 的 tile 坐标：

```cpp
int tile_row = vector_index * 4 / BN;
int tile_col = vector_index * 4 % BN;
```

`tile_col` 始终是 4 的倍数。global memory 中读取一个 `float4` 后，分别写入 `tile_col+0..3`。

## 为什么需要 fast path 条件

`cudaMalloc` 返回的 allocation 起点通常满足足够高的对齐，但 kernel 接口将来可能接收带 offset 的指针，因此 launcher 仍显式检查 A/B 指针是否满足 `alignof(float4)`。

对于 row-major A `[M,K]`，每行 stride 由 K 决定。要保证每一行起点都保持 16-byte 对齐，需要：

$$
K\bmod4=0
$$

对于 B `[K,N]`，每行 stride 由 N 决定，因此需要：

$$
N\bmod4=0
$$

内部参数还需要：

```cpp
static_assert(BK % 4 == 0);
static_assert(BN % 4 == 0);
```

当前 fast path 条件为：

```text
A 基地址 16B 对齐
B 基地址 16B 对齐
K % 4 == 0
N % 4 == 0
```

M 不影响每一行的 stride，所以 M 不被 4 整除时仍然可以走 `fast-float4`，只需在不存在的行上向 shared memory 写零。

## 为什么选择 kernel 级 fallback

如果 fast path 条件不满足，launcher 会调用已经验收过的 Register Tiling：

```text
满足所有条件  → path=fast-float4
不满足条件    → path=fallback-register
```

也可以在同一个 kernel 内对每个地址动态选择 `float4` 或 scalar，但非对齐行会大量退化，还会产生分支发散。首版使用 kernel 级 fallback，让 fast path 的约束、执行路径和性能更容易验证。

## 路径与正确性测试

| Shape | 预期路径 | 结果 |
| --- | --- | --- |
| `128×128×128` | `fast-float4` | PASS |
| `130×128×128` | `fast-float4` | PASS |
| `128×130×128` | `fallback-register` | PASS |
| `128×128×130` | `fallback-register` | PASS |

这组用例分别验证：

- 完全对齐的 fast path。
- M 尾块不影响 vector alignment。
- N 不被 4 整除时 fallback。
- K 不被 4 整除时 fallback。

CTest 当前为 14/14 PASS。

## Sanitizer

```text
fast-path memcheck             0 errors
N fallback memcheck           0 errors
K fallback memcheck           0 errors
fast-path racecheck           0 hazards
fast-path synccheck           0 errors
fast-path initcheck           0 errors
```

这些检查同时覆盖了 16-byte load 的地址合法性、shared memory 同步和 fallback 安全性。

## 编译资源

当前 `ptxas -v` 结果：

```text
83 registers/thread
8512 bytes shared memory/block
0 bytes spill stores
0 bytes spill loads
128 threads/block
```

Register 版本此前使用 72 registers/thread。清理 fast kernel 中不可达的 scalar tail 后，Vectorized 最终版本使用 83 registers/thread。这个差异说明编译器生成代码发生了明显变化，但不能仅凭寄存器数量直接解释性能，需要结合 ncu 的 theoretical/achieved occupancy 和 stall 指标。

## SASS 对照

从最终二进制提取函数后统计 global load：

```text
Register：   14 LDG.E
Vectorized： 14 LDG.E.128
```

最终 Vectorized kernel 的 global load 全部显示为 `LDG.E.128`，而 Register kernel 对应位置为标量 `LDG.E`。这证明源码中的 `float4` 主加载确实被编译成了 128-bit global load。

这里统计的是静态指令数量，不是运行时只执行了多少次 load。指令位于循环和分支中，会动态执行多次。

## `2048³` 阶段结果

使用相同的 `warmup=10`、`iterations=50`：

| 版本 | 延迟 | GFLOPS | TFLOPS | 相对 cuBLAS |
| --- | ---: | ---: | ---: | ---: |
| Register tiled | 2.645443 ms | 6494.138184 | 6.49 | 36.8% |
| Vectorized | 1.336320 ms | 12856.103851 | 12.86 | 72.8% |
| cuBLAS pedantic FP32 | 0.973271 ms | 17651.680482 | 17.65 | 100% |

Vectorized 相对 Register：

$$
\frac{2.645443}{1.336320}\approx1.980
$$

即阶段测试约快 1.98 倍。当前手写版本距离 cuBLAS pedantic FP32 约还有 1.37 倍。

这组结果与 `LDG.E.128` 证据一致，说明向量化加载已经生效。但性能提升不能全部归因于指令宽度：最终版本的寄存器数量也从 Register 的 72 变成 83，occupancy 和调度可能同时发生变化。后续需要用 Nsight Compute 比较 global load 指令、registers、occupancy、memory throughput 和 warp stall。

以上仍是同一次会话中的阶段数据，不是最终简历数字。正式结果会重复多轮取中位数，并记录完整实验环境和 Git commit。
