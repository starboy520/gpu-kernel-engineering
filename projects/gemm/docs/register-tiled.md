# 2D Register Tiled GEMM

## 为什么继续做 register tiling

Shared 版本已经把 A、B 的 global memory 数据搬到 shared memory，并在 block 内复用。但每个 thread 仍然只计算一个 C 元素：沿 K 维每走一步，都要从 shared memory 读取一个 A 和一个 B，只完成一次 FMA。

Register tiling 让一个 thread 同时计算一小块 C。A、B 从 shared memory 读进寄存器后，会被当前 thread 的多个输出重复使用。

## 当前参数

```text
BM = 64
BN = 64
BK = 16
TM = 8
TN = 4
```

含义：

- 一个 block 计算 C 的 `64×64` tile。
- K 每轮处理 16 个元素。
- 一个 thread 计算 `8×4=32` 个 C 元素。

block 维度由输出 tile 和 thread tile 决定：

$$
\text{blockDim.x}=\frac{BN}{TN}=16
$$

$$
\text{blockDim.y}=\frac{BM}{TM}=8
$$

一个 block 共 128 threads，也就是 4 个 warps。

源码中用 `static_assert` 固定这些内部约束：

```cpp
static_assert(BM % TM == 0);
static_assert(BN % TN == 0);
static_assert((BM / TM) * (BN / TN) <= 1024);
```

输入的 M、N、K 不需要整除这些参数。

## 一个 thread 负责哪些输出

thread 在 block tile 内的起点为：

```text
local_row_base = threadIdx.y × TM
local_col_base = threadIdx.x × TN
```

`reg_acc[i][j]` 对应的 global C 坐标是：

```text
row = blockIdx.y × BM + threadIdx.y × TM + i
col = blockIdx.x × BN + threadIdx.x × TN + j
```

其中：

```text
i = 0..7
j = 0..3
```

写回时对 32 个元素逐个检查 M/N 边界，因此一个 thread 负责的区域可以一部分有效、一部分越界。

## 协作加载

计算映射和搬运映射是分开的。先把 block 内的二维 thread 编号打平：

```cpp
int tid = threadIdx.y * blockDim.x + threadIdx.x;
int stride = blockDim.x * blockDim.y;
```

然后用 grid-stride 方式搬运：

```text
A tile：BM×BK = 64×16 = 1024 个元素
B tile：BK×BN = 16×64 = 1024 个元素
block：128 threads
每个 thread 分别搬 8 个 A 和 8 个 B
```

A/B 越界位置写 `0.0f`，因此支持 M、N、K 尾块。

## thread 内的外积

每个 `kk`：

1. 从 shared memory 读取 8 个 A，放到 `reg_t_a[TM]`。
2. 从 shared memory 读取 4 个 B，放到 `reg_t_b[TN]`。
3. 计算一个 `8×4` 外积，更新 32 个累加器。

也就是读取：

$$
TM+TN=12\text{ 个 float}
$$

完成：

$$
TM\times TN=32\text{ 个 FMA}
$$

相比一 thread 一输出的 Shared 版本，单次 shared load 可以服务更多 FMA。

## 同步与尾块

每轮 K tile 仍然需要两次 block 同步：

```text
协作写入 A/B shared tile
              ↓
       __syncthreads()
              ↓
读取 shared tile，完成 register outer product
              ↓
       __syncthreads()
              ↓
下一轮覆盖 shared tile
```

没有 thread 在 barrier 前提前 return。越界 thread 会参与补零和同步，最后只跳过无效 C 元素的写回。

## Padding 的当前状态

当前 shared memory 定义为：

```cpp
s_a[BM][BK + 1]
s_b[BK][BN + 1]
```

这仍属于待 profile 的参数选择，不能仅凭源码断言已经消除 bank conflict。

- A 的 `+1` 会改变不同 shared 行之间的 bank 映射。
- B 在计算阶段按 `threadIdx.x * TN` 读取，列方向 stride 为 4；仅增加行宽不一定能解决这种访问冲突。

后续需要用 Nsight Compute 的 shared wavefront/bank-conflict 指标比较 padding 前后，再决定最终布局。

## 正确性与安全检查

验收覆盖：

```text
单输出 tile、单 K tile：64×64×16       PASS
矩形尾块、多轮 K：      65×127×33      PASS
完整正确性矩阵：        11/11          PASS
CTest：                 7/7            PASS
memcheck：              0 errors
racecheck：             0 hazards
synccheck：             0 errors
initcheck：             0 errors
```

代表性命令：

```bash
./build/projects/gemm/gemm_runner \
  --kernel register \
  --m 65 --n 127 --k 33 \
  --mode validate \
  --warmup 1
```

完整验收方法见 [CUDA GEMM Kernel 验收手册](kernel-verification-guide.md)。

## 编译资源

单独使用 `ptxas -v` 编译当前版本：

```text
72 registers/thread
8512 bytes shared memory/block
0 bytes spill stores
0 bytes spill loads
128 threads/block
```

仅按 A100 每个 SM 的 65536 个 32-bit registers 粗略估算：

$$
\left\lfloor\frac{65536}{72\times128}\right\rfloor=7\text{ blocks/SM}
$$

每个 block 有 4 warps，因此寄存器对应的理论 occupancy 上限约为：

$$
\frac{7\times4}{64}=43.75\%
$$

这只是静态估算，最终以 Nsight Compute 的 theoretical/achieved occupancy 为准。

## 为什么 `384³` 反而更慢

相同计时参数下：

| 版本 | 延迟 | GFLOPS |
| --- | ---: | ---: |
| Naive | 0.048579 ms | 2331.20 |
| Shared | 0.036024 ms | 3143.60 |
| Register | 0.149791 ms | 756.03 |

Register 每个 block 计算 `64×64`，所以 `384×384` 只产生：

$$
\left(\frac{384}{64}\right)^2=36\text{ blocks}
$$

A100 有 108 个 SM，这个 grid 连一轮 SM 都填不满。这个结果主要说明测试 shape 不适合当前 kernel，不能直接说明 register tiling 无效。

## 扩大 grid 后的阶段结果

使用 `1024×1024×64`：

| 版本 | 延迟 | GFLOPS |
| --- | ---: | ---: |
| Naive | 0.042312 ms | 3172.12 |
| Shared | 0.032072 ms | 4184.93 |
| Register | 0.030986 ms | 4331.53 |

Register 相对 Shared：

$$
\frac{4331.53}{4184.93}\approx1.035
$$

阶段测试约快 3.5%；相对 Naive 约 1.37 倍。

这组早期结果不是正式性能数据：K 只有 64，Register 只有 256 个 blocks，当时也尚未加入 cuBLAS 大规模 reference。它保留在这里，用来说明测试 shape 会怎样影响结论。

## 接入 cuBLAS 后的 `2048³` 对比

cuBLAS pedantic FP32 reference 接入后，Register 版本通过 `2048³` 大矩阵对拍。使用相同的 `warmup=10` 和 `iterations=50`：

| 版本 | 延迟 | 性能 |
| --- | ---: | ---: |
| Naive | 4.697006 ms | 3.66 TFLOPS |
| Shared tiled | 3.250483 ms | 5.29 TFLOPS |
| Register tiled | 2.645443 ms | 6.49 TFLOPS |
| cuBLAS pedantic FP32 | 0.973271 ms | 17.65 TFLOPS |

Register 相比 Shared：

$$
\frac{6494.14}{5285.33}\approx1.229
$$

即提升约 22.9%。相比 Naive 约为 1.78 倍，达到当前 cuBLAS pedantic FP32 基线的约 36.8%。

这个结果比 `384³` 更适合评价当前 kernel：`2048×2048` 会产生 `32×32=1024` 个 Register blocks，不再存在只有 36 个 blocks、无法填满 108 个 SM 的明显问题。具体性能差距仍需 ncu 验证，暂不把 bank conflict、occupancy 或 global load 指令中的任何一项直接定为主瓶颈。
