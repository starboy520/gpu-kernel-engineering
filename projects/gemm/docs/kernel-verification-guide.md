# CUDA GEMM Kernel 验收手册

这份文档记录每写完一个 GEMM kernel 后的验收顺序。目标不是一次跑完一堆命令，而是逐层缩小问题范围：前一步通过，再进入下一步。

以下命令默认在仓库根目录执行。需要从任意子目录返回根目录时，可以运行：

```bash
cd "$(git rev-parse --show-toplevel)"
```

把命令里的 `<kernel>` 替换为实际名称，例如：

```text
naive
shared
register
vectorized
async
```

---

## 0. 验收前先检查代码

在运行前先确认：

- 编辑器中的文件已经保存。
- launcher 返回了正确的路径名称。
- launcher 使用 runner 传入的 CUDA stream。
- launcher 内没有 `cudaDeviceSynchronize()`。
- kernel launch 后检查了 launch error。
- 新 kernel 已加入 CMake 和 registry。

launcher 应只负责配置和发射 kernel：

```cpp
kernel<<<grid, block, 0, stream>>>(...);
CUDA_CHECK(cudaGetLastError());
return {"<kernel>", false};
```

同步和计时由 runner 统一完成。如果在 launcher 内同步，会破坏连续 launch 和 CUDA Event 计时。

---

## 1. 编译

```bash
cmake --build build -j
```

### 这一步检查什么

- CUDA/C++ 语法是否正确。
- launcher 声明和定义是否一致。
- registry 中的函数指针能否链接。
- CMake 是否包含新源文件。
- 是否出现编译警告、资源错误或 spill 提示。

### 通过标准

结尾出现：

```text
[100%] Built target gemm_runner
```

同时不应出现：

```text
error
warning
spill stores
spill loads
```

如果修改了 CMake 配置而普通构建没有重新生成，可以重新配置：

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build -j
```

---

## 2. 先跑最简单的完整 tile

以 tile 大小为 32 的 kernel 为例：

```bash
./build/projects/gemm/gemm_runner \
  --kernel <kernel> \
  --m 32 --n 32 --k 32 \
  --mode validate \
  --warmup 1
```

### 为什么先测这个 shape

它没有边界干扰：

- M、N、K 都正好是一个 tile。
- grid 通常只有一个 block。
- K 循环只有一轮。
- 不涉及尾块补零。
- 不涉及下一轮覆盖 shared memory。

如果它失败，优先检查：

1. thread 到 C 元素的映射。
2. A/B 的 global memory 下标。
3. shared memory 的写入和读取下标。
4. 内积循环。
5. C 的写回下标。

### 通过标准

```text
kernel=<kernel> path=<kernel> shape=32x32x32 status=PASS ...
```

注意：`validate` 模式不计时，所以：

```text
latency_ms=0.000000 gflops=0.000000
```

是正常结果。

---

## 3. 测试 M、N、K 尾块

```bash
./build/projects/gemm/gemm_runner \
  --kernel <kernel> \
  --m 33 --n 65 --k 17 \
  --mode validate \
  --warmup 1
```

这个 shape 同时检查：

| 维度 | 检查内容 |
| --- | --- |
| M=33 | 最后一个 block 只有部分行有效 |
| N=65 | 最后一个 block 只有部分列有效 |
| K=17 | 最后 15 个 tile 位置必须补零 |

如果完整 tile 通过、这个 shape 失败，优先检查：

- global load 的边界判断。
- 越界位置是否明确写 `0.0f`。
- C 写回是否检查 M/N 边界。
- K 尾块是否仍执行了错误的 global read。

---

## 4. 测试多轮 K tile

```bash
./build/projects/gemm/gemm_runner \
  --kernel <kernel> \
  --m 65 --n 127 --k 65 \
  --mode validate \
  --warmup 1
```

当 tile 大小是 32 时：

$$
\left\lceil\frac{65}{32}\right\rceil=3
$$

三轮分别处理：

```text
K = 0..31
K = 32..63
K = 64，其余位置补零
```

这一步主要检查：

- 多轮累加是否正确。
- 当前 tile 读完前，下一轮是否提前覆盖 shared memory。
- 最后一轮 K 尾块是否补零。
- M/N/K 尾块能否同时工作。

对于普通 shared-memory tiling，每轮通常需要：

```text
写 shared tile
    ↓
__syncthreads()
    ↓
读取 shared tile 并计算
    ↓
__syncthreads()
    ↓
下一轮覆盖 shared tile
```

---

## 5. 理解正确性输出

示例：

```text
kernel=shared path=shared shape=65x127x65 status=PASS \
max_abs=0.000001 max_rel=0.000502 latency_ms=0.000000 gflops=0.000000
```

| 字段 | 含义 |
| --- | --- |
| `kernel` | runner 选择的 kernel |
| `path` | launcher 实际采用的执行路径 |
| `shape` | M×N×K |
| `status` | 与 CPU reference 对拍是否通过 |
| `max_abs` | 全部元素中的最大绝对误差 |
| `max_rel` | 全部元素中的最大相对误差 |
| `latency_ms` | benchmark 模式下的平均延迟 |
| `gflops` | 根据 shape 和延迟计算的吞吐 |

当前小规模验证使用：

```text
atol = 1e-3
rtol = 1e-3
```

验证器采用保守规则：全部元素满足绝对误差，或者全部元素满足相对误差，即可通过。接近 0 的 reference 值容易放大相对误差，因此不能只看 `max_rel`。

---

## 6. memcheck：检查非法显存访问

```bash
compute-sanitizer \
  --tool memcheck \
  --error-exitcode=99 \
  ./build/projects/gemm/gemm_runner \
  --kernel <kernel> \
  --m 65 --n 127 --k 65 \
  --mode validate \
  --warmup 1
```

### 主要检查

- global memory 越界读写。
- 地址未对齐。
- 非法地址访问。
- CUDA API 和 kernel 执行错误。

### 为什么加 `--error-exitcode=99`

发现错误时让程序返回 99，便于脚本和 CI 判断失败。只搜索日志不够可靠。

### 通过标准

```text
status=PASS
========= ERROR SUMMARY: 0 errors
```

程序对拍通过但 sanitizer 报错，仍然不能验收。

---

## 7. racecheck：检查 shared memory 竞态

适用于使用 shared memory 的 kernel：

```bash
compute-sanitizer \
  --tool racecheck \
  --error-exitcode=99 \
  ./build/projects/gemm/gemm_runner \
  --kernel <kernel> \
  --m 65 --n 127 --k 65 \
  --mode validate \
  --warmup 1
```

### 常见竞态

| 类型 | 含义 |
| --- | --- |
| RAW | 写入尚未完成，其他 thread 已经读取 |
| WAR | 其他 thread 仍在读取，当前 thread 已覆盖写入 |
| WAW | 多个 thread 无序写入同一地址 |

多轮 shared tile 重点检查 WAR：某些 warp 还在读取当前 tile，另一些 warp 已经写入下一轮。

### 通过标准

```text
========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)
```

只测试 `K<tile_size` 不足以验证第二个 barrier，因为没有下一轮覆盖。racecheck 应使用多轮 K tile。

---

## 8. synccheck：检查 barrier 用法

```bash
compute-sanitizer \
  --tool synccheck \
  --error-exitcode=99 \
  ./build/projects/gemm/gemm_runner \
  --kernel <kernel> \
  --m 65 --n 127 --k 65 \
  --mode validate \
  --warmup 1
```

### 主要检查

- 只有部分 thread 到达 `__syncthreads()`。
- 分支导致 barrier divergence。
- warp-level barrier 的 mask 不合法。
- barrier 使用方式不正确。

危险写法：

```cpp
if (row < m && col < n) {
    __syncthreads();
}
```

边界 block 中只有部分 thread 进入分支，可能造成 barrier divergence。正确做法是让越界 thread 写零或跳过计算，但仍参加 block barrier。

### 通过标准

```text
========= ERROR SUMMARY: 0 errors
```

---

## 9. initcheck：检查未初始化 global memory

```bash
compute-sanitizer \
  --tool initcheck \
  --error-exitcode=99 \
  ./build/projects/gemm/gemm_runner \
  --kernel <kernel> \
  --m 65 --n 127 --k 65 \
  --mode validate \
  --warmup 1
```

`initcheck` 主要检查未初始化的 global memory 读取，不能代替 shared memory 的代码审查和 racecheck。

### 通过标准

```text
========= ERROR SUMMARY: 0 errors
```

---

## 10. CTest：确认旧功能没有被改坏

```bash
ctest --test-dir build --output-on-failure
```

CTest 负责固定的回归测试，例如：

- CPU reference 和误差计算。
- runner 参数与 registry。
- Naive 的矩形对拍。
- Shared 的矩形和 K 尾块对拍。

### 通过标准

```text
100% tests passed, 0 tests failed
```

单独验证新 kernel 通过，不代表旧 kernel 没有被破坏，因此每个阶段都要跑 CTest。

---

## 11. 跑完整 shape 矩阵

第一次验收新 kernel 时，建议逐行看输出：

```bash
tail -n +2 projects/gemm/tests/correctness_cases.csv |
while IFS=, read -r _ m n k _; do
    output=$(
        ./build/projects/gemm/gemm_runner \
          --kernel <kernel> \
          --m "$m" --n "$n" --k "$k" \
          --mode validate \
          --warmup 1
    ) || {
        printf '%s\n' "$output"
        exit 1
    }

    printf '%s\n' "$output"

    [[ "$output" == *"status=PASS"* &&
       "$output" == *"path=<kernel>"* ]] || exit 1
done
```

这 11 个 shape 覆盖：

- 极小矩阵。
- 非方阵。
- warp/tile 边界。
- M/N/K 尾块。
- 多轮 K tile。

对于有 fallback 的版本，不能统一检查 `path=<kernel>`；需要分别验证 fast path 和 fallback path 的名称。

---

## 12. benchmark 前先确认比较公平

同一轮比较必须保持一致：

- 相同 M、N、K。
- 相同输入 seed。
- 相同 warmup 次数。
- 相同 iterations。
- 相同编译配置。
- 都先通过正确性验证。

当前 cuBLAS 大规模 reference 尚未接入，CPU reference 的工作量限制为：

$$
M\times N\times K < 100,000,000
$$

因此阶段性 smoke benchmark 可以使用 `384³`：

$$
384^3=56,623,104
$$

### 基线

```bash
./build/projects/gemm/gemm_runner \
  --kernel naive \
  --m 384 --n 384 --k 384 \
  --mode benchmark \
  --warmup 10 \
  --iterations 50
```

### 新版本

```bash
./build/projects/gemm/gemm_runner \
  --kernel <kernel> \
  --m 384 --n 384 --k 384 \
  --mode benchmark \
  --warmup 10 \
  --iterations 50
```

### 计算加速比

使用延迟：

$$
\text{Speedup}=\frac{T_{baseline}}{T_{new}}
$$

或者使用吞吐：

$$
\text{Speedup}=\frac{P_{new}}{P_{baseline}}
$$

两种算法的结果应基本一致。

GEMM 的计算量约为：

$$
2MNK
$$

因此：

$$
\text{GFLOPS}=\frac{2MNK}{\text{latency(ms)}\times10^6}
$$

---

## 13. 不要把一次 smoke benchmark 当最终数据

阶段测试只能说明当前环境下的方向，不能直接写进简历。正式结果至少还需要：

- 更大的代表性 shape。
- cuBLAS pedantic FP32 公平基线。
- 多轮重复并取中位数。
- GPU 型号、CUDA、编译参数和 Git commit。
- 确认没有其他 GPU workload。
- Nsight Compute 和 SASS 证据。

阶段文档应该写：

> 在当前 shape 和统一计时设置下，新版本比基线快约 X 倍；具体瓶颈尚未 profile。

不要在没有 profiler 证据时直接写：

> 已经从 memory-bound 变成 compute-bound。

---

## 14. 出错时如何缩小范围

| 现象 | 优先检查 |
| --- | --- |
| 编译失败 | 声明/定义、CMake、registry、模板参数 |
| 完整 tile FAIL | 核心索引、thread 映射、内积、写回 |
| 只有尾块 FAIL | 越界 load、补零、边界 store |
| 只有多轮 K FAIL | 第二个 barrier、tile 覆盖、累加初始化 |
| 对拍 PASS，memcheck FAIL | 越界或未对齐访问 |
| 对拍偶尔 FAIL，racecheck 报错 | shared memory 竞态 |
| synccheck FAIL | barrier 放在分支中或参与线程不一致 |
| path 名称错误 | launcher 返回值或 fallback 逻辑 |
| 性能异常但正确 | block 数、occupancy、寄存器、同步、cache、计时口径 |

一次只验证一个假设，不要同时改多个地方。

---

## 15. 每个 kernel 的完成清单

```text
[ ] Release 构建通过，无警告和 spill 提示
[ ] 完整 tile 对拍通过
[ ] M/N/K 尾块对拍通过
[ ] 多轮 K tile 对拍通过
[ ] 完整 shape 矩阵通过
[ ] memcheck 通过
[ ] racecheck 通过（使用 shared memory 时）
[ ] synccheck 通过（使用 barrier 时）
[ ] initcheck 通过
[ ] CTest 全部通过
[ ] 与上一版本做相同参数的 smoke benchmark
[ ] 计算并复核 speedup
[ ] 中文实验记录包含命令、结果、失败过程和限制
[ ] 提交前检查 git diff，只提交当前阶段文件
```

一句话记忆：

> 先证明算对，再证明没有越界和竞态，最后才讨论为什么快。
