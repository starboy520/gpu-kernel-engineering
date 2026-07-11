# cuBLAS FP32 基线

这个基线有两个用途：一是作为可计时的 vendor baseline，二是在矩阵较大、CPU 对拍成本过高时生成验证结果。它不是作者实现的 CUDA kernel，列表中的 `author_kernel` 标记为 `false`。

## 行主序映射

runner 的输入是 row-major：$A$ 为 $m\times k$，$B$ 为 $k\times n$，目标是 $C=A B$。cuBLAS 默认按 column-major 解释内存。同一段 row-major 数据换一个视角，可以分别看成 column-major 的 $A^T$、$B^T$ 和 $C^T$，所以实际计算的是

$$
C^T = B^T A^T.
$$

因此 `cublasSgemm` 的参数顺序是 B 在前、A 在后，维度传 `(n, m, k)`，leading dimensions 分别为 `(n, k, n)`。这里不需要真实转置，也没有额外的转置 kernel 或临时矩阵。

每次调用都会把传入的 CUDA stream 设置到复用的 thread-local cuBLAS handle，并显式设置 `CUBLAS_PEDANTIC_MATH`。这样 FP32 基线不会走 TF32 快速路径。`alpha=1`、`beta=0`，调用保持异步；同步由 runner 的验证阶段负责。

## 验证分工

工作量 $m n k < 100,000,000$ 时，expected output 仍由 CPU 三重循环生成，使用 `atol=1e-3`、`rtol=1e-3`。这条路径会独立检查 cuBLAS 本身，小矩形测试包括 `17x19x23` 和 `31x7x13`。

工作量达到或超过阈值时，runner 先在 device 上用 pedantic FP32 cuBLAS 生成 expected output，同步并复制回 host，再运行被测 kernel。large reference 使用固定 `atol=1e-3`、`rtol=2e-3`；现有 `passes()` 的聚合 OR 语义不变。这组固定阈值相对正确实现已观测到的约 `1e-5` 量级 `max_abs` 仍有充足余量，但不会让明显的全局偏差仅靠绝对误差分支通过。结果行末尾会显示 `reference=cpu` 或 `reference=cublas-pedantic-fp32`，便于区分 expected output 的来源。

cuBLAS handle 的首次使用和 large reference 生成都发生在验证阶段。device allocation、输入复制、reference 生成、同步和回拷均不进入 benchmark event 区间；计时只包围 warmup 之后的迭代 launch。handle 会跨调用复用，不会在每个 timed iteration 中创建。

## 命令

从仓库根目录执行：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_COMPILE_WARNING_AS_ERROR=ON
cmake --build build -j
ctest --test-dir build --output-on-failure

./build/projects/gemm/gemm_runner --list
./build/projects/gemm/gemm_runner --kernel cublas-fp32 --m 17 --n 19 --k 23 --mode validate
./build/projects/gemm/gemm_runner --kernel cublas-fp32 --m 512 --n 512 --k 512 --mode benchmark --warmup 5 --iterations 20
./build/projects/gemm/gemm_runner --kernel register --m 512 --n 512 --k 512 --mode validate
./build/projects/gemm/gemm_runner --kernel register --m 2048 --n 2048 --k 2048 --mode validate

compute-sanitizer --tool memcheck ./build/projects/gemm/gemm_runner --kernel cublas-fp32 --m 17 --n 19 --k 23 --mode validate
compute-sanitizer --tool memcheck ./build/projects/gemm/gemm_runner --kernel register --m 512 --n 512 --k 512 --mode validate
```

## `2048³` 阶段结果

下面四个版本使用相同 shape、`warmup=10` 和 `iterations=50`，在同一次实验会话中完成：

| 版本 | 延迟 | 性能 | 相对 cuBLAS |
| --- | ---: | ---: | ---: |
| Naive | 4.697006 ms | 3.66 TFLOPS | 20.7% |
| Shared tiled | 3.250483 ms | 5.29 TFLOPS | 29.9% |
| Register tiled | 2.645443 ms | 6.49 TFLOPS | 36.8% |
| cuBLAS pedantic FP32 | 0.973271 ms | 17.65 TFLOPS | 100% |

当前优化阶梯的阶段性收益：

```text
Shared / Naive      ≈ 1.45×
Register / Shared   ≈ 1.23×
Register / Naive    ≈ 1.78×
cuBLAS / Register   ≈ 2.72×
```

这组数据说明 register reuse 在能充分产生 blocks、K 维也足够长时确实有效，但当前手写版本距离 cuBLAS 仍有明显差距。后续需要结合 vectorized load、异步流水和 ncu 指标定位差距，不能只根据 GFLOPS 断言具体瓶颈。

这仍是单次阶段结果，不是最终简历数字。正式结果会重复多轮取中位数，并记录 GPU 型号、频率状态、CUDA 版本、编译参数和 Git commit。

小尺寸 cuBLAS 已由 CPU reference 独立验证。大尺寸路径为了控制验证时间，会信任同一个 vendor implementation 生成 expected output；它能有效检查手写 kernel，但不是第二套完全独立的数值实现。对 `cublas-fp32` 本身进行大矩阵验证时，expected 和 actual 来自同一实现，属于恒等检查；真正有意义的是手写 kernel 与 cuBLAS expected 的比较。