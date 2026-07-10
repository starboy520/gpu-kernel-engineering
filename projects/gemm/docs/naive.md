# Naive FP32 GEMM

## 实现

这是整个优化阶梯的起点：一个 CUDA thread 负责一个输出元素。thread 沿 K 维读取 `A[row * k + i]` 和 `B[i * n + column]`，在寄存器中完成 FP32 累加，最后写回 `C[row * n + column]`。

当前 block 配置为 `dim3(32, 32)`。grid 使用向上取整，kernel 写回前检查 `row < m && column < n`，因此可以处理矩形矩阵和非 block 对齐的尺寸。

这一版没有显式的数据复用。A 和 B 会被不同 thread 重复读取，后续需要用 profiler 确认实际访存表现，再和 shared memory 版本比较。

## 构建与验证

以下命令都在仓库根目录执行。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build -j
ctest --test-dir build --output-on-failure
```

跑完整正确性矩阵，并检查实际执行路径和结果：

```bash
tail -n +2 projects/gemm/tests/correctness_cases.csv | while IFS=, read -r _ m n k _; do
    output=$(./build/projects/gemm/gemm_runner --kernel naive --m "$m" --n "$n" --k "$k" --mode validate) || exit 1
    printf '%s\n' "$output"
    grep -q 'status=PASS' <<< "$output" || exit 1
    grep -q 'path=naive' <<< "$output" || exit 1
done
```

做一次小规模 benchmark 冒烟测试。runner 会先对拍，再开始计时：

```bash
./build/projects/gemm/gemm_runner --kernel naive --m 128 --n 192 --k 64 --mode benchmark --warmup 2 --iterations 5
```

用一个矩形、非对齐尺寸检查越界访问：

```bash
compute-sanitizer --tool memcheck --error-exitcode=99 ./build/projects/gemm/gemm_runner --kernel naive --m 130 --n 127 --k 65 --mode validate
```

## 当前验收结果

- 正确性矩阵：11/11 通过
- `compute-sanitizer --tool memcheck`：0 error
- 矩形和非对齐尺寸均通过 CPU 对拍

这里暂不发布正式性能数据。完整优化阶梯完成后，会在同一实验环境下重新测试，并补充 Nsight Compute 和 SASS 证据。