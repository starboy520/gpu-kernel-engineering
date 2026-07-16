# M2 Warp-per-query 临时执行清单

> 本文件只用于 M2 实操跟踪；M2 全部证据收口后删除。

完整推导位于同级 `cuda_study` 仓库：
`docs/courses/attention/M2_WarpPerQuery_FP32_SIMT_Attention完整学习资料.md`。

## 当前目标

不再拆分调试 Checkpoint。直接完成完整 `warp_per_query_kernel()`，然后使用最终 Attention 输出对拍。

```text
M1 query-tiled                    [冻结]
M2 完整 Kernel                    [完成]
M2 128-case correctness           [完成]
M2 representative sanitizer       [完成]
M1 vs M2 benchmark/ncu/SASS       [开发期完成，正式重采待提交]
```

## Kernel 实现清单

只修改 `kernels/warp_per_query.cu` 中的六组 TODO。

### TODO 1：Q register fragment

- `warp_id = threadIdx.x / 32`；
- `lane_id = threadIdx.x % 32`；
- `global_query = blockIdx.x * 4 + warp_id`；
- 每 Lane 持有 feature `lane + slot*32`，最多四槽；
- 无效 Query Warp 不读取 Q，但不能提前退出。

### TODO 2：K/V cooperative load

- 128 threads 合作加载唯一一份 `K/V[valid_kv,D]`；
- Global stride 使用 `D`；
- Shared stride 使用 `MAX_D=128`；
- Barrier A 在加载之后；Barrier B 在所有 Warp 消费之后。

### TODO 3：Warp QK 和 lane-owned Score

对每个 `local_key`：

1. 32 lanes 分摊 feature dot；
2. Warp shuffle sum reduction；
3. scale 为 `rsqrtf(D)`；
4. causal 使用全局 Query/Key；
5. 将完整 Score 交给 `lane_id==local_key` 保存。

无 feature 的 Lane partial 为 0；无效/masked Key 的 Score 为 `-INFINITY`。

### TODO 4：Warp Online Softmax

- Lane 0～15 持有 Score；
- Warp max 的无效单位元为 `-INFINITY`；
- Warp sum 的无效单位元为 0；
- 每个 Warp 独立维护 `row_m/row_l`；
- all-mask tile：`alpha=1`、weights=0、状态不变；
- 非空 tile：更新 `m_new/alpha/weight/tile_l/row_l`。

### TODO 5：register O_acc

每 Lane 最多四个 `o_frag`：

```text
feature = lane + slot*32
```

每个 Key 的 Weight 从 `src_lane=local_key` 广播。每个 tile 先统一执行：

```text
o_frag *= alpha
```

再累加：

```text
o_frag += weight * V
```

不要在每个 Key 内重复缩放旧 `o_frag`。

### TODO 6：最终写回

仅有效 Query Warp、有效 feature 写：

```text
output[global_query*D + feature] = o_frag[slot] / row_l
```

## 验证顺序

先只跑最小用例：

```bash
build/projects/attention_prefill/attention_prefill_runner \
    --implementation warp-per-query --n 1 --d 1 --causal 0
```

通过后按顺序扩大：

```text
3x2 non-causal
5x2 causal
17x65 non-causal
33x65 causal
full 128 cases
full sanitizer
```

完整 correctness：

```bash
projects/attention_prefill/tests/test_warp_per_query.sh
```

## 完成门槛

- 126 个 shape + 2 个特殊输入通过；
- memcheck/racecheck/synccheck/initcheck 通过；
- 与 M1 使用同一 CUDA Event 协议；
- ncu 比较 registers、Shared Memory、occupancy、eligible warps、long/short scoreboard、barrier；
- SASS 证明 `SHFL`/FP32 `FFMA` 路径并检查 spill；
- 正收益、回退和无结论结果全部保留。

一句话记忆：

> 一个 Warp 拥有一条 Query；Score 沿 Key 分给 Lane，输出沿 Feature 分给 Lane，所有跨 Lane 状态都必须遵守同一个 collective mask。
