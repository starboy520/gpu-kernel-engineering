# FlashAttention 实验结果

四类证据严格分离：

- `raw/`：canonical benchmark CSV，正常 CUDA Event 墙钟的唯一数字来源；
- `generated/`：由 CSV 自动渲染的展示表，不手工维护；
- `profiles/`：本地 ncu report 与文本摘要，不进入 Git；
- `sass/`：本地完整 SASS，不进入 Git；
- `evidence/`：可提交的小型 SASS 静态计数与关键片段。

正式协议：

```text
GPU：NVIDIA A100 80GB PCIe
dtype：FP32
batch / head：1 / 1
N：512、768、1024
D：64、128
causal：0、1
kernel：naive、tiled、tiled-parallel、tiled-async
input：random，seed 1234
warmup：10
iterations：50
repeats：3
统计量：latency 中位数，同时保存 min/max/spread
计时：CUDA Event
reference：CPU double
```

正式运行要求干净工作树；非正式参数自动写入 smoke 文件。经 review 的 canonical CSV 可以显式加入 Git，临时 raw/generated/profile/SASS 产物保持忽略。
