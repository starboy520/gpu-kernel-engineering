# Attention Prefill M1 实验结果

M1 的 Br1 baseline 与 Br4 Query-tiled 实现使用统一 evidence runner 和统一 CUDA Event 协议。四类本地产物严格分离：

- `raw/`：benchmark CSV，canonical 性能数字的唯一来源；
- `generated/`：由 CSV 自动生成的 Markdown，不手工维护；
- `profiles/`：Nsight Compute 报告，不进入延迟表；
- `sass/`：本地完整 SASS；

正式协议：

```text
GPU：NVIDIA A100 80GB
implementation：br1、br4
N：128、256、512、1024、2048
D：64、128
causal：0、1
dtype：FP32
batch / heads：1 / 1
layout：row-major
input：random，seed 1234
warmup：10
iterations：50
repeats：3
统计量：latency 中位数，同时保存 min/max/spread
计时：CUDA Event
reference：CPU double
总行数：40
```

正式运行要求同时满足以下 provenance guard：

- 工作树干净，`AP_M1_ALLOW_DIRTY` 不能绕过 canonical 检查；
- canonical 调用不接受任何 runner 位置参数，只使用仓库标准绝对路径 `build/projects/attention_prefill/attention_prefill_evidence_runner`，而且该路径必须是普通文件、不能是 symlink；即使参数是指向标准 runner 的 alias symlink 也会拒绝。smoke 才允许传入自定义 runner；
- CMake 与 `source_fingerprint.py` 使用同一确定性 filesystem manifest。固定文件为 root/common/FlashAttention/Attention Prefill 的 `CMakeLists.txt` 与 `common/validation.cpp`；递归目录为 common include/src、FlashAttention include/kernels、Attention Prefill include/kernels/evidence；只纳入 `CMakeLists.txt`、`.cmake`、`.cu`、`.cpp`、`.hpp`、`.h`，排除 build/results/docs/tests/`__pycache__`。因此未 tracked 的 evidence 源文件同样参与；
- CMake 对排序后的相对路径逐文件计算 SHA-256，再对 `relative-path:file-sha256` manifest 计算最终 SHA-256。`CONFIGURE_DEPENDS` glob 监视相关目录集合，新增或删除匹配文件会触发重新配置；
- runner 编译时嵌入 `source_sha256`、`build_contract=release-sm80-<payload-hash-prefix>` 和完整 `build_contract_payload_sha256`。CMake 以固定字段顺序生成 schema 1 payload，字段为 `schema_version`、`build_type`、`cuda_architectures`、`cuda_compiler_id`、`cuda_compiler_version`、`cuda_compiler_realpath`、`cmake_cuda_flags`、`cmake_cuda_flags_release`，以及 evidence support、Attention Prefill kernel、FlashAttention kernels 三个 target 的 compile options；完整 payload 的原始 UTF-8 字节参与 SHA-256；
- CMake 生成 `build/projects/attention_prefill/attention_prefill_build_attestation.txt`，文件同时保存完整 payload、完整 payload SHA-256 和短 hash build contract。Canonical benchmark 要求它是非 symlink 普通文件、mtime 不早于 runner，并由独立 Python parser 验证 schema 1、`build_type=Release`、arch 精确为 `80`、compiler ID 为 `NVIDIA`、compiler version/realpath 非空、通用 `CMAKE_CUDA_FLAGS` 为空、Release flags 的规范化 token set 精确为 `-O3` 与 `-DNDEBUG`、三个 target options 精确匹配项目固定定义；随后重算完整 payload SHA-256，并与 runner metadata 的完整 hash 和 contract 短 hash 对拍。Compiler version/path 只记录并要求非空，不固定具体版本；benchmark 仍逐 repeat 对拍 source/build/device metadata；
- CSV 同时记录 runner binary SHA-256 与 source SHA-256。mtime freshness 可保留为辅助检查，但不能替代 fingerprint；
- `device_index`、`gpu_uuid`、`gpu_name`、`sm` 和 CUDA driver version 全部来自 runner 对实际 CUDA device 的查询。脚本不再用 `CUDA_VISIBLE_DEVICES` 推导 `nvidia-smi` index，也不以 `nvidia-smi` 的第一张卡代替真实执行设备；
- canonical 要求 runner 实际报告的 `gpu_name` token 同时包含 `A100` 和 `80GB`，且 `sm=8.0`；A100 40GB 不满足正式协议。

修改 shape、causal 或 timing 参数后，脚本自动写入 `smoke.csv` 与 `smoke.md`；工作树不干净时还必须显式设置 `AP_M1_ALLOW_DIRTY=1`。Smoke 文件不能覆盖 canonical 文件，自定义路径也必须保留 `smoke.csv` / `smoke.md` 文件名。Canonical CSV 和 Markdown 必须精确写入上述 official 路径。

CSV header 和数据行统一由 Python `csv.writer` 以固定 `lineterminator="\n"` 写入，因此 compiler 或 smoke build preset 中的逗号、引号和换行不会破坏列边界。Renderer 先以 `newline=''` 和 strict `DictReader` 解析，再用固定 field order、dialect 与 line terminator 规范序列化；原始文本与规范字节不一致即拒绝，包括裸 quote、冗余 quote 和 CRLF。该规则对 canonical 与 smoke 相同。每个 repeat 的 `max_abs` / `max_rel` 都会参与聚合，CSV 保存各自最大值，而不是最后一次运行的值。

渲染表使用以下定义：

- `speedup = Br1 latency / Br4 latency`；
- `delta = 100 × (speedup - 1)`；
- 任一路 `spread > 3%` 为 `inconclusive`；
- 否则 `speedup >= 1.05` 为 `benefit`，`speedup <= 0.95` 为 `regression`，其余为 `near-parity`；
- `requested reduction = Br1 requested_kv_elements / Br4 requested_kv_elements`。

Renderer 对 canonical 文件验证完整 40 行固定矩阵、A100 80GB / SM 8.0、`source_sha256`、`build_contract`、`device_index` 和全部正式协议字段，也验证 `path`、CTA 与 requested K/V 公式。所有 latency、spread 和误差字段必须有限；latency 必须为正，误差必须非负，且 `spread = 100 × (max - min) / median`（允许六位小数舍入误差）。Markdown table cell 中的管道符和换行会安全转义。

## 威胁模型

这些 guard 的目标是阻止日常实验中的 stale runner、错误 build 目录、错误配置、遗漏的未 tracked source、标准路径 symlink 和非规范 CSV 意外混入 canonical 结果。Source fingerprint、runner SHA-256、build contract 与 attestation 是一致性检查和可复现性元数据，不是 cryptographic attestation，也没有独立信任根。

它们不防御能够修改本地脚本、CMake、attestation、runner binary 或同时精确伪造全部 metadata 的恶意本地攻击者；smoke 自定义 runner 也不提供这一保证。若攻击者能替换验证者和被验证对象，本项目不声称检测该篡改。

> `requested_kv_elements` 是实现映射导出的理论请求元素数，不等于实际 DRAM bytes。Nsight Compute 的 duration 不进入 canonical 延迟表。

## Nsight Compute 与 SASS 复现

单点 profile 使用统一 evidence runner；脚本固定 validate 单 launch，并以 exact demangled regex 只采集对应 kernel：

```text
projects/attention_prefill/scripts/profile_m1.sh br1 256 64 0
projects/attention_prefill/scripts/profile_m1.sh br4 256 64 0
```

每次生成 `.ncu-rep`、完整命令/runner metadata `.txt`，以及 normalized `-metrics.csv`。只有工作树干净、非 symlink 标准 runner、embedded/current source fingerprint 一致、canonical build attestation 验证通过、默认 official profiles 目录时，normalized identity 才记录 `evidence_kind=canonical`、Git commit 并使用正式 stem。任何 dirty 工作树、`AP_M1_EVIDENCE_RUNNER` override 或 `AP_M1_ALLOW_PROFILE_SMOKE=1` 都必须显式设置路径名包含 `smoke` 的 `AP_M1_PROFILE_OUTPUT_DIR`，并使用 `-smoke` stem 与 `evidence_kind=smoke`，不能覆盖 official profiles。Canonical SASS 同样要求干净工作树并记录 Git commit。

正式 ncu 支持范围固定为 `2026.2.*`，其他版本在 canonical 与 smoke 中都 fail closed。脚本在 profile 前分别查询 launch metrics 和指定 profiling metrics；查询失败或缺少任一 metric 时停止。Canonical 只接受 `256x64 causal=0`、`1024x128 causal=0`、`1024x128 causal=1` 的 Br1/Br4 六点，其他 shape 必须隔离到路径名含 `smoke` 的输出目录。

Raw CSV 精确接受 ncu 2026.2 固定身份列 `ID`、`Process ID`、`Process Name`、`Host Name`、`Kernel Name`、`Context`、`Stream`、`Block Size`、`Grid Size`、`Device`、`CC`，以及按名称排序的请求 metrics；缺列、额外列、顺序变化、units/data 行额外字段或缺字段全部拒绝。report、summary、raw CSV 和 normalized CSV 全部先写入同一临时目录，profile、export、normalize 与 identity 合同全部通过后才替换目标文件；失败时旧 artifacts 保持不变。

六个 normalized CSV 传给 `summarize_m1_ncu.py`，默认严格要求完整 canonical matrix，且正式 compact 输出名只能是 `m1-ncu-summary.md`；仅验证一个 pair 时显式使用 `--allow-smoke-pair`，其 `--output` basename 必须包含 `smoke`，不能覆盖正式文件。表中保留 ncu 原始显示单位，ratio 会先按 ncu decimal scales 归一化：时间支持 `ns/us/ms/s`，bytes 支持 `byte/Kbyte/Mbyte/Gbyte`（$K=1000$），percent 只接受 `%`，count metrics 只接受声明的 sector/request/conflict/wavefront/register/thread/block/warp/inst/cycle/wave 等单位；不兼容或非法空单位会拒绝。`ncu duration` 只用于 profiler 内部比较，不等于 CUDA Event 或端到端 wall-clock。

SASS 从统一 runner binary 只 dump 一次，按 `c++filt` 后的完整 basename+参数签名精确分离两个 kernel：

```text
projects/attention_prefill/scripts/extract_m1_sass.sh
```

无参数且使用标准 runner 时，完整 dump 写入 ignored 的 `results/sass/full.sass`，两个完整函数写入 `results/sass/{br1,br4}.sass`，正式 compact 证据写入 `results/evidence/m1-sass.md`。任何 binary 参数或 runner override 都是 smoke，必须设置路径名含 `smoke` 的 `AP_M1_SASS_OUTPUT_DIR`，compact 文件固定为 `m1-sass-smoke.md`，不能覆盖正式证据。full dump、两个函数和 compact evidence 全部先写临时目录，canonical source/build/device attestation、完整签名与 ISA 合同通过后才替换旧文件。M1 合同要求两条路径 `FFMA>0`、`HMMA=0`、`LDGSTS=0`；`LDL/STL` 不强制为 0，但会产生醒目的 spill warning。静态 opcode 数量不是 runtime 执行次数。
