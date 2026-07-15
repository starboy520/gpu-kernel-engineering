#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${AP_M1_REPO_ROOT:-$(cd -- "$script_dir/../../.." && pwd)}"
repo_root="$(cd -- "$repo_root" && pwd)"
source "$repo_root/common/scripts/common.sh"
runner="${1:-$repo_root/build/projects/attention_prefill/attention_prefill_evidence_runner}"
canonical_runner="$repo_root/build/projects/attention_prefill/attention_prefill_evidence_runner"
build_attestation="$repo_root/build/projects/attention_prefill/attention_prefill_build_attestation.txt"
renderer="$script_dir/render_m1_results.py"
csv_appender="$script_dir/append_m1_csv.py"
fingerprint_script="$script_dir/source_fingerprint.py"
attestation_validator="$script_dir/validate_build_attestation.py"
official_csv="$repo_root/projects/attention_prefill/results/raw/a100-fp32-m1.csv"
official_md="$repo_root/projects/attention_prefill/results/generated/a100-fp32-m1.md"
output_csv="${AP_M1_OUTPUT_CSV:-$official_csv}"
output_md="${AP_M1_OUTPUT_MD:-$official_md}"

csv_columns=(
    timestamp git_commit runner_sha256 source_sha256 build_contract
    build_contract_payload_sha256 device_index
    gpu gpu_uuid sm driver cuda nvcc build_preset dtype batch heads layout
    implementation path n d causal
    input_pattern seed warmup iterations repeats latency_ms latency_min_ms
    latency_max_ms spread_pct passed max_abs max_rel workspace_bytes reference
    timing cta_count requested_kv_elements
)
default_shapes=(128x64 256x64 512x64 1024x64 2048x64 128x128 256x128 512x128 1024x128 2048x128)
default_causal=(0 1)
implementations=(br1 br4)
default_warmup=10
default_iterations=50
default_repeats=3
seed=1234

usage() {
    printf '用法: %s [smoke-runner]\n' "${0##*/}" >&2
    printf '覆盖变量: AP_M1_SHAPES AP_M1_CAUSAL AP_M1_WARMUP AP_M1_ITERATIONS AP_M1_REPEATS AP_M1_OUTPUT_CSV AP_M1_OUTPUT_MD AP_M1_BUILD_PRESET AP_M1_ALLOW_DIRTY\n' >&2
}

die() { gpu_die benchmark_m1 "$@"; }
require_command() { gpu_require_command benchmark_m1 "$1"; }

normalize_path() {
    python3 - "$1" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve(strict=False))
PY
}

parse_shape() {
    [[ $1 =~ ^([1-9][0-9]*)x([1-9][0-9]*)$ ]] || die "shape 必须为 NxD: $1"
    shape_n="${BASH_REMATCH[1]}"
    shape_d="${BASH_REMATCH[2]}"
}

extract_field() {
    local text="$1"
    local name="$2"
    local token
    local -a tokens
    read -r -a tokens <<< "$text"
    for token in "${tokens[@]}"; do
        if [[ $token == "$name="* ]]; then
            printf '%s' "${token#*=}"
            return 0
        fi
    done
    return 1
}

require_field() {
    local text="$1"
    local name="$2"
    local value
    value="$(extract_field "$text" "$name")" || die "runner 输出缺少字段: $name"
    [[ -n $value ]] || die "runner 输出字段为空: $name"
    printf '%s' "$value"
}

[[ $# -le 1 ]] || { usage; exit 2; }
[[ -f $renderer ]] || die "renderer 不存在: $renderer"
[[ -f $csv_appender ]] || die "CSV appender 不存在: $csv_appender"
[[ -f $fingerprint_script ]] || die "fingerprint script 不存在: $fingerprint_script"
[[ -f $attestation_validator ]] || die "attestation validator 不存在: $attestation_validator"
for command_name in git python3 nvcc sha256sum; do
    require_command "$command_name"
done

read -r -a shapes <<< "${AP_M1_SHAPES:-${default_shapes[*]}}"
read -r -a causal_modes <<< "${AP_M1_CAUSAL:-${default_causal[*]}}"
warmup="${AP_M1_WARMUP:-$default_warmup}"
iterations="${AP_M1_ITERATIONS:-$default_iterations}"
repeats="${AP_M1_REPEATS:-$default_repeats}"
build_preset="${AP_M1_BUILD_PRESET:-release-sm80}"

(( ${#shapes[@]} > 0 )) || die 'AP_M1_SHAPES 至少需要一个 shape'
(( ${#causal_modes[@]} > 0 )) || die 'AP_M1_CAUSAL 至少需要一个 causal mode'
gpu_nonnegative_integer "$warmup" || die 'AP_M1_WARMUP 必须为非负整数'
gpu_positive_integer "$iterations" || die 'AP_M1_ITERATIONS 必须为正整数'
gpu_positive_integer "$repeats" || die 'AP_M1_REPEATS 必须为正整数'
for shape in "${shapes[@]}"; do
    parse_shape "$shape"
done
for causal in "${causal_modes[@]}"; do
    [[ $causal == 0 || $causal == 1 ]] || die 'AP_M1_CAUSAL 只能包含 0/1'
done

official=1
[[ $warmup == "$default_warmup" && $iterations == "$default_iterations" && $repeats == "$default_repeats" ]] || official=0
[[ "${shapes[*]}" == "${default_shapes[*]}" && "${causal_modes[*]}" == "${default_causal[*]}" ]] || official=0

if (( official )); then
    (( $# == 0 )) || die 'canonical benchmark 不接受 runner 位置参数；只允许无参数标准 runner'
    [[ -f $canonical_runner && ! -L $canonical_runner ]] || \
        die 'canonical 标准 runner 必须为普通文件且不能是 symlink'
fi

[[ -x $runner ]] || die "runner 不存在或不可执行: $runner"

official_csv="$(normalize_path "$official_csv")"
official_md="$(normalize_path "$official_md")"
canonical_runner="$(normalize_path "$canonical_runner")"
runner="$(normalize_path "$runner")"
output_csv="$(normalize_path "$output_csv")"
output_md="$(normalize_path "$output_md")"
[[ $output_csv != "$output_md" ]] || die 'AP_M1_OUTPUT_CSV 与 AP_M1_OUTPUT_MD 不能相同'

if (( official )); then
    [[ $output_csv == "$official_csv" && $output_md == "$official_md" ]] || \
        die 'canonical 输出路径必须使用 official paths'
    [[ $build_preset == release-sm80 ]] || \
        die 'canonical benchmark 的 AP_M1_BUILD_PRESET 必须为 release-sm80'
    [[ $runner == "$canonical_runner" ]] || \
        die 'canonical benchmark 只允许标准 runner 绝对路径'
else
    if [[ -z ${AP_M1_OUTPUT_CSV+x} ]]; then
        output_csv="$repo_root/projects/attention_prefill/results/raw/smoke.csv"
    fi
    if [[ -z ${AP_M1_OUTPUT_MD+x} ]]; then
        output_md="$repo_root/projects/attention_prefill/results/generated/smoke.md"
    fi
    output_csv="$(normalize_path "$output_csv")"
    output_md="$(normalize_path "$output_md")"
    [[ $output_csv != "$official_csv" && $output_csv != "$official_md" && \
       $output_md != "$official_csv" && $output_md != "$official_md" ]] || \
        die 'smoke 不得覆盖任一 canonical 文件'
    [[ ${output_csv##*/} == smoke.csv ]] || die '非正式参数的 AP_M1_OUTPUT_CSV 必须命名为 smoke.csv'
    [[ ${output_md##*/} == smoke.md ]] || die '非正式参数的 AP_M1_OUTPUT_MD 必须命名为 smoke.md'
fi

if [[ -n $(git -C "$repo_root" status --short) ]]; then
    if (( official )); then
        die 'canonical benchmark 要求干净工作树，AP_M1_ALLOW_DIRTY 不能覆盖此要求'
    fi
    [[ ${AP_M1_ALLOW_DIRTY:-0} == 1 ]] || \
        die 'smoke benchmark 的工作树不干净；设置 AP_M1_ALLOW_DIRTY=1 才可继续'
fi

if (( official )); then
    mapfile -t source_files < <(
        python3 "$fingerprint_script" --repo-root "$repo_root" --print-files
    )
    (( ${#source_files[@]} > 0 )) || die '找不到用于 freshness 检查的 source'
    for source_path in "${source_files[@]}"; do
        if [[ $repo_root/$source_path -nt $runner ]]; then
            die "runner 早于最新 source: $source_path"
        fi
    done
fi

expected_rows=$(( ${#shapes[@]} * ${#causal_modes[@]} * ${#implementations[@]} ))
if (( official )); then
    [[ $expected_rows == 40 ]] || die "canonical 结果必须为 40 行，当前为 $expected_rows"
fi

timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
git_commit="$(git -C "$repo_root" rev-parse HEAD)"
runner_sha256="$(sha256sum "$runner" | awk '{print $1}')"
source_sha256="$(python3 "$fingerprint_script" --repo-root "$repo_root")" || \
    die '无法计算 source fingerprint'
metadata_output="$("$runner" --metadata-only 2>&1)" || {
    printf '%s\n' "$metadata_output"
    die 'runner metadata-only 失败'
}
embedded_source_sha256="$(require_field "$metadata_output" source_sha256)"
build_contract="$(require_field "$metadata_output" build_contract)"
build_contract_payload_sha256="$(require_field "$metadata_output" build_contract_payload_sha256)"
device_index="$(require_field "$metadata_output" device_index)"
gpu_uuid="$(require_field "$metadata_output" gpu_uuid)"
gpu="$(require_field "$metadata_output" gpu_name)"
sm="$(require_field "$metadata_output" sm)"
driver="$(require_field "$metadata_output" driver)"
[[ $embedded_source_sha256 == "$source_sha256" ]] || \
    die "runner source fingerprint 不一致: current=$source_sha256 embedded=$embedded_source_sha256"
[[ $device_index =~ ^[0-9]+$ ]] || die 'runner device_index 必须为非负整数'
[[ $source_sha256 =~ ^[0-9a-f]{64}$ && \
    $build_contract_payload_sha256 =~ ^[0-9a-f]{64}$ && \
    $gpu_uuid == GPU-* && \
   $gpu =~ ^[^[:space:],]+$ && $sm =~ ^[0-9]+\.[0-9]+$ && \
   $driver =~ ^[0-9]+$ ]] || die 'runner 返回的实际 CUDA device metadata 无效'
if (( official )); then
    [[ -f $build_attestation && ! -L $build_attestation ]] || \
        die 'canonical build attestation 必须为普通文件且不能是 symlink'
    [[ ! $runner -nt $build_attestation ]] || \
        die 'canonical build attestation mtime 早于 runner'
    attestation_output="$(python3 "$attestation_validator" "$build_attestation" 2>&1)" || {
        printf '%s\n' "$attestation_output" >&2
        die 'canonical build attestation payload 验证失败'
    }
    attested_build_contract="$(require_field "$(tr '\n' ' ' <<< "$attestation_output")" build_contract)"
    attested_payload_sha256="$(require_field "$(tr '\n' ' ' <<< "$attestation_output")" build_contract_payload_sha256)"
    [[ $attested_build_contract == "$build_contract" && \
       $attested_payload_sha256 == "$build_contract_payload_sha256" ]] || \
        die 'canonical build attestation 内容与 runner metadata 不一致'
    [[ $build_contract =~ ^release-sm80-[0-9a-f]{16}$ ]] || \
        die "canonical build_contract 必须匹配 release-sm80-<hash>: actual=$build_contract"
    [[ ${build_contract##*-} == "${build_contract_payload_sha256:0:16}" ]] || \
        die 'canonical build_contract hash prefix 与 payload SHA-256 不一致'
    [[ $gpu == *A100* && $gpu == *80GB* && $sm == 8.0 ]] || \
        die "canonical benchmark 要求 A100 80GB（gpu_name 同时包含 A100 和 80GB）且 SM 8.0: gpu=$gpu sm=$sm"
fi
nvcc_output="$(nvcc --version)"
cuda_summary="$(printf '%s\n' "$nvcc_output" | grep 'release ' | tail -n1 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//;s/,/;/g')"
[[ -n $cuda_summary ]] || die '无法从 nvcc --version 读取 CUDA release'
nvcc_summary="$(printf '%s' "$nvcc_output" | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g;s/[[:space:]]$//')"

mkdir -p "$(dirname -- "$output_csv")" "$(dirname -- "$output_md")"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT
csv_tmp="$tmp_dir/result.csv"
markdown_tmp="$tmp_dir/result.md"
python3 "$csv_appender" --header "$csv_tmp" "${csv_columns[@]}"

printf '[benchmark_m1] mode=%s rows=%d runner_sha256=%s\n' \
    "$([[ $official == 1 ]] && printf canonical || printf smoke)" \
    "$expected_rows" "$runner_sha256"
printf '[benchmark_m1] gpu=%s uuid=%s sm=%s commit=%s\n' \
    "$gpu" "$gpu_uuid" "$sm" "$git_commit"

for shape in "${shapes[@]}"; do
    parse_shape "$shape"
    for causal in "${causal_modes[@]}"; do
        for implementation in "${implementations[@]}"; do
            latencies=()
            paths=()
            cta_counts=()
            requested_counts=()
            max_abs_values=()
            max_rel_values=()
            repeat_metadata=()
            for ((repeat_id = 1; repeat_id <= repeats; ++repeat_id)); do
                printf '[benchmark_m1] implementation=%s shape=%s causal=%s repeat=%d/%d\n' \
                    "$implementation" "$shape" "$causal" "$repeat_id" "$repeats"
                if ! output="$(
                    "$runner" \
                        --implementation "$implementation" \
                        --n "$shape_n" \
                        --d "$shape_d" \
                        --causal "$causal" \
                        --mode benchmark \
                        --warmup "$warmup" \
                        --iterations "$iterations" \
                        --seed "$seed" 2>&1
                )"; then
                    printf '%s\n' "$output"
                    die "runner 失败: implementation=$implementation shape=$shape causal=$causal repeat=$repeat_id"
                fi
                printf '%s\n' "$output"

                [[ $(require_field "$output" implementation) == "$implementation" ]] || die 'runner implementation 与请求不一致'
                [[ $(require_field "$output" shape) == "$shape" ]] || die 'runner shape 与请求不一致'
                [[ $(require_field "$output" causal) == "$causal" ]] || die 'runner causal 与请求不一致'
                [[ $(require_field "$output" input_pattern) == random ]] || die 'runner input_pattern 不是 random'
                [[ $(require_field "$output" status) == PASS ]] || die 'correctness 未通过'
                [[ $(require_field "$output" workspace_bytes) == 0 ]] || die 'workspace_bytes 必须为 0'

                run_source_sha256="$(require_field "$output" source_sha256)"
                run_build_contract="$(require_field "$output" build_contract)"
                run_build_contract_payload_sha256="$(require_field "$output" build_contract_payload_sha256)"
                run_device_index="$(require_field "$output" device_index)"
                run_gpu_uuid="$(require_field "$output" gpu_uuid)"
                run_gpu="$(require_field "$output" gpu_name)"
                run_sm="$(require_field "$output" sm)"
                run_driver="$(require_field "$output" driver)"
                run_metadata="$run_source_sha256|$run_build_contract|$run_build_contract_payload_sha256|$run_device_index|$run_gpu_uuid|$run_gpu|$run_sm|$run_driver"
                repeat_metadata+=("$run_metadata")
                [[ $run_metadata == "$source_sha256|$build_contract|$build_contract_payload_sha256|$device_index|$gpu_uuid|$gpu|$sm|$driver" ]] || \
                    die 'runner repeat 的 source/build/device metadata 与预检不一致'

                latencies+=("$(require_field "$output" latency_ms)")
                paths+=("$(require_field "$output" path)")
                cta_counts+=("$(require_field "$output" cta_count)")
                requested_counts+=("$(require_field "$output" requested_kv_elements)")
                max_abs_values+=("$(require_field "$output" max_abs)")
                max_rel_values+=("$(require_field "$output" max_rel)")
            done

            first_path="${paths[0]}"
            first_cta="${cta_counts[0]}"
            first_requested="${requested_counts[0]}"
            for path_value in "${paths[@]}"; do
                [[ $path_value == "$first_path" ]] || die '重复运行 path 不一致'
            done
            for cta_value in "${cta_counts[@]}"; do
                [[ $cta_value == "$first_cta" ]] || die '重复运行 cta_count 不一致'
            done
            for requested_value in "${requested_counts[@]}"; do
                [[ $requested_value == "$first_requested" ]] || die '重复运行 requested_kv_elements 不一致'
            done
            for metadata_value in "${repeat_metadata[@]}"; do
                [[ $metadata_value == "${repeat_metadata[0]}" ]] || \
                    die 'runner repeat metadata 不一致'
            done
            [[ $first_path == "$implementation" ]] || die 'runner path 与 implementation 不一致'
            if [[ $implementation == br1 ]]; then
                expected_cta="$shape_n"
            else
                expected_cta=$(( (shape_n + 3) / 4 ))
            fi
            expected_requested=$(( 2 * expected_cta * shape_n * shape_d ))
            [[ $first_cta == "$expected_cta" ]] || die 'runner cta_count 与理论值不一致'
            [[ $first_requested == "$expected_requested" ]] || \
                die 'runner requested_kv_elements 与理论值不一致'

            stats="$(python3 - "$repeats" \
                "${latencies[@]}" "${max_abs_values[@]}" "${max_rel_values[@]}" <<'PY'
import math
import statistics
import sys
n = int(sys.argv[1])
values = [float(value) for value in sys.argv[2:2 + n]]
absolute_errors = [float(value) for value in sys.argv[2 + n:2 + 2 * n]]
relative_errors = [float(value) for value in sys.argv[2 + 2 * n:2 + 3 * n]]
if any(not math.isfinite(value) or value <= 0.0 for value in values):
    raise SystemExit("latency_ms must be positive")
if any(not math.isfinite(value) or value < 0.0
       for value in absolute_errors + relative_errors):
    raise SystemExit("error metrics must be finite and nonnegative")
median = statistics.median(values)
minimum = min(values)
maximum = max(values)
spread = 100.0 * (maximum - minimum) / median
print(
    f"{median:.6f} {minimum:.6f} {maximum:.6f} {spread:.6f} "
    f"{max(absolute_errors):.6f} {max(relative_errors):.6f}"
)
PY
)" || die 'latency 统计失败'
            read -r median_ms minimum_ms maximum_ms spread_pct max_abs max_rel <<< "$stats"

            python3 "$csv_appender" --row "$csv_tmp" \
                "$timestamp" "$git_commit" "$runner_sha256" "$source_sha256" \
                "$build_contract" "$build_contract_payload_sha256" \
                "$device_index" "$gpu" "$gpu_uuid" "$sm" \
                "$driver" "$cuda_summary" "$nvcc_summary" \
                "$build_preset" fp32 1 1 row-major "$implementation" \
                "$first_path" "$shape_n" "$shape_d" "$causal" random "$seed" \
                "$warmup" "$iterations" "$repeats" "$median_ms" "$minimum_ms" \
                "$maximum_ms" "$spread_pct" true "$max_abs" "$max_rel" 0 \
                cpu-double cuda-event "$first_cta" "$first_requested"
        done
    done
done

actual_rows="$(python3 - "$csv_tmp" <<'PY'
import csv
import sys
with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    print(sum(1 for _ in csv.DictReader(handle)))
PY
)"
[[ $actual_rows == "$expected_rows" ]] || \
    die "结果行数错误: expected=$expected_rows actual=$actual_rows"
python3 "$renderer" "$csv_tmp" "$markdown_tmp" "$output_csv"
mv "$csv_tmp" "$output_csv"
mv "$markdown_tmp" "$output_md"
printf '[benchmark_m1] wrote rows=%s csv=%s markdown=%s\n' \
    "$actual_rows" "$output_csv" "$output_md"
