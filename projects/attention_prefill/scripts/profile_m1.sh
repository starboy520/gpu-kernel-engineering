#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${AP_M1_REPO_ROOT:-$(cd -- "$script_dir/../../.." && pwd)}"
repo_root="$(cd -- "$repo_root" && pwd)"
canonical_runner="$repo_root/build/projects/attention_prefill/attention_prefill_evidence_runner"
runner="${AP_M1_EVIDENCE_RUNNER:-$canonical_runner}"
official_dir="$repo_root/projects/attention_prefill/results/profiles"
output_dir="${AP_M1_PROFILE_OUTPUT_DIR:-$official_dir}"
normalizer="$script_dir/summarize_m1_ncu.py"
fingerprint_script="$script_dir/source_fingerprint.py"
attestation_validator="$script_dir/validate_build_attestation.py"
build_attestation="$repo_root/build/projects/attention_prefill/attention_prefill_build_attestation.txt"

usage() { printf '用法: %s <br1|br4> [N D causal]\n' "${0##*/}" >&2; }
die() { printf 'profile_m1: %s\n' "$*" >&2; exit 1; }
positive_integer() { [[ $1 =~ ^[1-9][0-9]*$ ]]; }
normalize_path() {
    python3 - "$1" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve(strict=False))
PY
}
require_field() {
    local text="$1" name="$2" token
    for token in $text; do
        [[ $token == "$name="* ]] && { printf '%s' "${token#*=}"; return 0; }
    done
    return 1
}
valid_sha256() { [[ $1 =~ ^[0-9a-f]{64}$ ]]; }
metric_available() {
    local query_output="$1" metric="$2"
    awk -v expected="$metric" '$1 == expected { found = 1 } END { exit !found }' \
        <<< "$query_output"
}
metadata_identity() {
    local text="$1"
    printf '%s|%s|%s|%s|%s|%s|%s|%s' \
        "$(require_field "$text" source_sha256)" "$(require_field "$text" build_contract)" \
        "$(require_field "$text" build_contract_payload_sha256)" "$(require_field "$text" device_index)" \
        "$(require_field "$text" gpu_uuid)" "$(require_field "$text" gpu_name)" \
        "$(require_field "$text" sm)" "$(require_field "$text" driver)"
}

[[ $# == 1 || $# == 4 ]] || { usage; exit 2; }
implementation="$1"
case "$implementation" in
    br1) function_name=tiled_attention_kernel ;;
    br4) function_name=query_tiled_kernel ;;
    *) die 'implementation 只能是 br1 或 br4' ;;
esac
n="${2:-1024}"; d="${3:-128}"; causal="${4:-0}"
positive_integer "$n" || die 'N 必须为正整数'
positive_integer "$d" || die 'D 必须为正整数'
[[ $causal == 0 || $causal == 1 ]] || die 'causal 必须为 0 或 1'
for command_name in git ncu sha256sum python3; do command -v "$command_name" >/dev/null || die "找不到 $command_name"; done
[[ -f $runner && ! -L $runner && -x $runner ]] || die "runner 必须是可执行普通文件且不能是 symlink: $runner"
[[ -f $normalizer ]] || die "normalizer 不存在: $normalizer"

canonical_runner="$(normalize_path "$canonical_runner")"; runner="$(normalize_path "$runner")"
official_dir="$(normalize_path "$official_dir")"; output_dir="$(normalize_path "$output_dir")"
if [[ -v AP_M1_EVIDENCE_RUNNER || ${AP_M1_ALLOW_PROFILE_SMOKE:-0} == 1 ]]; then
    [[ -v AP_M1_PROFILE_OUTPUT_DIR ]] || die 'custom runner 或 AP_M1_ALLOW_PROFILE_SMOKE=1 必须设置 AP_M1_PROFILE_OUTPUT_DIR'
    [[ ${output_dir,,} == *smoke* && $output_dir != "$official_dir" ]] || die 'smoke profile 输出目录路径必须包含 smoke 且不能是 official profiles'
    evidence_kind=smoke
else
    [[ $runner == "$canonical_runner" ]] || die 'canonical profile 只允许标准 runner'
    [[ $output_dir == "$official_dir" ]] || die 'canonical profile 默认输出必须是 official profiles'
    evidence_kind=canonical
fi
canonical_problem=0
case "$n,$d,$causal" in
    256,64,0|1024,128,0|1024,128,1) canonical_problem=1 ;;
esac
if [[ $evidence_kind == canonical ]]; then
    (( canonical_problem == 1 )) || \
        die 'canonical profile shape 只允许 256x64 causal=0、1024x128 causal=0、1024x128 causal=1；其他 shape 必须显式使用 smoke 输出'
    [[ -z $(git -C "$repo_root" status --short) ]] || \
        die 'canonical profile 要求干净工作树；dirty 采集必须显式使用 smoke 输出'
fi
git_commit="$(git -C "$repo_root" rev-parse HEAD)"

runner_sha256="$(sha256sum "$runner" | awk '{print $1}')"; valid_sha256 "$runner_sha256" || die 'runner SHA-256 无效'
metadata_output="$("$runner" --metadata-only 2>&1)" || { printf '%s\n' "$metadata_output" >&2; die 'runner metadata-only 失败'; }
source_sha256="$(require_field "$metadata_output" source_sha256)"
build_contract="$(require_field "$metadata_output" build_contract)"
build_contract_payload_sha256="$(require_field "$metadata_output" build_contract_payload_sha256)"
device_index="$(require_field "$metadata_output" device_index)"; gpu_uuid="$(require_field "$metadata_output" gpu_uuid)"
gpu_name="$(require_field "$metadata_output" gpu_name)"; sm="$(require_field "$metadata_output" sm)"; driver="$(require_field "$metadata_output" driver)"
valid_sha256 "$source_sha256" || die 'source_sha256 无效'; valid_sha256 "$build_contract_payload_sha256" || die 'build payload SHA-256 无效'
[[ $build_contract =~ ^release-sm80-[0-9a-f]{16}$ && ${build_contract##*-} == "${build_contract_payload_sha256:0:16}" ]] || die 'build_contract 与 payload SHA-256 不一致'
[[ $device_index =~ ^[0-9]+$ && $gpu_uuid =~ ^GPU-[0-9A-Fa-f-]+$ && $gpu_name =~ ^[^[:space:],]+$ && $sm =~ ^[0-9]+\.[0-9]+$ && $driver =~ ^[0-9]+$ ]] || die 'runner CUDA device metadata 无效'

if [[ $evidence_kind == canonical ]]; then
    [[ $gpu_name == *A100* && $gpu_name == *80GB* && $sm == 8.0 ]] || die "canonical profile 要求 A100 80GB sm=8.0: gpu=$gpu_name sm=$sm"
    current_source_sha256="$(python3 "$fingerprint_script" --repo-root "$repo_root")" || die '无法计算当前 source fingerprint'
    [[ $source_sha256 == "$current_source_sha256" ]] || die 'canonical runner embedded source fingerprint 与当前 source 不一致'
    [[ -f $build_attestation && ! -L $build_attestation ]] || die 'canonical build attestation 必须是非 symlink 普通文件'
    attestation_output="$(python3 "$attestation_validator" "$build_attestation" 2>&1)" || { printf '%s\n' "$attestation_output" >&2; die 'canonical build attestation 验证失败'; }
    attestation_line="$(tr '\n' ' ' <<< "$attestation_output")"
    [[ $(require_field "$attestation_line" build_contract) == "$build_contract" && $(require_field "$attestation_line" build_contract_payload_sha256) == "$build_contract_payload_sha256" ]] || die 'canonical build attestation 与 runner metadata 不一致'
elif [[ ${AP_M1_ALLOW_NON_A100_SMOKE:-0} != 1 ]]; then
    [[ $gpu_name == *A100* && $gpu_name == *80GB* && $sm == 8.0 ]] || die "default profile 要求 A100 80GB sm=8.0: gpu=$gpu_name sm=$sm"
fi

ncu_version_output="$(ncu --version 2>&1)" || die 'ncu --version 失败'
ncu_version="$(sed -nE 's/^Version ([0-9]+(\.[0-9]+)+).*/\1/p' <<< "$ncu_version_output" | head -n1)"
[[ $ncu_version =~ ^[0-9]+(\.[0-9]+)+$ ]] || die '无法读取 ncu 版本'
[[ $ncu_version == 2026.2.* ]] || \
    die "仅支持 ncu 2026.2.*；当前=$ncu_version"

metrics=(gpu__time_duration.sum launch__registers_per_thread launch__shared_mem_per_block_static launch__waves_per_multiprocessor launch__occupancy_limit_shared_mem launch__occupancy_limit_registers sm__warps_active.avg.pct_of_peak_sustained_active smsp__warps_eligible.avg.per_cycle_active smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio smsp__average_warp_latency_per_inst_issued.ratio sm__throughput.avg.pct_of_peak_sustained_elapsed dram__throughput.avg.pct_of_peak_sustained_elapsed lts__throughput.avg.pct_of_peak_sustained_elapsed l1tex__throughput.avg.pct_of_peak_sustained_elapsed dram__bytes_read.sum dram__bytes_write.sum lts__t_sectors_op_read.sum lts__t_sectors_op_write.sum lts__t_sector_hit_rate.pct l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum l1tex__t_requests_pipe_lsu_mem_global_op_st.sum l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum l1tex__data_pipe_lsu_wavefronts_mem_shared.sum)
metrics_csv="$(IFS=,; printf '%s' "${metrics[*]}")"
launch_query="$(ncu --query-metrics-collection launch 2>&1)" || die 'ncu launch metric query 失败'
profile_metrics=()
for metric in "${metrics[@]}"; do
    if [[ $metric == launch__* ]]; then metric_available "$launch_query" "$metric" || die "ncu metric 不可用: $metric"; else profile_metrics+=("$metric"); fi
done
profile_query="$(ncu --query-metrics-mode all --metrics "$(IFS=,; printf '%s' "${profile_metrics[*]}")" --query-metrics 2>&1)" || die 'ncu profiling metric query 失败'
for metric in "${profile_metrics[@]}"; do metric_available "$profile_query" "$metric" || die "ncu metric 不可用: $metric"; done

profile_timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"; base_stem="${implementation}-${n}x${d}-causal${causal}"; stem="$base_stem"
[[ $evidence_kind == smoke ]] && stem="${base_stem}-smoke"
mkdir -p "$output_dir"; tmp_dir="$(mktemp -d "$output_dir/.profile-${stem}.XXXXXX")"; trap 'rm -rf "$tmp_dir"' EXIT
report_tmp="$tmp_dir/${stem}.ncu-rep"; summary_tmp="$tmp_dir/${stem}.txt"; raw_tmp="$tmp_dir/${stem}-raw.csv"; normalized_tmp="$tmp_dir/${stem}-metrics.csv"
report="$output_dir/${stem}.ncu-rep"; summary="$output_dir/${stem}.txt"; raw_csv="$output_dir/${stem}-raw.csv"; normalized_csv="$output_dir/${stem}-metrics.csv"
kernel_filter="regex:^.*::${function_name}\\(const float \\*, const float \\*, const float \\*, float \\*, int, int, bool\\)$"
runner_command=("$runner" --implementation "$implementation" --n "$n" --d "$d" --causal "$causal" --mode validate --warmup 0 --iterations 1 --seed 1234)
profile_command=(ncu --force-overwrite --replay-mode kernel --cache-control all --clock-control base --kernel-name-base demangled --kernel-name "$kernel_filter" --launch-count 1 --metrics "$metrics_csv" --export "$report_tmp" "${runner_command[@]}")
export_command=(ncu --import "$report_tmp" --page raw --csv --log-file "$raw_tmp" --metrics "$metrics_csv")
{
    printf '[profile_m1] evidence_kind=%s implementation=%s function=%s shape=%sx%s causal=%s\n' "$evidence_kind" "$implementation" "$function_name" "$n" "$d" "$causal"
    printf '[profile_m1] report=%s\n[profile_m1] normalized_csv=%s\n[profile_m1] kernel_filter=%s\n' "$report" "$normalized_csv" "$kernel_filter"
    printf '[profile_m1] profile_timestamp=%s runner_sha256=%s ncu_version=%s\n' "$profile_timestamp" "$runner_sha256" "$ncu_version"
    printf '[profile_m1] source_sha256=%s build_contract=%s build_contract_payload_sha256=%s\n' "$source_sha256" "$build_contract" "$build_contract_payload_sha256"
    printf '[profile_m1] device_index=%s gpu_uuid=%s gpu_name=%s sm=%s driver=%s\n' "$device_index" "$gpu_uuid" "$gpu_name" "$sm" "$driver"
    printf '[profile_m1] command='; printf '%q ' "${profile_command[@]}"; printf '\n[profile_m1] runner_command='; printf '%q ' "${runner_command[@]}"; printf '\n'
} > "$summary_tmp"
set +e; "${profile_command[@]}" 2>&1 | tee -a "$summary_tmp"; profile_status=${PIPESTATUS[0]}; set -e
(( profile_status == 0 )) || die "ncu profile 失败，exit=$profile_status"
runner_line="$(grep -E '(^|[[:space:]])implementation=' "$summary_tmp" | tail -n1 || true)"; [[ -n $runner_line ]] || die 'runner 输出缺少 metadata/result 行'
[[ $(require_field "$runner_line" implementation) == "$implementation" && $(require_field "$runner_line" shape) == "${n}x${d}" && $(require_field "$runner_line" causal) == "$causal" && $(require_field "$runner_line" status) == PASS ]] || die 'runner result identity/correctness 不匹配'
[[ $(metadata_identity "$runner_line") == "$(metadata_identity "$metadata_output")" ]] || die 'runner result metadata 与 metadata-only 不一致'
"${export_command[@]}" 2>&1 | tee -a "$summary_tmp"
python3 "$normalizer" --normalize-raw "$raw_tmp" --normalized-output "$normalized_tmp" --evidence-kind "$evidence_kind" --implementation "$implementation" --n "$n" --d "$d" --causal "$causal" --profile-timestamp "$profile_timestamp" --git-commit "$git_commit" --runner-sha256 "$runner_sha256" --source-sha256 "$source_sha256" --build-contract "$build_contract" --build-contract-payload-sha256 "$build_contract_payload_sha256" --device-index "$device_index" --gpu-uuid "$gpu_uuid" --gpu-name "$gpu_name" --sm "$sm" --driver "$driver" --ncu-version "$ncu_version"
[[ -s $report_tmp && -s $summary_tmp && -s $raw_tmp && -s $normalized_tmp ]] || die 'profile transaction 产物不完整'
mv -f "$report_tmp" "$report"; mv -f "$summary_tmp" "$summary"; mv -f "$raw_tmp" "$raw_csv"; mv -f "$normalized_tmp" "$normalized_csv"
printf '[profile_m1] wrote report=%s summary=%s raw_csv=%s normalized_csv=%s\n' "$report" "$summary" "$raw_csv" "$normalized_csv"
