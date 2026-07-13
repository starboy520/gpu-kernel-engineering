#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../../.." && pwd)"
runner="${FA_RUNNER:-$repo_root/build/projects/flash_attention/flash_attention_runner}"

usage() {
    printf '用法: %s <tiled|tiled-async> [N D causal]\n' "${0##*/}" >&2
}

die() { printf 'profile: %s\n' "$*" >&2; exit 1; }
positive_int() { [[ $1 =~ ^[1-9][0-9]*$ ]]; }

[[ $# == 1 || $# == 4 ]] || { usage; exit 2; }
kernel="$1"
case "$kernel" in
    tiled) function_name=tiled_attention_kernel ;;
    tiled-async) function_name=tiled_async_attention_kernel ;;
    *) die "只支持 tiled 或 tiled-async" ;;
esac
n="${2:-1024}"
d="${3:-128}"
causal="${4:-0}"
positive_int "$n" || die 'N 必须为正整数'
positive_int "$d" || die 'D 必须为正整数'
[[ $causal == 0 || $causal == 1 ]] || die 'causal 必须为 0 或 1'
command -v ncu >/dev/null || die '找不到 ncu'
[[ -x "$runner" ]] || die "runner 不可执行: $runner"

shape="${n}x${d}-causal${causal}"
output_dir="$repo_root/projects/flash_attention/results/profiles"
report="$output_dir/${kernel}-${shape}.ncu-rep"
summary="$output_dir/${kernel}-${shape}.txt"
metrics=(
    launch__registers_per_thread
    launch__shared_mem_per_block_static
    launch__waves_per_multiprocessor
    launch__occupancy_limit_shared_mem
    launch__occupancy_limit_registers
    sm__warps_active.avg.pct_of_peak_sustained_active
    smsp__warps_eligible.avg.per_cycle_active
    smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio
    smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio
    smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio
    smsp__average_warp_latency_per_inst_issued.ratio
    sm__throughput.avg.pct_of_peak_sustained_elapsed
    dram__throughput.avg.pct_of_peak_sustained_elapsed
    lts__throughput.avg.pct_of_peak_sustained_elapsed
    l1tex__throughput.avg.pct_of_peak_sustained_elapsed
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld
)
metrics_csv="$(IFS=,; printf '%s' "${metrics[*]}")"
kernel_filter="regex:.*${function_name}\\("
command=(
    ncu --force-overwrite --replay-mode kernel --cache-control all
    --clock-control base --kernel-name-base demangled
    --kernel-name "$kernel_filter" --launch-count 1 --metrics "$metrics_csv"
    --export "$report" "$runner" --kernel "$kernel" --n "$n" --d "$d"
    --causal "$causal" --input-pattern random --mode validate --warmup 0
    --iterations 1 --seed 1234
)

mkdir -p "$output_dir"
{
    printf '[profile] kernel=%s function=%s shape=%s\n' "$kernel" "$function_name" "$shape"
    printf '[profile] report=%s\n' "$report"
    printf '[profile] command='
    printf '%q ' "${command[@]}"
    printf '\n'
} > "$summary"
"${command[@]}" 2>&1 | tee -a "$summary"
printf '[profile] wrote report=%s summary=%s\n' "$report" "$summary"
