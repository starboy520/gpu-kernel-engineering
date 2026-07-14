#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../../.." && pwd)"
source "$repo_root/common/scripts/common.sh"
runner="${GEMM_RUNNER:-$repo_root/build/projects/gemm/gemm_runner}"

usage() {
    printf '用法: %s <kernel> [M N K]\n' "${0##*/}" >&2
    printf 'kernel: naive | shared | register | vectorized | async-16b\n' >&2
}

die() { gpu_die profile "$@"; }

[[ $# == 1 || $# == 4 ]] || {
    usage
    exit 2
}

kernel="$1"
case "$kernel" in
    naive) function_name=naive_kernel ;;
    shared) function_name=shared_tiled_kernel ;;
    register) function_name=register_tiled_kernel ;;
    vectorized) function_name=vectorized_tiled_kernel ;;
    async-16b) function_name=double_buffer_kernel ;;
    *) die "不支持的 kernel: $kernel" ;;
esac

m="${2:-2048}"
n="${3:-2048}"
k="${4:-2048}"
gpu_positive_integer "$m" || die "M 必须是正整数，当前值: $m"
gpu_positive_integer "$n" || die "N 必须是正整数，当前值: $n"
gpu_positive_integer "$k" || die "K 必须是正整数，当前值: $k"

gpu_require_command profile ncu
[[ -x "$runner" ]] || die "runner 不存在或不可执行: $runner"

shape="${m}x${n}x${k}"
output_dir="$repo_root/projects/gemm/results/profiles"
report="$output_dir/${kernel}-${shape}.ncu-rep"
summary="$output_dir/${kernel}-${shape}.txt"
metrics=(
    launch__registers_per_thread
    sm__throughput.avg.pct_of_peak_sustained_elapsed
    sm__warps_active.avg.pct_of_peak_sustained_active
    dram__throughput.avg.pct_of_peak_sustained_elapsed
    l1tex__throughput.avg.pct_of_peak_sustained_elapsed
    lts__throughput.avg.pct_of_peak_sustained_elapsed
    smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio
    smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld
)
metrics_csv="$(IFS=,; printf '%s' "${metrics[*]}")"
kernel_filter="regex:^${function_name}\\("
ncu_command=(
    ncu
    --force-overwrite
    --kernel-name-base demangled
    --kernel-name "$kernel_filter"
    --launch-count 1
    --metrics "$metrics_csv"
    --export "$report"
    "$runner"
    --kernel "$kernel"
    --m "$m"
    --n "$n"
    --k "$k"
    --mode validate
    --warmup 0
    --iterations 1
)

mkdir -p "$output_dir"
{
    printf '[profile] kernel=%s function=%s shape=%s\n' "$kernel" "$function_name" "$shape"
    printf '[profile] runner=%s\n' "$runner"
    printf '[profile] report=projects/gemm/results/profiles/%s\n' "${report##*/}"
    printf '[profile] metrics=%s\n' "$metrics_csv"
    printf '[profile] command='
    printf '%q ' "${ncu_command[@]}"
    printf '\n'
} > "$summary"

"${ncu_command[@]}" 2>&1 | tee -a "$summary"
printf '[profile] wrote report=%s summary=%s\n' "$report" "$summary"