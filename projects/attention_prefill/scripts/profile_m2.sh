#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../../.." && pwd)"
runner="$repo_root/build/projects/attention_prefill/attention_prefill_evidence_runner"
output_dir="$repo_root/projects/attention_prefill/results/profiles/m2"

[[ $# == 4 ]] || {
    printf '用法: %s <br4|m2> N D causal\n' "${0##*/}" >&2
    exit 2
}

implementation="$1"
n="$2"
d="$3"
causal="$4"
case "$implementation" in
    br4) symbol=query_tiled_kernel ;;
    m2) symbol=warp_per_query_kernel ;;
    *) printf 'implementation 必须是 br4 或 m2\n' >&2; exit 2 ;;
esac

metrics=(
    gpu__time_duration.sum
    launch__registers_per_thread
    launch__shared_mem_per_block_static
    launch__waves_per_multiprocessor
    sm__warps_active.avg.pct_of_peak_sustained_active
    smsp__warps_eligible.avg.per_cycle_active
    smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio
    smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio
    smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio
    smsp__average_warp_latency_per_inst_issued.ratio
    sm__throughput.avg.pct_of_peak_sustained_elapsed
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum
    l1tex__data_pipe_lsu_wavefronts_mem_shared.sum
)
metrics_csv="$(IFS=,; printf '%s' "${metrics[*]}")"
stem="${implementation}-${n}x${d}-causal${causal}"
mkdir -p "$output_dir"

ncu --force-overwrite --replay-mode kernel --cache-control all \
    --clock-control base --kernel-name-base demangled \
    --kernel-name "regex:.*${symbol}\\(" --launch-count 1 \
    --metrics "$metrics_csv" --export "$output_dir/${stem}.ncu-rep" \
    "$runner" --implementation "$implementation" --n "$n" --d "$d" \
    --causal "$causal" --mode validate --warmup 0 --iterations 1 --seed 1234 \
    2>&1 | tee "$output_dir/${stem}.txt"

printf '[profile_m2] wrote %s\n' "$output_dir/${stem}.txt"
