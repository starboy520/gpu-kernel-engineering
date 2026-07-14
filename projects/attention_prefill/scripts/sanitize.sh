#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../../.." && pwd)"
# shellcheck source=../../../common/scripts/common.sh
source "$repo_root/common/scripts/common.sh"

usage() {
    printf '用法: %s quick|full [runner]\n' "${0##*/}" >&2
}

die() { gpu_die attention-prefill-sanitize "$@"; }

[[ $# -ge 1 && $# -le 2 ]] || {
    usage
    exit 2
}
mode="$1"
[[ $mode == quick || $mode == full ]] || {
    usage
    die "无效 mode '$mode'，必须是 quick 或 full"
}

runner="${2:-$repo_root/build/projects/attention_prefill/attention_prefill_runner}"
[[ -x "$runner" ]] || die "runner 不存在或不可执行: $runner"
gpu_require_command attention-prefill-sanitize compute-sanitizer

command_count=0

run_sanitizer() {
    local tool="$1"
    local input_pattern="$2"
    local output
    local exit_code

    printf '[attention_prefill_sanitize] tool=%s shape=33x65 causal=1 input_pattern=%s\n' \
        "$tool" "$input_pattern"
    if output="$(compute-sanitizer --tool "$tool" --error-exitcode=99 \
        "$runner" --n 33 --d 65 --causal 1 \
        --input-pattern "$input_pattern" 2>&1)"; then
        exit_code=0
    else
        exit_code=$?
    fi
    printf '%s\n' "$output"
    ((command_count += 1))

    ((exit_code == 0)) || die "sanitizer 失败: tool=$tool exit=$exit_code"
    gpu_has_token "$output" "input_pattern=$input_pattern" || \
        die "输出缺少 input_pattern=$input_pattern"
    gpu_has_token "$output" 'status=PASS' || die '输出缺少 status=PASS'
}

run_sanitizer memcheck random
run_sanitizer memcheck zero-qk
run_sanitizer memcheck rising-logits

if [[ $mode == full ]]; then
    for tool in racecheck synccheck initcheck; do
        run_sanitizer "$tool" random
        run_sanitizer "$tool" rising-logits
    done
fi

printf '[attention_prefill_sanitize] summary mode=%s commands=%d status=PASS\n' \
    "$mode" "$command_count"
