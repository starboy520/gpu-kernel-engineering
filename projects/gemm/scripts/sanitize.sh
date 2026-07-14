#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../../.." && pwd)"
# shellcheck source=../../../common/scripts/common.sh
source "$repo_root/common/scripts/common.sh"

usage() {
    printf '用法: %s quick|full [runner]\n' "${0##*/}" >&2
}

die() { gpu_die sanitize "$@"; }

[[ $# -ge 1 && $# -le 2 ]] || {
    usage
    exit 2
}
mode="$1"
[[ $mode == quick || $mode == full ]] || {
    usage
    die "无效 mode '$mode'，必须是 quick 或 full"
}

runner="${2:-$repo_root/build/projects/gemm/gemm_runner}"
[[ -x "$runner" ]] || die "runner 不存在或不可执行: $runner"
gpu_require_command sanitize compute-sanitizer

command_count=0

run_sanitizer() {
    local tool="$1"
    local kernel="$2"
    local m="$3"
    local n="$4"
    local k="$5"
    local expected_path="$6"
    local intent="$7"
    local output
    local exit_code

    printf '[sanitize] tool=%s kernel=%s shape=%sx%sx%s expected_path=%s intent=%s\n' \
        "$tool" "$kernel" "$m" "$n" "$k" "$expected_path" "$intent"
    if output="$(compute-sanitizer --tool "$tool" --error-exitcode=99 \
        "$runner" --kernel "$kernel" --m "$m" --n "$n" --k "$k" \
        --mode validate --warmup 1 --iterations 1 2>&1)"; then
        exit_code=0
    else
        exit_code=$?
    fi
    printf '%s\n' "$output"
    ((command_count += 1))

    ((exit_code == 0)) || die "sanitizer 失败: tool=$tool kernel=$kernel exit=$exit_code"
    gpu_has_token "$output" "kernel=$kernel" || die "输出缺少 kernel=$kernel"
    gpu_has_token "$output" "path=$expected_path" || die "输出缺少 path=$expected_path"
    gpu_has_token "$output" 'status=PASS' || die '输出缺少 status=PASS'
}

run_quick() {
    run_sanitizer memcheck naive 65 128 48 naive representative
    run_sanitizer memcheck shared 65 128 48 shared representative
    run_sanitizer memcheck register 65 128 48 register representative
    run_sanitizer memcheck vectorized 65 128 48 fast-float4 aligned-fast
    run_sanitizer memcheck vectorized 65 130 48 fallback-register n-nonmultiple-fallback
    run_sanitizer memcheck async-16b 65 128 48 fast-pipeline-16b aligned-fast
    run_sanitizer memcheck async-16b 65 130 48 fallback-register n-nonmultiple-fallback
}

run_full_tools() {
    local kernel
    local expected_path

    for kernel in shared register vectorized async-16b; do
        case "$kernel" in
            shared|register) expected_path="$kernel" ;;
            vectorized) expected_path=fast-float4 ;;
            async-16b) expected_path=fast-pipeline-16b ;;
        esac
        run_sanitizer racecheck "$kernel" 65 128 48 "$expected_path" multi-k-shared-memory
        run_sanitizer synccheck "$kernel" 65 128 48 "$expected_path" multi-k-shared-memory
        run_sanitizer initcheck "$kernel" 65 128 48 "$expected_path" multi-k-shared-memory
    done
}

run_quick
if [[ $mode == full ]]; then
    run_full_tools
fi

printf '[sanitize] summary mode=%s commands=%d status=PASS\n' "$mode" "$command_count"