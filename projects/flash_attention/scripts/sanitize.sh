#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../../.." && pwd)"

usage() {
    printf '用法: %s quick|full [runner]\n' "${0##*/}" >&2
}

die() {
    printf 'sanitize: %s\n' "$*" >&2
    exit 1
}

has_token() {
    local text="$1"
    local token="$2"
    local word
    for word in $text; do
        [[ $word == "$token" ]] && return 0
    done
    return 1
}

[[ $# -ge 1 && $# -le 2 ]] || {
    usage
    exit 2
}
mode="$1"
[[ $mode == quick || $mode == full ]] || {
    usage
    die "无效 mode '$mode'，必须是 quick 或 full"
}

runner="${2:-$repo_root/build/projects/flash_attention/flash_attention_runner}"
[[ -x "$runner" ]] || die "runner 不存在或不可执行: $runner"
command -v compute-sanitizer >/dev/null 2>&1 || die '找不到 compute-sanitizer'

command_count=0

run_sanitizer() {
    local tool="$1"
    local kernel="$2"
    local n="$3"
    local d="$4"
    local causal="$5"
    local input_pattern="$6"
    local intent="$7"
    local output
    local exit_code

    printf '[sanitize] tool=%s kernel=%s shape=%sx%s causal=%s input_pattern=%s intent=%s\n' \
        "$tool" "$kernel" "$n" "$d" "$causal" "$input_pattern" "$intent"
    if output="$(compute-sanitizer --tool "$tool" --error-exitcode=99 \
        "$runner" --kernel "$kernel" --n "$n" --d "$d" --causal "$causal" \
        --input-pattern "$input_pattern" --mode validate --warmup 0 \
        --iterations 1 2>&1)"; then
        exit_code=0
    else
        exit_code=$?
    fi
    printf '%s\n' "$output"
    ((command_count += 1))

    ((exit_code == 0)) || die "sanitizer 失败: tool=$tool exit=$exit_code"
    has_token "$output" "kernel=$kernel" || die "输出缺少 kernel=$kernel"
    has_token "$output" "path=$kernel" || die "输出缺少 path=$kernel"
    has_token "$output" 'status=PASS' || die '输出缺少 status=PASS'
}

run_sanitizer memcheck naive 37 24 1 random tail-causal
run_sanitizer memcheck naive 37 24 0 negative-scores all-negative-softmax
run_sanitizer memcheck tiled 37 24 1 random tail-causal-zero-workspace
run_sanitizer memcheck tiled 37 24 0 negative-scores all-negative-online-softmax

if [[ $mode == full ]]; then
    for kernel in naive tiled; do
        run_sanitizer racecheck "$kernel" 37 24 1 random shared-state
        run_sanitizer synccheck "$kernel" 37 24 1 random block-synchronization
        # initcheck covers uninitialized device-global reads, not shared memory.
        run_sanitizer initcheck "$kernel" 37 24 1 random global-memory-initialization
    done
fi

printf '[sanitize] summary mode=%s commands=%d status=PASS\n' \
    "$mode" "$command_count"
