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
    local n="$2"
    local d="$3"
    local causal="$4"
    local input_pattern="$5"
    local intent="$6"
    local output
    local exit_code

    printf '[sanitize] tool=%s kernel=naive shape=%sx%s causal=%s input_pattern=%s intent=%s\n' \
        "$tool" "$n" "$d" "$causal" "$input_pattern" "$intent"
    if output="$(compute-sanitizer --tool "$tool" --error-exitcode=99 \
        "$runner" --kernel naive --n "$n" --d "$d" --causal "$causal" \
        --input-pattern "$input_pattern" --mode validate --warmup 0 \
        --iterations 1 2>&1)"; then
        exit_code=0
    else
        exit_code=$?
    fi
    printf '%s\n' "$output"
    ((command_count += 1))

    ((exit_code == 0)) || die "sanitizer 失败: tool=$tool exit=$exit_code"
    has_token "$output" 'kernel=naive' || die '输出缺少 kernel=naive'
    has_token "$output" 'path=naive' || die '输出缺少 path=naive'
    has_token "$output" 'status=PASS' || die '输出缺少 status=PASS'
}

run_sanitizer memcheck 37 24 1 random tail-causal
run_sanitizer memcheck 37 24 0 negative-scores all-negative-softmax

if [[ $mode == full ]]; then
    run_sanitizer racecheck 37 24 1 random shared-reduction
    run_sanitizer synccheck 37 24 1 random shared-reduction
    # initcheck covers uninitialized device-global reads, not shared memory.
    run_sanitizer initcheck 37 24 1 random global-memory-initialization
fi

printf '[sanitize] summary mode=%s commands=%d status=PASS\n' \
    "$mode" "$command_count"
