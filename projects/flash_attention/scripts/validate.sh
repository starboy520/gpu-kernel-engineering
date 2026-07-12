#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../../.." && pwd)"
[[ $# -le 1 ]] || {
    printf '用法: %s [runner]\n' "${0##*/}" >&2
    exit 2
}
runner="${1:-$repo_root/build/projects/flash_attention/flash_attention_runner}"
cases_file="${FLASH_ATTENTION_CASES_FILE:-$repo_root/projects/flash_attention/tests/correctness_cases.csv}"

die() {
    printf 'validate: %s\n' "$*" >&2
    exit 1
}

trim() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "$value"
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

[[ -x "$runner" ]] || die "runner 不存在或不可执行: $runner"
[[ -r "$cases_file" ]] || die "cases CSV 不可读: $cases_file"

if [[ ${FLASH_ATTENTION_SKIP_CTEST:-0} != 1 ]]; then
    printf '[validate] stage=ctest build=%s\n' "$repo_root/build"
    ctest --test-dir "$repo_root/build" --output-on-failure
fi

executions=0
passed=0
header_seen=0
line_number=0
declare -A seen_cases=()

while IFS= read -r raw_line || [[ -n $raw_line ]]; do
    ((line_number += 1))
    raw_line="${raw_line%$'\r'}"
    line="$(trim "$raw_line")"
    [[ -z $line || $line == \#* ]] && continue

    comma_text="${line//[^,]/}"
    ((${#comma_text} == 3)) || \
        die "malformed CSV row at line $line_number: expected 4 fields"

    IFS=',' read -r n d causal input_pattern <<< "$line"
    n="$(trim "$n")"
    d="$(trim "$d")"
    causal="$(trim "$causal")"
    input_pattern="$(trim "$input_pattern")"

    if ((header_seen == 0)); then
        [[ $n == n && $d == d && $causal == causal && $input_pattern == input_pattern ]] || \
            die "malformed CSV header at line $line_number"
        header_seen=1
        continue
    fi

    [[ $n =~ ^[1-9][0-9]*$ && $d =~ ^[1-9][0-9]*$ ]] || \
        die "malformed CSV row at line $line_number: n,d must be positive integers"
    [[ $causal == 0 || $causal == 1 ]] || \
        die "malformed CSV row at line $line_number: causal must be 0 or 1"
    [[ $input_pattern =~ ^(random|zero-qk|negative-scores)$ ]] || \
        die "malformed CSV row at line $line_number: invalid input pattern"

    case_key="$n,$d,$causal,$input_pattern"
    [[ -z ${seen_cases[$case_key]+present} ]] || \
        die "malformed CSV row at line $line_number: duplicate case '$case_key'"
    seen_cases[$case_key]=1

    printf '[validate] kernel=naive shape=%sx%s causal=%s input_pattern=%s\n' \
        "$n" "$d" "$causal" "$input_pattern"
    if output="$("$runner" --kernel naive --n "$n" --d "$d" \
        --causal "$causal" --input-pattern "$input_pattern" \
        --mode validate --warmup 0 --iterations 1 2>&1)"; then
        exit_code=0
    else
        exit_code=$?
    fi
    printf '%s\n' "$output"
    ((executions += 1))

    ((exit_code == 0)) || die "runner 失败: case=$case_key exit=$exit_code"
    has_token "$output" 'kernel=naive' || die '输出缺少 kernel=naive'
    has_token "$output" 'path=naive' || die '输出缺少 path=naive'
    has_token "$output" "shape=${n}x${d}" || die "输出缺少 shape=${n}x${d}"
    has_token "$output" "causal=$causal" || die "输出缺少 causal=$causal"
    has_token "$output" "input_pattern=$input_pattern" || \
        die "输出缺少 input_pattern=$input_pattern"
    has_token "$output" 'status=PASS' || die '输出缺少 status=PASS'
    ((passed += 1))
done < "$cases_file"

((header_seen == 1)) || die 'malformed CSV: missing header'
((executions > 0)) || die 'malformed CSV: no test cases'
printf '[validate] summary executions=%d pass=%d status=PASS\n' \
    "$executions" "$passed"
