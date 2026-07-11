#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../../.." && pwd)"
runner="${1:-$repo_root/build/projects/gemm/gemm_runner}"
cases_file="${GEMM_CASES_FILE:-$repo_root/projects/gemm/tests/correctness_cases.csv}"
author_kernels=(naive shared register vectorized async-16b)

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

printf '[validate] stage=ctest build=%s\n' "$repo_root/build"
ctest --test-dir "$repo_root/build" --output-on-failure

executions=0
passed=0
header_seen=0
line_number=0
declare -A seen_cases=()

run_case() {
    local kernel="$1"
    local m="$2"
    local n="$3"
    local k="$4"
    local expected_path="$5"
    local output
    local exit_code

    printf '[validate] kernel=%s shape=%sx%sx%s expected_path=%s\n' \
        "$kernel" "$m" "$n" "$k" "$expected_path"
    if output="$($runner --kernel "$kernel" --m "$m" --n "$n" --k "$k" \
        --mode validate --warmup 1 --iterations 1 2>&1)"; then
        exit_code=0
    else
        exit_code=$?
    fi
    printf '%s\n' "$output"
    ((executions += 1))

    ((exit_code == 0)) || die "runner 失败: kernel=$kernel shape=${m}x${n}x${k} exit=$exit_code"
    has_token "$output" "kernel=$kernel" || die "输出缺少 kernel=$kernel"
    has_token "$output" 'status=PASS' || die "输出缺少 status=PASS"
    if [[ $expected_path != any ]]; then
        has_token "$output" "path=$expected_path" || \
            die "路径不符: kernel=$kernel expected_path=$expected_path"
    fi

    ((passed += 1))
}

while IFS= read -r raw_line || [[ -n $raw_line ]]; do
    ((line_number += 1))
    raw_line="${raw_line%$'\r'}"
    line="$(trim "$raw_line")"
    [[ -z $line || $line == \#* ]] && continue

    comma_text="${line//[^,]/}"
    ((${#comma_text} == 4)) || die "malformed CSV row at line $line_number: expected 5 fields"

    IFS=',' read -r kernel m n k expected_path <<< "$line"
    kernel="$(trim "$kernel")"
    m="$(trim "$m")"
    n="$(trim "$n")"
    k="$(trim "$k")"
    expected_path="$(trim "$expected_path")"

    if ((header_seen == 0)); then
        [[ $kernel == kernel && $m == m && $n == n && $k == k && $expected_path == expected_path ]] || \
            die "malformed CSV header at line $line_number: expected kernel,m,n,k,expected_path"
        header_seen=1
        continue
    fi

    [[ $kernel =~ ^(all|naive|shared|register|vectorized|async-16b|cublas)$ ]] || \
        die "malformed CSV row at line $line_number: unknown kernel '$kernel'"
    [[ $m =~ ^[1-9][0-9]*$ && $n =~ ^[1-9][0-9]*$ && $k =~ ^[1-9][0-9]*$ ]] || \
        die "malformed CSV row at line $line_number: m,n,k must be positive integers"
    [[ $expected_path == any || $expected_path =~ ^[A-Za-z0-9._-]+$ ]] || \
        die "malformed CSV row at line $line_number: invalid expected_path '$expected_path'"

    case_key="$kernel,$m,$n,$k"
    [[ -z ${seen_cases[$case_key]+present} ]] || \
        die "malformed CSV row at line $line_number: duplicate case '$case_key'"
    seen_cases[$case_key]=1

    if [[ $kernel == all ]]; then
        for author_kernel in "${author_kernels[@]}"; do
            run_case "$author_kernel" "$m" "$n" "$k" "$expected_path"
        done
    else
        run_case "$kernel" "$m" "$n" "$k" "$expected_path"
    fi
done < "$cases_file"

((header_seen == 1)) || die 'malformed CSV: missing header'
((executions > 0)) || die 'malformed CSV: no test cases'
printf '[validate] summary executions=%d pass=%d status=PASS\n' "$executions" "$passed"