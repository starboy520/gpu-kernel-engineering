#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../../.." && pwd)"
runner="${1:-$repo_root/build/projects/gemm_tensorcore/gemm_tensorcore_runner}"

[[ -x "$runner" ]] || {
    printf 'test_wmma_single: runner 不存在或不可执行: %s\n' "$runner" >&2
    exit 1
}

patterns=(ones identity random)
passed=0
for pattern in "${patterns[@]}"; do
    printf '[test_wmma_single] input=%s\n' "$pattern"
    if output="$("$runner" --input "$pattern" --seed 1234 2>&1)"; then
        exit_code=0
    else
        exit_code=$?
    fi
    printf '%s\n' "$output"
    ((exit_code == 0))
    [[ $output == *"experiment=wmma-single"* ]]
    [[ $output == *"shape=16x16x16"* ]]
    [[ $output == *"input=${pattern}"* ]]
    [[ $output == *"status=PASS"* ]]
    ((passed += 1))
done

printf '[test_wmma_single] summary executions=%d pass=%d status=PASS\n' \
    "${#patterns[@]}" "$passed"
