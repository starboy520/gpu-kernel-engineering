#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../../.." && pwd)"
runner="${1:-$repo_root/build/projects/gemm_tensorcore/gemm_tensorcore_direct_runner}"

[[ -x "$runner" ]] || {
    printf 'test_wmma_direct: runner 不存在或不可执行: %s\n' "$runner" >&2
    exit 1
}

cases=(
    '16 16 16'
    '16 16 32'
    '16 16 48'
    '32 16 16'
    '16 32 16'
    '32 32 32'
    '17 19 23'
    '31 33 47'
)
patterns=(ones identity random)
passed=0
executions=0

for case_spec in "${cases[@]}"; do
    read -r m n k <<<"$case_spec"
    for pattern in "${patterns[@]}"; do
        ((executions += 1))
        printf '[test_wmma_direct] shape=%sx%sx%s input=%s\n' \
            "$m" "$n" "$k" "$pattern"
        if output="$($runner --m "$m" --n "$n" --k "$k" \
            --input "$pattern" --seed 1234 2>&1)"; then
            exit_code=0
        else
            exit_code=$?
        fi
        printf '%s\n' "$output"
        ((exit_code == 0))
        [[ $output == *"experiment=wmma-direct"* ]]
        [[ $output == *"shape=${m}x${n}x${k}"* ]]
        [[ $output == *"layout=A.row,B.col,C.row"* ]]
        [[ $output == *"input=${pattern}"* ]]
        [[ $output == *"status=PASS"* ]]
        ((passed += 1))
    done
done

printf '[test_wmma_direct] summary executions=%d pass=%d status=PASS\n' \
    "$executions" "$passed"
