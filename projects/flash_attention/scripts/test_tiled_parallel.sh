#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../../.." && pwd)"
runner="${1:-$repo_root/build/projects/flash_attention/flash_attention_runner}"

[[ -x "$runner" ]] || {
    printf 'test_tiled_parallel: runner 不存在或不可执行: %s\n' "$runner" >&2
    exit 1
}

cases=(
    "1 1 0 random"
    "1 1 1 random"
    "3 2 0 random"
    "3 2 1 random"
    "17 127 0 random"
    "17 128 1 zero-qk"
    "33 128 1 negative-scores"
    "37 24 0 random"
    "37 24 1 random"
    "37 24 0 negative-scores"
    "37 24 1 zero-qk"
    "128 64 0 random"
    "128 64 1 random"
)

passed=0
for item in "${cases[@]}"; do
    read -r n d causal input_pattern <<< "$item"
    printf '[test_tiled_parallel] shape=%sx%s causal=%s input_pattern=%s\n' \
        "$n" "$d" "$causal" "$input_pattern"
    if output="$("$runner" --kernel tiled-parallel --n "$n" --d "$d" \
        --causal "$causal" --input-pattern "$input_pattern" \
        --mode validate --warmup 0 --iterations 1 2>&1)"; then
        exit_code=0
    else
        exit_code=$?
    fi
    printf '%s\n' "$output"
    ((exit_code == 0))
    [[ $output == *"kernel=tiled-parallel"* ]]
    [[ $output == *"path=tiled-parallel"* ]]
    [[ $output == *"shape=${n}x${d}"* ]]
    [[ $output == *"causal=${causal}"* ]]
    [[ $output == *"input_pattern=${input_pattern}"* ]]
    [[ $output == *"status=PASS"* ]]
    [[ $output == *"workspace_bytes=0"* ]]
    ((passed += 1))
done

printf '[test_tiled_parallel] summary executions=%d pass=%d status=PASS\n' \
    "${#cases[@]}" "$passed"
