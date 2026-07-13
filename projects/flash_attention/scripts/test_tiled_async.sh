#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../../.." && pwd)"
runner="${1:-$repo_root/build/projects/flash_attention/flash_attention_runner}"

[[ -x "$runner" ]] || {
    printf 'test_tiled_async: runner 不存在或不可执行: %s\n' "$runner" >&2
    exit 1
}

# n d causal input_pattern expected_path
cases=(
    "1 1 0 random fallback-tiled"
    "1 4 1 random fast-pipeline-16b"
    "3 2 1 random fallback-tiled"
    "3 4 0 random fast-pipeline-16b"
    "17 127 0 random fallback-tiled"
    "17 128 1 zero-qk fast-pipeline-16b"
    "33 128 1 negative-scores fast-pipeline-16b"
    "37 24 0 random fast-pipeline-16b"
    "37 24 1 random fast-pipeline-16b"
    "37 24 0 negative-scores fast-pipeline-16b"
    "37 24 1 zero-qk fast-pipeline-16b"
    "128 64 0 random fast-pipeline-16b"
    "128 64 1 random fast-pipeline-16b"
)

passed=0
for item in "${cases[@]}"; do
    read -r n d causal input_pattern expected_path <<< "$item"
    printf '[test_tiled_async] shape=%sx%s causal=%s input_pattern=%s expected_path=%s\n' \
        "$n" "$d" "$causal" "$input_pattern" "$expected_path"
    if output="$("$runner" --kernel tiled-async --n "$n" --d "$d" \
        --causal "$causal" --input-pattern "$input_pattern" \
        --mode validate --warmup 0 --iterations 1 2>&1)"; then
        exit_code=0
    else
        exit_code=$?
    fi
    printf '%s\n' "$output"
    ((exit_code == 0))
    [[ $output == *"kernel=tiled-async"* ]]
    [[ $output == *"path=${expected_path}"* ]]
    [[ $output == *"shape=${n}x${d}"* ]]
    [[ $output == *"causal=${causal}"* ]]
    [[ $output == *"input_pattern=${input_pattern}"* ]]
    [[ $output == *"status=PASS"* ]]
    [[ $output == *"workspace_bytes=0"* ]]
    ((passed += 1))
done

printf '[test_tiled_async] summary executions=%d pass=%d status=PASS\n' \
    "${#cases[@]}" "$passed"