#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../../.." && pwd)"
runner="${1:-$repo_root/build/projects/attention_prefill/attention_prefill_runner}"

[[ -x "$runner" ]] || {
    printf 'test_warp_per_query: runner 不存在或不可执行: %s\n' "$runner" >&2
    exit 1
}

n_values=(1 3 4 5 15 16 17 31 33)
d_values=(1 2 63 64 65 127 128)
causal_values=(0 1)

passed=0
for n in "${n_values[@]}"; do
    for d in "${d_values[@]}"; do
        for causal in "${causal_values[@]}"; do
            printf '[test_warp_per_query] shape=%sx%s causal=%s\n' \
                "$n" "$d" "$causal"
            if output="$("$runner" --implementation warp-per-query \
                --n "$n" --d "$d" --causal "$causal" 2>&1)"; then
                exit_code=0
            else
                exit_code=$?
            fi
            printf '%s\n' "$output"
            ((exit_code == 0))
            [[ $output == *"kernel=warp-per-query"* ]]
            [[ $output == *"shape=${n}x${d}"* ]]
            [[ $output == *"causal=${causal}"* ]]
            [[ $output == *"input_pattern=random"* ]]
            [[ $output == *"status=PASS"* ]]
            ((passed += 1))
        done
    done
done

special_executions=0
for input_pattern in zero-qk rising-logits; do
    printf '[test_warp_per_query] special shape=33x65 causal=1 input_pattern=%s\n' \
        "$input_pattern"
    output="$("$runner" --implementation warp-per-query \
        --n 33 --d 65 --causal 1 --input-pattern "$input_pattern" 2>&1)"
    printf '%s\n' "$output"
    [[ $output == *"kernel=warp-per-query"* ]]
    [[ $output == *"input_pattern=${input_pattern}"* ]]
    [[ $output == *"status=PASS"* ]]
    ((passed += 1))
    ((special_executions += 1))
done

printf '[test_warp_per_query] summary shape_executions=%d special_executions=%d pass=%d status=PASS\n' \
    "$(( ${#n_values[@]} * ${#d_values[@]} * ${#causal_values[@]} ))" \
    "$special_executions" "$passed"
