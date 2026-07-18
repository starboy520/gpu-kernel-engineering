#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../../.." && pwd)"
runner="${1:-$repo_root/build/projects/gemm_tensorcore/gemm_tensorcore_direct_runner}"

[[ -x "$runner" ]] || {
    printf 'wmma_direct sanitize: runner 不存在或不可执行: %s\n' "$runner" >&2
    exit 1
}

for tool_name in memcheck racecheck synccheck initcheck; do
    printf '[wmma_direct_sanitize] tool=%s\n' "$tool_name"
    compute-sanitizer --tool "$tool_name" --error-exitcode=99 \
        "$runner" --m 17 --n 19 --k 23 --input random --seed 1234
done

printf '[wmma_direct_sanitize] summary tools=4 status=PASS\n'
