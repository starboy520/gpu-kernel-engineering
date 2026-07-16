#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../../.." && pwd)"
runner="${1:-$repo_root/build/projects/gemm_tensorcore/gemm_tensorcore_runner}"

[[ -x "$runner" ]] || {
    printf 'gemm_tensorcore sanitize: runner 不存在: %s\n' "$runner" >&2
    exit 1
}

for tool in memcheck racecheck synccheck initcheck; do
    printf '[gemm_tensorcore_sanitize] tool=%s\n' "$tool"
    compute-sanitizer --tool "$tool" --error-exitcode=99 \
        "$runner" --input random --seed 1234
done

printf '[gemm_tensorcore_sanitize] summary tools=4 status=PASS\n'
