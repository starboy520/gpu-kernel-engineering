#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../../.." && pwd)"
binary="${1:-$repo_root/build/projects/gemm_tensorcore/gemm_tensorcore_direct_runner}"
output_dir="$repo_root/projects/gemm_tensorcore/results/sass"
evidence="$repo_root/projects/gemm_tensorcore/results/evidence/wmma-direct-sass.md"

[[ -r "$binary" ]] || {
    printf 'binary 不可读: %s\n' "$binary" >&2
    exit 1
}
command -v cuobjdump >/dev/null || {
    printf '找不到 cuobjdump\n' >&2
    exit 1
}

mkdir -p "$output_dir" "$(dirname -- "$evidence")"
full="$output_dir/direct-full.sass"
target="$output_dir/wmma-direct.sass"
cuobjdump --dump-sass "$binary" > "$full"
awk '
    /^[[:space:]]*Function[[:space:]]*:/ {
        if (found) exit
        if (index($0, "wmma_direct_kernel")) found=1
    }
    found { print }
    END { if (!found) exit 42 }
' "$full" > "$target"

hmma="$(grep -E -c '[[:space:]]HMMA(\.[A-Z0-9_]+)*[[:space:]]' "$target" || true)"
ldg="$(grep -E -c '[[:space:]]LDG(\.[A-Z0-9_]+)*[[:space:]]' "$target" || true)"
stg="$(grep -E -c '[[:space:]]STG(\.[A-Z0-9_]+)*[[:space:]]' "$target" || true)"
movm="$(grep -E -c '[[:space:]]MOVM(\.[A-Z0-9_]+)*[[:space:]]' "$target" || true)"
ldl="$(grep -E -c '[[:space:]]LDL(\.[A-Z0-9_]+)*[[:space:]]' "$target" || true)"
stl="$(grep -E -c '[[:space:]]STL(\.[A-Z0-9_]+)*[[:space:]]' "$target" || true)"

{
    printf '# G1.5 Direct WMMA SASS\n\n'
    printf '> 静态 opcode 数量不等于 runtime 执行次数。Direct 版本使用 Global → fragment，不包含 Shared Memory 或异步复制。\n\n'
    printf '| Opcode | Count | 解释 |\n'
    printf '| --- | ---: | --- |\n'
    printf '| `HMMA` | %s | Tensor Core matrix multiply-accumulate |\n' "$hmma"
    printf '| `LDG` | %s | Global input load |\n' "$ldg"
    printf '| `STG` | %s | Global output store |\n' "$stg"
    printf '| `MOVM` | %s | Matrix operand register rearrangement |\n' "$movm"
    printf '| `LDL` | %s | Local-memory load / spill evidence |\n' "$ldl"
    printf '| `STL` | %s | Local-memory store / spill evidence |\n' "$stl"
} > "$evidence"

((hmma > 0)) || {
    printf 'Direct WMMA 未生成 HMMA。\n' >&2
    exit 1
}
((ldl == 0 && stl == 0)) || {
    printf 'Direct WMMA 检测到 local-memory load/store，请检查 spill。\n' >&2
    exit 1
}

printf '[wmma_direct_sass] wrote %s\n' "$evidence"
