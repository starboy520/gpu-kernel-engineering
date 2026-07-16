#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../../.." && pwd)"
binary="${1:-$repo_root/build/projects/gemm_tensorcore/gemm_tensorcore_runner}"
output_dir="$repo_root/projects/gemm_tensorcore/results/sass"
evidence="$repo_root/projects/gemm_tensorcore/results/evidence/wmma-single-sass.md"

[[ -r "$binary" ]] || { printf 'binary 不可读: %s\n' "$binary" >&2; exit 1; }
command -v cuobjdump >/dev/null || { printf '找不到 cuobjdump\n' >&2; exit 1; }

mkdir -p "$output_dir" "$(dirname -- "$evidence")"
full="$output_dir/full.sass"
target="$output_dir/wmma-single.sass"
cuobjdump --dump-sass "$binary" > "$full"
awk '
    /^[[:space:]]*Function[[:space:]]*:/ {
        if (found) exit
        if (index($0, "wmma_single_kernel")) found=1
    }
    found { print }
    END { if (!found) exit 42 }
' "$full" > "$target"

hmma="$(grep -E -c '[[:space:]]HMMA(\.[A-Z0-9_]+)*[[:space:]]' "$target" || true)"
ldl="$(grep -E -c '[[:space:]]LDL(\.[A-Z0-9_]+)*[[:space:]]' "$target" || true)"
stl="$(grep -E -c '[[:space:]]STL(\.[A-Z0-9_]+)*[[:space:]]' "$target" || true)"

{
    printf '# G1 WMMA Single Tile SASS\n\n'
    printf '> 静态 opcode 数量不等于 runtime 执行次数。\n\n'
    printf '| Opcode | Count |\n| --- | ---: |\n'
    printf '| `HMMA` | %s |\n' "$hmma"
    printf '| `LDL` | %s |\n' "$ldl"
    printf '| `STL` | %s |\n' "$stl"
} > "$evidence"

(( hmma > 0 )) || {
    printf 'G1 尚未生成 HMMA；请先完成 WMMA Kernel。\n' >&2
    exit 1
}
printf '[gemm_tensorcore_sass] wrote %s\n' "$evidence"
