#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../../.." && pwd)"
binary="$repo_root/build/projects/attention_prefill/attention_prefill_evidence_runner"
output_dir="$repo_root/projects/attention_prefill/results/sass/m2"
evidence="$repo_root/projects/attention_prefill/results/evidence/m2-sass-development.md"

[[ -x "$binary" ]] || { printf 'binary 不存在: %s\n' "$binary" >&2; exit 1; }
command -v cuobjdump >/dev/null || { printf '找不到 cuobjdump\n' >&2; exit 1; }

mkdir -p "$output_dir" "$(dirname -- "$evidence")"
full="$output_dir/full.sass"
cuobjdump --dump-sass "$binary" > "$full"

extract_symbol() {
    local symbol="$1"
    local destination="$2"
    awk -v target="$symbol" '
        /^[[:space:]]*Function[[:space:]]*:/ {
            if (found) exit
            if (index($0, target)) found=1
        }
        found { print }
        END { if (!found) exit 42 }
    ' "$full" > "$destination"
    [[ -s "$destination" ]] || { printf '未找到函数: %s\n' "$symbol" >&2; exit 1; }
}

extract_symbol query_tiled_kernel "$output_dir/m1-query-tiled.sass"
extract_symbol warp_per_query_kernel "$output_dir/m2-warp-per-query.sass"

count_opcode() {
    local opcode="$1"
    local file="$2"
    grep -E -c "[[:space:]]${opcode}(\.[A-Z0-9_]+)*[[:space:]]" "$file" || true
}

opcodes=(FFMA SHFL BAR LDG STG LDS STS LDL STL HMMA LDGSTS)
{
    printf '# M1 Query-tiled vs M2 Warp-per-query SASS（开发期）\n\n'
    printf '> 静态 opcode 数量不等于 runtime 执行次数；正式发布前需在 clean commit 上重采。\n\n'
    printf '| Opcode | M1 | M2 |\n| --- | ---: | ---: |\n'
    for opcode in "${opcodes[@]}"; do
        printf '| `%s` | %s | %s |\n' "$opcode" \
            "$(count_opcode "$opcode" "$output_dir/m1-query-tiled.sass")" \
            "$(count_opcode "$opcode" "$output_dir/m2-warp-per-query.sass")"
    done
} > "$evidence"

printf '[extract_m2_sass] wrote %s\n' "$evidence"
