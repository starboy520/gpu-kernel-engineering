#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../../.." && pwd)"

usage() { printf '用法: %s <tiled|tiled-async> [binary]\n' "${0##*/}" >&2; }
die() { printf 'extract_sass: %s\n' "$*" >&2; exit 1; }
count_opcode() { grep -E -c "$1" "$2" || true; }

[[ $# -ge 1 && $# -le 2 ]] || { usage; exit 2; }
kernel="$1"
case "$kernel" in
    tiled) function_name=tiled_attention_kernel ;;
    tiled-async) function_name=tiled_async_attention_kernel ;;
    *) die "只支持 tiled 或 tiled-async" ;;
esac
binary="${2:-$repo_root/build/projects/flash_attention/flash_attention_runner}"
command -v cuobjdump >/dev/null || die '找不到 cuobjdump'
[[ -r "$binary" ]] || die "binary 不可读: $binary"

sass_dir="$repo_root/projects/flash_attention/results/sass"
evidence_dir="$repo_root/projects/flash_attention/results/evidence"
sass_file="$sass_dir/${kernel}.sass"
evidence_file="$evidence_dir/${kernel}-sass.md"
mkdir -p "$sass_dir" "$evidence_dir"
full_sass="$(mktemp)"
trap 'rm -f "$full_sass"' EXIT
cuobjdump --dump-sass "$binary" > "$full_sass"
awk -v target="$function_name" '
    /^[[:space:]]*Function[[:space:]]*:/ {
        if (found) exit
        if (index($0, target)) found=1
    }
    found { print }
    END { if (!found) exit 42 }
' "$full_sass" > "$sass_file" || die "未找到函数: $function_name"

ldgsts="$(count_opcode '[[:space:]]LDGSTS(\.[A-Z0-9]+)*[[:space:]]' "$sass_file")"
ldgsts128="$(count_opcode '[[:space:]]LDGSTS\.E\.BYPASS\.128[[:space:]]' "$sass_file")"
arrives="$(count_opcode '[[:space:]]ARRIVES\.LDGSTSBAR(\.[A-Z0-9]+)*[[:space:]]' "$sass_file")"
ldl="$(count_opcode '[[:space:]]LDL(\.[A-Z0-9]+)*[[:space:]]' "$sass_file")"
stl="$(count_opcode '[[:space:]]STL(\.[A-Z0-9]+)*[[:space:]]' "$sass_file")"

{
    printf '# %s SASS 静态证据\n\n' "$kernel"
    printf '计数来自目标函数的静态 SASS，不代表运行时执行次数。\n\n'
    printf '| Opcode | 静态数量 |\n| --- | ---: |\n'
    printf '| `LDGSTS*` | %s |\n' "$ldgsts"
    printf '| `LDGSTS.E.BYPASS.128` | %s |\n' "$ldgsts128"
    printf '| `ARRIVES.LDGSTSBAR*` | %s |\n' "$arrives"
    printf '| `LDL*` | %s |\n' "$ldl"
    printf '| `STL*` | %s |\n\n' "$stl"
    printf '## 异步搬运片段\n\n```text\n'
    grep -E 'LDGSTS|LDGSTSBAR' "$sass_file" || true
    printf '```\n'
} > "$evidence_file"
printf '[extract_sass] wrote sass=%s evidence=%s\n' "$sass_file" "$evidence_file"
