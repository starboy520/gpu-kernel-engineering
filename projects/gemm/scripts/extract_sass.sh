#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../../.." && pwd)"

usage() {
    printf '用法: %s <kernel> [binary]\n' "${0##*/}" >&2
    printf 'kernel: naive | shared | register | vectorized | async-16b\n' >&2
}

die() {
    printf 'extract_sass: %s\n' "$*" >&2
    exit 1
}

count_opcode() {
    local pattern="$1"
    local input="$2"
    local count
    count="$(grep -E -c "$pattern" "$input" || true)"
    printf '%s' "$count"
}

[[ $# -ge 1 && $# -le 2 ]] || {
    usage
    exit 2
}

kernel="$1"
case "$kernel" in
    naive) function_name=naive_kernel ;;
    shared) function_name=shared_tiled_kernel ;;
    register) function_name=register_tiled_kernel ;;
    vectorized) function_name=vectorized_tiled_kernel ;;
    async-16b) function_name=double_buffer_kernel ;;
    *) die "不支持的 kernel: $kernel" ;;
esac

binary="${2:-$repo_root/build/projects/gemm/gemm_runner}"
command -v cuobjdump >/dev/null 2>&1 || die '找不到 cuobjdump'
[[ -r "$binary" ]] || die "binary 不存在或不可读: $binary"

sass_dir="$repo_root/projects/gemm/results/sass"
evidence_dir="$repo_root/projects/gemm/results/evidence"
sass_file="$sass_dir/${kernel}.sass"
evidence_file="$evidence_dir/${kernel}-sass.md"
mkdir -p "$sass_dir" "$evidence_dir"
full_sass="$(mktemp "$sass_dir/.cuobjdump.XXXXXX.sass")"
function_tmp="$(mktemp "$sass_dir/.function.XXXXXX.sass")"
cuobjdump_error="$(mktemp "$sass_dir/.cuobjdump.XXXXXX.err")"
trap 'rm -f "$full_sass" "$function_tmp" "$cuobjdump_error"' EXIT

if ! cuobjdump --dump-sass "$binary" > "$full_sass" 2>"$cuobjdump_error"; then
    cat "$cuobjdump_error" >&2
    die "cuobjdump 执行失败: $binary"
fi
if ! awk -v target="$function_name" '
    /^[[:space:]]*Function[[:space:]]*:/ {
        if (found) {
            exit
        }
        if (index($0, target) != 0) {
            found = 1
        }
    }
    found { print }
    END { if (!found) exit 42 }
' "$full_sass" > "$function_tmp"; then
    die "未找到目标函数: $function_name"
fi
[[ -s "$function_tmp" ]] || die "未找到目标函数: $function_name"
mv "$function_tmp" "$sass_file"

ldg_count="$(count_opcode '[[:space:]]LDG(\.[A-Z0-9]+)*[[:space:]]' "$sass_file")"
ldg_128_count="$(count_opcode '[[:space:]]LDG\.E\.128[[:space:]]' "$sass_file")"
ldgsts_count="$(count_opcode '[[:space:]]LDGSTS(\.[A-Z0-9]+)*[[:space:]]' "$sass_file")"
ldgsts_128_count="$(count_opcode '[[:space:]]LDGSTS\.E\.BYPASS\.128[[:space:]]' "$sass_file")"
ldgstsbar_count="$(count_opcode '(^|[.])LDGSTSBAR(\.[A-Z0-9]+)*[[:space:]]' "$sass_file")"
ffma_count="$(count_opcode '[[:space:]]FFMA[[:space:]]' "$sass_file")"
ldl_count="$(count_opcode '[[:space:]]LDL(\.[A-Z0-9]+)*[[:space:]]' "$sass_file")"
stl_count="$(count_opcode '[[:space:]]STL(\.[A-Z0-9]+)*[[:space:]]' "$sass_file")"

{
    printf '# %s SASS 静态证据\n\n' "$kernel"
    printf '由 `projects/gemm/scripts/extract_sass.sh %s` 从当前构建的 `gemm_runner` 生成。计数是目标函数内的静态指令数，不代表运行时执行次数。\n\n' "$kernel"
    printf '| Opcode | 静态数量 |\n'
    printf '| --- | ---: |\n'
    printf '| `LDG*` | %s |\n' "$ldg_count"
    printf '| `LDG.E.128` | %s |\n' "$ldg_128_count"
    printf '| `LDGSTS*` | %s |\n' "$ldgsts_count"
    printf '| `LDGSTS.E.BYPASS.128` | %s |\n' "$ldgsts_128_count"
    printf '| `LDGSTSBAR*` | %s |\n' "$ldgstsbar_count"
    printf '| `FFMA` | %s |\n' "$ffma_count"
    printf '| `LDL*` | %s |\n' "$ldl_count"
    printf '| `STL*` | %s |\n\n' "$stl_count"
    printf '`LDL`/`STL` 是 local-memory spill 的静态风险信号；最终仍需结合 ptxas 和 profiler 判断。\n\n'
    printf '## 宽加载与异步搬运片段\n\n'
    printf '```text\n'
    grep -E '([[:space:]]LDG\.E\.128|[[:space:]]LDGSTS|[.]LDGSTSBAR)(\.[A-Z0-9]+)*[[:space:]]' "$sass_file" || true
    printf '```\n'
} > "$evidence_file"

printf '[extract_sass] kernel=%s function=%s\n' "$kernel" "$function_name"
printf '[extract_sass] wrote sass=%s evidence=%s\n' "$sass_file" "$evidence_file"