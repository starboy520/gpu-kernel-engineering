#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../../.." && pwd)"
runner="${1:-$repo_root/build/projects/flash_attention/flash_attention_runner}"
renderer="$script_dir/render_results.py"
official_csv="$repo_root/projects/flash_attention/results/raw/a100-fp32.csv"
official_md="$repo_root/projects/flash_attention/results/generated/a100-fp32.md"
canonical_csv="${FA_OUTPUT_CSV:-$official_csv}"
generated_md="${FA_OUTPUT_MD:-$official_md}"

csv_header='timestamp,git_commit,runner_sha256,gpu,gpu_uuid,sm,driver,cuda,nvcc,build_preset,dtype,batch,heads,layout,kernel,path,n,d,causal,input_pattern,seed,warmup,iterations,repeats,latency_ms,latency_min_ms,latency_max_ms,spread_pct,passed,max_abs,max_rel,workspace_bytes,reference,timing'
default_shapes=(512x64 768x64 1024x64 512x128 768x128 1024x128)
default_causal=(0 1)
kernels=(naive tiled tiled-parallel tiled-async)
default_warmup=10
default_iterations=50
default_repeats=3
seed=1234

usage() {
    printf '用法: %s [runner]\n' "${0##*/}" >&2
    printf '覆盖变量: FA_SHAPES FA_CAUSAL FA_WARMUP FA_ITERATIONS FA_REPEATS FA_OUTPUT_CSV FA_OUTPUT_MD FA_BUILD_PRESET FA_ALLOW_DIRTY\n' >&2
}

die() {
    printf 'benchmark: %s\n' "$*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "找不到命令: $1"
}

normalize_path() {
    python3 - "$1" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve(strict=False))
PY
}

positive_int() { [[ $1 =~ ^[1-9][0-9]*$ ]]; }
nonnegative_int() { [[ $1 =~ ^[0-9]+$ ]]; }

parse_shape() {
    [[ $1 =~ ^([1-9][0-9]*)x([1-9][0-9]*)$ ]] || die "shape 必须为 NxD: $1"
    shape_n="${BASH_REMATCH[1]}"
    shape_d="${BASH_REMATCH[2]}"
}

extract_field() {
    local text="$1"
    local name="$2"
    local token
    for token in $text; do
        if [[ $token == "$name="* ]]; then
            printf '%s' "${token#*=}"
            return 0
        fi
    done
    return 1
}

[[ $# -le 1 ]] || { usage; exit 2; }
[[ -x "$runner" ]] || die "runner 不存在或不可执行: $runner"
[[ -f "$renderer" ]] || die "renderer 不存在: $renderer"
for command_name in git python3 nvidia-smi nvcc sha256sum; do require_command "$command_name"; done

read -r -a shapes <<< "${FA_SHAPES:-${default_shapes[*]}}"
read -r -a causal_modes <<< "${FA_CAUSAL:-${default_causal[*]}}"
warmup="${FA_WARMUP:-$default_warmup}"
iterations="${FA_ITERATIONS:-$default_iterations}"
repeats="${FA_REPEATS:-$default_repeats}"
official_csv="$(normalize_path "$official_csv")"
official_md="$(normalize_path "$official_md")"
canonical_csv="$(normalize_path "$canonical_csv")"
generated_md="$(normalize_path "$generated_md")"
nonnegative_int "$warmup" || die "FA_WARMUP 必须为非负整数"
positive_int "$iterations" || die "FA_ITERATIONS 必须为正整数"
positive_int "$repeats" || die "FA_REPEATS 必须为正整数"

for causal in "${causal_modes[@]}"; do
    [[ $causal == 0 || $causal == 1 ]] || die "FA_CAUSAL 只能包含 0/1"
done

official=1
[[ $warmup == "$default_warmup" && $iterations == "$default_iterations" && $repeats == "$default_repeats" ]] || official=0
[[ "${shapes[*]}" == "${default_shapes[*]}" && "${causal_modes[*]}" == "${default_causal[*]}" ]] || official=0

if (( ! official )); then
    if [[ -n ${FA_OUTPUT_CSV+x} ]]; then
        [[ $canonical_csv != "$official_csv" && $canonical_csv != "$official_md" ]] || \
            die 'smoke 不得写入任一 canonical 文件'
    else
        canonical_csv="$repo_root/projects/flash_attention/results/raw/smoke.csv"
    fi
    if [[ -n ${FA_OUTPUT_MD+x} ]]; then
        [[ $generated_md != "$official_csv" && $generated_md != "$official_md" ]] || \
            die 'smoke 不得写入任一 canonical 文件'
    else
        generated_md="$repo_root/projects/flash_attention/results/generated/smoke.md"
    fi
fi

if [[ -n $(git -C "$repo_root" status --short) ]]; then
    if (( official )); then
        die 'canonical benchmark 要求干净工作树'
    fi
    [[ ${FA_ALLOW_DIRTY:-0} == 1 ]] || die 'smoke benchmark 的工作树不干净；设置 FA_ALLOW_DIRTY=1 才可继续'
fi

expected_rows=$(( ${#shapes[@]} * ${#causal_modes[@]} * ${#kernels[@]} ))
timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
git_commit="$(git -C "$repo_root" rev-parse HEAD)"
runner_sha256="$(sha256sum "$runner" | awk '{print $1}')"
IFS=',' read -r gpu gpu_uuid sm driver <<< "$(nvidia-smi --query-gpu=name,uuid,compute_cap,driver_version --format=csv,noheader | head -n1 | sed 's/, /,/g')"
nvcc_output="$(nvcc --version)"
cuda_summary="$(printf '%s\n' "$nvcc_output" | tail -n1 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//;s/,/;/g')"
nvcc_summary="$(printf '%s' "$nvcc_output" | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g;s/[[:space:]]$//;s/,/;/g')"

mkdir -p "$(dirname -- "$canonical_csv")" "$(dirname -- "$generated_md")"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT
output_tmp="$tmp_dir/result.csv"
markdown_tmp="$tmp_dir/result.md"
printf '%s\n' "$csv_header" > "$output_tmp"

printf '[benchmark] mode=%s rows=%d runner_sha256=%s\n' "$([[ $official == 1 ]] && printf canonical || printf smoke)" "$expected_rows" "$runner_sha256"
printf '[benchmark] gpu=%s sm=%s commit=%s\n' "$gpu" "$sm" "$git_commit"

for shape in "${shapes[@]}"; do
    parse_shape "$shape"
    for causal in "${causal_modes[@]}"; do
        for kernel in "${kernels[@]}"; do
            latencies=()
            paths=()
            last_output=''
            for ((repeat_id=1; repeat_id<=repeats; ++repeat_id)); do
                printf '[benchmark] kernel=%s shape=%s causal=%s repeat=%d/%d\n' "$kernel" "$shape" "$causal" "$repeat_id" "$repeats"
                output="$("$runner" --kernel "$kernel" --n "$shape_n" --d "$shape_d" --causal "$causal" --input-pattern random --mode benchmark --warmup "$warmup" --iterations "$iterations" --seed "$seed")" || die "runner 失败: $kernel $shape causal=$causal"
                printf '%s\n' "$output"
                [[ $(extract_field "$output" status) == PASS ]] || die "correctness 未通过"
                latency="$(extract_field "$output" latency_ms)" || die '缺少 latency_ms'
                path="$(extract_field "$output" path)" || die '缺少 path'
                latencies+=("$latency")
                paths+=("$path")
                last_output="$output"
            done
            first_path="${paths[0]}"
            for path in "${paths[@]}"; do [[ $path == "$first_path" ]] || die "重复运行 path 不一致"; done
            if [[ $kernel == tiled-async && $shape_d =~ ^(64|128)$ ]]; then
                [[ $first_path == fast-pipeline-16b ]] || die "canonical Async 命中 fallback: $shape"
            fi
            stats="$(python3 - "${latencies[@]}" <<'PY'
import statistics, sys
values = [float(value) for value in sys.argv[1:]]
median = statistics.median(values)
minimum = min(values)
maximum = max(values)
spread = 100.0 * (maximum - minimum) / median
print(f"{median:.6f} {minimum:.6f} {maximum:.6f} {spread:.6f}")
PY
)"
            read -r median_ms minimum_ms maximum_ms spread_pct <<< "$stats"
            max_abs="$(extract_field "$last_output" max_abs)"
            max_rel="$(extract_field "$last_output" max_rel)"
            workspace_bytes="$(extract_field "$last_output" workspace_bytes)"
            printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,fp32,1,1,row-major,%s,%s,%s,%s,%s,random,%s,%s,%s,%s,%s,%s,%s,%s,true,%s,%s,%s,cpu-double,cuda-event\n' \
                "$timestamp" "$git_commit" "$runner_sha256" "$gpu" "$gpu_uuid" "$sm" "$driver" "$cuda_summary" "$nvcc_summary" \
                "${FA_BUILD_PRESET:-release-sm80}" \
                "$kernel" "$first_path" "$shape_n" "$shape_d" "$causal" "$seed" "$warmup" "$iterations" "$repeats" \
                "$median_ms" "$minimum_ms" "$maximum_ms" "$spread_pct" "$max_abs" "$max_rel" "$workspace_bytes" >> "$output_tmp"
        done
    done
done

actual_rows="$(($(wc -l < "$output_tmp") - 1))"
[[ $actual_rows == "$expected_rows" ]] || die "结果行数错误: expected=$expected_rows actual=$actual_rows"
python3 "$renderer" "$output_tmp" "$markdown_tmp" "$canonical_csv"
mv "$output_tmp" "$canonical_csv"
mv "$markdown_tmp" "$generated_md"
printf '[benchmark] wrote rows=%s csv=%s markdown=%s\n' "$actual_rows" "$canonical_csv" "$generated_md"
