#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../../.." && pwd)"
runner="${1:-$repo_root/build/projects/gemm/gemm_runner}"
render_script="$repo_root/projects/gemm/scripts/render_results.py"
canonical_csv="${GEMM_OUTPUT_CSV:-$repo_root/projects/gemm/results/raw/a100-fp32.csv}"
generated_md="${GEMM_OUTPUT_MD:-$repo_root/projects/gemm/results/generated/a100-fp32.md}"

csv_header='timestamp,git_commit,gpu,cuda,nvcc,kernel,path,m,n,k,warmup,iterations,latency_ms,gflops,passed,max_abs,max_rel,reference'
default_shapes=(512x512x512 1024x1024x1024 2048x2048x2048)
kernels=(naive shared register vectorized async-16b cublas-fp32)
default_warmup=10
default_iterations=50
default_repeats=3
seed=1234

die() {
    printf 'benchmark: %s\n' "$*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "找不到命令: $1"
}

parse_positive_int() {
    local value="$1"
    local name="$2"
    [[ $value =~ ^[1-9][0-9]*$ ]] || die "$name 必须是正整数，当前值: $value"
}

parse_nonnegative_int() {
    local value="$1"
    local name="$2"
    [[ $value =~ ^[0-9]+$ ]] || die "$name 必须是非负整数，当前值: $value"
}

shape_is_official() {
    local candidate="$1"
    local item
    for item in "${default_shapes[@]}"; do
        [[ $candidate == "$item" ]] && return 0
    done
    return 1
}

parse_shape() {
    local shape="$1"
    [[ $shape =~ ^([1-9][0-9]*)x([1-9][0-9]*)x([1-9][0-9]*)$ ]] || \
        die "shape 格式必须为 MxNxK，当前值: $shape"
    shape_m="${BASH_REMATCH[1]}"
    shape_n="${BASH_REMATCH[2]}"
    shape_k="${BASH_REMATCH[3]}"
}

extract_trial_info() {
    local trial_csv="$1"
    python3 - "$trial_csv" <<'PY'
import csv
import sys

path = sys.argv[1]
with open(path, newline="", encoding="utf-8") as handle:
    raw_lines = [line.rstrip("\n") for line in handle]
with open(path, newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))

if len(rows) != 1:
    raise SystemExit(f"expected exactly one CSV data row, found {len(rows)}")
if len(raw_lines) < 2:
    raise SystemExit("expected CSV header plus one data row")

row = rows[0]
print(row["latency_ms"])
print(row["path"])
print(row["reference"])
print(raw_lines[-1])
PY
}

select_median_row() {
    local rows_csv="$1"
    python3 - "$rows_csv" <<'PY'
import csv
import sys

path = sys.argv[1]
with open(path, newline="", encoding="utf-8") as handle:
    raw_lines = [line.rstrip("\n") for line in handle]
with open(path, newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))

data_lines = raw_lines[1:]
if len(rows) != len(data_lines):
    raise SystemExit("CSV raw/data row count mismatch")
if not rows:
    raise SystemExit("median selection requires at least one row")

ranked = sorted(
    (float(row["latency_ms"]), index, data_lines[index])
    for index, row in enumerate(rows)
)
median = ranked[len(ranked) // 2]
print(f"{median[0]:.6f}")
print(median[2])
PY
}

[[ -x "$runner" ]] || die "runner 不存在或不可执行: $runner"
[[ -f "$render_script" ]] || die "render_results.py 不存在: $render_script"

require_command git
require_command python3
require_command nvidia-smi
require_command nvcc

read -r -a shapes <<< "${GEMM_SHAPES:-${default_shapes[*]}}"
warmup="${GEMM_WARMUP:-$default_warmup}"
iterations="${GEMM_ITERATIONS:-$default_iterations}"
repeats="${GEMM_REPEATS:-$default_repeats}"

(( ${#shapes[@]} > 0 )) || die '至少需要一个 shape'
parse_nonnegative_int "$warmup" GEMM_WARMUP
parse_positive_int "$iterations" GEMM_ITERATIONS
parse_positive_int "$repeats" GEMM_REPEATS

official_protocol=1
if [[ $warmup != "$default_warmup" || $iterations != "$default_iterations" || $repeats != "$default_repeats" ]]; then
    official_protocol=0
fi
if (( ${#shapes[@]} != ${#default_shapes[@]} )); then
    official_protocol=0
else
    for index in "${!default_shapes[@]}"; do
        if [[ ${shapes[$index]} != "${default_shapes[$index]}" ]]; then
            official_protocol=0
            break
        fi
    done
fi

if (( ! official_protocol )); then
    if [[ -z ${GEMM_OUTPUT_CSV+x} ]]; then
        canonical_csv="$repo_root/projects/gemm/results/raw/smoke.csv"
    fi
    if [[ -z ${GEMM_OUTPUT_MD+x} ]]; then
        generated_md="$repo_root/projects/gemm/results/generated/smoke.md"
    fi
fi

if [[ ${GEMM_ALLOW_DIRTY:-0} != 1 ]]; then
    if [[ -n $(git -C "$repo_root" status --short) ]]; then
        die 'benchmark 需要干净工作树；请先提交/暂存修改，或显式设置 GEMM_ALLOW_DIRTY=1 覆盖'
    fi
fi

timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
git_commit="$(git -C "$repo_root" rev-parse HEAD)"
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | tr -d '\r')"
nvcc_output="$(nvcc --version)"
cuda_summary="$(printf '%s\n' "$nvcc_output" | tail -n1 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
nvcc_summary="$(printf '%s' "$nvcc_output" | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g;s/[[:space:]]$//')"

mkdir -p "$(dirname -- "$canonical_csv")" "$(dirname -- "$generated_md")"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT
canonical_tmp="$tmp_dir/canonical.csv"
printf '%s\n' "$csv_header" > "$canonical_tmp"

printf '[benchmark] mode=%s runner=%s csv=%s markdown=%s\n' \
    "$([[ $official_protocol == 1 ]] && printf official || printf smoke)" \
    "$runner" "$canonical_csv" "$generated_md"
printf '[benchmark] metadata timestamp=%s git_commit=%s gpu=%s\n' \
    "$timestamp" "$git_commit" "$gpu_name"

for shape in "${shapes[@]}"; do
    parse_shape "$shape"
    rows_csv_base="$tmp_dir/${shape//x/_}"

    for kernel in "${kernels[@]}"; do
        rows_csv="$rows_csv_base.$kernel.csv"
        printf '%s\n' "$csv_header" > "$rows_csv"

        for ((repeat_index = 1; repeat_index <= repeats; ++repeat_index)); do
            trial_csv="$tmp_dir/${shape//x/_}.${kernel}.repeat${repeat_index}.csv"
            rm -f "$trial_csv"

            printf '[benchmark] kernel=%s shape=%s repeat=%d/%d warmup=%s iterations=%s\n' \
                "$kernel" "$shape" "$repeat_index" "$repeats" "$warmup" "$iterations"

            if ! output="$(
                GEMM_BENCH_TIMESTAMP="$timestamp" \
                GEMM_GIT_COMMIT="$git_commit" \
                GEMM_GPU="$gpu_name" \
                GEMM_CUDA="$cuda_summary" \
                GEMM_NVCC="$nvcc_summary" \
                "$runner" \
                    --kernel "$kernel" \
                    --m "$shape_m" \
                    --n "$shape_n" \
                    --k "$shape_k" \
                    --mode benchmark \
                    --warmup "$warmup" \
                    --iterations "$iterations" \
                    --seed "$seed" \
                    --csv "$trial_csv" 2>&1
            )"; then
                printf '%s\n' "$output"
                die "runner 失败: kernel=$kernel shape=$shape repeat=$repeat_index"
            fi
            printf '%s\n' "$output"

            mapfile -t trial_info < <(extract_trial_info "$trial_csv")
            (( ${#trial_info[@]} == 4 )) || die "trial CSV 解析失败: $trial_csv"
            trial_latency="${trial_info[0]}"
            trial_path="${trial_info[1]}"
            trial_reference="${trial_info[2]}"
            trial_row="${trial_info[3]}"

            if shape_is_official "$shape" && [[ $kernel == vectorized || $kernel == async-16b ]]; then
                [[ $trial_path != fallback-* ]] || \
                    die "官方 shape 命中了 fallback 路径: kernel=$kernel shape=$shape path=$trial_path"
            fi

            printf '[benchmark] row kernel=%s shape=%s latency_ms=%s path=%s reference=%s\n' \
                "$kernel" "$shape" "$trial_latency" "$trial_path" "$trial_reference"
            printf '%s\n' "$trial_row" >> "$rows_csv"
        done

        mapfile -t median_info < <(select_median_row "$rows_csv")
        (( ${#median_info[@]} == 2 )) || die "中位数选择失败: kernel=$kernel shape=$shape"
        printf '[benchmark] median kernel=%s shape=%s latency_ms=%s\n' \
            "$kernel" "$shape" "${median_info[0]}"
        printf '%s\n' "${median_info[1]}" >> "$canonical_tmp"
    done
done

mv "$canonical_tmp" "$canonical_csv"
python3 "$render_script" "$canonical_csv" "$generated_md"

printf '[benchmark] wrote csv=%s markdown=%s\n' "$canonical_csv" "$generated_md"