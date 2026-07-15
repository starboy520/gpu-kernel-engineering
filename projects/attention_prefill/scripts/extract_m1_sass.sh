#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${AP_M1_REPO_ROOT:-$(cd -- "$script_dir/../../.." && pwd)}"
repo_root="$(cd -- "$repo_root" && pwd)"
canonical_binary="$repo_root/build/projects/attention_prefill/attention_prefill_evidence_runner"
binary="${1:-${AP_M1_EVIDENCE_RUNNER:-$canonical_binary}}"
fingerprint_script="$script_dir/source_fingerprint.py"
attestation_validator="$script_dir/validate_build_attestation.py"
build_attestation="$repo_root/build/projects/attention_prefill/attention_prefill_build_attestation.txt"

usage() { printf '用法: %s [evidence-runner-binary]\n' "${0##*/}" >&2; }
die() { printf 'extract_m1_sass: %s\n' "$*" >&2; exit 1; }
count_opcode() {
    local opcode="$1"
    local file_path="$2"
    grep -E -c "[[:space:]]${opcode}(\\.[A-Z0-9_]+)*[[:space:]]" "$file_path" || true
}
extract_function() {
    local symbol="$1"
    local expected_signature="$2"
    local destination="$3"
    local header_symbol demangled candidate_count=0 candidate= basename_match=0
    while IFS= read -r header_symbol; do
        demangled="$(c++filt -- "$header_symbol")"
        if [[ $demangled =~ (^|::)${symbol}\( ]]; then
            ((basename_match += 1))
        fi
        if [[ $demangled == "$expected_signature" || $demangled == *"::$expected_signature" ]]; then
            candidate="$header_symbol"
            ((candidate_count += 1))
        fi
    done < <(
        sed -nE 's/^[[:space:]]*Function[[:space:]]*:[[:space:]]*([^[:space:]]+).*/\1/p' \
            "$full_sass"
    )
    [[ $candidate_count == 1 ]] || \
        die "函数 $symbol 精确签名候选数量必须为 1，实际为 $candidate_count（同 basename=$basename_match）"
    awk -v target="$candidate" '
        /^[[:space:]]*Function[[:space:]]*:/ {
            if (found) exit
            header=$0
            sub(/^[[:space:]]*Function[[:space:]]*:[[:space:]]*/, "", header)
            sub(/[[:space:]].*$/, "", header)
            if (header == target) found=1
        }
        found { print }
        END { if (!found) exit 42 }
    ' "$full_sass" > "$destination" || die "未找到精确函数 header: $candidate"
    [[ -s $destination ]] || die "目标函数 SASS 为空: $symbol"
}
runner_field() {
    local name="$1"
    local token
    for token in $runner_metadata; do
        if [[ $token == "$name="* ]]; then
            printf '%s' "${token#*=}"
            return 0
        fi
    done
    return 1
}

[[ $# -le 1 ]] || { usage; exit 2; }
command -v cuobjdump >/dev/null 2>&1 || die '找不到 cuobjdump'
command -v c++filt >/dev/null 2>&1 || die '找不到 c++filt'
command -v sha256sum >/dev/null 2>&1 || die '找不到 sha256sum'
command -v git >/dev/null 2>&1 || die '找不到 git'
[[ -f $binary && ! -L $binary && -x $binary ]] || \
    die "binary 必须是可执行普通文件且不能是 symlink: $binary"

canonical_binary="$(python3 - "$canonical_binary" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve(strict=False))
PY
)"
binary="$(python3 - "$binary" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve(strict=False))
PY
)"
if [[ $# == 0 && ! -v AP_M1_EVIDENCE_RUNNER && $binary == "$canonical_binary" ]]; then
    evidence_kind=canonical
    sass_dir="$repo_root/projects/attention_prefill/results/sass"
    evidence_dir="$repo_root/projects/attention_prefill/results/evidence"
    evidence_file="$evidence_dir/m1-sass.md"
else
    [[ -v AP_M1_SASS_OUTPUT_DIR ]] || \
        die '自定义 binary/override 为 smoke，必须设置 AP_M1_SASS_OUTPUT_DIR'
    sass_dir="$(python3 - "${AP_M1_SASS_OUTPUT_DIR}" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve(strict=False))
PY
)"
    official_sass="$repo_root/projects/attention_prefill/results/sass"
    official_evidence="$repo_root/projects/attention_prefill/results/evidence"
    [[ ${sass_dir,,} == *smoke* && $sass_dir != "$official_sass" && $sass_dir != "$official_evidence" ]] || \
        die 'smoke SASS 输出目录必须包含 smoke 且不能是 official sass/evidence'
    evidence_kind=smoke
    evidence_dir="$sass_dir"
    evidence_file="$evidence_dir/m1-sass-smoke.md"
fi

runner_metadata="$("$binary" --metadata-only 2>&1)" || {
    printf '%s\n' "$runner_metadata" >&2
    die 'binary --metadata-only 失败'
}
source_sha256="$(runner_field source_sha256)" || die 'metadata 缺少 source_sha256'
build_contract="$(runner_field build_contract)" || die 'metadata 缺少 build_contract'
build_payload_sha256="$(runner_field build_contract_payload_sha256)" || \
    die 'metadata 缺少 build_contract_payload_sha256'
device_index="$(runner_field device_index)" || die 'metadata 缺少 device_index'
gpu_uuid="$(runner_field gpu_uuid)" || die 'metadata 缺少 gpu_uuid'
gpu_name="$(runner_field gpu_name)" || die 'metadata 缺少 gpu_name'
sm="$(runner_field sm)" || die 'metadata 缺少 sm'
driver="$(runner_field driver)" || die 'metadata 缺少 driver'
[[ $source_sha256 =~ ^[0-9a-f]{64}$ ]] || die 'metadata source_sha256 无效'
[[ $build_contract =~ ^release-sm80-[0-9a-f]{16}$ ]] || \
    die 'metadata build_contract 无效'
[[ $build_payload_sha256 =~ ^[0-9a-f]{64}$ ]] || \
    die 'metadata build_contract_payload_sha256 无效'
[[ ${build_contract##*-} == "${build_payload_sha256:0:16}" ]] || \
    die 'metadata build_contract 与 payload SHA-256 不一致'
[[ $device_index =~ ^[0-9]+$ ]] || die 'metadata device_index 无效'
[[ $gpu_uuid =~ ^GPU-[0-9A-Fa-f-]+$ ]] || die 'metadata gpu_uuid 无效'
[[ $gpu_name =~ ^[^[:space:],]+$ ]] || die 'metadata gpu_name 无效'
[[ $sm =~ ^[0-9]+\.[0-9]+$ ]] || die 'metadata sm 无效'
[[ $driver =~ ^[0-9]+$ ]] || die 'metadata driver 无效'
if [[ $evidence_kind == canonical ]]; then
    [[ -z $(git -C "$repo_root" status --short) ]] || \
        die 'canonical SASS 要求干净工作树；dirty 提取必须显式使用 smoke 输出'
    [[ $gpu_name == *A100* && $gpu_name == *80GB* && $sm == 8.0 ]] || \
        die "canonical SASS 要求 A100 80GB sm=8.0: gpu=$gpu_name sm=$sm"
    current_source_sha256="$(python3 "$fingerprint_script" --repo-root "$repo_root")" || \
        die '无法计算当前 source fingerprint'
    [[ $source_sha256 == "$current_source_sha256" ]] || \
        die 'canonical binary embedded source fingerprint 与当前 source 不一致'
    [[ -f $build_attestation && ! -L $build_attestation ]] || \
        die 'canonical build attestation 必须是非 symlink 普通文件'
    attestation_output="$(python3 "$attestation_validator" "$build_attestation" 2>&1)" || {
        printf '%s\n' "$attestation_output" >&2
        die 'canonical build attestation 验证失败'
    }
    attestation_line="$(tr '\n' ' ' <<< "$attestation_output")"
    for token in $attestation_line; do
        case "$token" in
            build_contract=*) attested_contract="${token#*=}" ;;
            build_contract_payload_sha256=*) attested_payload_sha256="${token#*=}" ;;
        esac
    done
    [[ ${attested_contract:-} == "$build_contract" && \
       ${attested_payload_sha256:-} == "$build_payload_sha256" ]] || \
        die 'canonical build attestation 与 binary metadata 不一致'
fi
git_commit="$(git -C "$repo_root" rev-parse HEAD)"

mkdir -p "$sass_dir" "$evidence_dir"
tmp_dir="$(mktemp -d "$sass_dir/.sass-transaction.XXXXXX")"
trap 'rm -rf "$tmp_dir"' EXIT
full_sass="$tmp_dir/full.sass"
br1_tmp="$tmp_dir/br1.sass"
br4_tmp="$tmp_dir/br4.sass"
evidence_tmp="$tmp_dir/${evidence_file##*/}"
cuobjdump --dump-sass "$binary" > "$full_sass"
signature_suffix='(float const*, float const*, float const*, float*, int, int, bool)'
extract_function tiled_attention_kernel "tiled_attention_kernel${signature_suffix}" "$br1_tmp"
extract_function query_tiled_kernel "query_tiled_kernel${signature_suffix}" "$br4_tmp"
cmp -s "$br1_tmp" "$br4_tmp" && \
    die 'br1 与 br4 SASS 提取结果不能相同'

opcodes=(FFMA BAR LDG STG LDS STS LDL STL HMMA LDGSTS)
declare -A br1_counts br4_counts
for opcode in "${opcodes[@]}"; do
    if [[ $opcode == BAR ]]; then
        br1_counts[$opcode]="$(count_opcode 'BAR' "$br1_tmp")"
        br4_counts[$opcode]="$(count_opcode 'BAR' "$br4_tmp")"
    else
        br1_counts[$opcode]="$(count_opcode "$opcode" "$br1_tmp")"
        br4_counts[$opcode]="$(count_opcode "$opcode" "$br4_tmp")"
    fi
done

(( br1_counts[FFMA] > 0 )) || die 'br1 FFMA 必须大于 0'
(( br4_counts[FFMA] > 0 )) || die 'br4 FFMA 必须大于 0'
for implementation in br1 br4; do
    declare -n counts="${implementation}_counts"
    (( counts[HMMA] == 0 )) || die "$implementation M1 合同违反: HMMA=${counts[HMMA]}"
    (( counts[LDGSTS] == 0 )) || die "$implementation M1 合同违反: LDGSTS=${counts[LDGSTS]}"
done

binary_sha256="$(sha256sum "$binary" | awk '{print $1}')"
[[ $binary_sha256 =~ ^[0-9a-f]{64}$ ]] || die 'binary SHA-256 无效'

spill_warning='Spill warning: none'
if (( br1_counts[LDL] > 0 || br1_counts[STL] > 0 || br4_counts[LDL] > 0 || br4_counts[STL] > 0 )); then
    spill_warning='SPILL WARNING: 发现 LDL/STL；M1 不强制为 0，但需结合 ptxas 与 runtime 指标调查 local-memory spill。'
fi

binary_display="$binary"
if [[ $binary == "$repo_root/"* ]]; then
    binary_display="${binary#"$repo_root/"}"
fi
sass_dir_display="$sass_dir"
if [[ $sass_dir == "$repo_root/"* ]]; then
    sass_dir_display="${sass_dir#"$repo_root/"}"
fi

{
    printf '# Attention Prefill M1 SASS 静态证据\n\n'
    printf '> 下表是目标函数的静态指令数量；静态数量不等于 runtime 执行次数。\n\n'
    printf '| Metadata | Value |\n| --- | --- |\n'
    printf '| Evidence kind | `%s` |\n' "$evidence_kind"
    printf '| Git commit | `%s` |\n' "$git_commit"
    printf '| Binary | `%s` |\n' "$binary_display"
    printf '| Binary SHA-256 | `%s` |\n' "$binary_sha256"
    printf '| Source fingerprint | `%s` |\n' "$source_sha256"
    printf '| Build contract | `%s` |\n' "$build_contract"
    printf '| Build payload SHA-256 | `%s` |\n' "$build_payload_sha256"
    printf '| Device index | `%s` |\n' "$device_index"
    printf '| GPU UUID | `%s` |\n' "$gpu_uuid"
    printf '| GPU / SM | `%s` / `%s` |\n' "$gpu_name" "$sm"
    printf '| CUDA driver | `%s` |\n' "$driver"
    printf '| Full SASS directory | `%s` |\n' "$sass_dir_display"
    printf '| Br1 signature | `tiled_attention_kernel%s` |\n' "$signature_suffix"
    printf '| Br4 signature | `query_tiled_kernel%s` |\n\n' "$signature_suffix"
    printf '| Opcode | Br1 | Br4 |\n| --- | ---: | ---: |\n'
    for opcode in "${opcodes[@]}"; do
        printf '| `%s` | %s | %s |\n' "$opcode" \
            "${br1_counts[$opcode]}" "${br4_counts[$opcode]}"
    done
    printf '\nM1 ISA 合同：`HMMA=0` 且 `LDGSTS=0`；两条路径 `FFMA>0`。\n\n'
    printf '**%s**\n' "$spill_warning"
} > "$evidence_tmp"

[[ -s $full_sass && -s $br1_tmp && -s $br4_tmp && -s $evidence_tmp ]] || \
    die 'SASS transaction 产物不完整'
mv -f "$full_sass" "$sass_dir/full.sass"
mv -f "$br1_tmp" "$sass_dir/br1.sass"
mv -f "$br4_tmp" "$sass_dir/br4.sass"
mv -f "$evidence_tmp" "$evidence_file"

printf '[extract_m1_sass] wrote full=%s br1=%s br4=%s evidence=%s\n' \
    "$sass_dir/full.sass" "$sass_dir/br1.sass" "$sass_dir/br4.sass" "$evidence_file"
