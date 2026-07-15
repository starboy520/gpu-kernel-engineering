#!/usr/bin/env python3

import argparse
import csv
import math
import re
import sys
from pathlib import Path


IDENTITY_COLUMNS = (
    "evidence_kind",
    "implementation",
    "n",
    "d",
    "causal",
    "kernel_name",
    "process_name",
    "block_size",
    "grid_size",
    "profile_timestamp",
    "git_commit",
    "runner_sha256",
    "source_sha256",
    "build_contract",
    "build_contract_payload_sha256",
    "device_index",
    "gpu_uuid",
    "gpu_name",
    "sm",
    "driver",
    "ncu_version",
)
OPTIONAL_IDENTITY_COLUMNS = ("profile_timestamp",)
STRICT_IDENTITY_COLUMNS = (
    "git_commit",
    "runner_sha256",
    "source_sha256",
    "build_contract",
    "build_contract_payload_sha256",
    "device_index",
    "gpu_uuid",
    "gpu_name",
    "sm",
    "driver",
    "ncu_version",
)
CANONICAL_PROBLEMS = ((256, 64, 0), (1024, 128, 0), (1024, 128, 1))
IMPLEMENTATIONS = ("br1", "br4")
SYMBOLS = {"br1": "tiled_attention_kernel", "br4": "query_tiled_kernel"}
KERNEL_ARGUMENTS = (
    "const float *, const float *, const float *, float *, int, int, bool"
)
NCU_RAW_IDENTITY_COLUMNS = (
    "ID",
    "Process ID",
    "Process Name",
    "Host Name",
    "Kernel Name",
    "Context",
    "Stream",
    "Block Size",
    "Grid Size",
    "Device",
    "CC",
)


class Metric:
    def __init__(self, label, name, dimension, unit_scales, test_unit):
        self.label = label
        self.name = name
        self.dimension = dimension
        self.unit_scales = unit_scales
        self.test_unit = test_unit


class Record:
    def __init__(self, identity, values, units):
        self.identity = identity
        self.values = values
        self.units = units


TIME_UNITS = {"ns": 1.0e-9, "us": 1.0e-6, "ms": 1.0e-3, "s": 1.0}
BYTE_UNITS = {"byte": 1.0, "Kbyte": 1.0e3, "Mbyte": 1.0e6, "Gbyte": 1.0e9}
PERCENT_UNITS = {"%": 1.0}


def count_units(*units):
    return {unit: 1.0 for unit in ("", *units)}


METRICS = (
    Metric("ncu duration", "gpu__time_duration.sum", "time", TIME_UNITS, "ns"),
    Metric("Registers/thread", "launch__registers_per_thread", "register-count", count_units("register/thread"), "register/thread"),
    Metric("Static SMEM/block", "launch__shared_mem_per_block_static", "bytes-per-block", {f"{unit}/block": scale for unit, scale in BYTE_UNITS.items()}, "byte/block"),
    Metric("Waves/SM", "launch__waves_per_multiprocessor", "dimensionless", {"": 1.0, "wave": 1.0}, "wave"),
    Metric("SMEM limit blocks/SM", "launch__occupancy_limit_shared_mem", "block-count", count_units("block"), "block"),
    Metric("Register limit blocks/SM", "launch__occupancy_limit_registers", "block-count", count_units("block"), "block"),
    Metric("Achieved occupancy", "sm__warps_active.avg.pct_of_peak_sustained_active", "percent", PERCENT_UNITS, "%"),
    Metric("Eligible warps/cycle", "smsp__warps_eligible.avg.per_cycle_active", "warp-count", count_units("warp"), "warp"),
    Metric("Long scoreboard", "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio", "stall-count", count_units("warp", "inst"), "warp"),
    Metric("Short scoreboard", "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio", "stall-count", count_units("warp", "inst"), "warp"),
    Metric("Barrier stall", "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio", "stall-count", count_units("warp", "inst"), "warp"),
    Metric("Warp latency", "smsp__average_warp_latency_per_inst_issued.ratio", "cycle-count", count_units("cycle"), "cycle"),
    Metric("SM throughput", "sm__throughput.avg.pct_of_peak_sustained_elapsed", "percent", PERCENT_UNITS, "%"),
    Metric("DRAM throughput", "dram__throughput.avg.pct_of_peak_sustained_elapsed", "percent", PERCENT_UNITS, "%"),
    Metric("L2 throughput", "lts__throughput.avg.pct_of_peak_sustained_elapsed", "percent", PERCENT_UNITS, "%"),
    Metric("L1 throughput", "l1tex__throughput.avg.pct_of_peak_sustained_elapsed", "percent", PERCENT_UNITS, "%"),
    Metric("DRAM read bytes", "dram__bytes_read.sum", "bytes", BYTE_UNITS, "byte"),
    Metric("DRAM write bytes", "dram__bytes_write.sum", "bytes", BYTE_UNITS, "byte"),
    Metric("L2 read sectors", "lts__t_sectors_op_read.sum", "sector-count", count_units("sector"), "sector"),
    Metric("L2 write sectors", "lts__t_sectors_op_write.sum", "sector-count", count_units("sector"), "sector"),
    Metric("L2 sector hit rate", "lts__t_sector_hit_rate.pct", "percent", PERCENT_UNITS, "%"),
    Metric("Global load sectors", "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum", "sector-count", count_units("sector"), "sector"),
    Metric("Global load requests", "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum", "request-count", count_units("request"), "request"),
    Metric("Global store sectors", "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum", "sector-count", count_units("sector"), "sector"),
    Metric("Global store requests", "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum", "request-count", count_units("request"), "request"),
    Metric("Shared bank conflicts", "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum", "conflict-count", count_units("conflict"), "conflict"),
    Metric("Shared wavefronts", "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum", "wavefront-count", count_units("wavefront"), "wavefront"),
)


def normalized_columns():
    columns = list(IDENTITY_COLUMNS)
    for metric in METRICS:
        if metric.name not in columns:
            columns.extend((metric.name, f"{metric.name}__unit"))
    return columns


def _identity_key(record):
    identity = record.identity
    try:
        return (
            int(identity["n"]),
            int(identity["d"]),
            int(identity["causal"]),
            identity["implementation"],
        )
    except (KeyError, ValueError) as error:
        raise ValueError(f"invalid profile identity: {identity}") from error


def _validate_record(record):
    missing_identity = [
        name
        for name in IDENTITY_COLUMNS
        if name not in OPTIONAL_IDENTITY_COLUMNS and not record.identity.get(name)
    ]
    if missing_identity:
        raise ValueError(f"missing required identity: {', '.join(missing_identity)}")
    implementation = record.identity["implementation"]
    if record.identity["evidence_kind"] not in ("canonical", "smoke"):
        raise ValueError("evidence_kind must be canonical or smoke")
    if implementation not in IMPLEMENTATIONS:
        raise ValueError(f"unknown implementation: {implementation}")
    symbol = SYMBOLS[implementation]
    expected_suffix = f"::{symbol}({KERNEL_ARGUMENTS})"
    if not record.identity["kernel_name"].endswith(expected_suffix):
        raise ValueError(
            f"kernel symbol mismatch for {implementation}: "
            f"expected *{expected_suffix}, found {record.identity['kernel_name']}"
        )
    if record.identity["process_name"] != "attention_prefill_evidence_runner":
        raise ValueError(
            "process identity mismatch: expected attention_prefill_evidence_runner"
        )
    n, d, causal, _ = _identity_key(record)
    if n <= 0 or d <= 0 or causal not in (0, 1):
        raise ValueError(f"invalid problem identity: N={n} D={d} causal={causal}")
    if record.identity["block_size"] != "(128, 1, 1)":
        raise ValueError(
            f"block identity mismatch: expected (128, 1, 1), "
            f"found {record.identity['block_size']}"
        )
    expected_grid = n if implementation == "br1" else math.ceil(n / 4)
    if record.identity["grid_size"] != f"({expected_grid}, 1, 1)":
        raise ValueError(
            f"grid identity mismatch for {implementation}: expected "
            f"({expected_grid}, 1, 1), found {record.identity['grid_size']}"
        )
    token_patterns = {
        "git_commit": r"[0-9a-f]{40,64}",
        "runner_sha256": r"[0-9a-f]{64}",
        "source_sha256": r"[0-9a-f]{64}",
        "build_contract": r"[A-Za-z0-9._-]+",
        "build_contract_payload_sha256": r"[0-9a-f]{64}",
        "device_index": r"[0-9]+",
        "gpu_uuid": r"GPU-[0-9A-Fa-f-]+",
        "gpu_name": r"[^\s,]+",
        "sm": r"[0-9]+\.[0-9]+",
        "driver": r"[0-9]+",
        "ncu_version": r"[0-9]+(?:\.[0-9]+)+",
    }
    for field, pattern in token_patterns.items():
        value = record.identity[field]
        if re.fullmatch(pattern, value) is None:
            raise ValueError(f"invalid {field}: {value}")
    timestamp = record.identity.get("profile_timestamp", "")
    if timestamp and re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z",
        timestamp,
    ) is None:
        raise ValueError(f"invalid profile_timestamp: {timestamp}")
    payload_sha256 = record.identity["build_contract_payload_sha256"]
    build_contract = record.identity["build_contract"]
    if build_contract.startswith("release-sm80-") and not build_contract.endswith(
        payload_sha256[:16]
    ):
        raise ValueError(
            "build_contract does not match build_contract_payload_sha256 prefix"
        )
    for metric in METRICS:
        if metric.name not in record.values or record.values[metric.name] == "":
            raise ValueError(f"missing required metric: {metric.name}")
        try:
            value = float(record.values[metric.name])
        except ValueError as error:
            raise ValueError(
                f"metric {metric.name} is not numeric: {record.values[metric.name]}"
            ) from error
        if not math.isfinite(value):
            raise ValueError(f"metric {metric.name} must be finite")
        if value < 0.0:
            raise ValueError(f"metric {metric.name} must be nonnegative")
        if metric.name not in record.units:
            raise ValueError(f"missing metric unit field: {metric.name}")
        unit = record.units[metric.name]
        if unit not in metric.unit_scales:
            allowed = ", ".join(repr(item) for item in metric.unit_scales)
            raise ValueError(
                f"metric {metric.name} has incompatible unit {unit!r}; "
                f"allowed units: {allowed}"
            )


def validate_records(records, require_canonical):
    if not records:
        raise ValueError("no profile records supplied")
    seen = set()
    for record in records:
        _validate_record(record)
        key = _identity_key(record)
        if key in seen:
            raise ValueError(f"duplicate profile identity: {key}")
        seen.add(key)
    for field in STRICT_IDENTITY_COLUMNS:
        values = {record.identity[field] for record in records}
        if len(values) != 1:
            category = {
                "git_commit": "source",
                "runner_sha256": "binary",
                "source_sha256": "source",
                "build_contract": "build",
                "build_contract_payload_sha256": "build",
                "device_index": "device",
                "gpu_uuid": "device",
                "gpu_name": "device",
                "sm": "device",
                "driver": "device",
                "ncu_version": "ncu",
            }[field]
            raise ValueError(
                f"{category} identity mismatch for {field}: {sorted(values)}"
            )
    if require_canonical:
        identity = records[0].identity
        if any(record.identity["evidence_kind"] != "canonical" for record in records):
            raise ValueError("canonical summary requires evidence_kind=canonical")
        if any(not record.identity["ncu_version"].startswith("2026.2.") for record in records):
            raise ValueError("canonical profiles support only Nsight Compute 2026.2.*")
        if re.fullmatch(r"release-sm80-[0-9a-f]{16}", identity["build_contract"]) is None:
            raise ValueError(
                "canonical profiles require build_contract="
                "release-sm80-<16 lowercase hex>"
            )
        if identity["build_contract"].rsplit("-", 1)[1] != identity[
            "build_contract_payload_sha256"
        ][:16]:
            raise ValueError(
                "canonical build_contract must match payload hash prefix"
            )
        if not (
            "A100" in identity["gpu_name"]
            and "80GB" in identity["gpu_name"]
            and identity["sm"] == "8.0"
        ):
            raise ValueError(
                "canonical profiles require A100 80GB with sm=8.0: "
                f"gpu={identity['gpu_name']} sm={identity['sm']}"
            )
        expected = {
            (*problem, implementation)
            for problem in CANONICAL_PROBLEMS
            for implementation in IMPLEMENTATIONS
        }
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        if missing or extra:
            raise ValueError(
                f"missing canonical profile or unexpected identity: "
                f"missing={missing} extra={extra}"
            )
    by_problem = {}
    for n, d, causal, implementation in seen:
        by_problem.setdefault((n, d, causal), set()).add(implementation)
    for problem, implementations in sorted(by_problem.items()):
        missing = set(IMPLEMENTATIONS) - implementations
        if missing:
            raise ValueError(
                f"missing implementation pair for {problem}: {', '.join(sorted(missing))}"
            )


def normalize_raw_csv(raw_path, implementation, n, d, causal, provenance):
    with raw_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, strict=True)
        fieldnames = reader.fieldnames or []
        required = [*NCU_RAW_IDENTITY_COLUMNS, *sorted(metric.name for metric in METRICS)]
        missing = [name for name in required if name not in fieldnames]
        if missing:
            raise ValueError(f"raw ncu CSV missing required columns: {', '.join(missing)}")
        unexpected = [name for name in fieldnames if name not in required]
        if unexpected:
            raise ValueError(
                f"raw ncu CSV has unexpected columns: {', '.join(unexpected)}"
            )
        if fieldnames != required:
            raise ValueError("raw ncu CSV column order does not match schema")
        try:
            units_row = next(reader)
        except StopIteration as error:
            raise ValueError("raw ncu CSV is missing its units row") from error
        rows = list(reader)
    if None in units_row:
        raise ValueError("raw ncu CSV units row has extra fields")
    if any(units_row.get(name) is None for name in required):
        raise ValueError("raw ncu CSV units row has missing fields")
    if len(rows) != 1:
        raise ValueError(f"expected exactly one kernel data row, found {len(rows)}")
    row = rows[0]
    if None in row:
        raise ValueError("raw ncu CSV data row has extra fields")
    if any(row.get(name) is None for name in required):
        raise ValueError("raw ncu CSV data row has missing fields")
    record = Record(
        identity={
            "implementation": implementation,
            "n": str(n),
            "d": str(d),
            "causal": str(causal),
            "kernel_name": (row.get("Kernel Name") or "").strip(),
            "process_name": (row.get("Process Name") or "").strip(),
            "block_size": (row.get("Block Size") or "").strip(),
            "grid_size": (row.get("Grid Size") or "").strip(),
            **provenance,
        },
        values={metric.name: (row.get(metric.name) or "").strip() for metric in METRICS},
        units={
            metric.name: (units_row.get(metric.name) or "").strip()
            for metric in METRICS
        },
    )
    _validate_record(record)
    return record


def write_normalized_csv(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = normalized_columns()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for record in records:
            row = dict(record.identity)
            for metric in METRICS:
                row[metric.name] = record.values[metric.name]
                row[f"{metric.name}__unit"] = record.units[metric.name]
            writer.writerow(row)


def load_normalized_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, strict=True)
        fieldnames = reader.fieldnames or []
        missing = [name for name in normalized_columns() if name not in fieldnames]
        if missing:
            raise ValueError(
                f"normalized CSV missing required columns: {', '.join(missing)}"
            )
        unexpected = [name for name in fieldnames if name not in normalized_columns()]
        if unexpected:
            raise ValueError(
                f"normalized CSV has unexpected columns: {', '.join(unexpected)}"
            )
        if fieldnames != normalized_columns():
            raise ValueError("normalized CSV column order does not match schema")
        rows = list(reader)
    for row_number, row in enumerate(rows, start=2):
        if None in row:
            raise ValueError(f"normalized CSV row {row_number} has extra fields")
        if any(row.get(name) is None for name in normalized_columns()):
            raise ValueError(f"normalized CSV row {row_number} has missing fields")
    records = []
    for row in rows:
        records.append(
            Record(
                identity={name: (row.get(name) or "").strip() for name in IDENTITY_COLUMNS},
                values={
                    metric.name: (row.get(metric.name) or "").strip()
                    for metric in METRICS
                },
                units={
                    metric.name: (row.get(f"{metric.name}__unit") or "").strip()
                    for metric in METRICS
                },
            )
        )
    return records


def load_inputs(paths, require_canonical):
    records = []
    for path in paths:
        records.extend(load_normalized_csv(path))
    validate_records(records, require_canonical=require_canonical)
    return records


def _display(record, metric):
    value = record.values[metric.name]
    unit = record.units[metric.name]
    return f"{value} {unit}".strip()


def _ratio(br1, br4, metric):
    baseline_unit = br1.units[metric.name]
    candidate_unit = br4.units[metric.name]
    if baseline_unit not in metric.unit_scales or candidate_unit not in metric.unit_scales:
        raise ValueError(f"metric {metric.name} has incompatible unit")
    baseline = float(br1.values[metric.name]) * metric.unit_scales[baseline_unit]
    candidate = float(br4.values[metric.name]) * metric.unit_scales[candidate_unit]
    if baseline == 0.0:
        return "-" if candidate == 0.0 else "∞"
    return f"{candidate / baseline:.3f}x"


def portable_source(path):
    marker = "projects/attention_prefill/"
    normalized = path.as_posix()
    if marker in normalized:
        return marker + normalized.split(marker, 1)[1]
    return normalized


def render(records, sources):
    by_problem = {}
    for record in records:
        n, d, causal, implementation = _identity_key(record)
        by_problem.setdefault((n, d, causal), {})[implementation] = record
    identity = records[0].identity
    timestamps = sorted(
        {
            record.identity["profile_timestamp"]
            for record in records
            if record.identity.get("profile_timestamp")
        }
    )
    evidence_kinds = sorted({record.identity["evidence_kind"] for record in records})
    is_smoke = evidence_kinds == ["smoke"] or len(records) == 2
    title_suffix = " Smoke（单点诊断，非六点 canonical）" if is_smoke else ""
    lines = [
        f"# Attention Prefill M1 Nsight Compute 摘要{title_suffix}",
        "",
        "> ncu duration 仅用于 profiler 内部对照，不等于 CUDA Event 或端到端 wall-clock；单位来自 ncu CSV，不静默换算。",
        "",
        f"> Evidence kind：`{', '.join(evidence_kinds)}`；" +
        ("当前只包含一个 Br1/Br4 pair，不是正式六点 canonical 结论。" if is_smoke else "包含完整六点 canonical profile。"),
        "",
        "输入：" + "、".join(f"`{portable_source(path)}`" for path in sources),
        "",
        "## 环境与构建身份",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Git commit | `{identity['git_commit']}` |",
        f"| Runner SHA-256 | `{identity['runner_sha256']}` |",
        f"| Source SHA-256 | `{identity['source_sha256']}` |",
        f"| Build contract | `{identity['build_contract']}` |",
        f"| Build payload SHA-256 | `{identity['build_contract_payload_sha256']}` |",
        f"| Device index | `{identity['device_index']}` |",
        f"| GPU UUID | `{identity['gpu_uuid']}` |",
        f"| GPU | `{identity['gpu_name']}` |",
        f"| SM | `{identity['sm']}` |",
        f"| CUDA driver | `{identity['driver']}` |",
        f"| Nsight Compute | `{identity['ncu_version']}` |",
        f"| Profile timestamps | `{', '.join(timestamps) if timestamps else 'not-recorded'}` |",
        "",
    ]
    for problem in sorted(by_problem):
        n, d, causal = problem
        br1 = by_problem[problem]["br1"]
        br4 = by_problem[problem]["br4"]
        lines.extend(
            [
                f"## {n}x{d} causal={causal}",
                "",
                f"Kernel：Br1 `{br1.identity['kernel_name']}`；Br4 `{br4.identity['kernel_name']}`。",
                "",
                "| Metric | Br1 | Br4 | Br4 / Br1 |",
                "| --- | ---: | ---: | ---: |",
                f"| Block | {br1.identity['block_size']} | {br4.identity['block_size']} | - |",
                f"| Grid | {br1.identity['grid_size']} | {br4.identity['grid_size']} | - |",
            ]
        )
        for metric in METRICS:
            lines.append(
                f"| {metric.label} | {_display(br1, metric)} | "
                f"{_display(br4, metric)} | {_ratio(br1, br4, metric)} |"
            )
        lines.append("")
    return "\n".join(lines)


def _parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="*")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-smoke-pair", action="store_true")
    parser.add_argument("--normalize-raw", type=Path)
    parser.add_argument("--normalized-output", type=Path)
    parser.add_argument("--implementation", choices=IMPLEMENTATIONS)
    parser.add_argument("--n", type=int)
    parser.add_argument("--d", type=int)
    parser.add_argument("--causal", type=int, choices=(0, 1))
    for name in (
        "evidence-kind",
        "profile-timestamp",
        "git-commit",
        "runner-sha256",
        "source-sha256",
        "build-contract",
        "build-contract-payload-sha256",
        "device-index",
        "gpu-uuid",
        "gpu-name",
        "sm",
        "driver",
        "ncu-version",
    ):
        parser.add_argument(f"--{name}")
    return parser.parse_args()


def main():
    arguments = _parse_arguments()
    try:
        if arguments.normalize_raw is not None:
            required = (
                arguments.normalized_output,
                arguments.implementation,
                arguments.n,
                arguments.d,
                arguments.causal,
            )
            if any(value is None for value in required):
                raise ValueError(
                    "normalization requires --normalized-output, --implementation, "
                    "--n, --d, and --causal"
                )
            record = normalize_raw_csv(
                arguments.normalize_raw,
                arguments.implementation,
                arguments.n,
                arguments.d,
                arguments.causal,
                {
                    name: getattr(arguments, name)
                    for name in (
                        "evidence_kind",
                        "profile_timestamp",
                        "git_commit",
                        "runner_sha256",
                        "source_sha256",
                        "build_contract",
                        "build_contract_payload_sha256",
                        "device_index",
                        "gpu_uuid",
                        "gpu_name",
                        "sm",
                        "driver",
                        "ncu_version",
                    )
                },
            )
            write_normalized_csv(arguments.normalized_output, [record])
            return 0
        paths = [Path(value) for value in arguments.csv]
        expected_count = 2 if arguments.allow_smoke_pair else 6
        if len(paths) != expected_count:
            raise ValueError(f"expected exactly {expected_count} normalized CSV inputs")
        records = load_inputs(paths, require_canonical=not arguments.allow_smoke_pair)
        text = render(records, paths)
        if not text.endswith("\n"):
            text += "\n"
        if arguments.output is None:
            sys.stdout.write(text)
        else:
            basename = arguments.output.name
            if arguments.allow_smoke_pair:
                if "smoke" not in basename.lower() or basename == "m1-ncu-summary.md":
                    raise ValueError(
                        "--allow-smoke-pair --output basename must contain smoke "
                        "and cannot be m1-ncu-summary.md"
                    )
            elif basename != "m1-ncu-summary.md":
                raise ValueError(
                    "canonical compact evidence output must be m1-ncu-summary.md"
                )
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            arguments.output.write_text(text, encoding="utf-8")
        return 0
    except (OSError, csv.Error, ValueError) as error:
        print(f"summarize_m1_ncu: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
