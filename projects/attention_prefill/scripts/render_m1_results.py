#!/usr/bin/env python3

import csv
import io
import math
import re
import sys
from collections import Counter
from pathlib import Path


IMPLEMENTATION_ORDER = ("br1", "br4")
CANONICAL_N = (128, 256, 512, 1024, 2048)
CANONICAL_D = (64, 128)
CANONICAL_CAUSAL = (0, 1)
CANONICAL_PROTOCOL = {
    "sm": "8.0",
    "build_preset": "release-sm80",
    "dtype": "fp32",
    "batch": "1",
    "heads": "1",
    "layout": "row-major",
    "input_pattern": "random",
    "seed": "1234",
    "warmup": "10",
    "iterations": "50",
    "repeats": "3",
    "workspace_bytes": "0",
    "reference": "cpu-double",
    "timing": "cuda-event",
}
SPREAD_ABSOLUTE_TOLERANCE = 1.0e-3
REQUIRED_COLUMNS = [
    "timestamp", "git_commit", "runner_sha256", "source_sha256",
    "build_contract", "build_contract_payload_sha256", "device_index",
    "gpu", "gpu_uuid", "sm", "driver",
    "cuda", "nvcc", "build_preset", "dtype", "batch", "heads", "layout",
    "implementation", "path", "n", "d", "causal",
    "input_pattern", "seed", "warmup", "iterations", "repeats",
    "latency_ms", "latency_min_ms", "latency_max_ms", "spread_pct",
    "passed", "max_abs", "max_rel", "workspace_bytes", "reference",
    "timing", "cta_count", "requested_kv_elements",
]
CONSISTENT_COLUMNS = [
    "timestamp", "git_commit", "runner_sha256", "source_sha256",
    "build_contract", "build_contract_payload_sha256", "device_index",
    "gpu", "gpu_uuid", "sm", "driver",
    "cuda", "nvcc", "build_preset", "dtype", "batch", "heads", "layout",
    "input_pattern", "seed", "warmup", "iterations", "repeats",
    "workspace_bytes", "reference", "timing",
]
SAFE_TOKEN = re.compile(r"^[^\s,]+$")
SHA256_TOKEN = re.compile(r"^[0-9a-f]{64}$")
CANONICAL_BUILD_CONTRACT = re.compile(r"^release-sm80-[0-9a-f]{16}$")


def speedup(br1_ms: float, br4_ms: float) -> float:
    if br1_ms <= 0.0 or br4_ms <= 0.0:
        raise ValueError("latency_ms must be positive")
    return br1_ms / br4_ms


def delta_percent(br1_ms: float, br4_ms: float) -> float:
    return 100.0 * (speedup(br1_ms, br4_ms) - 1.0)


def requested_reduction(br1_elements: int, br4_elements: int) -> float:
    if br1_elements <= 0 or br4_elements <= 0:
        raise ValueError("requested_kv_elements must be positive")
    return br1_elements / br4_elements


def classify(pair_speedup: float, br1_spread_pct: float, br4_spread_pct: float) -> str:
    if br1_spread_pct > 3.0 or br4_spread_pct > 3.0:
        return "inconclusive"
    if pair_speedup >= 1.05:
        return "benefit"
    if pair_speedup <= 0.95:
        return "regression"
    return "near-parity"


def is_smoke_source(source: str) -> bool:
    return Path(source).name == "smoke.csv"


def _parse_int(row: dict[str, str], column: str) -> int:
    try:
        return int(row[column])
    except ValueError as error:
        raise ValueError(f"invalid integer in {column}: {row[column]}") from error


def _parse_float(row: dict[str, str], column: str) -> float:
    try:
        value = float(row[column])
    except ValueError as error:
        raise ValueError(f"invalid number in {column}: {row[column]}") from error
    if not math.isfinite(value):
        raise ValueError(f"{column} must be finite")
    return value


def _problem_key(row: dict[str, str]) -> tuple[int, int, int]:
    return _parse_int(row, "n"), _parse_int(row, "d"), _parse_int(row, "causal")


def _validate_canonical(rows: list[dict[str, str]]) -> None:
    if "A100" not in rows[0]["gpu"] or "80GB" not in rows[0]["gpu"]:
        raise ValueError("canonical protocol requires an A100 80GB GPU")
    for column, expected in CANONICAL_PROTOCOL.items():
        if rows[0][column] != expected:
            raise ValueError(
                f"canonical protocol requires {column}={expected}; "
                f"found {rows[0][column]}"
            )
    actual = {
        (_problem_key(row), row["implementation"])
        for row in rows
    }
    expected = {
        ((n, d, causal), implementation)
        for n in CANONICAL_N
        for d in CANONICAL_D
        for causal in CANONICAL_CAUSAL
        for implementation in IMPLEMENTATION_ORDER
    }
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"canonical matrix mismatch: missing={missing} extra={extra}"
        )


def load_rows(csv_path: Path, source_display: str | None = None) -> list[dict[str, str]]:
    try:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            raw_text = handle.read()
            reader = csv.DictReader(io.StringIO(raw_text, newline=""), strict=True)
            fieldnames = reader.fieldnames or []
            missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
            if missing:
                raise ValueError(f"missing required columns: {', '.join(missing)}")
            unexpected = [column for column in fieldnames if column not in REQUIRED_COLUMNS]
            if unexpected:
                raise ValueError(f"unexpected columns: {', '.join(unexpected)}")
            if len(fieldnames) != len(REQUIRED_COLUMNS):
                raise ValueError("CSV header column count mismatch")
            rows = []
            for row_number, row in enumerate(reader, start=2):
                if None in row:
                    raise ValueError(f"row {row_number} has extra fields")
                if any(row[column] is None for column in REQUIRED_COLUMNS):
                    raise ValueError(f"row {row_number} column count mismatch")
                rows.append(row)
            canonical = io.StringIO(newline="")
            writer = csv.DictWriter(
                canonical,
                fieldnames=REQUIRED_COLUMNS,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
            if raw_text != canonical.getvalue():
                raise ValueError("non-canonical CSV encoding")
    except csv.Error as error:
        raise ValueError(f"malformed CSV: {error}") from error

    if not rows:
        raise ValueError("result CSV contains no rows")
    for row_number, row in enumerate(rows, start=2):
        empty = [column for column in REQUIRED_COLUMNS if not (row.get(column) or "").strip()]
        if empty:
            raise ValueError(
                f"row {row_number} has empty required values: {', '.join(empty)}"
            )

    source = source_display or csv_path.as_posix()
    smoke = is_smoke_source(source)
    if not smoke and len(rows) != 40:
        raise ValueError(
            f"canonical result CSV must contain exactly 40 rows; found {len(rows)}"
        )

    for column in CONSISTENT_COLUMNS:
        values = {row[column] for row in rows}
        if len(values) != 1:
            raise ValueError(f"inconsistent {column} values")

    first = rows[0]
    if not SHA256_TOKEN.fullmatch(first["source_sha256"]):
        raise ValueError("source_sha256 must be 64 lowercase hex characters")
    if not SAFE_TOKEN.fullmatch(first["build_contract"]):
        raise ValueError("build_contract must be a safe no-space token")
    if not SHA256_TOKEN.fullmatch(first["build_contract_payload_sha256"]):
        raise ValueError(
            "build_contract_payload_sha256 must be 64 lowercase hex characters"
        )
    device_index = _parse_int(first, "device_index")
    if device_index < 0:
        raise ValueError("device_index must be nonnegative")

    by_problem: dict[tuple[int, int, int], dict[str, dict[str, str]]] = {}
    for row in rows:
        key = _problem_key(row)
        n, d, causal = key
        if n <= 0 or d <= 0:
            raise ValueError(f"N and D must be positive: {n}x{d}")
        if causal not in (0, 1):
            raise ValueError(f"causal must be 0 or 1: {causal}")
        implementation = row["implementation"]
        if implementation not in IMPLEMENTATION_ORDER:
            raise ValueError(f"unknown implementation: {implementation}")
        implementations = by_problem.setdefault(key, {})
        if implementation in implementations:
            raise ValueError(
                "duplicate implementation pair for "
                f"N={n} D={d} causal={causal} implementation={implementation}"
            )
        implementations[implementation] = row

        latency = _parse_float(row, "latency_ms")
        latency_min = _parse_float(row, "latency_min_ms")
        latency_max = _parse_float(row, "latency_max_ms")
        spread = _parse_float(row, "spread_pct")
        max_abs = _parse_float(row, "max_abs")
        max_rel = _parse_float(row, "max_rel")
        if latency <= 0.0 or latency_min <= 0.0 or latency_max <= 0.0:
            raise ValueError("latency values must be positive")
        if latency_min > latency or latency > latency_max:
            raise ValueError("latency must satisfy min <= median <= max")
        if spread < 0.0:
            raise ValueError("spread_pct must be nonnegative")
        expected_spread = 100.0 * (latency_max - latency_min) / latency
        if not math.isclose(
            spread, expected_spread, rel_tol=0.0,
            abs_tol=SPREAD_ABSOLUTE_TOLERANCE,
        ):
            raise ValueError(
                "spread_pct does not match 100*(max-min)/median: "
                f"reported={spread} expected={expected_spread}"
            )
        if max_abs < 0.0:
            raise ValueError("max_abs must be nonnegative")
        if max_rel < 0.0:
            raise ValueError("max_rel must be nonnegative")
        if row["passed"].lower() != "true":
            raise ValueError(f"row did not pass correctness: N={n} D={d} causal={causal}")
        if row["path"] != implementation:
            raise ValueError("path must match implementation")
        expected_cta = n if implementation == "br1" else (n + 3) // 4
        cta_count = _parse_int(row, "cta_count")
        if cta_count != expected_cta:
            raise ValueError(
                f"cta_count mismatch: expected={expected_cta} actual={cta_count}"
            )
        requested = _parse_int(row, "requested_kv_elements")
        expected_requested = 2 * expected_cta * n * d
        if requested != expected_requested:
            raise ValueError(
                "requested_kv_elements mismatch: "
                f"expected={expected_requested} actual={requested}"
            )
        if _parse_int(row, "workspace_bytes") < 0:
            raise ValueError("workspace_bytes must be nonnegative")

    for (n, d, causal), implementations in sorted(by_problem.items()):
        missing_implementations = [
            implementation for implementation in IMPLEMENTATION_ORDER
            if implementation not in implementations
        ]
        if missing_implementations:
            raise ValueError(
                "missing implementation pair for "
                f"N={n} D={d} causal={causal}: "
                f"{', '.join(missing_implementations)}"
            )
    if not smoke:
        if not CANONICAL_BUILD_CONTRACT.fullmatch(first["build_contract"]):
            raise ValueError(
                "canonical protocol requires build_contract="
                "release-sm80-<16 lowercase hex>"
            )
        if first["build_contract"].rsplit("-", 1)[1] != first[
            "build_contract_payload_sha256"
        ][:16]:
            raise ValueError(
                "canonical build_contract must match payload hash prefix"
            )
        _validate_canonical(rows)
    return rows


def portable_source(source: str) -> str:
    marker = "projects/attention_prefill/"
    normalized = source.replace("\\", "/")
    if marker in normalized:
        return marker + normalized.split(marker, 1)[1]
    return normalized


def markdown_cell(value: object) -> str:
    return (
        str(value)
        .replace("`", "'")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\n", "<br>")
        .replace("|", r"\|")
    )


def code_cell(value: object) -> str:
    return f"`{markdown_cell(value)}`"


def render(rows: list[dict[str, str]], source: str) -> str:
    by_problem: dict[tuple[int, int, int], dict[str, dict[str, str]]] = {}
    for row in rows:
        by_problem.setdefault(_problem_key(row), {})[row["implementation"]] = row

    first = rows[0]
    smoke = is_smoke_source(source)
    title = "Smoke（非性能证据）" if smoke else "Canonical Benchmark"
    classifications: Counter[str] = Counter()
    table_rows: list[str] = []

    for n, d, causal in sorted(by_problem, key=lambda key: (key[1], key[0], key[2])):
        pair = by_problem[(n, d, causal)]
        br1 = pair["br1"]
        br4 = pair["br4"]
        br1_ms = float(br1["latency_ms"])
        br4_ms = float(br4["latency_ms"])
        pair_speedup = speedup(br1_ms, br4_ms)
        br1_spread = float(br1["spread_pct"])
        br4_spread = float(br4["spread_pct"])
        classification = classify(pair_speedup, br1_spread, br4_spread)
        classifications[classification] += 1
        request_ratio = requested_reduction(
            int(br1["requested_kv_elements"]), int(br4["requested_kv_elements"])
        )
        table_rows.append(
            "| {n} | {d} | {causal} | {br1_path} | {br4_path} | "
            "{br1_ms:.6f} | {br4_ms:.6f} | {speedup:.3f}x | {delta:+.1f}% | "
            "{br1_spread:.2f}% / {br4_spread:.2f}% | {br1_cta} / {br4_cta} | "
            "{br1_requested} / {br4_requested} | {request_ratio:.3f}x | {classification} |".format(
                n=n,
                d=d,
                causal=causal,
                br1_path=markdown_cell(br1["path"]),
                br4_path=markdown_cell(br4["path"]),
                br1_ms=br1_ms,
                br4_ms=br4_ms,
                speedup=pair_speedup,
                delta=delta_percent(br1_ms, br4_ms),
                br1_spread=br1_spread,
                br4_spread=br4_spread,
                br1_cta=br1["cta_count"],
                br4_cta=br4["cta_count"],
                br1_requested=br1["requested_kv_elements"],
                br4_requested=br4["requested_kv_elements"],
                request_ratio=request_ratio,
                classification=classification,
            )
        )

    lines = [
        f"# Attention Prefill M1 Br1 vs Br4 {title}",
        "",
        f"来源 CSV：{code_cell(portable_source(source))}",
        "",
        "## 环境与协议",
        "",
        "| 项目 | 值 |",
        "| --- | --- |",
        f"| Timestamp | {markdown_cell(first['timestamp'])} |",
        f"| Git commit | {code_cell(first['git_commit'])} |",
        f"| Runner SHA-256 | {code_cell(first['runner_sha256'])} |",
        f"| Source SHA-256 | {code_cell(first['source_sha256'])} |",
        f"| Build contract | {code_cell(first['build_contract'])} |",
        f"| Build payload SHA-256 | {code_cell(first['build_contract_payload_sha256'])} |",
        f"| CUDA device index | {markdown_cell(first['device_index'])} |",
        f"| GPU | {markdown_cell(first['gpu'])} |",
        f"| GPU UUID | {code_cell(first['gpu_uuid'])} |",
        f"| SM | {markdown_cell(first['sm'])} |",
        f"| Driver | {markdown_cell(first['driver'])} |",
        f"| CUDA | {markdown_cell(first['cuda'])} |",
        f"| nvcc | {markdown_cell(first['nvcc'])} |",
        f"| Build preset | {markdown_cell(first['build_preset'])} |",
        "| Tensor contract | dtype={dtype}, batch={batch}, heads={heads}, "
        "layout={layout} |".format(**{
            key: markdown_cell(first[key])
            for key in ("dtype", "batch", "heads", "layout")
        }),
        "| Input | {pattern}, seed={seed} |".format(
            pattern=markdown_cell(first["input_pattern"]),
            seed=markdown_cell(first["seed"]),
        ),
        "| Timing | {timing}, warmup={warmup}, iterations={iterations}, "
        "repeats={repeats}, median/min/max |".format(**{
            key: markdown_cell(first[key])
            for key in ("timing", "warmup", "iterations", "repeats")
        }),
        f"| Correctness reference | {markdown_cell(first['reference'])} |",
        f"| Workspace | {markdown_cell(first['workspace_bytes'])} bytes |",
        "",
        "> `Speedup = Br1 latency / Br4 latency`；`Delta = 100 × (speedup - 1)`。任一路 spread > 3% 时分类为 inconclusive。理论 requested elements 不等于 DRAM bytes；ncu duration 不进入本表。",
        "",
        "## Br1 vs Br4",
        "",
        "| N | D | Causal | Br1 path | Br4 path | Br1 (ms) | Br4 (ms) | Speedup | Delta | Spread Br1 / Br4 | CTA Br1 / Br4 | Requested Br1 / Br4 | Requested reduction | 分类 |",
        "| ---: | ---: | :---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        *table_rows,
        "",
        "## 分类汇总",
        "",
        "| 分类 | 数量 |",
        "| --- | ---: |",
    ]
    for classification in ("benefit", "near-parity", "regression", "inconclusive"):
        lines.append(f"| {classification} | {classifications[classification]} |")
    lines.append("")
    return "\n".join(lines)


def render_file(csv_path: Path, markdown_path: Path, source_display: str | None = None) -> None:
    source = source_display or csv_path.as_posix()
    rows = load_rows(csv_path, source)
    text = render(rows, source)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(text + ("" if text.endswith("\n") else "\n"), encoding="utf-8")


def main() -> int:
    if len(sys.argv) not in (3, 4):
        print(
            "usage: render_m1_results.py <input.csv> <output.md> [source-display]",
            file=sys.stderr,
        )
        return 2
    try:
        render_file(
            Path(sys.argv[1]),
            Path(sys.argv[2]),
            sys.argv[3] if len(sys.argv) == 4 else None,
        )
    except (OSError, ValueError) as error:
        print(f"render_m1_results: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
