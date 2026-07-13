#!/usr/bin/env python3

import csv
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Record:
    kernel: str
    values: dict[str, str]
    units: dict[str, str]


METRICS = [
    ("Registers/thread", "launch__registers_per_thread"),
    ("Static SMEM", "launch__shared_mem_per_block_static"),
    ("Waves/SM", "launch__waves_per_multiprocessor"),
    ("SMEM limit blocks/SM", "launch__occupancy_limit_shared_mem"),
    ("Register limit blocks/SM", "launch__occupancy_limit_registers"),
    ("Active warps", "sm__warps_active.avg.pct_of_peak_sustained_active"),
    ("Eligible warps/cycle", "smsp__warps_eligible.avg.per_cycle_active"),
    ("Long scoreboard", "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"),
    ("Short scoreboard", "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio"),
    ("Barrier stall", "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio"),
    ("Warp latency", "smsp__average_warp_latency_per_inst_issued.ratio"),
    ("SM throughput", "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("DRAM throughput", "dram__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("L2 throughput", "lts__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("L1TEX throughput", "l1tex__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("Shared load conflicts", "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"),
    ("Shared load wavefronts", "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum"),
]


def load_raw_csv(csv_path: Path) -> Record:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        required = ["Kernel Name", *[metric for _, metric in METRICS]]
        missing = [field for field in required if field not in fields]
        if missing:
            raise ValueError(f"missing required columns: {', '.join(missing)}")
        try:
            units = next(reader)
        except StopIteration as error:
            raise ValueError("raw CSV is missing the units row") from error
        rows = list(reader)
    if len(rows) != 1:
        raise ValueError(f"expected exactly one kernel data row, found {len(rows)}")
    row = rows[0]
    kernel = row.get("Kernel Name", "").strip()
    if not kernel:
        raise ValueError("kernel name is empty")
    return Record(kernel, row, units)


def report_to_record(report: Path) -> Record:
    with tempfile.NamedTemporaryFile(suffix=".csv") as temporary:
        result = subprocess.run(
            ["ncu", "--import", str(report), "--page", "raw", "--csv"],
            check=False,
            stdout=temporary,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
        temporary.flush()
        return load_raw_csv(Path(temporary.name))


def wavefront_factor(record: Record) -> float | None:
    conflicts = record.values.get(
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"
    )
    wavefronts = record.values.get(
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum"
    )
    if not conflicts or not wavefronts:
        return None
    try:
        conflict_value = float(conflicts)
        wavefront_value = float(wavefronts)
    except ValueError:
        return None
    ideal = wavefront_value - conflict_value
    return wavefront_value / ideal if ideal > 0 else None


def display(record: Record, metric: str) -> str:
    value = record.values.get(metric, "")
    if value in ("", None):
        return "-"
    unit = record.units.get(metric, "")
    return f"{value} {unit}".strip()


def short_kernel_name(name: str) -> str:
    if "tiled_async_attention_kernel" in name:
        return "Tiled Async"
    if "tiled_attention_kernel" in name:
        return "Tiled"
    return name.split("(", 1)[0]


def render_comparison(left: Record, right: Record) -> str:
    lines = [
        f"| Metric | {short_kernel_name(left.kernel)} | {short_kernel_name(right.kernel)} |",
        "| --- | ---: | ---: |",
    ]
    for label, metric in METRICS:
        lines.append(f"| {label} | {display(left, metric)} | {display(right, metric)} |")
    left_factor = wavefront_factor(left)
    right_factor = wavefront_factor(right)
    lines.append(
        "| Shared load wavefront factor | {left} | {right} |".format(
            left=f"{left_factor:.3f}x" if left_factor is not None else "-",
            right=f"{right_factor:.3f}x" if right_factor is not None else "-",
        )
    )
    return "\n".join(lines)


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: summarize_ncu.py <baseline.ncu-rep> <candidate.ncu-rep>", file=sys.stderr)
        return 2
    try:
        left = report_to_record(Path(sys.argv[1]))
        right = report_to_record(Path(sys.argv[2]))
    except (OSError, RuntimeError, StopIteration, ValueError) as error:
        print(f"summarize_ncu: {error}", file=sys.stderr)
        return 1
    print(render_comparison(left, right))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
