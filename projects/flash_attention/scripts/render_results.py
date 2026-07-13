#!/usr/bin/env python3

import csv
import sys
from collections import defaultdict
from pathlib import Path


KERNEL_ORDER = ["naive", "tiled", "tiled-parallel", "tiled-async"]
REQUIRED_COLUMNS = [
    "timestamp", "git_commit", "runner_sha256", "gpu", "gpu_uuid", "sm",
    "driver", "cuda", "nvcc", "build_preset", "dtype", "batch", "heads",
    "layout", "kernel", "path", "n", "d", "causal", "input_pattern",
    "seed", "warmup", "iterations", "repeats", "latency_ms",
    "latency_min_ms", "latency_max_ms", "spread_pct", "passed", "max_abs",
    "max_rel", "workspace_bytes", "reference", "timing",
]


def speedup(baseline_ms: float, candidate_ms: float) -> float:
    return baseline_ms / candidate_ms


def delta_percent(baseline_ms: float, candidate_ms: float) -> float:
    return 100.0 * (speedup(baseline_ms, candidate_ms) - 1.0)


def classify(async_speedup: float, spread_pct: float) -> str:
    if spread_pct > 3.0:
        return "inconclusive"
    if async_speedup >= 1.05:
        return "benefit"
    if async_speedup <= 0.95:
        return "regression"
    return "near-parity"


def summarize_boundary(samples: list[tuple[int, str]]) -> tuple[int | None, str, int | None, str]:
    non_benefit = [n for n, kind in samples if kind in ("regression", "near-parity")]
    inconclusive = [str(n) for n, kind in samples if kind == "inconclusive"]
    benefits = [n for n, kind in samples if kind == "benefit"]
    first_benefit = min(benefits, default=None)
    last_non = max((n for n in non_benefit if first_benefit is None or n < first_benefit), default=None)
    bracket = f"{last_non}–{first_benefit}" if last_non is not None and first_benefit is not None else "未定位"
    return last_non, ", ".join(inconclusive) or "-", first_benefit, bracket


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [name for name in REQUIRED_COLUMNS if name not in fieldnames]
        if missing:
            raise ValueError(f"missing required columns: {', '.join(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("result CSV contains no rows")
    return rows


def fmt_delta(value: float) -> str:
    return f"{value:+.1f}%"


def portable_source(source: str) -> str:
    marker = "projects/flash_attention/"
    normalized = source.replace("\\", "/")
    if marker in normalized:
        return marker + normalized.split(marker, 1)[1]
    return normalized


def render(rows: list[dict[str, str]], source: str) -> str:
    by_problem: dict[tuple[int, int, int], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        key = (int(row["causal"]), int(row["d"]), int(row["n"]))
        by_problem[key][row["kernel"]] = row

    first = rows[0]
    smoke = Path(source).name == "smoke.csv"
    title = "Smoke（非性能证据）" if smoke else "Canonical Benchmark"
    lines = [
        f"# A100 FP32 FlashAttention {title}",
        "",
        f"来源 CSV：`{portable_source(source)}`",
        "",
        ("协议：CUDA Event，warmup {warmup}，iterations {iterations}，"
         "repeats {repeats} 取 latency 中位数；seed {seed}。"
         ).format(**first),
        "",
        (f"环境：{first['gpu']}，SM {first['sm']}，driver {first['driver']}，"
         f"commit `{first['git_commit']}`。"),
        "",
        "> `Async Δ = 100 × (T_tiled / T_async - 1)`：正数表示 Async 更快，负数表示回退。它是 speedup delta，不是常规延迟增幅；ncu 时间不进入本表。",
        "",
    ]

    for causal in (0, 1):
        lines.extend([f"## causal={causal}", ""])
        for d in sorted({key[1] for key in by_problem if key[0] == causal}):
            lines.extend([
                f"### D={d}",
                "",
                "| N | Naive (ms) | Tiled (ms) | Parallel (ms) | Async (ms) | Async Δ | Async spread | 分类 |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ])
            for key in sorted(k for k in by_problem if k[0] == causal and k[1] == d):
                n = key[2]
                problem = by_problem[key]
                if any(kernel not in problem for kernel in KERNEL_ORDER):
                    continue
                tiled_ms = float(problem["tiled"]["latency_ms"])
                async_ms = float(problem["tiled-async"]["latency_ms"])
                async_spread = float(problem["tiled-async"]["spread_pct"])
                async_speedup = speedup(tiled_ms, async_ms)
                lines.append(
                    "| {n} | {naive:.6f} | {tiled:.6f} | {parallel:.6f} | "
                    "{async_ms:.6f} | {delta} | {spread:.2f}% | {kind} |".format(
                        n=n,
                        naive=float(problem["naive"]["latency_ms"]),
                        tiled=tiled_ms,
                        parallel=float(problem["tiled-parallel"]["latency_ms"]),
                        async_ms=async_ms,
                        delta=fmt_delta(delta_percent(tiled_ms, async_ms)),
                        spread=async_spread,
                        kind=classify(async_speedup, async_spread),
                    )
                )
            lines.append("")

    lines.extend([
        "## Async 收益边界",
        "",
        "| D | Causal | 最后回退/持平 N | Inconclusive N | 首个收益 N | 边界区间 |",
        "| ---: | :---: | ---: | --- | ---: | --- |",
    ])
    groups = sorted({(key[1], key[0]) for key in by_problem})
    for d, causal in groups:
        samples: list[tuple[int, str]] = []
        for key in sorted(k for k in by_problem if k[0] == causal and k[1] == d):
            problem = by_problem[key]
            if "tiled" not in problem or "tiled-async" not in problem:
                continue
            tiled_ms = float(problem["tiled"]["latency_ms"])
            async_ms = float(problem["tiled-async"]["latency_ms"])
            spread = float(problem["tiled-async"]["spread_pct"])
            samples.append((key[2], classify(speedup(tiled_ms, async_ms), spread)))
        last_non, inconclusive, first_benefit, bracket = summarize_boundary(samples)
        lines.append(
            f"| {d} | {causal} | {last_non if last_non is not None else '-'} | "
            f"{inconclusive} | {first_benefit if first_benefit is not None else '-'} | {bracket} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_file(csv_path: Path, markdown_path: Path, source_display: str | None = None) -> None:
    rows = load_rows(csv_path)
    text = render(rows, source_display or csv_path.as_posix())
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(text + ("" if text.endswith("\n") else "\n"), encoding="utf-8")


def main() -> int:
    if len(sys.argv) not in (3, 4):
        print("usage: render_results.py <input.csv> <output.md> [source-display]", file=sys.stderr)
        return 2
    try:
        render_file(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] if len(sys.argv) == 4 else None)
    except (OSError, ValueError) as error:
        print(f"render_results: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
