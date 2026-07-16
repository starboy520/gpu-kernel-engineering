#!/usr/bin/env python3

import argparse
import csv
import math
import statistics
import subprocess
from pathlib import Path


IMPLEMENTATIONS = ("br4", "m2")
DEFAULT_SHAPES = (128, 256, 512, 1024, 2048)
DEFAULT_D = (64, 128)
DEFAULT_CAUSAL = (0, 1)


def parse_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in text.split():
        key, separator, value = token.partition("=")
        if separator:
            fields[key] = value
    return fields


def run_once(runner: Path, implementation: str, n: int, d: int, causal: int,
             warmup: int, iterations: int) -> dict[str, str]:
    completed = subprocess.run(
        [
            str(runner), "--implementation", implementation,
            "--n", str(n), "--d", str(d), "--causal", str(causal),
            "--mode", "benchmark", "--warmup", str(warmup),
            "--iterations", str(iterations), "--seed", "1234",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    fields = parse_fields(completed.stdout)
    if fields.get("status") != "PASS":
        raise RuntimeError(f"correctness failed: {completed.stdout.strip()}")
    if fields.get("implementation") != implementation:
        raise RuntimeError(f"wrong implementation: {completed.stdout.strip()}")
    return fields


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runner", type=Path,
        default=Path("build/projects/attention_prefill/attention_prefill_evidence_runner"),
    )
    parser.add_argument("--output-csv", type=Path,
                        default=Path("projects/attention_prefill/results/raw/m2-development.csv"))
    parser.add_argument("--output-md", type=Path,
                        default=Path("projects/attention_prefill/results/generated/m2-development.md"))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--shapes", nargs="*", type=int, default=DEFAULT_SHAPES)
    arguments = parser.parse_args()

    if not arguments.runner.is_file():
        raise SystemExit(f"runner not found: {arguments.runner}")
    if arguments.warmup < 0 or arguments.iterations <= 0 or arguments.repeats <= 0:
        raise SystemExit("invalid timing controls")

    rows: list[dict[str, object]] = []
    for d in DEFAULT_D:
        for n in arguments.shapes:
            for causal in DEFAULT_CAUSAL:
                samples: dict[str, list[float]] = {name: [] for name in IMPLEMENTATIONS}
                errors: dict[str, tuple[str, str]] = {}
                for implementation in IMPLEMENTATIONS:
                    for repeat in range(arguments.repeats):
                        print(
                            f"[compare_m2] implementation={implementation} "
                            f"shape={n}x{d} causal={causal} "
                            f"repeat={repeat + 1}/{arguments.repeats}",
                            flush=True,
                        )
                        fields = run_once(
                            arguments.runner, implementation, n, d, causal,
                            arguments.warmup, arguments.iterations,
                        )
                        latency = float(fields["latency_ms"])
                        if not math.isfinite(latency) or latency <= 0.0:
                            raise RuntimeError("invalid latency")
                        samples[implementation].append(latency)
                        errors[implementation] = (fields["max_abs"], fields["max_rel"])

                m1 = statistics.median(samples["br4"])
                m2 = statistics.median(samples["m2"])
                speedup = m1 / m2
                rows.append({
                    "n": n,
                    "d": d,
                    "causal": causal,
                    "m1_query_tiled_ms": m1,
                    "m2_warp_per_query_ms": m2,
                    "speedup": speedup,
                    "delta_pct": 100.0 * (speedup - 1.0),
                    "m1_spread_pct": 100.0 * (max(samples["br4"]) - min(samples["br4"])) / m1,
                    "m2_spread_pct": 100.0 * (max(samples["m2"]) - min(samples["m2"])) / m2,
                    "m1_max_abs": errors["br4"][0],
                    "m2_max_abs": errors["m2"][0],
                })

    arguments.output_csv.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_md.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# M1 Query-tiled vs M2 Warp-per-query Development Benchmark",
        "",
        f"协议：A100 FP32，CUDA Event，warmup={arguments.warmup}，"
        f"iterations={arguments.iterations}，repeats={arguments.repeats} 取中位数。",
        "",
        "> `Speedup = M1 / M2`；大于 1 表示 M2 更快。当前文件是开发期结果，正式发布前需在 clean commit 上重采。",
        "",
        "| N | D | Causal | M1 (ms) | M2 (ms) | Speedup | Delta | M1 spread | M2 spread |",
        "| ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['n']} | {row['d']} | {row['causal']} | "
            f"{row['m1_query_tiled_ms']:.6f} | {row['m2_warp_per_query_ms']:.6f} | "
            f"{row['speedup']:.3f}x | {row['delta_pct']:+.1f}% | "
            f"{row['m1_spread_pct']:.2f}% | {row['m2_spread_pct']:.2f}% |"
        )
    arguments.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[compare_m2] wrote rows={len(rows)} csv={arguments.output_csv} md={arguments.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
