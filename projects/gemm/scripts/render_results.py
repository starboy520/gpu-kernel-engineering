#!/usr/bin/env python3

import csv
import os
import sys
from collections import defaultdict


KERNEL_ORDER = [
    "naive",
    "shared",
    "register",
    "vectorized",
    "async-16b",
    "cublas-fp32",
]

REQUIRED_COLUMNS = [
    "timestamp",
    "git_commit",
    "gpu",
    "cuda",
    "nvcc",
    "kernel",
    "path",
    "m",
    "n",
    "k",
    "warmup",
    "iterations",
    "latency_ms",
    "gflops",
    "passed",
    "max_abs",
    "max_rel",
    "reference",
]


def fail(message: str) -> int:
    sys.stderr.write(f"{message}\n")
    return 1


def parse_args() -> tuple[str, str]:
    if len(sys.argv) != 3:
        raise SystemExit("usage: render_results.py <input_csv> <output_md>")
    return sys.argv[1], sys.argv[2]


def load_rows(csv_path: str) -> list[dict[str, str]]:
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        for column in REQUIRED_COLUMNS:
            if column not in fieldnames:
                raise ValueError(f"missing required column: {column}")
        return list(reader)


def shape_key(row: dict[str, str]) -> tuple[int, int, int]:
    return int(row["m"]), int(row["n"]), int(row["k"])


def format_speedup(previous_latency: float | None, current_latency: float) -> str:
    if previous_latency is None or previous_latency <= 0.0 or current_latency <= 0.0:
        return "-"
    return f"{previous_latency / current_latency:.2f}x"


def format_cublas_share(current_gflops: float, cublas_gflops: float | None) -> str:
    if cublas_gflops is None or cublas_gflops <= 0.0 or current_gflops <= 0.0:
        return "-"
    return f"{100.0 * current_gflops / cublas_gflops:.1f}%"


def render_markdown(rows: list[dict[str, str]], source_csv: str) -> str:
    grouped: dict[tuple[int, int, int], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        grouped[shape_key(row)][row["kernel"]] = row

    lines = [
        "# A100 FP32 GEMM 基准结果",
        "",
        f"来源 CSV：`{source_csv}`",
        "",
        "说明：只渲染 CSV 中真实存在的行，不为缺失 kernel 补值；`Path` 表示运行时实际选择的实现路径，`Reference` 表示正确性对拍来源。",
        "",
    ]

    for shape in sorted(grouped):
        m, n, k = shape
        rows_by_kernel = grouped[shape]
        cublas_row = rows_by_kernel.get("cublas-fp32")
        cublas_gflops = None
        if cublas_row is not None:
            cublas_gflops = float(cublas_row["gflops"])

        lines.extend(
            [
                f"## {m}x{n}x{k}",
                "",
                "| Kernel | Path | Reference | Latency (ms) | GFLOPS | Prev Speedup | % of cuBLAS |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )

        previous_latency = None
        for kernel in KERNEL_ORDER:
            row = rows_by_kernel.get(kernel)
            if row is None:
                continue

            latency = float(row["latency_ms"])
            gflops = float(row["gflops"])
            lines.append(
                "| {kernel} | {path} | {reference} | {latency:.6f} | {gflops:.2f} | {speedup} | {share} |".format(
                    kernel=kernel,
                    path=row["path"] or "-",
                    reference=row["reference"] or "-",
                    latency=latency,
                    gflops=gflops,
                    speedup=format_speedup(previous_latency, latency),
                    share=format_cublas_share(gflops, cublas_gflops),
                )
            )
            previous_latency = latency

        lines.append("")

    return "\n".join(lines)


def main() -> int:
    input_csv, output_md = parse_args()
    try:
        rows = load_rows(input_csv)
    except ValueError as error:
        return fail(str(error))

    output_directory = os.path.dirname(output_md)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    rendered = render_markdown(rows, input_csv)
    with open(output_md, "w", encoding="utf-8") as handle:
        handle.write(rendered)
        if not rendered.endswith("\n"):
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())