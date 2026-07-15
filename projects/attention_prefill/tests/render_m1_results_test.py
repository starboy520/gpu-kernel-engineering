#!/usr/bin/env python3

import csv
import importlib.util
import math
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "render_m1_results.py"
PROVENANCE_COLUMNS = [
    "source_sha256", "build_contract", "build_contract_payload_sha256",
    "device_index",
]


def load_renderer():
    spec = importlib.util.spec_from_file_location("render_m1_results", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RenderM1ResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.renderer = load_renderer()

    def make_row(self, implementation, **overrides):
        n = int(overrides.get("n", 128))
        d = int(overrides.get("d", 64))
        cta_count = n if implementation == "br1" else math.ceil(n / 4)
        latency = 2.0 if implementation == "br1" else 1.0
        row = {
            column: "x"
            for column in dict.fromkeys(
                [*self.renderer.REQUIRED_COLUMNS, *PROVENANCE_COLUMNS]
            )
        }
        row.update({
            "timestamp": "2026-07-15T00:00:00Z",
            "git_commit": "0123456789abcdef",
            "runner_sha256": "a" * 64,
            "source_sha256": "b" * 64,
            "build_contract": "release-sm80-cccccccccccccccc",
            "build_contract_payload_sha256": "c" * 64,
            "device_index": "0",
            "gpu": "NVIDIA A100-SXM4-80GB",
            "gpu_uuid": "GPU-test",
            "sm": "8.0",
            "driver": "575.57.08",
            "cuda": "Cuda compilation tools; release 12.8; V12.8.93",
            "nvcc": "nvcc test",
            "build_preset": "release-sm80",
            "dtype": "fp32",
            "batch": "1",
            "heads": "1",
            "layout": "row-major",
            "implementation": implementation,
            "path": implementation,
            "n": str(n),
            "d": str(d),
            "causal": "0",
            "input_pattern": "random",
            "seed": "1234",
            "warmup": "10",
            "iterations": "50",
            "repeats": "3",
            "latency_ms": str(latency),
            "latency_min_ms": str(latency * 0.99),
            "latency_max_ms": str(latency * 1.01),
            "spread_pct": "2.0",
            "passed": "true",
            "max_abs": "1e-6",
            "max_rel": "2e-6",
            "workspace_bytes": "0",
            "reference": "cpu-double",
            "timing": "cuda-event",
            "cta_count": str(cta_count),
            "requested_kv_elements": str(2 * cta_count * n * d),
        })
        row.update({key: str(value) for key, value in overrides.items()})
        return row

    def canonical_rows(self):
        return [
            self.make_row(implementation, n=n, d=d, causal=causal)
            for n in (128, 256, 512, 1024, 2048)
            for d in (64, 128)
            for causal in (0, 1)
            for implementation in ("br1", "br4")
        ]

    def write_csv(self, csv_path, rows, columns=None):
        fieldnames = columns or list(dict.fromkeys([
            *self.renderer.REQUIRED_COLUMNS, *PROVENANCE_COLUMNS,
        ]))
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fieldnames, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)

    def render_smoke(self, rows):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        csv_path = Path(directory.name) / "smoke.csv"
        markdown_path = Path(directory.name) / "smoke.md"
        self.write_csv(csv_path, rows)
        self.renderer.render_file(csv_path, markdown_path)
        return markdown_path.read_text(encoding="utf-8")

    def load_smoke(self, rows):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        csv_path = Path(directory.name) / "smoke.csv"
        self.write_csv(csv_path, rows)
        return self.renderer.load_rows(csv_path)

    def pair_line(self, br1_latency, br4_latency):
        rows = []
        for implementation, latency in (("br1", br1_latency), ("br4", br4_latency)):
            rows.append(self.make_row(
                implementation,
                latency_ms=latency,
                latency_min_ms=latency,
                latency_max_ms=latency,
                spread_pct=0,
            ))
        text = self.render_smoke(rows)
        return next(line for line in text.splitlines() if line.startswith("| 128 |"))

    def test_complete_pair_renders_speedup_delta_classification_and_request_metrics(self):
        text = self.render_smoke([self.make_row("br1"), self.make_row("br4")])

        self.assertIn("2.000x", text)
        self.assertIn("+100.0%", text)
        self.assertIn("benefit", text)
        self.assertIn("128 / 32", text)
        self.assertIn("4.000x", text)
        self.assertIn("理论 requested elements 不等于 DRAM bytes", text)
        self.assertIn("ncu duration 不进入本表", text)

    def test_spread_above_three_percent_on_either_path_is_inconclusive(self):
        text = self.render_smoke([
            self.make_row(
                "br1", latency_ms=1, latency_min_ms=0.9849,
                latency_max_ms=1.0151, spread_pct=3.02,
            ),
            self.make_row(
                "br4", latency_ms=1, latency_min_ms=1,
                latency_max_ms=1, spread_pct=0,
            ),
        ])

        line = next(line for line in text.splitlines() if line.startswith("| 128 |"))
        self.assertTrue(line.endswith("| inconclusive |"))

    def test_missing_required_columns_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "smoke.csv"
            columns = [
                column
                for column in dict.fromkeys([
                    *self.renderer.REQUIRED_COLUMNS, *PROVENANCE_COLUMNS,
                ])
                if column != "runner_sha256"
            ]
            rows = [
                {key: value for key, value in self.make_row("br1").items() if key in columns},
                {key: value for key, value in self.make_row("br4").items() if key in columns},
            ]
            self.write_csv(csv_path, rows, columns)

            with self.assertRaisesRegex(ValueError, "missing required columns: runner_sha256"):
                self.renderer.load_rows(csv_path)

    def test_unclosed_quote_is_rejected_as_malformed_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "smoke.csv"
            csv_path.write_text(
                ",".join(dict.fromkeys([
                    *self.renderer.REQUIRED_COLUMNS, *PROVENANCE_COLUMNS,
                ])) + "\n\"unterminated",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "malformed CSV"):
                self.renderer.load_rows(csv_path)

    def test_noncanonical_csv_quoting_and_bare_quote_are_rejected(self):
        rows = [self.make_row("br1"), self.make_row("br4")]
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "smoke.csv"
            self.write_csv(csv_path, rows)
            original = csv_path.read_text(encoding="utf-8")
            csv_path.write_text(
                original.replace(",random,", ',"random",', 1),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "non-canonical CSV encoding"):
                self.renderer.load_rows(csv_path)

            csv_path.write_text(
                original.replace(",random,", ',ran"dom,', 1),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "non-canonical CSV encoding"):
                self.renderer.load_rows(csv_path)

    def test_extra_header_and_extra_row_fields_are_rejected(self):
        row = self.make_row("br1")
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "smoke.csv"
            columns = [
                *dict.fromkeys([
                    *self.renderer.REQUIRED_COLUMNS, *PROVENANCE_COLUMNS,
                ]),
                "unexpected",
            ]
            row["unexpected"] = "value"
            self.write_csv(csv_path, [row], columns)
            with self.assertRaisesRegex(ValueError, "unexpected columns"):
                self.renderer.load_rows(csv_path)

        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "smoke.csv"
            columns = list(dict.fromkeys([
                *self.renderer.REQUIRED_COLUMNS, *PROVENANCE_COLUMNS,
            ]))
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(columns)
                writer.writerow([row[column] for column in columns] + ["extra"])
            with self.assertRaisesRegex(ValueError, "extra fields"):
                self.renderer.load_rows(csv_path)

    def test_row_with_missing_field_is_rejected_as_column_count_mismatch(self):
        row = self.make_row("br1")
        columns = list(dict.fromkeys([
            *self.renderer.REQUIRED_COLUMNS, *PROVENANCE_COLUMNS,
        ]))
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "smoke.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(columns)
                writer.writerow([row[column] for column in columns[:-1]])
            with self.assertRaisesRegex(ValueError, "column count mismatch"):
                self.renderer.load_rows(csv_path)

    def test_source_build_and_device_provenance_are_required_and_validated(self):
        cases = (
            ("source_sha256", "short", "source_sha256"),
            ("build_contract", "release sm80", "build_contract"),
            ("build_contract_payload_sha256", "short", "payload"),
            ("device_index", "-1", "device_index"),
        )
        for column, value, message in cases:
            with self.subTest(column=column):
                with self.assertRaisesRegex(ValueError, message):
                    self.load_smoke([
                        self.make_row("br1", **{column: value}),
                        self.make_row("br4", **{column: value}),
                    ])

    def test_duplicate_implementation_for_problem_is_rejected(self):
        duplicate = self.make_row("br1", latency_ms="2.1")
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "smoke.csv"
            self.write_csv(csv_path, [self.make_row("br1"), duplicate, self.make_row("br4")])

            with self.assertRaisesRegex(ValueError, "duplicate implementation pair"):
                self.renderer.load_rows(csv_path)

    def test_missing_implementation_pair_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "smoke.csv"
            self.write_csv(csv_path, [self.make_row("br1")])

            with self.assertRaisesRegex(ValueError, "missing implementation pair.*br4"):
                self.renderer.load_rows(csv_path)

    def test_empty_csv_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "smoke.csv"
            self.write_csv(csv_path, [])

            with self.assertRaisesRegex(ValueError, "contains no rows"):
                self.renderer.load_rows(csv_path)

    def test_non_smoke_canonical_csv_must_have_exactly_forty_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "a100-fp32-m1.csv"
            self.write_csv(csv_path, [self.make_row("br1"), self.make_row("br4")])

            with self.assertRaisesRegex(ValueError, "canonical result CSV must contain exactly 40 rows"):
                self.renderer.load_rows(csv_path)

    def test_classification_exact_thresholds_and_near_parity_are_in_data_rows(self):
        cases = (
            (1.05, 1.0, "benefit"),
            (0.95, 1.0, "regression"),
            (1.049, 1.0, "near-parity"),
            (0.951, 1.0, "near-parity"),
        )
        for br1_latency, br4_latency, expected in cases:
            with self.subTest(speedup=br1_latency / br4_latency):
                self.assertTrue(
                    self.pair_line(br1_latency, br4_latency).endswith(
                        f"| {expected} |"
                    )
                )

    def test_spread_must_match_min_max_and_median_with_rounding_tolerance(self):
        self.load_smoke([
            self.make_row(
                "br1", latency_ms="3.000000", latency_min_ms="2.999999",
                latency_max_ms="3.000001", spread_pct="0.000067",
            ),
            self.make_row("br4"),
        ])

        with self.assertRaisesRegex(ValueError, "spread_pct does not match"):
            self.load_smoke([
                self.make_row("br1", spread_pct="1.5"),
                self.make_row("br4"),
            ])

    def test_all_floating_metrics_must_be_finite(self):
        columns = (
            "latency_ms", "latency_min_ms", "latency_max_ms",
            "spread_pct", "max_abs", "max_rel",
        )
        for column in columns:
            for value in ("nan", "inf", "-inf"):
                with self.subTest(column=column, value=value):
                    with self.assertRaisesRegex(
                        ValueError, f"{column} must be finite"
                    ):
                        self.load_smoke([
                            self.make_row("br1", **{column: value}),
                            self.make_row("br4"),
                        ])

    def test_error_metrics_must_be_nonnegative(self):
        for column in ("max_abs", "max_rel"):
            with self.subTest(column=column):
                with self.assertRaisesRegex(
                    ValueError, f"{column} must be nonnegative"
                ):
                    self.load_smoke([
                        self.make_row("br1", **{column: -0.1}),
                        self.make_row("br4"),
                    ])

    def test_canonical_requires_exact_fixed_matrix_not_only_forty_rows(self):
        rows = self.canonical_rows()
        for row in rows:
            if row["n"] == "2048":
                row["n"] = "4096"
                cta = 4096 if row["implementation"] == "br1" else 1024
                row["cta_count"] = str(cta)
                row["requested_kv_elements"] = str(
                    2 * cta * 4096 * int(row["d"])
                )

        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "a100-fp32-m1.csv"
            self.write_csv(csv_path, rows)
            with self.assertRaisesRegex(ValueError, "canonical matrix mismatch"):
                self.renderer.load_rows(csv_path)

    def test_canonical_requires_a100_sm80_and_fixed_protocol(self):
        cases = {
            "gpu": "NVIDIA H100",
            "sm": "9.0",
            "build_preset": "debug-sm80",
            "dtype": "fp16",
            "batch": "2",
            "heads": "8",
            "layout": "column-major",
            "input_pattern": "ones",
            "seed": "7",
            "warmup": "9",
            "iterations": "49",
            "repeats": "2",
            "workspace_bytes": "4",
            "reference": "gpu-reference",
            "timing": "wall-clock",
        }
        for column, value in cases.items():
            with self.subTest(column=column):
                rows = self.canonical_rows()
                for row in rows:
                    row[column] = value
                with tempfile.TemporaryDirectory() as directory:
                    csv_path = Path(directory) / "a100-fp32-m1.csv"
                    self.write_csv(csv_path, rows)
                    with self.assertRaisesRegex(ValueError, "canonical protocol"):
                        self.renderer.load_rows(csv_path)

    def test_canonical_rejects_a100_40gb(self):
        rows = self.canonical_rows()
        for row in rows:
            row["gpu"] = "NVIDIA A100-SXM4-40GB"
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "a100-fp32-m1.csv"
            self.write_csv(csv_path, rows)
            with self.assertRaisesRegex(ValueError, "A100 80GB"):
                self.renderer.load_rows(csv_path)

    def test_canonical_build_contract_prefix_matches_payload_hash(self):
        rows = self.canonical_rows()
        for row in rows:
            row["build_contract"] = "release-sm80-0000000000000000"
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "a100-fp32-m1.csv"
            self.write_csv(csv_path, rows)
            with self.assertRaisesRegex(ValueError, "payload hash prefix"):
                self.renderer.load_rows(csv_path)

    def test_path_cta_and_requested_elements_match_implementation(self):
        cases = (
            ("path", "wrong-path", "path must match implementation"),
            ("cta_count", "31", "cta_count mismatch"),
            ("requested_kv_elements", "1", "requested_kv_elements mismatch"),
        )
        for column, value, message in cases:
            with self.subTest(column=column):
                with self.assertRaisesRegex(ValueError, message):
                    self.load_smoke([
                        self.make_row("br1"),
                        self.make_row("br4", **{column: value}),
                    ])

    def test_markdown_escapes_table_pipes_newlines_and_backticks(self):
        special = "dev|preset\nline `quoted`"
        text = self.render_smoke([
            self.make_row("br1", build_preset=special),
            self.make_row("br4", build_preset=special),
        ])

        self.assertIn("dev\\|preset<br>line 'quoted'", text)
        self.assertNotIn("dev|preset\nline", text)


if __name__ == "__main__":
    unittest.main()
