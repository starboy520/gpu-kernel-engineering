#!/usr/bin/env python3

import csv
import importlib.util
import math
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "summarize_m1_ncu.py"


def load_module():
    spec = importlib.util.spec_from_file_location("summarize_m1_ncu", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SummarizeM1NcuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = load_module()

    def make_record(self, implementation, n, d, causal, scale=1.0, **identity_overrides):
        symbol = (
            "tiled_attention_kernel"
            if implementation == "br1"
            else "query_tiled_kernel"
        )
        identity = {
            "evidence_kind": "canonical",
            "implementation": implementation,
            "n": str(n),
            "d": str(d),
            "causal": str(causal),
            "kernel_name": (
                f"<unnamed>::{symbol}(const float *, const float *, "
                "const float *, float *, int, int, bool)"
            ),
            "process_name": "attention_prefill_evidence_runner",
            "block_size": "(128, 1, 1)",
            "grid_size": f"({n if implementation == 'br1' else math.ceil(n / 4)}, 1, 1)",
            "profile_timestamp": "2026-07-15T12:34:56Z",
            "git_commit": "a" * 40,
            "runner_sha256": "1" * 64,
            "source_sha256": "2" * 64,
            "build_contract": "release-sm80-" + "3" * 16,
            "build_contract_payload_sha256": "3" * 64,
            "device_index": "0",
            "gpu_uuid": "GPU-6b24bab9-c81f-0e0e-ff68-a11dcf17018b",
            "gpu_name": "NVIDIA_A100_80GB_PCIe",
            "sm": "8.0",
            "driver": "13030",
            "ncu_version": "2026.2.0.0",
        }
        identity.update(identity_overrides)
        values = {
            metric.name: str((index + 1) * scale)
            for index, metric in enumerate(self.summary.METRICS)
        }
        units = {metric.name: metric.test_unit for metric in self.summary.METRICS}
        return self.summary.Record(identity=identity, values=values, units=units)

    def write_normalized(self, path, records):
        self.summary.write_normalized_csv(path, records)

    def canonical_records(self):
        return [
            self.make_record(implementation, n, d, causal, 1.0 if implementation == "br1" else 0.5)
            for n, d, causal in ((256, 64, 0), (1024, 128, 0), (1024, 128, 1))
            for implementation in ("br1", "br4")
        ]

    def test_six_files_render_paired_compact_markdown_with_units_and_caveat(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for record in self.canonical_records():
                identity = record.identity
                path = Path(directory) / (
                    f"{identity['implementation']}-{identity['n']}x{identity['d']}-"
                    f"causal{identity['causal']}-metrics.csv"
                )
                self.write_normalized(path, [record])
                paths.append(path)

            records = self.summary.load_inputs(paths, require_canonical=True)
            rendered = self.summary.render(records, paths)

        self.assertEqual(len(records), 6)
        self.assertIn("256x64 causal=0", rendered)
        self.assertIn("1024x128 causal=1", rendered)
        self.assertIn("Registers/thread", rendered)
        self.assertIn("Grid", rendered)
        self.assertIn("Achieved occupancy", rendered)
        self.assertIn("Eligible warps/cycle", rendered)
        self.assertIn("Long scoreboard", rendered)
        self.assertIn("Short scoreboard", rendered)
        self.assertIn("Barrier stall", rendered)
        self.assertIn("Warp latency", rendered)
        self.assertIn("SM throughput", rendered)
        self.assertIn("DRAM read bytes", rendered)
        self.assertIn("DRAM write bytes", rendered)
        self.assertIn("L2 read sectors", rendered)
        self.assertIn("L2 write sectors", rendered)
        self.assertIn("Global load sectors", rendered)
        self.assertIn("Global store sectors", rendered)
        self.assertIn("Global store requests", rendered)
        self.assertIn("Shared bank conflicts", rendered)
        self.assertIn("Br4 / Br1", rendered)
        self.assertIn("register/thread", rendered)
        self.assertIn("ncu duration 仅用于 profiler", rendered)
        self.assertIn("不等于 CUDA Event 或端到端 wall-clock", rendered)
        self.assertIn("## 环境与构建身份", rendered)
        self.assertIn("Runner SHA-256", rendered)
        self.assertIn("Source SHA-256", rendered)
        self.assertIn("Build contract", rendered)
        self.assertIn("GPU UUID", rendered)
        self.assertIn("NVIDIA_A100_80GB_PCIe", rendered)
        self.assertIn("2026.2.0.0", rendered)

    def test_missing_pair_duplicate_and_wrong_symbol_are_rejected(self):
        records = self.canonical_records()
        with self.assertRaisesRegex(ValueError, "missing canonical profile"):
            self.summary.validate_records(records[:-1], require_canonical=True)

        with self.assertRaisesRegex(ValueError, "duplicate profile identity"):
            self.summary.validate_records([*records, records[0]], require_canonical=True)

        wrong = self.make_record("br4", 256, 64, 0)
        wrong.identity["kernel_name"] = "tiled_attention_kernel(float *)"
        with self.assertRaisesRegex(ValueError, "kernel symbol mismatch"):
            self.summary.validate_records([wrong], require_canonical=False)

        suffix_collision = self.make_record("br4", 256, 64, 0)
        suffix_collision.identity["kernel_name"] = (
            "<unnamed>::query_tiled_kernel_extra(const float *, const float *, "
            "const float *, float *, int, int, bool)"
        )
        with self.assertRaisesRegex(ValueError, "kernel symbol mismatch"):
            self.summary.validate_records(
                [suffix_collision], require_canonical=False
            )

        wrong_signature = self.make_record("br4", 256, 64, 0)
        wrong_signature.identity["kernel_name"] = (
            "<unnamed>::query_tiled_kernel(float const*, float const*, "
            "float const*, float*, int, int, int)"
        )
        with self.assertRaisesRegex(ValueError, "kernel symbol mismatch"):
            self.summary.validate_records([wrong_signature], require_canonical=False)

    def test_missing_metric_and_nonfinite_value_are_rejected(self):
        missing = self.make_record("br1", 256, 64, 0)
        missing.values.pop(self.summary.METRICS[-1].name)
        with self.assertRaisesRegex(ValueError, "missing required metric"):
            self.summary.validate_records([missing], require_canonical=False)

        for text in ("nan", "inf", "-inf"):
            with self.subTest(value=text):
                invalid = self.make_record("br1", 256, 64, 0)
                invalid.values[self.summary.METRICS[0].name] = text
                with self.assertRaisesRegex(ValueError, "must be finite"):
                    self.summary.validate_records([invalid], require_canonical=False)

    def test_pair_rejects_different_gpu_binary_and_build_identity(self):
        mismatch_cases = (
            ({"gpu_uuid": "GPU-00000000-0000-0000-0000-000000000000"}, "device"),
            ({"runner_sha256": "f" * 64}, "binary"),
            ({
                "build_contract": "release-sm80-" + "4" * 16,
                "build_contract_payload_sha256": "4" * 64,
            }, "build"),
            ({"ncu_version": "2025.3.1.0"}, "ncu"),
        )
        for overrides, expected_message in mismatch_cases:
            with self.subTest(overrides=overrides):
                br1 = self.make_record("br1", 256, 64, 0)
                br4 = self.make_record("br4", 256, 64, 0, **overrides)
                with self.assertRaisesRegex(ValueError, expected_message):
                    self.summary.validate_records(
                        [br1, br4], require_canonical=False
                    )

    def test_canonical_rejects_smoke_kind_and_unverified_ncu_version(self):
        smoke = self.canonical_records()
        for record in smoke:
            record.identity["evidence_kind"] = "smoke"
        with self.assertRaisesRegex(ValueError, "evidence_kind=canonical"):
            self.summary.validate_records(smoke, require_canonical=True)

        unsupported = self.canonical_records()
        for record in unsupported:
            record.identity["ncu_version"] = "2027.1.0"
        with self.assertRaisesRegex(ValueError, "2026\\.2"):
            self.summary.validate_records(unsupported, require_canonical=True)

    def test_canonical_rejects_non_release_sm80_build_contract(self):
        records = self.canonical_records()
        for record in records:
            record.identity["build_contract"] = "debug-sm90-evil"

        with self.assertRaisesRegex(
            ValueError, "canonical profiles require build_contract"
        ):
            self.summary.validate_records(records, require_canonical=True)

    def test_ratio_converts_time_and_decimal_byte_units(self):
        br1 = self.make_record("br1", 256, 64, 0)
        br4 = self.make_record("br4", 256, 64, 0)
        duration = self.summary.METRICS[0]
        read_bytes = next(
            metric for metric in self.summary.METRICS
            if metric.name == "dram__bytes_read.sum"
        )
        br1.values[duration.name], br1.units[duration.name] = "900", "us"
        br4.values[duration.name], br4.units[duration.name] = "1.1", "ms"
        br1.values[read_bytes.name], br1.units[read_bytes.name] = "1000", "byte"
        br4.values[read_bytes.name], br4.units[read_bytes.name] = "1", "Kbyte"

        self.assertEqual(self.summary._ratio(br1, br4, duration), "1.222x")
        self.assertEqual(self.summary._ratio(br1, br4, read_bytes), "1.000x")

    def test_ratio_normalizes_declared_count_unit_aliases_to_the_same_base(self):
        br1 = self.make_record("br1", 256, 64, 0)
        br4 = self.make_record("br4", 256, 64, 0)
        sectors = next(
            metric for metric in self.summary.METRICS
            if metric.name == "lts__t_sectors_op_read.sum"
        )
        br1.values[sectors.name], br1.units[sectors.name] = "1000", ""
        br4.values[sectors.name], br4.units[sectors.name] = "500", "sector"

        self.assertEqual(self.summary._ratio(br1, br4, sectors), "0.500x")

    def test_incompatible_and_illegal_empty_metric_units_are_rejected(self):
        duration = self.summary.METRICS[0]
        count_metric = next(
            metric for metric in self.summary.METRICS
            if metric.name == "lts__t_sectors_op_read.sum"
        )
        for metric, unit in ((duration, "byte"), (duration, "")):
            with self.subTest(metric=metric.name, unit=unit):
                record = self.make_record("br1", 256, 64, 0)
                record.units[metric.name] = unit
                with self.assertRaisesRegex(ValueError, "unit"):
                    self.summary._validate_record(record)

                count = self.make_record("br1", 256, 64, 0)
                count.units[count_metric.name] = ""
                self.summary._validate_record(count)

    def test_identity_tokens_are_valid_and_timestamp_is_optional(self):
        record = self.make_record("br1", 256, 64, 0, profile_timestamp="")
        self.summary._validate_record(record)

        invalid_fields = {
            "runner_sha256": "not-a-hash",
            "source_sha256": "A" * 64,
            "build_contract": "release sm80",
            "build_contract_payload_sha256": "b" * 63,
            "device_index": "-1",
            "gpu_uuid": "not-a-gpu-uuid",
            "gpu_name": "NVIDIA A100 80GB",
            "sm": "sm_80",
            "driver": "13.3",
            "ncu_version": "Version 2026.2.0.0",
            "profile_timestamp": "2026/07/15 12:34:56",
        }
        for field, value in invalid_fields.items():
            with self.subTest(field=field):
                invalid = self.make_record("br1", 256, 64, 0, **{field: value})
                with self.assertRaisesRegex(ValueError, re.escape(field)):
                    self.summary._validate_record(invalid)

    def test_canonical_profiles_require_a100_80gb_sm80(self):
        records = self.canonical_records()
        for record in records:
            record.identity["gpu_name"] = "NVIDIA_A100-PCIE-40GB"
        with self.assertRaisesRegex(ValueError, "A100 80GB"):
            self.summary.validate_records(records, require_canonical=True)

    def test_block_and_grid_identity_must_match_implementation(self):
        wrong_block = self.make_record("br1", 256, 64, 0)
        wrong_block.identity["block_size"] = "(64, 1, 1)"
        with self.assertRaisesRegex(ValueError, "block identity mismatch"):
            self.summary._validate_record(wrong_block)

        wrong_grid = self.make_record("br4", 256, 64, 0)
        wrong_grid.identity["grid_size"] = "(256, 1, 1)"
        with self.assertRaisesRegex(ValueError, "grid identity mismatch"):
            self.summary._validate_record(wrong_grid)

    def test_raw_ncu_csv_normalizes_one_kernel_and_preserves_units(self):
        record = self.make_record("br4", 256, 64, 0)
        fields = [
            *self.summary.NCU_RAW_IDENTITY_COLUMNS,
            *sorted(metric.name for metric in self.summary.METRICS),
        ]
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "raw.csv"
            normalized_path = Path(directory) / "normalized.csv"
            with raw_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(fields)
                writer.writerow([
                    *("" for _ in self.summary.NCU_RAW_IDENTITY_COLUMNS),
                    *[
                        record.units[name]
                        for name in sorted(record.units)
                    ],
                ])
                identity_values = {
                    "Process Name": record.identity["process_name"],
                    "Kernel Name": record.identity["kernel_name"],
                    "Block Size": record.identity["block_size"],
                    "Grid Size": record.identity["grid_size"],
                }
                writer.writerow([
                    *(identity_values.get(name, "x") for name in self.summary.NCU_RAW_IDENTITY_COLUMNS),
                    *[
                        record.values[name]
                        for name in sorted(record.values)
                    ],
                ])

            provenance = {
                name: value
                for name, value in record.identity.items()
                if name not in {
                    "implementation", "n", "d", "causal", "kernel_name",
                    "process_name", "block_size", "grid_size",
                }
            }
            normalized = self.summary.normalize_raw_csv(
                raw_path, "br4", 256, 64, 0, provenance
            )
            self.summary.write_normalized_csv(normalized_path, [normalized])
            loaded = self.summary.load_normalized_csv(normalized_path)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].identity["implementation"], "br4")
        metric = self.summary.METRICS[0].name
        self.assertEqual(loaded[0].units[metric], record.units[metric])

    def test_raw_ncu_csv_rejects_zero_or_multiple_kernel_rows(self):
        fields = [
            *self.summary.NCU_RAW_IDENTITY_COLUMNS,
            *sorted(metric.name for metric in self.summary.METRICS),
        ]
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "raw.csv"
            with raw_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(fields)
                writer.writerow([""] * len(fields))
                for index in range(2):
                    identity_values = {
                        "Kernel Name": f"query_tiled_kernel_{index}()",
                        "Process Name": "runner",
                        "Block Size": "(128,1,1)",
                        "Grid Size": "(64,1,1)",
                    }
                    writer.writerow([
                        *(identity_values.get(name, "x") for name in self.summary.NCU_RAW_IDENTITY_COLUMNS),
                        *(["1"] * len(self.summary.METRICS)),
                    ])

            with self.assertRaisesRegex(ValueError, "exactly one kernel data row"):
                self.summary.normalize_raw_csv(
                    raw_path, "br4", 256, 64, 0,
                    {
                        name: value
                        for name, value in self.make_record(
                            "br4", 256, 64, 0
                        ).identity.items()
                        if name not in {
                            "implementation", "n", "d", "causal",
                            "kernel_name", "process_name", "block_size",
                            "grid_size",
                        }
                    },
                )

    def test_raw_and_normalized_csv_reject_extra_and_missing_row_fields(self):
        record = self.make_record("br4", 256, 64, 0)
        raw_fields = [
            *self.summary.NCU_RAW_IDENTITY_COLUMNS,
            *sorted(metric.name for metric in self.summary.METRICS),
        ]
        provenance = {
            name: value for name, value in record.identity.items()
            if name not in {
                "implementation", "n", "d", "causal", "kernel_name",
                "process_name", "block_size", "grid_size",
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw.csv"
            with raw.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(raw_fields)
                writer.writerow([
                    *("" for _ in self.summary.NCU_RAW_IDENTITY_COLUMNS),
                    *[record.units[name] for name in sorted(record.units)],
                ])
                identity_values = {
                    "Kernel Name": record.identity["kernel_name"],
                    "Process Name": record.identity["process_name"],
                    "Block Size": record.identity["block_size"],
                    "Grid Size": record.identity["grid_size"],
                }
                writer.writerow([
                    *(identity_values.get(name, "x") for name in self.summary.NCU_RAW_IDENTITY_COLUMNS),
                    *[record.values[name] for name in sorted(record.values)], "extra",
                ])
            with self.assertRaisesRegex(ValueError, "extra fields"):
                self.summary.normalize_raw_csv(raw, "br4", 256, 64, 0, provenance)

            normalized = Path(directory) / "normalized.csv"
            self.write_normalized(normalized, [record])
            with normalized.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
            rows[1] = rows[1][:-1]
            with normalized.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle, lineterminator="\n").writerows(rows)
            with self.assertRaisesRegex(ValueError, "missing fields"):
                self.summary.load_normalized_csv(normalized)

    def test_raw_ncu_csv_rejects_unexpected_header_columns(self):
        record = self.make_record("br4", 256, 64, 0)
        fields = [
            *self.summary.NCU_RAW_IDENTITY_COLUMNS,
            *sorted(metric.name for metric in self.summary.METRICS),
            "Unexpected Extra",
        ]
        provenance = {
            name: value for name, value in record.identity.items()
            if name not in {
                "implementation", "n", "d", "causal", "kernel_name",
                "process_name", "block_size", "grid_size",
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw.csv"
            with raw.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(fields)
                writer.writerow([
                    *("" for _ in self.summary.NCU_RAW_IDENTITY_COLUMNS),
                    *[record.units[name] for name in sorted(record.units)], "",
                ])
                identity_values = {
                    "Kernel Name": record.identity["kernel_name"],
                    "Process Name": record.identity["process_name"],
                    "Block Size": record.identity["block_size"],
                    "Grid Size": record.identity["grid_size"],
                }
                writer.writerow([
                    *(identity_values.get(name, "x") for name in self.summary.NCU_RAW_IDENTITY_COLUMNS),
                    *[record.values[name] for name in sorted(record.values)], "extra",
                ])

            with self.assertRaisesRegex(ValueError, "unexpected columns"):
                self.summary.normalize_raw_csv(
                    raw, "br4", 256, 64, 0, provenance
                )

    def test_normalized_csv_rejects_unexpected_columns_for_stable_schema(self):
        record = self.make_record("br4", 256, 64, 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            self.write_normalized(path, [record])
            lines = path.read_text(encoding="utf-8").splitlines()
            lines[0] += ",unexpected"
            lines[1] += ",value"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError, "normalized CSV has unexpected columns: unexpected"
            ):
                self.summary.load_normalized_csv(path)

    def test_normalized_csv_requires_fixed_column_order(self):
        record = self.make_record("br4", 256, 64, 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            self.write_normalized(path, [record])
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
            for row in rows:
                row[0], row[1] = row[1], row[0]
            with path.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle, lineterminator="\n").writerows(rows)

            with self.assertRaisesRegex(
                ValueError, "normalized CSV column order does not match schema"
            ):
                self.summary.load_normalized_csv(path)

    def test_smoke_output_name_cannot_overwrite_canonical_compact_evidence(self):
        records = [
            self.make_record("br1", 256, 64, 0, evidence_kind="smoke"),
            self.make_record("br4", 256, 64, 0, evidence_kind="smoke"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for record in records:
                path = Path(directory) / f"{record.identity['implementation']}-smoke.csv"
                self.write_normalized(path, [record])
                paths.append(path)
            output = Path(directory) / "m1-ncu-summary.md"
            result = subprocess.run(
                [sys.executable, str(SCRIPT), *map(str, paths),
                 "--allow-smoke-pair", "--output", str(output)],
                text=True, capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("basename must contain smoke", result.stderr)
            self.assertFalse(output.exists())

    def test_published_smoke_summary_matches_declared_profile_inputs(self):
        project = Path(__file__).parents[1]
        profiles = project / "results" / "profiles" / "smoke"
        sources = [
            profiles / "br1-256x64-causal0-smoke-metrics.csv",
            profiles / "br4-256x64-causal0-smoke-metrics.csv",
        ]
        published = project / "results" / "evidence" / "m1-ncu-smoke.md"
        if not all(path.is_file() for path in sources):
            self.skipTest("local ignored ncu profile inputs are unavailable")

        records = self.summary.load_inputs(sources, require_canonical=False)
        expected = self.summary.render(records, sources)
        if not expected.endswith("\n"):
            expected += "\n"
        self.assertEqual(published.read_text(encoding="utf-8"), expected)


if __name__ == "__main__":
    unittest.main()