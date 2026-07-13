#!/usr/bin/env python3

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "summarize_ncu.py"


def load_module():
    spec = importlib.util.spec_from_file_location("summarize_ncu", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SummarizeNcuTests(unittest.TestCase):
    def test_loads_requested_metrics_and_units(self):
        module = load_module()
        metric_names = [metric for _, metric in module.METRICS]
        fields = ["Kernel Name", *metric_names]
        values = {metric: "1" for metric in metric_names}
        values["launch__registers_per_thread"] = "39"
        values["l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"] = "100"
        values["l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum"] = "250"
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "raw.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(fields)
                writer.writerow(["", "register/thread", *([""] * (len(metric_names) - 1))])
                writer.writerow(["kernel(float*)", *[values[metric] for metric in metric_names]])
            record = module.load_raw_csv(csv_path)

        self.assertEqual(record.values["launch__registers_per_thread"], "39")
        self.assertEqual(record.units["launch__registers_per_thread"], "register/thread")
        self.assertAlmostEqual(module.wavefront_factor(record), 250 / 150)

    def test_compare_renders_missing_metrics_without_crashing(self):
        module = load_module()
        left = module.Record("tiled", {"launch__registers_per_thread": "31"}, {})
        right = module.Record("async", {"launch__registers_per_thread": "39"}, {})
        rendered = module.render_comparison(left, right)
        self.assertIn("Registers/thread", rendered)
        self.assertIn("31", rendered)
        self.assertIn("39", rendered)
        self.assertIn("-", rendered)

    def test_shortens_known_kernel_names(self):
        module = load_module()
        self.assertEqual(
            module.short_kernel_name("<unnamed>::tiled_attention_kernel(const float *)"),
            "Tiled",
        )
        self.assertEqual(
            module.short_kernel_name("<unnamed>::tiled_async_attention_kernel(const float *)"),
            "Tiled Async",
        )

    def test_rejects_multi_kernel_raw_csv(self):
        module = load_module()
        fields = ["Kernel Name", *[metric for _, metric in module.METRICS]]
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "raw.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(fields)
                writer.writerow(["", *([""] * len(module.METRICS))])
                writer.writerow(["first()", *(["1"] * len(module.METRICS))])
                writer.writerow(["second()", *(["2"] * len(module.METRICS))])
            with self.assertRaisesRegex(ValueError, "exactly one kernel"):
                module.load_raw_csv(csv_path)

    def test_rejects_missing_required_metric(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "raw.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["Kernel Name", "launch__registers_per_thread"])
                writer.writerow(["", "register/thread"])
                writer.writerow(["kernel()", "39"])
            with self.assertRaisesRegex(ValueError, "missing required columns"):
                module.load_raw_csv(csv_path)

    def test_na_bank_counter_has_no_wavefront_factor(self):
        module = load_module()
        record = module.Record(
            "async",
            {
                "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum": "n/a",
                "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum": "100",
            },
            {},
        )
        self.assertIsNone(module.wavefront_factor(record))

        record.values[
            "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"
        ] = "10"
        record.values[
            "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum"
        ] = "n/a"
        self.assertIsNone(module.wavefront_factor(record))

    def test_rejects_empty_kernel_name(self):
        module = load_module()
        metric_names = [metric for _, metric in module.METRICS]
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "raw.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["Kernel Name", *metric_names])
                writer.writerow(["", *([""] * len(metric_names))])
                writer.writerow(["", *(["1"] * len(metric_names))])
            with self.assertRaisesRegex(ValueError, "kernel name is empty"):
                module.load_raw_csv(csv_path)


if __name__ == "__main__":
    unittest.main()
