#!/usr/bin/env python3

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "render_results.py"


def load_renderer():
    spec = importlib.util.spec_from_file_location("render_results", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RenderResultsTests(unittest.TestCase):
    def setUp(self):
        self.renderer = load_renderer()

    def test_async_delta_is_positive_for_improvement(self):
        self.assertAlmostEqual(self.renderer.speedup(2.0, 1.0), 2.0)
        self.assertAlmostEqual(self.renderer.delta_percent(2.0, 1.0), 100.0)
        self.assertAlmostEqual(self.renderer.delta_percent(1.0, 2.0), -50.0)

    def test_classification_requires_five_percent_and_low_spread(self):
        self.assertEqual(self.renderer.classify(1.10, 2.0), "benefit")
        self.assertEqual(self.renderer.classify(0.90, 2.0), "regression")
        self.assertEqual(self.renderer.classify(1.02, 1.0), "near-parity")
        self.assertEqual(self.renderer.classify(1.10, 4.0), "inconclusive")

    def test_inconclusive_sample_is_not_treated_as_regression_or_parity(self):
        summary = self.renderer.summarize_boundary([
            (512, "regression"),
            (768, "inconclusive"),
            (1024, "benefit"),
        ])
        self.assertEqual(summary, (512, "768", 1024, "512–1024"))

    def test_render_preserves_improvement_and_regression_shapes(self):
        columns = self.renderer.REQUIRED_COLUMNS
        rows = []
        for n, tiled_ms, async_ms in ((512, 1.0, 1.2), (1024, 2.0, 1.6)):
            for kernel, latency in (("naive", tiled_ms * 2),
                                    ("tiled", tiled_ms),
                                    ("tiled-parallel", tiled_ms * 1.01),
                                    ("tiled-async", async_ms)):
                row = {column: "x" for column in columns}
                row.update({
                    "kernel": kernel,
                    "path": "fast-pipeline-16b" if kernel == "tiled-async" else kernel,
                    "n": str(n),
                    "d": "128",
                    "causal": "0",
                    "latency_ms": str(latency),
                    "latency_min_ms": str(latency),
                    "latency_max_ms": str(latency),
                    "spread_pct": "1.0",
                    "workspace_bytes": "0",
                })
                rows.append(row)

        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "input.csv"
            md_path = Path(directory) / "output.md"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
            self.renderer.render_file(csv_path, md_path)
            text = md_path.read_text(encoding="utf-8")

        self.assertIn("regression", text)
        self.assertIn("benefit", text)
        self.assertIn("-16.7%", text)
        self.assertIn("+25.0%", text)
        self.assertIn("512–1024", text)


if __name__ == "__main__":
    unittest.main()
