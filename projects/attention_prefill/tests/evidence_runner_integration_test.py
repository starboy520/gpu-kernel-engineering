#!/usr/bin/env python3

import argparse
import math
import subprocess
import unittest
from pathlib import Path


def parse_fields(output: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in output.split():
        key, separator, value = token.partition("=")
        if not separator or not key or not value:
            raise AssertionError(f"invalid runner field token: {token!r}")
        if key in fields:
            raise AssertionError(f"duplicate runner field: {key}")
        fields[key] = value
    return fields


class EvidenceRunnerIntegrationTests(unittest.TestCase):
    runner: Path

    def invoke(self, implementation: str, mode: str) -> dict[str, str]:
        completed = subprocess.run(
            [
                str(self.runner),
                "--implementation", implementation,
                "--n", "128",
                "--d", "64",
                "--causal", "0",
                "--mode", mode,
                "--warmup", "0",
                "--iterations", "1",
                "--seed", "1234",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return parse_fields(completed.stdout)

    def assert_common_fields(self, fields: dict[str, str], implementation: str) -> None:
        required = {
            "source_sha256", "build_contract", "build_contract_payload_sha256",
            "device_index", "gpu_uuid", "gpu_name", "sm", "driver",
            "implementation", "path", "shape", "causal", "input_pattern",
            "status", "max_abs", "max_rel", "latency_ms", "cta_count",
            "requested_kv_elements", "workspace_bytes",
        }
        self.assertEqual(required - fields.keys(), set())
        self.assertEqual(fields["implementation"], implementation)
        self.assertEqual(fields["path"], implementation)
        self.assertEqual(fields["shape"], "128x64")
        self.assertEqual(fields["causal"], "0")
        self.assertEqual(fields["status"], "PASS")
        self.assertEqual(fields["workspace_bytes"], "0")
        for metric in ("max_abs", "max_rel"):
            value = float(fields[metric])
            self.assertTrue(math.isfinite(value))
            self.assertGreaterEqual(value, 0.0)

    def test_real_br1_and_br4_validate_and_benchmark_dispatches(self) -> None:
        validation: dict[str, dict[str, str]] = {}
        benchmark: dict[str, dict[str, str]] = {}
        for implementation in ("br1", "br4"):
            validation[implementation] = self.invoke(implementation, "validate")
            benchmark[implementation] = self.invoke(implementation, "benchmark")
            self.assert_common_fields(validation[implementation], implementation)
            self.assert_common_fields(benchmark[implementation], implementation)
            latency = float(benchmark[implementation]["latency_ms"])
            self.assertTrue(math.isfinite(latency))
            self.assertGreater(latency, 0.0)

        self.assertEqual(validation["br1"]["max_abs"], validation["br4"]["max_abs"])
        self.assertEqual(validation["br1"]["max_rel"], validation["br4"]["max_rel"])
        self.assertEqual(benchmark["br1"]["max_abs"], benchmark["br4"]["max_abs"])
        self.assertEqual(benchmark["br1"]["max_rel"], benchmark["br4"]["max_rel"])
        self.assertEqual(validation["br1"]["cta_count"], "128")
        self.assertEqual(validation["br4"]["cta_count"], "32")
        self.assertEqual(validation["br1"]["requested_kv_elements"], "2097152")
        self.assertEqual(validation["br4"]["requested_kv_elements"], "524288")


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--runner", required=True, type=Path)
    arguments, remaining = parser.parse_known_args()
    EvidenceRunnerIntegrationTests.runner = arguments.runner
    unittest.main(argv=[__file__, *remaining])


if __name__ == "__main__":
    main()
