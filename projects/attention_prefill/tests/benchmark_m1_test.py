#!/usr/bin/env python3

import csv
import hashlib
import os
import stat
import subprocess
import tempfile
import textwrap
import time
import unittest
from pathlib import Path


PROJECT = Path(__file__).parents[1]
BENCHMARK = PROJECT / "scripts" / "benchmark_m1.sh"
APPENDER = PROJECT / "scripts" / "append_m1_csv.py"
FINGERPRINT = PROJECT / "scripts" / "source_fingerprint.py"
ATTESTATION_FIELDS = (
    "schema_version",
    "build_type",
    "cuda_architectures",
    "cuda_compiler_id",
    "cuda_compiler_version",
    "cuda_compiler_realpath",
    "cmake_cuda_flags",
    "cmake_cuda_flags_release",
    "target_compile_options_evidence_support",
    "target_compile_options_attention_kernel",
    "target_compile_options_flash_kernels",
)
DEFAULT_PAYLOAD = {
    "schema_version": "1",
    "build_type": "Release",
    "cuda_architectures": "80",
    "cuda_compiler_id": "NVIDIA",
    "cuda_compiler_version": "12.8.93",
    "cuda_compiler_realpath": "/usr/local/cuda/bin/nvcc",
    "cmake_cuda_flags": "",
    "cmake_cuda_flags_release": "-O3 -DNDEBUG",
    "target_compile_options_evidence_support": (
        "-O3;-lineinfo;-Xcompiler=-Wall,-Wextra"
    ),
    "target_compile_options_attention_kernel": (
        "-O3;-lineinfo;-Xptxas=-warn-spills;-Xcompiler=-Wall,-Wextra"
    ),
    "target_compile_options_flash_kernels": (
        "-O3;-lineinfo;-Xptxas=-warn-spills;-Xcompiler=-Wall,-Wextra"
    ),
}


def payload_text(payload):
    return "".join(f"{field}={payload[field]}\n" for field in ATTESTATION_FIELDS)


DEFAULT_PAYLOAD_SHA256 = hashlib.sha256(
    payload_text(DEFAULT_PAYLOAD).encode("utf-8")
).hexdigest()
DEFAULT_BUILD_CONTRACT = f"release-sm80-{DEFAULT_PAYLOAD_SHA256[:16]}"


class BenchmarkM1Tests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.repo = Path(self.temporary_directory.name) / "repo"
        self.fake_bin = Path(self.temporary_directory.name) / "bin"
        self.fake_bin.mkdir(parents=True)
        self._make_repo()
        self._make_commands()
        self.source_sha256 = self._source_fingerprint()
        self.runner = (
            self.repo / "build/projects/attention_prefill/attention_prefill_evidence_runner"
        )
        self._write_executable(self.runner, self._runner_program())
        self.attestation = (
            self.repo
            / "build/projects/attention_prefill/attention_prefill_build_attestation.txt"
        )
        self._write_attestation()
        future = time.time() + 10
        os.utime(self.runner, (future, future))
        os.utime(self.attestation, (future + 1, future + 1))

    def _write_executable(self, path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)

    def _run(self, command):
        return subprocess.run(
            command,
            cwd=self.repo,
            text=True,
            capture_output=True,
            check=True,
        )

    def _make_repo(self):
        (self.repo / ".gitignore").parent.mkdir(parents=True, exist_ok=True)
        (self.repo / ".gitignore").write_text("/build/\n", encoding="utf-8")
        common = self.repo / "common/scripts/common.sh"
        common.parent.mkdir(parents=True)
        common.write_text(textwrap.dedent("""
            gpu_die() { local prefix="$1"; shift; printf '%s: %s\n' "$prefix" "$*" >&2; exit 1; }
            gpu_require_command() { command -v "$2" >/dev/null 2>&1 || gpu_die "$1" "missing command: $2"; }
            gpu_positive_integer() { [[ $1 =~ ^[1-9][0-9]*$ ]]; }
            gpu_nonnegative_integer() { [[ $1 =~ ^[0-9]+$ ]]; }
        """).lstrip(), encoding="utf-8")
        sources = (
            "CMakeLists.txt",
            "projects/attention_prefill/CMakeLists.txt",
            "projects/attention_prefill/include/api.hpp",
            "projects/attention_prefill/evidence/main.cu",
            "projects/attention_prefill/kernels/query_tiled.cu",
            "projects/attention_prefill/runner/main.cu",
            "projects/flash_attention/include/api.hpp",
            "projects/flash_attention/kernels/tiled.cu",
        )
        for relative in sources:
            path = self.repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"tracked: {relative}\n", encoding="utf-8")
        source_script = Path(__file__).parents[1] / "scripts" / "source_fingerprint.py"
        fixture_script = self.repo / "projects/attention_prefill/scripts/source_fingerprint.py"
        fixture_script.parent.mkdir(parents=True, exist_ok=True)
        fixture_script.write_bytes(source_script.read_bytes())
        self._run(["git", "init", "-q"])
        self._run(["git", "config", "user.email", "test@example.com"])
        self._run(["git", "config", "user.name", "Test User"])
        self._run(["git", "add", "."])
        self._run(["git", "commit", "-qm", "fixture"])

    def _make_commands(self):
        self._write_executable(self.fake_bin / "nvidia-smi", r"""
            #!/usr/bin/env bash
            case "$*" in
                *query-gpu=name*) printf '%s\n' "${FAKE_GPU_NAME:-NVIDIA A100-SXM4-80GB}" ;;
                *query-gpu=uuid*) printf '%s\n' "${FAKE_GPU_UUID:-GPU-test}" ;;
                *query-gpu=compute_cap*) printf '%s\n' "${FAKE_GPU_SM:-8.0}" ;;
                *query-gpu=driver_version*) printf '%s\n' "${FAKE_GPU_DRIVER:-575.57.08}" ;;
                *) exit 2 ;;
            esac
        """)
        self._write_executable(self.fake_bin / "nvcc", r"""
            #!/usr/bin/env bash
            printf '%s\n' 'nvcc: NVIDIA (R) Cuda compiler driver'
            printf '%s\n' 'Cuda compilation tools, release 12.8, V12.8.93'
        """)

    def _source_fingerprint(self):
        return subprocess.run(
            ["python3", str(FINGERPRINT), "--repo-root", str(self.repo)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def _write_attestation(
        self, *, contract=None, payload_sha256=None, payload_overrides=None
    ):
        payload = {**DEFAULT_PAYLOAD, **(payload_overrides or {})}
        calculated_sha256 = hashlib.sha256(
            payload_text(payload).encode("utf-8")
        ).hexdigest()
        payload_sha256 = payload_sha256 or calculated_sha256
        contract = contract or (
            f"{payload['build_type'].lower()}-sm{payload['cuda_architectures']}-"
            f"{payload_sha256[:16]}"
        )
        self.attestation.parent.mkdir(parents=True, exist_ok=True)
        self.attestation.write_text(
            payload_text(payload)
            + f"build_contract={contract}\n"
            f"build_contract_payload_sha256={payload_sha256}\n",
            encoding="utf-8",
        )
        return contract, payload_sha256

    def _runner_program(self):
        return r"""
            #!/usr/bin/env python3
            import math
            import os
            import sys
            from pathlib import Path

            metadata = {
                "source_sha256": os.environ.get("FAKE_SOURCE_SHA256", "missing"),
                "build_contract": os.environ.get(
                    "FAKE_BUILD_CONTRACT", "release-sm80-cccccccccccccccc"
                ),
                "build_contract_payload_sha256": os.environ.get(
                    "FAKE_BUILD_CONTRACT_PAYLOAD_SHA256", "c" * 64
                ),
                "device_index": os.environ.get("FAKE_DEVICE_INDEX", "0"),
                "gpu_uuid": os.environ.get(
                    "FAKE_RUNNER_GPU_UUID",
                    "GPU-01234567-89ab-cdef-0123-456789abcdef",
                ),
                "gpu_name": os.environ.get("FAKE_RUNNER_GPU_NAME", "NVIDIA_A100-SXM4-80GB"),
                "sm": os.environ.get("FAKE_RUNNER_SM", "8.0"),
                "driver": os.environ.get("FAKE_RUNNER_DRIVER", "12080"),
            }
            if "--metadata-only" in sys.argv:
                print(" ".join(f"{key}={value}" for key, value in metadata.items()))
                raise SystemExit(0)

            arguments = dict(zip(sys.argv[1::2], sys.argv[2::2]))
            implementation = arguments["--implementation"]
            n = int(arguments["--n"])
            d = int(arguments["--d"])
            causal = arguments["--causal"]
            cta = n if implementation == "br1" else math.ceil(n / 4)
            count = 0
            counter_file = os.environ.get("FAKE_COUNTER_FILE")
            if counter_file:
                counter = Path(counter_file)
                count = int(counter.read_text() if counter.exists() else "0") + 1
                counter.write_text(str(count))
            values = {
                **metadata,
                "implementation": implementation,
                "path": implementation,
                "shape": f"{n}x{d}",
                "causal": causal,
                "input_pattern": "random",
                "status": "PASS",
                "max_abs": str((0.1, 0.3, 0.2, 0.4)[(count - 1) % 4] if count else 0.1),
                "max_rel": str((0.01, 0.03, 0.02, 0.04)[(count - 1) % 4] if count else 0.01),
                "latency_ms": os.environ.get("FAKE_LATENCY", "1.0"),
                "cta_count": str(cta),
                "requested_kv_elements": str(2 * cta * n * d),
                "workspace_bytes": "0",
            }
            missing = os.environ.get("FAKE_MISSING_FIELD")
            print(" ".join(f"{key}={value}" for key, value in values.items() if key != missing))
        """

    def environment(self, **overrides):
        environment = os.environ.copy()
        environment.update({
            "PATH": f"{self.fake_bin}{os.pathsep}{environment['PATH']}",
            "AP_M1_REPO_ROOT": str(self.repo),
            "PYTHONDONTWRITEBYTECODE": "1",
            "FAKE_SOURCE_SHA256": self.source_sha256,
            "FAKE_BUILD_CONTRACT": DEFAULT_BUILD_CONTRACT,
            "FAKE_BUILD_CONTRACT_PAYLOAD_SHA256": DEFAULT_PAYLOAD_SHA256,
        })
        environment.update({key: str(value) for key, value in overrides.items()})
        return environment

    def smoke_environment(self, **overrides):
        output = Path(self.temporary_directory.name) / "custom"
        environment = self.environment(
            AP_M1_SHAPES="128x64",
            AP_M1_CAUSAL="0",
            AP_M1_WARMUP="0",
            AP_M1_ITERATIONS="1",
            AP_M1_REPEATS="2",
            AP_M1_OUTPUT_CSV=output / "smoke.csv",
            AP_M1_OUTPUT_MD=output / "smoke.md",
        )
        environment.update({key: str(value) for key, value in overrides.items()})
        return environment

    def run_benchmark(self, environment, runner=None, include_runner_argument=None):
        command = [str(BENCHMARK)]
        if include_runner_argument is None:
            include_runner_argument = runner is not None
        if include_runner_argument:
            command.append(str(runner or self.runner))
        return subprocess.run(
            command,
            cwd=self.repo,
            env=environment,
            text=True,
            capture_output=True,
        )

    def test_append_helper_round_trips_commas_quotes_and_newlines(self):
        csv_path = Path(self.temporary_directory.name) / "special.csv"
        fields = ["plain", "comma,value", 'quote"value', "two\nlines"]

        subprocess.run(
            ["python3", str(APPENDER), "--header", str(csv_path), *fields],
            check=True,
        )
        subprocess.run(
            ["python3", str(APPENDER), "--row", str(csv_path), *fields],
            check=True,
        )

        with csv_path.open(newline="", encoding="utf-8") as handle:
            self.assertEqual(list(csv.reader(handle)), [fields, fields])

    def test_canonical_rejects_dirty_tree_even_with_override(self):
        (self.repo / "CMakeLists.txt").write_text("dirty\n", encoding="utf-8")
        result = self.run_benchmark(self.environment(AP_M1_ALLOW_DIRTY="1"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("canonical benchmark 要求干净工作树", result.stderr)

    def test_smoke_dirty_tree_requires_explicit_opt_in(self):
        (self.repo / "CMakeLists.txt").write_text("dirty\n", encoding="utf-8")
        rejected = self.run_benchmark(self.smoke_environment())
        accepted = self.run_benchmark(self.smoke_environment(
            AP_M1_ALLOW_DIRTY="1",
            FAKE_SOURCE_SHA256=self._source_fingerprint(),
        ))
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("AP_M1_ALLOW_DIRTY=1", rejected.stderr)
        self.assertEqual(accepted.returncode, 0, accepted.stderr)

    def test_canonical_output_paths_are_not_customizable(self):
        custom = Path(self.temporary_directory.name) / "custom" / "a100-fp32-m1.csv"
        result = self.run_benchmark(self.environment(AP_M1_OUTPUT_CSV=custom))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("canonical 输出路径必须使用 official paths", result.stderr)

    def test_smoke_custom_paths_must_keep_smoke_filenames(self):
        output = Path(self.temporary_directory.name) / "custom"
        result = self.run_benchmark(self.smoke_environment(
            AP_M1_OUTPUT_CSV=output / "ad-hoc.csv",
        ))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("必须命名为 smoke.csv", result.stderr)

    def test_canonical_rejects_runner_older_than_tracked_sources(self):
        old = time.time() - 100
        os.utime(self.runner, (old, old))
        source = self.repo / "projects/attention_prefill/include/api.hpp"
        new = time.time() + 100
        os.utime(source, (new, new))
        result = self.run_benchmark(self.environment())
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("runner 早于最新 source", result.stderr)

    def test_canonical_rejects_custom_runner_path(self):
        custom_runner = self.repo / "build/custom/evidence_runner"
        self._write_executable(custom_runner, self._runner_program())
        result = self.run_benchmark(self.environment(), custom_runner)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("canonical benchmark 不接受 runner 位置参数", result.stderr)

    def test_canonical_rejects_any_runner_positional_argument_including_alias_symlink(self):
        alias = self.repo / "build/custom/evidence_runner_alias"
        alias.parent.mkdir(parents=True, exist_ok=True)
        alias.symlink_to(self.runner)
        counter = Path(self.temporary_directory.name) / "counter"

        direct = self.run_benchmark(
            self.environment(FAKE_COUNTER_FILE=counter), self.runner
        )
        symlink = self.run_benchmark(
            self.environment(FAKE_COUNTER_FILE=counter), alias
        )

        for result in (direct, symlink):
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("canonical benchmark 不接受 runner 位置参数", result.stderr)
        self.assertFalse(counter.exists())

    def test_canonical_rejects_symlink_at_standard_runner_path(self):
        real_runner = self.repo / "build/custom/real_evidence_runner"
        self._write_executable(real_runner, self._runner_program())
        self.runner.unlink()
        self.runner.symlink_to(real_runner)

        result = self.run_benchmark(self.environment())

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("标准 runner 必须为普通文件且不能是 symlink", result.stderr)

    def test_runner_source_fingerprint_mismatch_is_rejected(self):
        result = self.run_benchmark(self.environment(
            FAKE_SOURCE_SHA256="0" * 64,
        ))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("source fingerprint 不一致", result.stderr)

    def test_canonical_build_contract_mismatch_is_rejected(self):
        result = self.run_benchmark(self.environment(
            FAKE_BUILD_CONTRACT="debug-sm80-cccccccccccccccc",
        ))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("build attestation 内容与 runner metadata 不一致", result.stderr)

    def test_canonical_build_contract_prefix_must_match_payload_hash(self):
        contract = "release-sm80-0000000000000000"
        self._write_attestation(contract=contract)
        future = time.time() + 10
        os.utime(self.runner, (future, future))
        os.utime(self.attestation, (future + 1, future + 1))

        result = self.run_benchmark(self.environment(FAKE_BUILD_CONTRACT=contract))

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("build_contract does not match payload", result.stderr)

    def test_canonical_rejects_wrong_attestation_flags_debug_and_arch_before_dispatch(self):
        cases = (
            ({"schema_version": "2"}, "schema_version=1"),
            ({"cmake_cuda_flags": "-use_fast_math"}, "cmake_cuda_flags"),
            ({"cmake_cuda_flags_release": "-O2 -DNDEBUG"}, "cmake_cuda_flags_release"),
            ({"build_type": "Debug"}, "build_type=Release"),
            ({"cuda_architectures": "80;90"}, "cuda_architectures=80"),
            ({"cuda_compiler_id": "Clang"}, "cuda_compiler_id=NVIDIA"),
            ({"cuda_compiler_version": ""}, "cuda_compiler_version"),
            ({"cuda_compiler_realpath": ""}, "cuda_compiler_realpath"),
            (
                {"target_compile_options_evidence_support": "-O2;-lineinfo"},
                "target_compile_options_evidence_support",
            ),
            (
                {"target_compile_options_attention_kernel": "-O2;-lineinfo"},
                "target_compile_options_attention_kernel",
            ),
            (
                {"target_compile_options_flash_kernels": "-O2;-lineinfo"},
                "target_compile_options_flash_kernels",
            ),
        )
        for payload_overrides, message in cases:
            with self.subTest(payload_overrides=payload_overrides):
                contract, payload_sha256 = self._write_attestation(
                    payload_overrides=payload_overrides
                )
                future = time.time() + 10
                os.utime(self.runner, (future, future))
                os.utime(self.attestation, (future + 1, future + 1))
                counter = Path(self.temporary_directory.name) / "counter"

                result = self.run_benchmark(
                    self.environment(
                        FAKE_BUILD_CONTRACT=contract,
                        FAKE_BUILD_CONTRACT_PAYLOAD_SHA256=payload_sha256,
                        FAKE_COUNTER_FILE=counter,
                    ),
                    include_runner_argument=False,
                )

                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
                self.assertFalse(counter.exists())

    def test_canonical_rejects_attestation_payload_hash_mismatch_before_dispatch(self):
        self._write_attestation(payload_sha256="0" * 64)
        future = time.time() + 10
        os.utime(self.runner, (future, future))
        os.utime(self.attestation, (future + 1, future + 1))
        counter = Path(self.temporary_directory.name) / "counter"

        result = self.run_benchmark(
            self.environment(
                FAKE_BUILD_CONTRACT="release-sm80-0000000000000000",
                FAKE_BUILD_CONTRACT_PAYLOAD_SHA256="0" * 64,
                FAKE_COUNTER_FILE=counter,
            )
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not match the complete payload", result.stderr)
        self.assertFalse(counter.exists())

    def test_canonical_rejects_missing_symlinked_stale_or_mismatched_attestation(self):
        cases = ("missing", "symlink", "stale", "mismatch")
        for case in cases:
            with self.subTest(case=case):
                self._write_attestation()
                future = time.time() + 10
                os.utime(self.runner, (future, future))
                os.utime(self.attestation, (future + 1, future + 1))
                if case == "missing":
                    self.attestation.unlink()
                elif case == "symlink":
                    target = self.attestation.with_suffix(".real")
                    self.attestation.rename(target)
                    self.attestation.symlink_to(target)
                elif case == "stale":
                    old = future - 100
                    os.utime(self.attestation, (old, old))
                else:
                    self._write_attestation(contract="release-sm80-0000000000000000")
                    os.utime(self.attestation, (future + 1, future + 1))

                result = self.run_benchmark(self.environment())

                self.assertNotEqual(result.returncode, 0)
                self.assertIn("build attestation", result.stderr)

    def test_cuda_visible_devices_is_not_used_as_reported_device_identity(self):
        environment = self.smoke_environment(
            CUDA_VISIBLE_DEVICES="GPU-remapped",
            FAKE_DEVICE_INDEX="4",
            FAKE_RUNNER_GPU_UUID="GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        )
        result = self.run_benchmark(environment)
        self.assertEqual(result.returncode, 0, result.stderr)
        with Path(environment["AP_M1_OUTPUT_CSV"]).open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["device_index"], "4")
        self.assertEqual(
            rows[0]["gpu_uuid"],
            "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        )

    def test_canonical_rejects_non_a100_before_running_benchmarks(self):
        counter = Path(self.temporary_directory.name) / "counter"
        result = self.run_benchmark(self.environment(
            FAKE_RUNNER_GPU_NAME="NVIDIA_H100",
            FAKE_RUNNER_SM="9.0",
            FAKE_COUNTER_FILE=counter,
        ))

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("A100 和 80GB", result.stderr)
        self.assertFalse(counter.exists())

    def test_canonical_rejects_a100_40gb_before_running_benchmarks(self):
        counter = Path(self.temporary_directory.name) / "counter"
        result = self.run_benchmark(
            self.environment(
                FAKE_RUNNER_GPU_NAME="NVIDIA_A100-SXM4-40GB",
                FAKE_COUNTER_FILE=counter,
            ),
            include_runner_argument=False,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("A100 80GB", result.stderr)
        self.assertFalse(counter.exists())

    def test_missing_runner_field_and_nonfinite_latency_are_rejected(self):
        missing = self.run_benchmark(self.smoke_environment(
            FAKE_MISSING_FIELD="requested_kv_elements",
        ))
        nonfinite = self.run_benchmark(self.smoke_environment(FAKE_LATENCY="nan"))
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("runner 输出缺少字段: requested_kv_elements", missing.stderr)
        self.assertNotEqual(nonfinite.returncode, 0)
        self.assertIn("latency 统计失败", nonfinite.stderr)

    def test_smoke_custom_paths_csv_special_characters_and_repeat_error_maxima(self):
        counter = Path(self.temporary_directory.name) / "counter"
        special_preset = "dev,qa\nline"
        runner_gpu = "NVIDIA_A100-SXM4-80GB"
        environment = self.smoke_environment(
            AP_M1_BUILD_PRESET=special_preset,
            FAKE_GPU_NAME="WRONG nvidia-smi device",
            FAKE_RUNNER_GPU_NAME=runner_gpu,
            FAKE_DEVICE_INDEX=3,
            FAKE_RUNNER_GPU_UUID="GPU-fedcba98-7654-3210-fedc-ba9876543210",
            FAKE_COUNTER_FILE=counter,
        )
        result = self.run_benchmark(environment)
        self.assertEqual(result.returncode, 0, result.stderr)
        csv_path = Path(environment["AP_M1_OUTPUT_CSV"])
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["build_preset"], special_preset)
        self.assertEqual(rows[0]["source_sha256"], self.source_sha256)
        self.assertEqual(rows[0]["build_contract"], DEFAULT_BUILD_CONTRACT)
        self.assertEqual(
            rows[0]["build_contract_payload_sha256"], DEFAULT_PAYLOAD_SHA256
        )
        self.assertEqual(rows[0]["device_index"], "3")
        self.assertEqual(rows[0]["gpu"], runner_gpu)
        self.assertEqual(
            rows[0]["gpu_uuid"],
            "GPU-fedcba98-7654-3210-fedc-ba9876543210",
        )
        self.assertEqual(rows[0]["sm"], "8.0")
        self.assertEqual(rows[0]["driver"], "12080")
        self.assertEqual(rows[0]["max_abs"], "0.300000")
        self.assertEqual(rows[0]["max_rel"], "0.030000")
        self.assertEqual(rows[1]["max_abs"], "0.400000")
        self.assertEqual(rows[1]["max_rel"], "0.040000")
        markdown = Path(environment["AP_M1_OUTPUT_MD"]).read_text(encoding="utf-8")
        self.assertIn(runner_gpu, markdown)


if __name__ == "__main__":
    unittest.main()
