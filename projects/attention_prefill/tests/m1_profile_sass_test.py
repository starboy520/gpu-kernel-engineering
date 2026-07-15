#!/usr/bin/env python3

import os
import csv
import stat
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


PROJECT = Path(__file__).parents[1]
PROFILE = PROJECT / "scripts" / "profile_m1.sh"
EXTRACT = PROJECT / "scripts" / "extract_m1_sass.sh"


class M1ProfileSassTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.repo = self.root / "repo"
        self.fake_bin = self.root / "bin"
        self.fake_bin.mkdir(parents=True)
        self.runner = (
            self.repo
            / "build/projects/attention_prefill/attention_prefill_evidence_runner"
        )
        self._write_executable(self.runner, """
            #!/usr/bin/env bash
            if [[ ${1:-} == --metadata-only ]]; then
                echo "source_sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa build_contract=release-sm80-bbbbbbbbbbbbbbbb build_contract_payload_sha256=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb device_index=0 gpu_uuid=GPU-6b24bab9-c81f-0e0e-ff68-a11dcf17018b gpu_name=NVIDIA_A100_80GB_PCIe sm=8.0 driver=13030"
                exit 0
            fi
            echo runner
        """)
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        subprocess.run(
            ["git", "-C", str(self.repo), "config", "user.email", "test@example.com"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(self.repo), "config", "user.name", "Test User"],
            check=True,
        )
        marker = self.repo / "fixture.txt"
        marker.write_text("fixture\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(self.repo), "add", "fixture.txt"], check=True)
        subprocess.run(
            ["git", "-C", str(self.repo), "commit", "-qm", "fixture"], check=True
        )

    def _write_executable(self, path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)

    def environment(self, **overrides):
        environment = os.environ.copy()
        environment.update({
            "PATH": f"{self.fake_bin}{os.pathsep}{environment['PATH']}",
            "AP_M1_REPO_ROOT": str(self.repo),
            "AP_M1_EVIDENCE_RUNNER": str(self.runner),
            "AP_M1_SASS_OUTPUT_DIR": str(self.repo / "projects/attention_prefill/results/smoke"),
            "PYTHONDONTWRITEBYTECODE": "1",
        })
        environment.update({key: str(value) for key, value in overrides.items()})
        return environment

    def test_profile_uses_exact_kernel_filter_single_launch_and_writes_normalized_csv(self):
        arguments_file = self.root / "ncu-arguments.txt"
        self._write_executable(self.fake_bin / "ncu", f"""
            #!/usr/bin/env python3
            import csv
            import pathlib
            import sys

            arguments = sys.argv[1:]
            if arguments == ["--version"]:
                print("NVIDIA (R) Nsight Compute Command Line Profiler")
                print("Version 2026.2.0.0 (build 37790515) (public-release)")
                raise SystemExit(0)
            if "--query-metrics-collection" in arguments:
                print("\\n".join([
                    "launch__registers_per_thread  Ratio",
                    "launch__shared_mem_per_block_static  Ratio",
                    "launch__waves_per_multiprocessor  Counter",
                    "launch__occupancy_limit_shared_mem  Counter",
                    "launch__occupancy_limit_registers  Counter",
                ]))
                raise SystemExit(0)
            if "--query-metrics" in arguments:
                print("\\n".join(
                    metric + "  Counter"
                    for metric in arguments[arguments.index("--metrics") + 1].split(",")
                ))
                raise SystemExit(0)
            pathlib.Path({str(arguments_file)!r}).write_text(
                " ".join(arguments), encoding="utf-8"
            )
            if "--import" in arguments:
                output = pathlib.Path(arguments[arguments.index("--log-file") + 1])
                metrics = arguments[arguments.index("--metrics") + 1].split(",")
                units = {{
                    "gpu__time_duration.sum": "ns",
                    "launch__registers_per_thread": "register/thread",
                    "launch__shared_mem_per_block_static": "byte/block",
                    "launch__waves_per_multiprocessor": "",
                    "launch__occupancy_limit_shared_mem": "block",
                    "launch__occupancy_limit_registers": "block",
                    "sm__warps_active.avg.pct_of_peak_sustained_active": "%",
                    "smsp__warps_eligible.avg.per_cycle_active": "warp",
                    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio": "inst",
                    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio": "inst",
                    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio": "inst",
                    "smsp__average_warp_latency_per_inst_issued.ratio": "cycle",
                    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "%",
                    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "%",
                    "lts__throughput.avg.pct_of_peak_sustained_elapsed": "%",
                    "l1tex__throughput.avg.pct_of_peak_sustained_elapsed": "%",
                    "dram__bytes_read.sum": "byte",
                    "dram__bytes_write.sum": "byte",
                    "lts__t_sectors_op_read.sum": "sector",
                    "lts__t_sectors_op_write.sum": "sector",
                    "lts__t_sector_hit_rate.pct": "%",
                    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": "sector",
                    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum": "request",
                    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum": "sector",
                    "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum": "request",
                    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum": "conflict",
                    "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum": "wavefront",
                }}
                with output.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle, lineterminator="\\n")
                    writer.writerow([
                        "ID", "Process ID", "Process Name", "Host Name",
                        "Kernel Name", "Context", "Stream", "Block Size",
                        "Grid Size", "Device", "CC", *sorted(metrics),
                    ])
                    writer.writerow([*("" for _ in range(11)), *[units[metric] for metric in sorted(metrics)]])
                    writer.writerow([
                        "1", "2", "attention_prefill_evidence_runner", "host",
                        "<unnamed>::query_tiled_kernel(const float *, const float *, "
                        "const float *, float *, int, int, bool)",
                        "1", "1", "(128, 1, 1)", "(64, 1, 1)",
                        "NVIDIA A100", "8.0", *(["1"] * len(metrics)),
                    ])
                raise SystemExit(0)
            report = pathlib.Path(arguments[arguments.index("--export") + 1])
            report.write_text("fake report", encoding="utf-8")
            print('==PROF== Profiling "<unnamed>::query_tiled_kernel(const float *, const float *, const float *, float *, int, int, bool)" - 8 passes')
            print(
                "source_sha256=" + "a" * 64
                + " build_contract=release-sm80-" + "b" * 16
                + " build_contract_payload_sha256=" + "b" * 64
                + " device_index=0 gpu_uuid=GPU-6b24bab9-c81f-0e0e-ff68-a11dcf17018b"
                + " gpu_name=NVIDIA_A100_80GB_PCIe sm=8.0 driver=13030"
                + " implementation=br4 path=br4 shape=256x64"
                + " causal=0 input_pattern=random status=PASS max_abs=0 max_rel=0"
                + " latency_ms=0 cta_count=64 requested_kv_elements=1 workspace_bytes=0"
            )
        """)

        result = subprocess.run(
            [str(PROFILE), "br4", "256", "64", "0"],
            cwd=self.repo,
            env=self.environment(AP_M1_PROFILE_OUTPUT_DIR=self.root / "profiles/smoke"),
            text=True,
            capture_output=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        summary = self.root / "profiles/smoke/br4-256x64-causal0-smoke.txt"
        normalized = self.root / "profiles/smoke/br4-256x64-causal0-smoke-metrics.csv"
        self.assertTrue(summary.is_file())
        self.assertTrue(normalized.is_file())
        self.assertTrue(
            (self.root / "profiles/smoke/br4-256x64-causal0-smoke-raw.csv").is_file()
        )
        text = summary.read_text(encoding="utf-8")
        self.assertIn(
            "kernel_filter=regex:^.*::query_tiled_kernel\\(const float \\*, "
            "const float \\*, const float \\*, float \\*, int, int, bool\\)$",
            text,
        )
        self.assertIn("--launch-count 1", text)
        self.assertIn("--replay-mode kernel", text)
        self.assertIn("--cache-control all", text)
        self.assertIn("--clock-control base", text)
        self.assertIn("--mode validate --warmup 0 --iterations 1 --seed 1234", text)
        self.assertIn("status=PASS", text)
        self.assertIn("gpu_name=NVIDIA_A100_80GB_PCIe", text)
        self.assertIn("runner_sha256=", text)
        self.assertIn("source_sha256=" + "a" * 64, text)
        self.assertIn("build_contract=release-sm80-" + "b" * 16, text)
        self.assertIn("device_index=0", text)
        self.assertIn("gpu_uuid=GPU-6b24bab9-c81f-0e0e-ff68-a11dcf17018b", text)
        self.assertIn("sm=8.0", text)
        self.assertIn("driver=13030", text)
        self.assertIn("ncu_version=2026.2.0.0", text)
        with normalized.open(newline="", encoding="utf-8") as handle:
            row = next(csv.DictReader(handle))
        for field in (
            "evidence_kind", "runner_sha256", "source_sha256", "build_contract",
            "build_contract_payload_sha256", "device_index", "gpu_uuid",
            "gpu_name", "sm", "driver", "ncu_version", "profile_timestamp",
        ):
            self.assertIn(field, row)
            self.assertTrue(row[field], field)
        self.assertEqual(row["evidence_kind"], "smoke")
        arguments = text
        for metric in (
            "dram__bytes_write.sum", "lts__t_sectors_op_write.sum",
            "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
            "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
        ):
            self.assertIn(metric, arguments)

    def test_profile_default_rejects_non_a100_80gb_device(self):
        self._write_executable(self.runner, """
            #!/usr/bin/env bash
            echo "source_sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa build_contract=release-sm80-bbbbbbbbbbbbbbbb build_contract_payload_sha256=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb device_index=0 gpu_uuid=GPU-6b24bab9-c81f-0e0e-ff68-a11dcf17018b gpu_name=NVIDIA_A100-PCIE-40GB sm=8.0 driver=13030"
        """)
        result = subprocess.run(
            [str(PROFILE), "br4", "256", "64", "0"],
            cwd=self.repo,
            env=self.environment(AP_M1_PROFILE_OUTPUT_DIR=self.root / "profiles/smoke"),
            text=True, capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("A100 80GB", result.stderr)

    def test_profile_override_requires_smoke_output_and_rejects_missing_metric(self):
        self._write_executable(self.fake_bin / "ncu", """
            #!/usr/bin/env bash
            if [[ $1 == --version ]]; then echo 'Version 2026.2.0.0'; exit 0; fi
            if [[ $1 == --query-metrics-collection ]]; then
                echo 'launch__registers_per_thread  Ratio'
                exit 0
            fi
            if [[ " $* " == *" --query-metrics "* ]]; then echo 'gpu__time_duration.sum  Counter'; exit 0; fi
            exit 99
        """)
        official = subprocess.run(
            [str(PROFILE), "br4", "256", "64", "0"], cwd=self.repo,
            env=self.environment(AP_M1_PROFILE_OUTPUT_DIR=self.repo / "projects/attention_prefill/results/profiles"),
            text=True, capture_output=True,
        )
        self.assertNotEqual(official.returncode, 0)
        self.assertIn("smoke", official.stderr)

        missing = subprocess.run(
            [str(PROFILE), "br4", "256", "64", "0"], cwd=self.repo,
            env=self.environment(AP_M1_PROFILE_OUTPUT_DIR=self.root / "profiles/smoke"),
            text=True, capture_output=True,
        )
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("ncu metric 不可用", missing.stderr)

    def test_profile_metric_query_rejects_regex_near_collision(self):
        self._write_executable(self.fake_bin / "ncu", """
            #!/usr/bin/env bash
            if [[ $1 == --version ]]; then echo 'Version 2026.2.0.0'; exit 0; fi
            if [[ $1 == --query-metrics-collection ]]; then
                printf '%s\n' \
                    'launch__registers_per_thread Counter' \
                    'launch__shared_mem_per_block_static Counter' \
                    'launch__waves_per_multiprocessor Counter' \
                    'launch__occupancy_limit_shared_mem Counter' \
                    'launch__occupancy_limit_registers Counter'
                exit 0
            fi
            if [[ " $* " == *" --query-metrics "* ]]; then
                while (( $# )); do
                    if [[ $1 == --metrics ]]; then
                        shift
                        metrics="$1"
                        break
                    fi
                    shift
                done
                IFS=, read -r -a names <<< "$metrics"
                for metric in "${names[@]}"; do
                    if [[ $metric == sm__throughput.avg.pct_of_peak_sustained_elapsed ]]; then
                        echo 'sm__throughputXavgXpct_of_peak_sustained_elapsed Counter'
                    else
                        echo "$metric Counter"
                    fi
                done
                exit 0
            fi
            exit 99
        """)

        result = subprocess.run(
            [str(PROFILE), "br4", "256", "64", "0"], cwd=self.repo,
            env=self.environment(
                AP_M1_PROFILE_OUTPUT_DIR=self.root / "profiles/smoke"
            ), text=True, capture_output=True,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "ncu metric 不可用: "
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            result.stderr,
        )

    def test_profile_failed_export_preserves_existing_artifacts(self):
        output = self.root / "profiles/smoke"
        output.mkdir(parents=True)
        old_paths = [
            output / "br4-256x64-causal0-smoke.ncu-rep",
            output / "br4-256x64-causal0-smoke.txt",
            output / "br4-256x64-causal0-smoke-raw.csv",
            output / "br4-256x64-causal0-smoke-metrics.csv",
        ]
        for path in old_paths:
            path.write_text("old", encoding="utf-8")
        self._write_executable(self.fake_bin / "ncu", """
            #!/usr/bin/env bash
            if [[ $1 == --version ]]; then echo 'Version 2026.2.0.0'; exit 0; fi
            if [[ $1 == --query-metrics ]]; then
                cat "$FAKE_METRICS_FILE"
                exit 0
            fi
            if [[ " $* " == *" --import "* ]]; then exit 41; fi
            while (( $# )); do
                if [[ $1 == --export ]]; then shift; printf report > "$1"; fi
                shift
            done
            echo 'source_sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa build_contract=release-sm80-bbbbbbbbbbbbbbbb build_contract_payload_sha256=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb device_index=0 gpu_uuid=GPU-6b24bab9-c81f-0e0e-ff68-a11dcf17018b gpu_name=NVIDIA_A100_80GB_PCIe sm=8.0 driver=13030 implementation=br4 path=br4 shape=256x64 causal=0 status=PASS'
        """)
        metrics_file = self.root / "metrics.txt"
        metrics_file.write_text("\n".join([
            metric for metric in (
                "gpu__time_duration.sum", "launch__registers_per_thread",
                "launch__shared_mem_per_block_static", "launch__waves_per_multiprocessor",
                "launch__occupancy_limit_shared_mem", "launch__occupancy_limit_registers",
                "sm__warps_active.avg.pct_of_peak_sustained_active",
                "smsp__warps_eligible.avg.per_cycle_active",
                "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
                "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
                "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
                "smsp__average_warp_latency_per_inst_issued.ratio",
                "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                "dram__throughput.avg.pct_of_peak_sustained_elapsed",
                "lts__throughput.avg.pct_of_peak_sustained_elapsed",
                "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
                "dram__bytes_read.sum", "dram__bytes_write.sum",
                "lts__t_sectors_op_read.sum", "lts__t_sectors_op_write.sum",
                "lts__t_sector_hit_rate.pct",
                "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
                "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
                "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
                "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
                "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",
                "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum",
            )
        ]) + "\n", encoding="utf-8")
        result = subprocess.run(
            [str(PROFILE), "br4", "256", "64", "0"], cwd=self.repo,
            env=self.environment(AP_M1_PROFILE_OUTPUT_DIR=output, FAKE_METRICS_FILE=metrics_file),
            text=True, capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual([path.read_text(encoding="utf-8") for path in old_paths], ["old"] * 4)

    def test_profile_rejects_unverified_ncu_version_even_for_smoke(self):
        self._write_executable(self.fake_bin / "ncu", """
            #!/usr/bin/env bash
            if [[ $1 == --version ]]; then echo 'Version 2027.1.0'; exit 0; fi
            exit 99
        """)
        rejected = subprocess.run(
            [str(PROFILE), "br4", "256", "64", "0"], cwd=self.repo,
            env=self.environment(AP_M1_PROFILE_OUTPUT_DIR=self.root / "profiles/smoke"),
            text=True, capture_output=True,
        )
        explicitly_requested = subprocess.run(
            [str(PROFILE), "br4", "256", "64", "0"], cwd=self.repo,
            env=self.environment(
                AP_M1_PROFILE_OUTPUT_DIR=self.root / "profiles/smoke",
                AP_M1_ALLOW_UNVERIFIED_NCU="1",
            ), text=True, capture_output=True,
        )
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("2026.2", rejected.stderr)
        self.assertNotEqual(explicitly_requested.returncode, 0)
        self.assertIn("2026.2", explicitly_requested.stderr)

    def test_profile_noncanonical_shape_cannot_enter_official_profiles(self):
        self._write_executable(self.fake_bin / "ncu", """
            #!/usr/bin/env bash
            echo unexpected-ncu-dispatch >&2
            exit 99
        """)

        environment = self.environment(
            AP_M1_PROFILE_OUTPUT_DIR=(
                self.repo / "projects/attention_prefill/results/profiles"
            ),
        )
        environment.pop("AP_M1_EVIDENCE_RUNNER")
        result = subprocess.run(
            [str(PROFILE), "br4", "512", "64", "0"],
            cwd=self.repo,
            env=environment,
            text=True,
            capture_output=True,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("canonical profile shape", result.stderr)
        self.assertNotIn("unexpected-ncu-dispatch", result.stderr)

    def test_profile_and_sass_support_runner_path_with_spaces(self):
        spaced_runner = self.repo / "build/custom space/evidence runner"
        self._write_executable(spaced_runner, self.runner.read_text(encoding="utf-8"))
        self._write_executable(self.fake_bin / "ncu", """
            #!/usr/bin/env bash
            if [[ $1 == --version ]]; then echo 'Version 2026.2.0.0'; exit 0; fi
            if [[ $1 == --query-metrics-collection ]]; then exit 17; fi
            exit 99
        """)
        profile = subprocess.run(
            [str(PROFILE), "br4", "256", "64", "0"], cwd=self.repo,
            env=self.environment(
                AP_M1_EVIDENCE_RUNNER=spaced_runner,
                AP_M1_PROFILE_OUTPUT_DIR=self.root / "profile space/smoke",
            ), text=True, capture_output=True,
        )
        self.assertNotIn("metadata-only", profile.stderr)
        self.assertIn("launch metric query", profile.stderr)

        self._write_executable(self.fake_bin / "cuobjdump", f"""
            #!/usr/bin/env bash
            cat <<'SASS'
            {textwrap.dedent(self._fake_sass()).strip()}
            SASS
        """)
        sass = subprocess.run(
            [str(EXTRACT), str(spaced_runner)], cwd=self.repo,
            env=self.environment(AP_M1_SASS_OUTPUT_DIR=self.root / "sass space/smoke"),
            text=True, capture_output=True,
        )
        self.assertEqual(sass.returncode, 0, sass.stderr)

    def _fake_sass(self, *, hmma=False, ldl=False):
        return f"""
            code for sm_80
                    Function : _Z22tiled_attention_kernelPKfS0_S0_Pfiib
                    /*0000*/ FFMA R1, R2, R3, R4;
                    /*0010*/ LDG.E R2, [R4];
                    /*0020*/ LDS R3, [R5];
                    /*0030*/ BAR.SYNC.DEFER_BLOCKING 0;
                    /*0040*/ STS [R5], R3;
                    /*0050*/ STG.E [R4], R1;
                    Function : _Z18query_tiled_kernelPKfS0_S0_Pfiib
                    /*0000*/ FFMA R1, R2, R3, R4;
                    /*0010*/ FFMA R5, R6, R7, R8;
                    /*0020*/ LDG.E R2, [R4];
                    /*0030*/ LDS R3, [R5];
                    /*0040*/ BAR.SYNC.DEFER_BLOCKING 0;
                    /*0050*/ STS [R5], R3;
                    /*0060*/ STG.E [R4], R1;
                    {"/*0070*/ HMMA.1688.F32 R1, R2, R3, R4;" if hmma else ""}
                    {"/*0080*/ LDL R9, [R10];" if ldl else ""}
        """

    def test_sass_extracts_both_symbols_counts_contract_and_spill_warning(self):
        self._write_executable(self.fake_bin / "cuobjdump", f"""
            #!/usr/bin/env bash
            cat <<'SASS'
            {textwrap.dedent(self._fake_sass(ldl=True)).strip()}
            SASS
        """)

        result = subprocess.run(
            [str(EXTRACT)], cwd=self.repo, env=self.environment(),
            text=True, capture_output=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        evidence = self.repo / "projects/attention_prefill/results/smoke/m1-sass-smoke.md"
        self.assertTrue(evidence.is_file())
        text = evidence.read_text(encoding="utf-8")
        self.assertIn("Binary SHA-256", text)
        self.assertIn("Source fingerprint", text)
        self.assertIn("Build payload SHA-256", text)
        self.assertIn("Device index", text)
        self.assertIn("GPU UUID", text)
        self.assertIn("CUDA driver", text)
        self.assertIn("tiled_attention_kernel", text)
        self.assertIn("query_tiled_kernel", text)
        self.assertIn(
            "tiled_attention_kernel(float const*, float const*, float const*, "
            "float*, int, int, bool)",
            text,
        )
        self.assertIn(
            "query_tiled_kernel(float const*, float const*, float const*, "
            "float*, int, int, bool)",
            text,
        )
        self.assertIn(
            "projects/attention_prefill/results/smoke",
            text,
        )
        self.assertIn("| `FFMA` | 1 | 2 |", text)
        self.assertIn("| `BAR` | 1 | 1 |", text)
        self.assertIn("| `HMMA` | 0 | 0 |", text)
        self.assertIn("| `LDGSTS` | 0 | 0 |", text)
        self.assertIn("spill warning", text.lower())
        self.assertIn("静态数量不等于 runtime 执行次数", text)
        self.assertTrue(
            (self.repo / "projects/attention_prefill/results/smoke/br1.sass").is_file()
        )
        self.assertTrue(
            (self.repo / "projects/attention_prefill/results/smoke/full.sass").is_file()
        )
        self.assertTrue(
            (self.repo / "projects/attention_prefill/results/smoke/br4.sass").is_file()
        )
        br1 = (self.repo / "projects/attention_prefill/results/smoke/br1.sass").read_text(encoding="utf-8")
        br4 = (self.repo / "projects/attention_prefill/results/smoke/br4.sass").read_text(encoding="utf-8")
        self.assertTrue(br1.strip())
        self.assertTrue(br4.strip())
        self.assertNotEqual(br1, br4)

    def test_sass_none_spill_message_is_explicit(self):
        self._write_executable(self.fake_bin / "cuobjdump", f"""
            #!/usr/bin/env bash
            cat <<'SASS'
            {textwrap.dedent(self._fake_sass()).strip()}
            SASS
        """)

        result = subprocess.run(
            [str(EXTRACT)], cwd=self.repo, env=self.environment(),
            text=True, capture_output=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        evidence = self.repo / "projects/attention_prefill/results/smoke/m1-sass-smoke.md"
        text = evidence.read_text(encoding="utf-8")
        self.assertIn("Spill warning: none", text)
        self.assertNotIn("**none**", text)

    def test_sass_exact_header_match_ignores_collision_symbol(self):
        sass = self._fake_sass().replace(
            "code for sm_80",
            "code for sm_80\n"
            "        Function : _Z28query_tiled_kernel_collisionv\n"
            "        /*0000*/ HMMA.1688.F32 R1, R2, R3, R4;",
        )
        self._write_executable(self.fake_bin / "cuobjdump", f"""
            #!/usr/bin/env bash
            cat <<'SASS'
            {textwrap.dedent(sass).strip()}
            SASS
        """)

        result = subprocess.run(
            [str(EXTRACT)], cwd=self.repo, env=self.environment(),
            text=True, capture_output=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        br4 = (self.repo / "projects/attention_prefill/results/smoke/br4.sass").read_text(encoding="utf-8")
        self.assertIn("Function : _Z18query_tiled_kernel", br4)
        self.assertNotIn("query_tiled_kernel_collision", br4)
        self.assertNotIn("HMMA", br4)

    def test_sass_matches_real_anonymous_namespace_headers(self):
        sass = """
            code for sm_80
                    Function : _ZN40_GLOBAL__N__9370347e_8_tiled_cu_23519e4d22tiled_attention_kernelEPKfS1_S1_Pfiib
                    /*0000*/ FFMA R1, R2, R3, R4;
                    Function : _ZN47_GLOBAL__N__0f6f3d1b_14_query_tiled_cu_d5dc129618query_tiled_kernelEPKfS1_S1_Pfiib
                    /*0000*/ FFMA R5, R6, R7, R8;
        """
        self._write_executable(self.fake_bin / "cuobjdump", f"""
            #!/usr/bin/env bash
            cat <<'SASS'
            {textwrap.dedent(sass).strip()}
            SASS
        """)

        result = subprocess.run(
            [str(EXTRACT)], cwd=self.repo, env=self.environment(),
            text=True, capture_output=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        br1 = (self.repo / "projects/attention_prefill/results/smoke/br1.sass").read_text(encoding="utf-8")
        br4 = (self.repo / "projects/attention_prefill/results/smoke/br4.sass").read_text(encoding="utf-8")
        self.assertIn("tiled_attention_kernel", br1)
        self.assertIn("query_tiled_kernel", br4)

    def test_sass_requires_executable_regular_binary_and_valid_metadata(self):
        self._write_executable(self.fake_bin / "cuobjdump", f"""
            #!/usr/bin/env bash
            cat <<'SASS'
            {textwrap.dedent(self._fake_sass()).strip()}
            SASS
        """)
        self.runner.chmod(stat.S_IRUSR | stat.S_IWUSR)
        result = subprocess.run(
            [str(EXTRACT)], cwd=self.repo, env=self.environment(),
            text=True, capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("可执行普通文件", result.stderr)

        self._write_executable(self.runner, "#!/usr/bin/env bash\nexit 9\n")
        result = subprocess.run(
            [str(EXTRACT)], cwd=self.repo, env=self.environment(),
            text=True, capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("metadata-only", result.stderr)

        self._write_executable(self.runner, """
            #!/usr/bin/env bash
            echo "source_sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa build_contract=debug-sm80 build_contract_payload_sha256=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb device_index=0 gpu_uuid=GPU-6b24bab9-c81f-0e0e-ff68-a11dcf17018b gpu_name=NVIDIA_A100_80GB_PCIe sm=8.0 driver=13030"
        """)
        result = subprocess.run(
            [str(EXTRACT)], cwd=self.repo, env=self.environment(),
            text=True, capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("build_contract", result.stderr)

    def test_sass_rejects_hmma_contract_violation(self):
        self._write_executable(self.fake_bin / "cuobjdump", f"""
            #!/usr/bin/env bash
            cat <<'SASS'
            {textwrap.dedent(self._fake_sass(hmma=True)).strip()}
            SASS
        """)

        result = subprocess.run(
            [str(EXTRACT)], cwd=self.repo, env=self.environment(),
            text=True, capture_output=True,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("HMMA", result.stderr)

    def test_sass_rejects_same_basename_with_wrong_signature(self):
        sass = self._fake_sass().replace(
            "Function : _Z18query_tiled_kernelPKfS0_S0_Pfiib",
            "Function : _Z18query_tiled_kernelPKfS0_S0_Pfiii",
        )
        self._write_executable(self.fake_bin / "cuobjdump", f"""
            #!/usr/bin/env bash
            cat <<'SASS'
            {textwrap.dedent(sass).strip()}
            SASS
        """)
        result = subprocess.run(
            [str(EXTRACT)], cwd=self.repo, env=self.environment(),
            text=True, capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("签名", result.stderr)

    def test_sass_failed_contract_preserves_all_old_artifacts(self):
        output = self.repo / "projects/attention_prefill/results/smoke"
        output.mkdir(parents=True, exist_ok=True)
        old_paths = [
            output / "full.sass", output / "br1.sass", output / "br4.sass",
            output / "m1-sass-smoke.md",
        ]
        for path in old_paths:
            path.write_text("old", encoding="utf-8")
        self._write_executable(self.fake_bin / "cuobjdump", f"""
            #!/usr/bin/env bash
            cat <<'SASS'
            {textwrap.dedent(self._fake_sass(hmma=True)).strip()}
            SASS
        """)
        result = subprocess.run([str(EXTRACT)], cwd=self.repo, env=self.environment(), text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual([path.read_text(encoding="utf-8") for path in old_paths], ["old"] * 4)


if __name__ == "__main__":
    unittest.main()