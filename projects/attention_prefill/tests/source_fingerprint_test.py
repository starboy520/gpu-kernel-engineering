#!/usr/bin/env python3

import argparse
import importlib.util
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


PROJECT = Path(__file__).parents[1]
SCRIPT = PROJECT / "scripts" / "source_fingerprint.py"


def load_fingerprint_module():
    spec = importlib.util.spec_from_file_location("source_fingerprint", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SourceFingerprintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fingerprint = load_fingerprint_module()

    def make_repo(self, root: Path) -> None:
        files = {
            "CMakeLists.txt": "root\n",
            "common/CMakeLists.txt": "common\n",
            "common/validation.cpp": "validation\n",
            "common/include/gpu_kernel/api.hpp": "api\n",
            "projects/flash_attention/CMakeLists.txt": "flash\n",
            "projects/flash_attention/include/flash_attention/api.hpp": "flash api\n",
            "projects/flash_attention/kernels/tiled.cu": "flash kernel\n",
            "projects/attention_prefill/CMakeLists.txt": "prefill\n",
            "projects/attention_prefill/include/attention_prefill/api.hpp": "prefill api\n",
            "projects/attention_prefill/kernels/query_tiled.cu": "prefill kernel\n",
            "projects/attention_prefill/evidence/main.cu": "untracked evidence\n",
        }
        for relative, content in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

    def test_filesystem_manifest_includes_untracked_evidence_and_excludes_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            self.make_repo(repo)
            included = repo / "projects/attention_prefill/evidence/untracked.hpp"
            included.write_text("included\n", encoding="utf-8")
            cuda_header = repo / "projects/flash_attention/include/flash_attention/common.cuh"
            cuda_header.write_text("cuda header\n", encoding="utf-8")
            excluded = repo / "projects/attention_prefill/results/raw/generated.cpp"
            excluded.parent.mkdir(parents=True)
            excluded.write_text("excluded\n", encoding="utf-8")

            sources = self.fingerprint.filesystem_sources(repo)

            self.assertIn(
                "projects/attention_prefill/evidence/untracked.hpp", sources
            )
            self.assertIn(
                "projects/flash_attention/include/flash_attention/common.cuh",
                sources,
            )
            self.assertNotIn(
                "projects/attention_prefill/results/raw/generated.cpp", sources
            )

    def test_matching_file_addition_and_removal_change_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            self.make_repo(repo)
            initial = self.fingerprint.source_fingerprint(
                repo, self.fingerprint.filesystem_sources(repo)
            )
            added = repo / "projects/attention_prefill/evidence/new_source.cpp"
            added.write_text("new source\n", encoding="utf-8")
            after_add = self.fingerprint.source_fingerprint(
                repo, self.fingerprint.filesystem_sources(repo)
            )
            added.unlink()
            after_remove = self.fingerprint.source_fingerprint(
                repo, self.fingerprint.filesystem_sources(repo)
            )

            self.assertNotEqual(initial, after_add)
            self.assertEqual(initial, after_remove)

    def test_tracking_and_committing_existing_source_does_not_change_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            self.make_repo(repo)
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(
                ["git", "-C", str(repo), "config", "user.email", "test@example.com"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(repo), "config", "user.name", "Test User"],
                check=True,
            )
            before = self.fingerprint.source_fingerprint(
                repo, self.fingerprint.filesystem_sources(repo)
            )
            subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
            subprocess.run(
                ["git", "-C", str(repo), "commit", "-qm", "fixture"],
                check=True,
            )
            after = self.fingerprint.source_fingerprint(
                repo, self.fingerprint.filesystem_sources(repo)
            )

            self.assertEqual(before, after)

    def test_cmake_configure_depends_reconfigures_for_add_and_remove(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            build = root / "build"
            self.make_repo(repo)
            fixture_script = repo / "source_fingerprint.py"
            fixture_script.write_bytes(SCRIPT.read_bytes())
            cmake_project = root / "cmake-project"
            cmake_project.mkdir()
            (cmake_project / "CMakeLists.txt").write_text(textwrap.dedent(f"""
                cmake_minimum_required(VERSION 3.25)
                project(fingerprint_configure_depends NONE)
                find_package(Python3 REQUIRED COMPONENTS Interpreter)
                file(GLOB_RECURSE candidates CONFIGURE_DEPENDS LIST_DIRECTORIES false
                    "{repo.as_posix()}/common/include/*"
                    "{repo.as_posix()}/projects/flash_attention/include/*"
                    "{repo.as_posix()}/projects/flash_attention/kernels/*"
                    "{repo.as_posix()}/projects/attention_prefill/include/*"
                    "{repo.as_posix()}/projects/attention_prefill/kernels/*"
                    "{repo.as_posix()}/projects/attention_prefill/evidence/*"
                )
                execute_process(
                    COMMAND ${{Python3_EXECUTABLE}} "{fixture_script.as_posix()}"
                            --repo-root "{repo.as_posix()}"
                    OUTPUT_VARIABLE fingerprint
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    COMMAND_ERROR_IS_FATAL ANY
                )
                file(WRITE "${{CMAKE_BINARY_DIR}}/fingerprint.txt" "${{fingerprint}}\n")
            """).lstrip(), encoding="utf-8")

            subprocess.run(
                ["cmake", "-S", str(cmake_project), "-B", str(build)],
                check=True, capture_output=True, text=True,
            )
            initial = (build / "fingerprint.txt").read_text(encoding="utf-8")
            added = repo / "projects/attention_prefill/evidence/configure_depends.hpp"
            added.write_text("added\n", encoding="utf-8")
            subprocess.run(
                ["cmake", "--build", str(build)],
                check=True, capture_output=True, text=True,
            )
            after_add = (build / "fingerprint.txt").read_text(encoding="utf-8")
            added.unlink()
            subprocess.run(
                ["cmake", "--build", str(build)],
                check=True, capture_output=True, text=True,
            )
            after_remove = (build / "fingerprint.txt").read_text(encoding="utf-8")

            self.assertNotEqual(initial, after_add)
            self.assertEqual(initial, after_remove)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--runner", type=Path)
    arguments, remaining = parser.parse_known_args()
    if (arguments.repo_root is None) != (arguments.runner is None):
        parser.error("--repo-root and --runner must be provided together")
    if arguments.repo_root is not None and arguments.runner is not None:
        class ActualEmbeddedFingerprintTest(unittest.TestCase):
            def test_actual_cmake_embedded_fingerprint_matches_python(self):
                expected = subprocess.run(
                    [
                        sys.executable, str(SCRIPT), "--repo-root",
                        str(arguments.repo_root),
                    ],
                    check=True, capture_output=True, text=True,
                ).stdout.strip()
                metadata = subprocess.run(
                    [str(arguments.runner), "--metadata-only"],
                    check=True, capture_output=True, text=True,
                ).stdout.split()
                fields = dict(token.split("=", 1) for token in metadata)
                self.assertEqual(fields["source_sha256"], expected)

    unittest.main(argv=[sys.argv[0], *remaining])