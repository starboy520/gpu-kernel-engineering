#!/usr/bin/env python3

import argparse
import hashlib
from pathlib import Path


SOURCE_SUFFIXES = {".cu", ".cuh", ".cpp", ".hpp", ".h", ".cmake"}
EXCLUDED_PARTS = {"build", "results", "docs", "tests", "__pycache__"}
EXPLICIT_FILES = (
    "CMakeLists.txt",
    "common/CMakeLists.txt",
    "common/validation.cpp",
    "projects/flash_attention/CMakeLists.txt",
    "projects/attention_prefill/CMakeLists.txt",
)
SOURCE_DIRECTORIES = (
    "common/include",
    "common/src",
    "projects/flash_attention/include",
    "projects/flash_attention/kernels",
    "projects/attention_prefill/include",
    "projects/attention_prefill/kernels",
    "projects/attention_prefill/evidence",
)


def is_source_file(path: Path) -> bool:
    return path.name == "CMakeLists.txt" or path.suffix in SOURCE_SUFFIXES


def filesystem_sources(repo_root: Path) -> list[str]:
    selected = {
        relative
        for relative in EXPLICIT_FILES
        if (repo_root / relative).is_file()
    }
    for directory in SOURCE_DIRECTORIES:
        root = repo_root / directory
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            relative_path = path.relative_to(repo_root)
            if (
                path.is_file()
                and not any(part in EXCLUDED_PARTS for part in relative_path.parts)
                and is_source_file(path)
            ):
                selected.add(relative_path.as_posix())
    sources = sorted(selected)
    if not sources:
        raise ValueError("no evidence sources found")
    return sources


def source_fingerprint(repo_root: Path, sources: list[str]) -> str:
    manifest = "".join(
        f"{relative}:{hashlib.sha256((repo_root / relative).read_bytes()).hexdigest()}\n"
        for relative in sources
    )
    return hashlib.sha256(manifest.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--print-files", action="store_true")
    arguments = parser.parse_args()

    repo_root = arguments.repo_root.resolve(strict=True)
    sources = filesystem_sources(repo_root)
    if arguments.print_files:
        print("\n".join(sources))
    else:
        print(source_fingerprint(repo_root, sources))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
