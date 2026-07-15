#!/usr/bin/env python3

import hashlib
import sys
from pathlib import Path


PAYLOAD_FIELDS = (
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
METADATA_FIELDS = (
    "build_contract",
    "build_contract_payload_sha256",
)
EXPECTED_KERNEL_OPTIONS = (
    "-O3;-lineinfo;-Xptxas=-warn-spills;-Xcompiler=-Wall,-Wextra"
)
EXPECTED_EVIDENCE_OPTIONS = "-O3;-lineinfo;-Xcompiler=-Wall,-Wextra"


def parse_attestation(path: Path) -> tuple[dict[str, str], str]:
    text = path.read_text(encoding="utf-8")
    if not text.endswith("\n") or "\r" in text:
        raise ValueError("attestation must use LF lines and end with a newline")
    lines = text.splitlines()
    expected_fields = (*PAYLOAD_FIELDS, *METADATA_FIELDS)
    if len(lines) != len(expected_fields):
        raise ValueError(
            f"attestation field count mismatch: expected={len(expected_fields)} "
            f"actual={len(lines)}"
        )

    values: dict[str, str] = {}
    for expected_field, line in zip(expected_fields, lines):
        field, separator, value = line.partition("=")
        if not separator or field != expected_field:
            raise ValueError(
                f"attestation field order mismatch: expected={expected_field} "
                f"actual={field or '<missing>'}"
            )
        values[field] = value

    payload = "".join(f"{field}={values[field]}\n" for field in PAYLOAD_FIELDS)
    return values, payload


def validate_attestation(path: Path) -> dict[str, str]:
    values, payload = parse_attestation(path)
    payload_sha256 = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    if values["schema_version"] != "1":
        raise ValueError("canonical attestation requires schema_version=1")
    if values["build_type"] != "Release":
        raise ValueError("canonical attestation requires build_type=Release")
    if values["cuda_architectures"] != "80":
        raise ValueError("canonical attestation requires cuda_architectures=80")
    if values["cuda_compiler_id"] != "NVIDIA":
        raise ValueError("canonical attestation requires cuda_compiler_id=NVIDIA")
    for field in ("cuda_compiler_version", "cuda_compiler_realpath"):
        if not values[field]:
            raise ValueError(f"canonical attestation requires nonempty {field}")
    if values["cmake_cuda_flags"]:
        raise ValueError("canonical attestation requires empty cmake_cuda_flags")
    if set(values["cmake_cuda_flags_release"].split()) != {"-O3", "-DNDEBUG"}:
        raise ValueError(
            "canonical attestation requires cmake_cuda_flags_release token set "
            "{-O3,-DNDEBUG}"
        )

    expected_options = {
        "target_compile_options_evidence_support": EXPECTED_EVIDENCE_OPTIONS,
        "target_compile_options_attention_kernel": EXPECTED_KERNEL_OPTIONS,
        "target_compile_options_flash_kernels": EXPECTED_KERNEL_OPTIONS,
    }
    for field, expected in expected_options.items():
        if values[field] != expected:
            raise ValueError(
                f"canonical attestation requires {field}={expected}; "
                f"actual={values[field]}"
            )

    if values["build_contract_payload_sha256"] != payload_sha256:
        raise ValueError(
            "build_contract_payload_sha256 does not match the complete payload"
        )
    expected_contract = f"release-sm80-{payload_sha256[:16]}"
    if values["build_contract"] != expected_contract:
        raise ValueError(
            f"build_contract does not match payload: expected={expected_contract} "
            f"actual={values['build_contract']}"
        )
    return values


def main() -> int:
    if len(sys.argv) != 2:
        print(
            "usage: validate_build_attestation.py <attestation>",
            file=sys.stderr,
        )
        return 2
    try:
        values = validate_attestation(Path(sys.argv[1]))
    except (OSError, UnicodeError, ValueError) as error:
        print(f"validate_build_attestation: {error}", file=sys.stderr)
        return 1
    print(f"build_contract={values['build_contract']}")
    print(
        "build_contract_payload_sha256="
        f"{values['build_contract_payload_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
