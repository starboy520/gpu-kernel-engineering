#!/usr/bin/env python3

import csv
import sys
from pathlib import Path


def append_fields(path: Path, fields: list[str], mode: str) -> None:
    if mode == "--header" and path.exists():
        raise ValueError(f"refusing to overwrite existing CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    file_mode = "x" if mode == "--header" else "a"
    with path.open(file_mode, newline="", encoding="utf-8") as handle:
        csv.writer(handle, lineterminator="\n").writerow(fields)


def main() -> int:
    if len(sys.argv) < 4 or sys.argv[1] not in ("--header", "--row"):
        print(
            "usage: append_m1_csv.py <--header|--row> <csv-path> <field>...",
            file=sys.stderr,
        )
        return 2
    try:
        append_fields(Path(sys.argv[2]), sys.argv[3:], sys.argv[1])
    except (OSError, ValueError) as error:
        print(f"append_m1_csv: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
