"""Lightweight archive and tabular-file audit for KuaiSAR.

The script intentionally does not infer CTR labels. It reports file names,
headers, row samples, time-like fields and value distributions so the label
contract can be decided from the actual release.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable


TEXT_SUFFIXES = {".csv", ".tsv", ".txt", ".dat", ".json", ".jsonl"}
TIME_WORDS = re.compile(r"time|timestamp|ts|date|day|hour|minute", re.I)
EVENT_WORDS = re.compile(r"event|action|behavior|behaviour|label|click|like|follow|watch|search|scene", re.I)


def md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_inventory(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        entries = []
        for info in archive.infolist():
            entries.append({"name": info.filename, "size": info.file_size, "compressed": info.compress_size})
        return {"file": str(path), "md5": md5(path), "entries": entries}


def split_line(raw: str, delimiter: str | None) -> list[str]:
    if delimiter:
        return next(csv.reader([raw], delimiter=delimiter))
    return re.split(r"\s+", raw.strip())


def sample_text_file(path: Path, max_lines: int = 5) -> dict:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        lines = [line.rstrip("\n\r") for _, line in zip(range(max_lines), handle)]
    if not lines:
        return {"path": str(path), "empty": True}
    suffix = path.suffix.lower()
    delimiter = "\t" if suffix == ".tsv" else "," if suffix == ".csv" else None
    rows = [split_line(line, delimiter) for line in lines]
    header = rows[0] if suffix in {".csv", ".tsv"} else None
    fields = header or []
    return {
        "path": str(path),
        "size": path.stat().st_size,
        "suffix": suffix,
        "delimiter": delimiter,
        "sample_lines": len(lines),
        "header": fields,
        "widths": [len(row) for row in rows],
        "time_like_fields": [field for field in fields if TIME_WORDS.search(field)],
        "event_like_fields": [field for field in fields if EVENT_WORDS.search(field)],
        "sample": rows[:3],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/dataset_audit.json"))
    args = parser.parse_args()

    report: dict = {"raw_dir": str(args.raw), "archives": [], "text_files": [], "notes": []}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for path in sorted(args.raw.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() == ".zip":
            try:
                report["archives"].append(archive_inventory(path))
            except zipfile.BadZipFile as exc:
                report["archives"].append({"file": str(path), "error": f"bad_zip: {exc}"})
        elif path.name.endswith(".zip.partial"):
            report["notes"].append({"partial_archive": str(path), "size": path.stat().st_size, "usable": False})
        elif path.suffix.lower() in TEXT_SUFFIXES:
            report["text_files"].append(sample_text_file(path))

    if not report["archives"] and not report["text_files"]:
        report["notes"].append("No readable archive or tabular text file found.")
    report["notes"].append("This report does not infer CTR labels or exposure semantics.")
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
