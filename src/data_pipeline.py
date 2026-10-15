"""Deterministic, leakage-aware splits for KuaiSAR interaction tables.

The module deliberately keeps labels close to their source columns:
recommendation samples use ``click`` and search samples use ``click_cnt``.
It accepts CSVs with either ``time`` or ``timestamp`` and writes compact
JSONL/CSV-free split summaries that can be checked into experiment logs.
"""

from __future__ import annotations

import argparse
import csv
import math
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Iterable


def _time_key(row: dict[str, str]) -> float:
    raw = row.get("timestamp") or row.get("time") or ""
    try:
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError('Non-finite timestamp')
        return value
    except ValueError:
        pass
    text = raw.strip().replace("Z", "+00:00")
    try:
        value = datetime.fromisoformat(text)
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone(timedelta(hours=8)))
        return value.timestamp()
    except ValueError as exc:
        raise ValueError(f"Cannot parse time/timestamp value: {raw!r}") from exc


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def split_rows(rows: Iterable[dict[str, str]], train_ratio: float = 0.8, valid_ratio: float = 0.1, *, cutoffs: tuple[float, float] | None = None) -> dict[str, list[dict[str, str]]]:
    if not 0 < train_ratio < 1 or not 0 < valid_ratio < 1 - train_ratio:
        raise ValueError('Ratios must be positive and sum to less than one')
    ordered = sorted(rows, key=lambda row: (_time_key(row), json.dumps(row, sort_keys=True, ensure_ascii=False)))
    result = {name: [] for name in ('train', 'valid', 'test')}
    if not ordered:
        return result
    if cutoffs is None:
        # Fixture convenience only; production CLI requires shared explicit cutoffs.
        cutoffs = (_time_key(ordered[int(len(ordered) * train_ratio)]),
                   _time_key(ordered[int(len(ordered) * (train_ratio + valid_ratio))]))
    a, b = cutoffs
    if not math.isfinite(a) or not math.isfinite(b) or a >= b:
        raise ValueError('Two finite, strictly increasing cutoffs are required')
    for row in ordered:
        t = _time_key(row)
        result['train' if t < a else 'valid' if t < b else 'test'].append(row)
    # Refuse split search sessions rather than silently introducing leakage.
    owners = {}
    for split, items in result.items():
        for row in items:
            session = row.get('search_session_id')
            if session:
                key = (row.get('user_id'), session)
                if key in owners and owners[key] != split:
                    raise ValueError(f'Search session crosses cutoff: {key}')
                owners[key] = split
    return result


def write_jsonl(path: Path, rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def summarize(name: str, splits: dict[str, list[dict[str, str]]]) -> dict:
    result = {"name": name, "counts": {key: len(value) for key, value in splits.items()}, "splits": {}}
    for split, rows in splits.items():
        labels = []
        for row in rows:
            value = row.get("click") if "click" in row else row.get("click_cnt")
            if value not in (None, ""):
                try:
                    labels.append(float(value))
                except ValueError:
                    pass
        result["splits"][split] = {
            "first_time": _time_key(rows[0]) if rows else None,
            "last_time": _time_key(rows[-1]) if rows else None,
            "label_count": len(labels),
            "positive_count": sum(value > 0 for value in labels),
            "label_mean": sum(labels) / len(labels) if labels else None,
        }
    return result


def process_table(path: Path, output_dir: Path, cutoffs: tuple[float, float]) -> dict:
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"Empty table: {path}")
    required_time = {"time", "timestamp"}
    if not required_time.intersection(rows[0]):
        raise ValueError(f"Missing time/timestamp column: {path}")
    splits = split_rows(rows, cutoffs=cutoffs)
    name = path.stem
    for split, split_rows_ in splits.items():
        write_jsonl(output_dir / f"{name}.{split}.jsonl", split_rows_)
    return summarize(name, splits)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw/extracted"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--summary", type=Path, default=Path("artifacts/split_summary.json"))
    parser.add_argument('--train-end', required=True, help='Exclusive train cutoff, same units as input timestamp; audit units first')
    parser.add_argument('--valid-end', required=True, help='Exclusive validation cutoff shared by both services')
    args = parser.parse_args()
    tables = [path for path in sorted(args.input_dir.rglob("*.csv")) if path.name in {"rec_inter.csv", "src_inter.csv"}]
    if not tables:
        raise SystemExit(f"No rec_inter.csv or src_inter.csv found below {args.input_dir}")
    cutoffs = (_time_key({'time': args.train_end}), _time_key({'time': args.valid_end}))
    report = {"split": "shared_explicit_time", "cutoffs": cutoffs,
              "status": "prototype_pending_real_data_audit",
              "tables": [process_table(path, args.output_dir, cutoffs) for path in tables]}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
