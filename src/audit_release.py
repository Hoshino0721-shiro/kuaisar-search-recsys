"""Stream the verified official archive without altering source CSVs."""
import csv
import hashlib
import io
import json
import zipfile
from decimal import Decimal
from collections import Counter
from pathlib import Path


def main():
    path = Path('data/raw/KuaiSAR.zip')
    digest = hashlib.md5()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != 'daea8cbf605db6bd5841740f0e4a12d9':
        raise ValueError('Not the verified official small release')
    report = {'archive': str(path), 'md5': digest.hexdigest(), 'tables': {},
              'limitations': ['Exact duplicate rows, item/user referential integrity and session consistency not checked in this first streaming pass.', 'Numeric timestamp units require comparison with text times before splitting.']}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not name.endswith('.csv'):
                continue
            with io.TextIOWrapper(z.open(name), encoding='utf-8-sig', newline='') as f:
                reader = csv.DictReader(f)
                fields = reader.fieldnames
                missing = Counter()
                distributions = {k: Counter() for k in fields if k in ('click', 'click_cnt', 'item_type', 'search_source')}
                time_fields = [k for k in fields if k in ('timestamp', 'time', 'search_session_timestamp', 'search_session_time')]
                bounds = {k: [None, None] for k in time_fields}
                days = Counter()
                users = set()
                malformed = 0
                n = 0
                examples = []
                for row in reader:
                    n += 1
                    if None in row or any(v is None for v in row.values()):
                        malformed += 1
                    for k in fields:
                        if row.get(k) in (None, ''):
                            missing[k] += 1
                    for k, counts in distributions.items():
                        counts[row.get(k)] += 1
                    for k in time_fields:
                        value = row.get(k)
                        if value:
                            value = int(Decimal(value)) if 'timestamp' in k else value
                            lo, hi = bounds[k]
                            bounds[k] = [value if lo is None else min(lo, value), value if hi is None else max(hi, value)]
                    text_time = row.get('time') or row.get('search_session_time')
                    if text_time:
                        days[text_time[:10]] += 1
                    if 'user_id' in row:
                        users.add(row['user_id'])
                    if n <= 2:
                        examples.append({k: row[k] for k in time_fields})
                    if n % 1000000 == 0:
                        print(name, n, flush=True)
            report['tables'][name] = {'rows': n, 'columns': fields,
                'empty_fields': dict(missing), 'malformed_rows': malformed,
                'distributions': {k: dict(v) for k, v in distributions.items()},
                'time_bounds': bounds, 'time_examples': examples, 'daily_rows': dict(sorted(days.items())),
                'unique_users': len(users) if 'user_id' in fields else None,
                'zip_crc': 'passed_by_full_member_read'}
            print(name, 'complete', n, flush=True)
    out = Path('artifacts/release_audit.json')
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(out, flush=True)


if __name__ == '__main__':
    main()
