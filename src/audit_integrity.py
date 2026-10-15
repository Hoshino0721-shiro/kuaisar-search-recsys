"""Second streaming pass: keys, clocks and sessions; no source mutation."""
import csv
import io
import json
import zipfile
from collections import Counter
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from pathlib import Path
from src.audit_dataset import md5

TZ = timezone(timedelta(hours=8))


def rows(z, name):
    with io.TextIOWrapper(z.open('KuaiSAR_final/' + name), encoding='utf-8-sig', newline='') as f:
        yield from csv.DictReader(f, strict=True)


def main():
    path = Path('data/raw/KuaiSAR.zip')
    checksum = md5(path)
    if checksum != 'daea8cbf605db6bd5841740f0e4a12d9':
        raise ValueError('Archive checksum mismatch')
    report = {'md5': checksum, 'tables': {}, 'limitations': ['Exact full-row duplicates and interaction composite-key duplicates not checked.']}
    with zipfile.ZipFile(path) as z:
        items = set()
        item_counts = Counter()
        anomalous = []
        for r in rows(z, 'item_features.csv'):
            key = int(r['item_id'])
            item_counts['rows'] += 1
            item_counts['duplicate_item_ids'] += key in items
            items.add(key)
            if r['item_type'] not in ('NORMAL', 'AD', 'UNKNOW', 'LIVEGAME_PLAYBACK'):
                anomalous.append({'item_id': r['item_id'], 'item_type': r['item_type']})
            item_counts['epoch_upload_date'] += r['upload_time'].startswith('1970-01-01')
        report['items'] = dict(item_counts, unique_items=len(items), anomalous_item_types=anomalous)
        print('Item keys complete', flush=True)
        users = {r['user_id'] for r in rows(z, 'user_features.csv')}
        for name in ('rec_inter.csv', 'src_inter.csv'):
            search = name.startswith('src')
            tscol = 'search_session_timestamp' if search else 'timestamp'
            textcol = 'search_session_time' if search else 'time'
            counts = Counter()
            missing_items = set()
            seen_items = set()
            sessions = {}
            text_cache = {}
            for r in rows(z, name):
                counts['rows'] += 1
                item = int(r['item_id'])
                seen_items.add(item)
                if item not in items:
                    counts['missing_item_feature_rows'] += 1
                    missing_items.add(item)
                counts['missing_user_feature_rows'] += r['user_id'] not in users
                milliseconds = Decimal(r[tscol])
                counts['fractional_milliseconds'] += milliseconds != milliseconds.to_integral_value()
                text = r[textcol]
                if text not in text_cache:
                    text_cache[text] = int(datetime.fromisoformat(text).replace(tzinfo=TZ).timestamp())
                counts['timestamp_text_mismatches'] += int(milliseconds // 1000) != text_cache[text]
                if search:
                    key = (r['user_id'], r['search_session_id'])
                    signature = (r[tscol], r['keyword'], r['search_source'])
                    if key not in sessions:
                        sessions[key] = [signature, False, False]
                    state = sessions[key]
                    state[1] |= int(r['click_cnt']) > 0
                    state[2] |= signature != state[0]
                if counts['rows'] % 1000000 == 0:
                    print(name, counts['rows'], flush=True)
            result = dict(counts, distinct_items=len(seen_items), distinct_missing_items=len(missing_items))
            if search:
                result.update(sessions=len(sessions), clicked_sessions=sum(s[1] for s in sessions.values()), inconsistent_sessions=sum(s[2] for s in sessions.values()))
            report['tables'][name] = result
    Path('artifacts/integrity_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report['tables'], indent=2), flush=True)


if __name__ == '__main__':
    main()
