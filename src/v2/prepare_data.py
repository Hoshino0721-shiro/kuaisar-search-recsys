"""Equivalent v1 data build with indexed audit lookups and explicit audit recovery.

The frozen v1 module remains unchanged. A fresh build compiles that exact source
with one additional index DDL before its expensive correlated UPDATE. Recovery
is restricted to the fully imported official small dataset and audit stage.
"""
import argparse
import json
import sqlite3
from pathlib import Path
from src import prepare_data as legacy
from src.audit_dataset import md5

INDEX_SQL = '''CREATE INDEX IF NOT EXISTS ambiguous_key_lookup
ON ambiguous_keys(scene,user_id,session_id,item_id,timestamp_ms);'''


def fresh():
    path = Path(legacy.__file__)
    source = path.read_text(encoding='utf-8')
    needle = "    UPDATE events SET exclusion='ambiguous_event_key'"
    if source.count(needle) != 1:
        raise RuntimeError('Legacy audit source changed; inspect before applying optimization')
    source = source.replace(needle, '    ' + INDEX_SQL + '\n' + needle)
    namespace = {'__name__': '__main__', '__file__': str(path)}
    exec(compile(source, str(path), 'exec'), namespace)


def resume_audit():
    root = legacy.ROOT
    if (root / 'manifest.json').exists():
        raise FileExistsError('Completed manifest exists; refusing to modify released data')
    archive = Path('data/raw/KuaiSAR.zip')
    checksum = md5(archive)
    if checksum != 'daea8cbf605db6bd5841740f0e4a12d9':
        raise ValueError('Official archive checksum mismatch')
    db = sqlite3.connect((root / 'dataset.sqlite').resolve().as_uri() + '?mode=rw', uri=True)
    db.execute('PRAGMA cache_size=-131072')
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    required = {'items', 'users', 'events', 'session_audit', 'duplicate_records', 'ambiguous_keys'}
    if not required.issubset(tables):
        raise ValueError('Recovery requires complete import and both duplicate audit tables')
    counts = dict(db.execute('SELECT scene,count(*) FROM events GROUP BY scene'))
    if counts != {'rec': 7493101, 'search': 3171231}:
        raise ValueError(f'Import is incomplete or not the official small dataset: {counts}')
    if db.execute('SELECT count(*) FROM items').fetchone()[0] != 4157218:
        raise ValueError('Item import is incomplete')
    if db.execute('SELECT count(*) FROM users').fetchone()[0] != 25877:
        raise ValueError('User import is incomplete')
    integrity = db.execute('PRAGMA integrity_check').fetchone()[0]
    if integrity != 'ok':
        raise ValueError(integrity)
    print('Validated complete import; resuming indexed audit', flush=True)
    db.executescript(INDEX_SQL + '''
    UPDATE events SET exclusion='ambiguous_event_key' WHERE exclusion IS NULL AND EXISTS
      (SELECT 1 FROM ambiguous_keys a WHERE a.scene=events.scene AND a.user_id=events.user_id
       AND a.session_id=events.session_id AND a.item_id=events.item_id AND a.timestamp_ms=events.timestamp_ms);
    CREATE INDEX IF NOT EXISTS session_audit_key ON session_audit(user_id,session_id);
    UPDATE events SET exclusion='inconsistent_session' WHERE scene='search' AND exclusion IS NULL AND EXISTS
      (SELECT 1 FROM session_audit s WHERE s.user_id=events.user_id AND s.session_id=events.session_id);
    CREATE INDEX IF NOT EXISTS event_split ON events(scene,split,exclusion,timestamp_ms);
    CREATE INDEX IF NOT EXISTS event_user_time ON events(user_id,timestamp_ms);
    CREATE VIEW IF NOT EXISTS clean_events AS SELECT e.*,CASE WHEN i.item_id IS NULL THEN 1 ELSE 0 END AS missing_item_features
      FROM events e LEFT JOIN items i ON i.item_id=e.item_id WHERE e.exclusion IS NULL;
    CREATE TABLE IF NOT EXISTS train_users AS SELECT DISTINCT user_id FROM events WHERE split='train' AND exclusion IS NULL;
    CREATE UNIQUE INDEX IF NOT EXISTS train_user_key ON train_users(user_id);
    CREATE TABLE IF NOT EXISTS train_items AS SELECT DISTINCT item_id FROM events WHERE split='train' AND exclusion IS NULL;
    CREATE UNIQUE INDEX IF NOT EXISTS train_item_key ON train_items(item_id);
    ''')
    for scene in ('rec', 'search'):
        for split in ('train', 'valid', 'test'):
            db.execute(f"CREATE VIEW IF NOT EXISTS {scene}_{split} AS SELECT * FROM clean_events WHERE scene='{scene}' AND split='{split}'")
    def query(sql):
        cur = db.execute(sql)
        return [dict(zip([d[0] for d in cur.description], row)) for row in cur]
    report = {'source_md5': checksum,
              'cutoffs_beijing': ['2023-05-29T00:00:00+08:00', '2023-05-30T00:00:00+08:00'],
              'cutoffs_ms': [legacy.TRAIN_END, legacy.VALID_END], 'status': 'building',
              'raw_counts': {'rec': counts['rec'], 'search': counts['search']}}
    report['split_counts'] = query('SELECT scene,split,count(*) AS rows,sum(label) AS positives,min(timestamp_ms) AS min_ms,max(timestamp_ms) AS max_ms,sum(missing_item_features) AS missing_item_features FROM clean_events GROUP BY scene,split')
    report['exclusions'] = query('SELECT scene,exclusion,count(*) AS rows FROM events WHERE exclusion IS NOT NULL GROUP BY scene,exclusion')
    report['ambiguous_keys'] = query('SELECT scene,count(*) AS groups,sum(min_label<>max_label) AS conflicting_label_groups FROM ambiguous_keys GROUP BY scene')
    report['sessions'] = query('SELECT * FROM session_audit')
    report['cold_start'] = query("SELECT e.scene,e.split,count(*) AS rows,sum(u.user_id IS NULL) AS new_user_rows,sum(i.item_id IS NULL) AS new_item_rows FROM clean_events e LEFT JOIN train_users u ON u.user_id=e.user_id LEFT JOIN train_items i ON i.item_id=e.item_id WHERE e.split<>'train' GROUP BY e.scene,e.split")
    report['checks'] = {
        'foreign_user_errors': db.execute('SELECT count(*) FROM events e LEFT JOIN users u ON u.user_id=e.user_id WHERE u.user_id IS NULL').fetchone()[0],
        'cross_split_sessions': db.execute("SELECT count(*) FROM (SELECT user_id,session_id FROM clean_events WHERE scene='search' GROUP BY user_id,session_id HAVING count(DISTINCT split)>1)").fetchone()[0],
        'sqlite_integrity': db.execute('PRAGMA integrity_check').fetchone()[0]}
    if report['checks'] != {'foreign_user_errors': 0, 'cross_split_sessions': 0, 'sqlite_integrity': 'ok'}:
        raise ValueError(report['checks'])
    if sum(x['rows'] for x in report['split_counts']) + sum(x['rows'] for x in report['exclusions']) != sum(counts.values()):
        raise ValueError('Excluded and clean event counts do not reconcile')
    report['status'] = 'ready'
    report['history_contract'] = 'Use only events.timestamp_ms < target timestamp. Never feed current click_count or current search_source feedback into rec predictors. Train vocab/statistics only on train. No histories materialized yet.'
    db.commit()
    db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    db.close()
    (root / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-audit', action='store_true')
    args = parser.parse_args()
    resume_audit() if args.resume_audit else fresh()


if __name__ == '__main__':
    main()
