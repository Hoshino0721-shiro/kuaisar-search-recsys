"""Build an auditable disk-backed dataset. Run from project root.

Outputs SQLite tables/views, no future-derived aggregates or negative sampling.
Refuses overwrite. Source row ordinals refer to CSV data records, not physical lines.
"""
import csv
import hashlib
import io
import json
import sqlite3
import zipfile
from collections import Counter
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from pathlib import Path
from src.audit_dataset import md5

ROOT = Path('data/processed/kuaisar_small_v1')
TZ = timezone(timedelta(hours=8))
TRAIN_END = int(datetime(2023, 5, 29, tzinfo=TZ).timestamp()*1000)
VALID_END = int(datetime(2023, 5, 30, tzinfo=TZ).timestamp()*1000)


def split_of(ts):
    return 'train' if ts < TRAIN_END else 'valid' if ts < VALID_END else 'test'


def milliseconds(raw):
    value = Decimal(raw)
    if not value.is_finite() or value != value.to_integral_value():
        raise ValueError('Non-integral timestamp')
    return int(value)


def records(z, name):
    with io.TextIOWrapper(z.open('KuaiSAR_final/' + name), encoding='utf-8-sig', newline='') as f:
        for n, row in enumerate(csv.DictReader(f, strict=True), 1):
            if None in row or any(v is None for v in row.values()):
                raise ValueError((name, n, 'Malformed row'))
            yield n, row


def main():
    archive = Path('data/raw/KuaiSAR.zip')
    checksum = md5(archive)
    if checksum != 'daea8cbf605db6bd5841740f0e4a12d9':
        raise ValueError('Archive checksum mismatch')
    ROOT.mkdir(parents=True, exist_ok=True)
    dbpath = ROOT/'dataset.sqlite'
    if dbpath.exists():
        raise FileExistsError('Refusing to overwrite dataset; inspect existing build')
    db = sqlite3.connect(dbpath)
    db.executescript('''
    PRAGMA journal_mode=WAL;
    PRAGMA synchronous=NORMAL;
    PRAGMA cache_size=-131072;
    PRAGMA temp_store=FILE;
    CREATE TABLE items(item_id INTEGER PRIMARY KEY, author_id TEXT, raw_type TEXT,
      feature_type TEXT, type_anomaly INTEGER, upload_time TEXT, upload_missing INTEGER,
      category_id TEXT);
    CREATE TABLE users(user_id INTEGER PRIMARY KEY);
    CREATE TABLE events(id INTEGER PRIMARY KEY, scene TEXT, source_row INTEGER,
      user_id INTEGER, item_id INTEGER, timestamp_ms INTEGER, label INTEGER,
      click_count INTEGER, session_id TEXT, keyword TEXT, search_source TEXT,
      result_type TEXT, split TEXT, fingerprint BLOB, exclusion TEXT);
    CREATE TABLE session_audit(user_id INTEGER, session_id TEXT, details TEXT);
    ''')
    report = {'source_md5': checksum, 'cutoffs_beijing': ['2023-05-29T00:00:00+08:00', '2023-05-30T00:00:00+08:00'],
              'cutoffs_ms': [TRAIN_END, VALID_END], 'status': 'building', 'raw_counts': {}}
    sessions = {}
    with zipfile.ZipFile(archive) as z:
        batch = []
        for n, r in records(z, 'item_features.csv'):
            typ = r['item_type']
            anomaly = typ not in ('NORMAL','AD','UNKNOW','LIVEGAME_PLAYBACK')
            missing = r['upload_time'].startswith('1970-01-01') or r['upload_time'] in ('','NULL','null','unknown')
            batch.append((int(r['item_id']), r['author_id'], typ, 'UNKNOWN' if anomaly or typ == 'UNKNOW' else typ,
                          int(anomaly), None if missing else r['upload_time'], int(missing), r['first_level_category_id']))
            if len(batch) == 50000:
                db.executemany('INSERT INTO items VALUES (?,?,?,?,?,?,?,?)', batch); db.commit(); batch=[]
        db.executemany('INSERT INTO items VALUES (?,?,?,?,?,?,?,?)', batch)
        db.executemany('INSERT INTO users VALUES (?)', ((int(r['user_id']),) for _,r in records(z,'user_features.csv')))
        db.commit()
        print('Features loaded', flush=True)
        for scene, name in [('rec','rec_inter.csv'), ('search','src_inter.csv')]:
            batch=[]
            for n,r in records(z,name):
                ts=milliseconds(r['timestamp'] if scene=='rec' else r['search_session_timestamp'])
                count=int(r['click'] if scene=='rec' else r['click_cnt'])
                if count < 0 or (scene=='rec' and count not in (0,1)):
                    raise ValueError((name,n,'Invalid label'))
                uid,iid=int(r['user_id']),int(r['item_id'])
                sid=r.get('search_session_id','')
                if scene=='search':
                    key=(uid,sid)
                    signature=(ts,r['keyword'],r['search_source'])
                    sessions.setdefault(key,set()).add(signature)
                # Hash is for exact-source-record deduplication; source ZIP retains every value.
                digest=hashlib.sha256(json.dumps(r,sort_keys=True,ensure_ascii=False).encode()).digest()
                batch.append((scene,n,uid,iid,ts,int(count>0),count,sid,r.get('keyword',''),r.get('search_source',''),r.get('item_type',''),split_of(ts),digest))
                if len(batch)==50000:
                    db.executemany('INSERT INTO events(scene,source_row,user_id,item_id,timestamp_ms,label,click_count,session_id,keyword,search_source,result_type,split,fingerprint) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',batch)
                    db.commit();batch=[]
                if n%1000000==0: print(scene,n,flush=True)
            db.executemany('INSERT INTO events(scene,source_row,user_id,item_id,timestamp_ms,label,click_count,session_id,keyword,search_source,result_type,split,fingerprint) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',batch)
            db.commit();report['raw_counts'][scene]=n
    bad=[(uid,sid,json.dumps(sorted(signatures),ensure_ascii=False)) for (uid,sid),signatures in sessions.items() if len(signatures)>1]
    del sessions
    db.executemany('INSERT INTO session_audit VALUES (?,?,?)',bad); db.commit()
    print('Auditing duplicate records and event keys',flush=True)
    db.executescript('''
    CREATE INDEX event_fingerprint ON events(scene,fingerprint);
    CREATE TABLE duplicate_records AS SELECT scene,fingerprint,min(id) AS kept_id,count(*) AS copies
      FROM events GROUP BY scene,fingerprint HAVING count(*)>1;
    UPDATE events SET exclusion='duplicate_source_record' WHERE id IN (
      SELECT e.id FROM events e JOIN duplicate_records d ON e.scene=d.scene AND e.fingerprint=d.fingerprint WHERE e.id<>d.kept_id);
    CREATE INDEX event_key ON events(scene,user_id,session_id,item_id,timestamp_ms);
    CREATE TABLE ambiguous_keys AS SELECT scene,user_id,session_id,item_id,timestamp_ms,
      count(*) AS records,min(label) AS min_label,max(label) AS max_label
      FROM events WHERE exclusion IS NULL GROUP BY scene,user_id,session_id,item_id,timestamp_ms HAVING count(*)>1;
    UPDATE events SET exclusion='ambiguous_event_key' WHERE exclusion IS NULL AND EXISTS
      (SELECT 1 FROM ambiguous_keys a WHERE a.scene=events.scene AND a.user_id=events.user_id
       AND a.session_id=events.session_id AND a.item_id=events.item_id AND a.timestamp_ms=events.timestamp_ms);
    CREATE INDEX session_audit_key ON session_audit(user_id,session_id);
    UPDATE events SET exclusion='inconsistent_session' WHERE scene='search' AND exclusion IS NULL AND EXISTS
      (SELECT 1 FROM session_audit s WHERE s.user_id=events.user_id AND s.session_id=events.session_id);
    CREATE INDEX event_split ON events(scene,split,exclusion,timestamp_ms);
    CREATE INDEX event_user_time ON events(user_id,timestamp_ms);
    CREATE VIEW clean_events AS SELECT e.*,CASE WHEN i.item_id IS NULL THEN 1 ELSE 0 END AS missing_item_features
      FROM events e LEFT JOIN items i ON i.item_id=e.item_id WHERE e.exclusion IS NULL;
    CREATE TABLE train_users AS SELECT DISTINCT user_id FROM events WHERE split='train' AND exclusion IS NULL;
    CREATE UNIQUE INDEX train_user_key ON train_users(user_id);
    CREATE TABLE train_items AS SELECT DISTINCT item_id FROM events WHERE split='train' AND exclusion IS NULL;
    CREATE UNIQUE INDEX train_item_key ON train_items(item_id);
    ''')
    for scene in ('rec','search'):
        for split in ('train','valid','test'):
            db.execute(f"CREATE VIEW {scene}_{split} AS SELECT * FROM clean_events WHERE scene='{scene}' AND split='{split}'")
    def query(sql):
        cur=db.execute(sql)
        return [dict(zip([d[0] for d in cur.description],r)) for r in cur]
    report['split_counts']=query('SELECT scene,split,count(*) AS rows,sum(label) AS positives,min(timestamp_ms) AS min_ms,max(timestamp_ms) AS max_ms,sum(missing_item_features) AS missing_item_features FROM clean_events GROUP BY scene,split')
    report['exclusions']=query('SELECT scene,exclusion,count(*) AS rows FROM events WHERE exclusion IS NOT NULL GROUP BY scene,exclusion')
    report['ambiguous_keys']=query('SELECT scene,count(*) AS groups,sum(min_label<>max_label) AS conflicting_label_groups FROM ambiguous_keys GROUP BY scene')
    report['sessions']=query('SELECT * FROM session_audit')
    report['cold_start']=query("SELECT e.scene,e.split,count(*) AS rows,sum(u.user_id IS NULL) AS new_user_rows,sum(i.item_id IS NULL) AS new_item_rows FROM clean_events e LEFT JOIN train_users u ON u.user_id=e.user_id LEFT JOIN train_items i ON i.item_id=e.item_id WHERE e.split<>'train' GROUP BY e.scene,e.split")
    report['checks']={
      'foreign_user_errors':db.execute('SELECT count(*) FROM events e LEFT JOIN users u ON u.user_id=e.user_id WHERE u.user_id IS NULL').fetchone()[0],
      'cross_split_sessions':db.execute("SELECT count(*) FROM (SELECT user_id,session_id FROM clean_events WHERE scene='search' GROUP BY user_id,session_id HAVING count(DISTINCT split)>1)").fetchone()[0],
      'sqlite_integrity':db.execute('PRAGMA integrity_check').fetchone()[0]}
    assert report['checks']=={'foreign_user_errors':0,'cross_split_sessions':0,'sqlite_integrity':'ok'},report['checks']
    assert sum(x['rows'] for x in report['split_counts'])+sum(x['rows'] for x in report['exclusions'])==sum(report['raw_counts'].values())
    report['status']='ready'
    report['history_contract']='Use only events.timestamp_ms < target timestamp. Never feed current click_count or current search_source feedback into rec predictors. Train vocab/statistics only on train. No histories materialized yet.'
    db.commit();db.execute('PRAGMA wal_checkpoint(TRUNCATE)');db.close()
    (ROOT/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True,indent=2),flush=True)


if __name__=='__main__':
    main()
