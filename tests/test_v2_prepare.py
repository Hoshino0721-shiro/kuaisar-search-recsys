import sqlite3
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from src.v2 import prepare_data


class AuditRecoveryTests(unittest.TestCase):
    def test_index_preserves_exclusions_and_removes_inner_table_scan(self):
        db = sqlite3.connect(':memory:')
        db.executescript('''
          CREATE TABLE events(id INTEGER PRIMARY KEY,scene TEXT,user_id INTEGER,
            session_id TEXT,item_id INTEGER,timestamp_ms INTEGER,exclusion TEXT);
          CREATE TABLE ambiguous_keys(scene TEXT,user_id INTEGER,session_id TEXT,
            item_id INTEGER,timestamp_ms INTEGER);
          INSERT INTO events VALUES(1,'search',1,'s',10,100,NULL);
          INSERT INTO events VALUES(2,'search',1,'s',10,100,NULL);
          INSERT INTO events VALUES(3,'search',1,'s',20,100,NULL);
          INSERT INTO events VALUES(4,'search',1,'s',10,100,'duplicate_source_record');
          INSERT INTO ambiguous_keys VALUES('search',1,'s',10,100);
        ''')
        sql = """UPDATE events SET exclusion='ambiguous_event_key'
            WHERE exclusion IS NULL AND EXISTS
            (SELECT 1 FROM ambiguous_keys a WHERE a.scene=events.scene
             AND a.user_id=events.user_id AND a.session_id=events.session_id
             AND a.item_id=events.item_id AND a.timestamp_ms=events.timestamp_ms)"""
        db.execute(sql)
        reference = db.execute('SELECT id,exclusion FROM events ORDER BY id').fetchall()
        db.execute("UPDATE events SET exclusion=NULL WHERE exclusion='ambiguous_event_key'")
        db.executescript(prepare_data.INDEX_SQL)
        plan = str(db.execute('EXPLAIN QUERY PLAN ' + sql).fetchall())
        self.assertIn('ambiguous_key_lookup', plan)
        db.execute(sql)
        self.assertEqual(reference, db.execute('SELECT id,exclusion FROM events ORDER BY id').fetchall())
        self.assertIsNone(reference[2][1])
        self.assertEqual(reference[3][1], 'duplicate_source_record')
        db.close()

    def test_recovery_never_opens_completed_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'manifest.json').write_text('{}')
            with patch.object(prepare_data.legacy, 'ROOT', root), patch.object(prepare_data, 'md5') as checksum:
                with self.assertRaises(FileExistsError):
                    prepare_data.resume_audit()
                checksum.assert_not_called()


if __name__ == '__main__':
    unittest.main()
