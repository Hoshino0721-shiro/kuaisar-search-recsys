from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
import numpy as np
from src.itemcf_index import IndexConfig, build, read_snapshot
from src.v2.recall_allocation import evaluate, load_population


class RecallAllocationTests(unittest.TestCase):
    def test_population_excludes_future_scores_and_preserves_event_multiplicity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = sqlite3.connect(root/'dataset.sqlite')
            db.execute('CREATE TABLE clean_events(user_id,item_id,timestamp_ms,label,scene,split)')
            db.executemany('INSERT INTO clean_events VALUES (?,?,?,?,?,?)', [
                (1,10,90,1,'rec','train'), (1,20,80,1,'rec','train'),
                (1,999,100,1,'rec','train'), (1,888,70,1,'search','train'),
                (1,30,110,1,'rec','valid'), (1,30,120,1,'rec','valid'),
                (2,10,110,0,'rec','valid'), (1,777,210,1,'rec','test')])
            db.commit()
            db.close()
            users, positives, popular, recent = load_population(root,100,20)
            self.assertEqual(users,[1,2])
            self.assertEqual(positives[1],{30:2})
            self.assertEqual(popular,[10,20])
            self.assertEqual(recent,[10,20])

    def test_evaluation_reports_zero_positive_partition_without_using_test_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            db=sqlite3.connect(root/'dataset.sqlite')
            db.execute('CREATE TABLE clean_events(user_id,item_id,timestamp_ms,label,scene,split)')
            db.executemany('INSERT INTO clean_events VALUES (?,?,?,?,?,?)',[
                (1,10,90,1,'rec','train'), (1,20,80,1,'rec','train'),
                (2,20,90,1,'rec','train'), (2,30,80,1,'rec','train'),
                (1,30,110,1,'rec','valid'), (9,10,120,0,'rec','valid'),
                (1,999,210,1,'rec','test')])
            db.commit()
            db.close()
            (root/'manifest.json').write_text(json.dumps({'status':'ready','cutoffs_ms':[100,200]}))
            spec={'route_limits':{'popular':2,'recent':2,'itemcf':2}}
            config={'itemcf_index':str(root/'index'),'budget':2,'seed':2026,
                    'variants':{'recent_only':{**spec,'weights':{'recent':1,'popular':0,'itemcf':0}},
                                'legacy_rrf':{**spec,'weights':{'recent':2,'popular':1,'itemcf':1}}}}
            with redirect_stdout(io.StringIO()):
                build(read_snapshot(root,IndexConfig(neighbors=3,max_user_items=3,seed_items=3)),
                      IndexConfig(neighbors=3,max_user_items=3,seed_items=3),root/'index')
                report=evaluate(config,root,root/'result')
            self.assertFalse(report['test_used'])
            self.assertEqual(report['metrics']['recent_only']['all']['positive_events'],1)
            self.assertIsNone(report['metrics']['recent_only']['no_train_clicks']['event_recall'])
            self.assertTrue(any(v['event_recall_delta'] is None
                                for v in report['selected_contrasts']['recent_only'].values()))
            with np.load(root/'result/paired_hits.npz') as saved:
                self.assertEqual(int(saved['positive_counts'].sum()),1)
            with self.assertRaises(FileExistsError):
                evaluate(config,root,root/'result')


if __name__=='__main__':
    unittest.main()
