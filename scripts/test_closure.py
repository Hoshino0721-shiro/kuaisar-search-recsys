"""Full fixture/unit regression with a machine-readable release prerequisite."""
import argparse
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.itemcf_index import digest
from src.v2.ranking_convergence import write_json


def run(output):
    temporary=Path('artifacts/environment_tmp').resolve()
    temporary.mkdir(parents=True,exist_ok=True)
    tempfile.tempdir=str(temporary)
    suite=unittest.defaultTestLoader.discover('tests')
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    report={'status':'passed' if result.wasSuccessful() else 'failed','tests_run':result.testsRun,
            'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
            'source_sha256':digest(Path(__file__)),
            'implementation_sources':{p.as_posix():digest(p) for folder in ('src','scripts')
                                      for p in sorted(Path(folder).rglob('*.py'))},
            'test_sources':{p.as_posix():digest(p) for p in sorted(Path('tests').glob('test_*.py'))},
            'scope':'unit/synthetic fixture tests only; no real-data training or final-test label evaluation'}
    output.parent.mkdir(parents=True,exist_ok=True)
    write_json(output,report)
    return result.wasSuccessful()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('artifacts/tower_closure_20261022/tests.json'))
    raise SystemExit(not run(p.parse_args().out))
