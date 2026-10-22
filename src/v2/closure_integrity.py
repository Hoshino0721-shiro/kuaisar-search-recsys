"""Preserve a complete local v1 archive without requiring its private weights."""
import json
from pathlib import Path

from src.itemcf_index import digest


def verify_v1_archive(freeze_file=Path('experiments/final_test/freeze.json')):
    record=json.loads(freeze_file.read_text(encoding='utf-8'))
    assets=record['asset_hashes']
    paths={name:Path(name.replace('\\','/')) for name in assets}
    missing=[name for name in assets if not paths[name].exists()]
    # An independent v2 reproduction need not recreate historical v1 weights or
    # byte-identical caches. Its own data and models have separate manifests.
    checked=0
    for name,expected in assets.items():
        if missing and not name.endswith('.py'): continue
        if digest(paths[name])!=expected: raise ValueError('Frozen v1 archive asset changed: '+name)
        checked+=1
    return {'complete':not missing,'verified_assets':checked,'missing_assets':missing,
            'scope':'all v1 archived assets' if not missing else 'v1 source only; incomplete historical archive, not fully verified'}
