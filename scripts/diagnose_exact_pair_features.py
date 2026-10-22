"""Descriptive feature distinguishability of actual paired exposures."""
import argparse
from pathlib import Path

import numpy as np

from src.ctr_features import FEATURES
from src.v2.ranking_convergence import write_json
from src.v2.temporal_route_features import ROUTE
from src.v2.train_rankers import sha


def diagnose(reference, prepared, pairs, output):
    if output.exists():
        raise FileExistsError('Preserve old diagnostics')
    columns = list(range(20))+list(range(28, 34))
    result = {}
    for split, path in [('train', prepared/'full_X.npy'), ('valid', reference/'valid_X.npy')]:
        x = np.load(path, mmap_mode='r')
        p, n = [np.load(pairs/f'{split}_{k}.npy') for k in ('positive', 'negative')]
        identical = 0
        differences = np.zeros(len(columns), dtype=np.int64)
        for lo in range(0, len(p), 100000):
            different = x[p[lo:lo+100000]][:, columns] != x[n[lo:lo+100000]][:, columns]
            identical += int((~different.any(axis=1)).sum())
            differences += different.sum(axis=0)
        result[split] = {'pairs': len(p), 'identical_active_features': identical,
                         'identical_fraction': identical/len(p),
                         'different_pair_counts': {name: int(count) for name, count in zip(FEATURES+ROUTE, differences)}}
        x._mmap.close()
    write_json(output, {'splits': result, 'test_used': False, 'source_sha256': sha(__file__),
               'pair_metadata_sha256': sha(pairs/'metadata.json'),
               'scope': 'exact raw feature equality within observed pairs; not a causal explanation or relevance bound'})
    print('Pair feature diagnostics written', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, default=Path('artifacts/full_exposure_local_20261022/reference'))
    parser.add_argument('--prepared', type=Path, default=Path('artifacts/full_exposure_local_20261022/full_cache'))
    parser.add_argument('--pairs', type=Path, default=Path('artifacts/exact_pair_finetune_20261022/pairs'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.reference, args.prepared, args.pairs, args.out)
