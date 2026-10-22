"""Seed-averaged matched user comparisons; reused validation, exploratory only."""
import argparse
import json
from pathlib import Path

import numpy as np

from src.v2.ranking_convergence import write_json
from src.v2.route_attribution import load_labels
from src.v2.temporal_route_evaluate import paired_intervals
from src.v2.train_rankers import sha


def compare(study, warm, reference, output):
    if output.exists():
        raise FileExistsError('Preserve old comparison')
    report = json.loads((study/'metrics.json').read_text(encoding='utf-8'))
    if report['status'] != 'complete' or report['test_used']:
        raise ValueError('Complete validation study required')
    cfg = report['protocol']['config']
    users = np.load(reference/'pool_users.npy')
    items = np.load(reference/'pool_items.npy', mmap_mode='r')
    _, positive, _, _ = load_labels(users, items)
    means = {}
    for key in cfg['objectives']+['warm']:
        arrays = []
        for seed in cfg['seeds']:
            path = warm/f'full_R_s{seed}_per_user.npz' if key == 'warm' else study/f'{key}_s{seed}_per_user.npz'
            with np.load(path) as data:
                arrays.append([data[k] for k in ('direct_hits', 'direct_ndcg', 'cascade_hits', 'cascade_ndcg')])
        means[key] = [np.mean([a[j] for a in arrays], axis=0) for j in range(4)]
    comparisons = {}
    for a, b in [('bce', 'warm'), ('endpoint_aux', 'warm'), ('pair_aux', 'warm'),
                 ('pair_aux', 'bce'), ('pair_aux', 'endpoint_aux')]:
        x, y = means[a], means[b]
        comparisons[a+'-'+b] = {
            'direct': paired_intervals(x[1]-y[1], x[0]-y[0], positive),
            'cascade_hit_difference_mean': float((x[2]-y[2]).sum()),
            'direct_hit_difference_mean': float((x[0]-y[0]).sum()),
            'cascade_significance_not_tested': True}
    result = {'comparisons': comparisons, 'aggregate': report['aggregate'],
              'warm_aggregate': report['warm_aggregate'], 'study_report_sha256': sha(study/'metrics.json'),
              'warm_report_sha256': sha(warm/'metrics.json'), 'source_sha256': sha(__file__),
              'test_used': False, 'default_changed': False,
              'scope': 'exploratory paired user bootstrap after averaging three seeds; reused validation, no multiplicity correction or independent temporal test'}
    write_json(output, result)
    items._mmap.close()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, default=Path('artifacts/exact_pair_finetune_20261022/study'))
    parser.add_argument('--warm', type=Path, default=Path('artifacts/ranking_convergence_20261022'))
    parser.add_argument('--reference', type=Path, default=Path('artifacts/full_exposure_local_20261022/reference'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    compare(args.study, args.warm, args.reference, args.out)
