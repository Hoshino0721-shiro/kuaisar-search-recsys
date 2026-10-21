"""Replay training days into memory, never overwrite caches or read test labels."""
import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path

import numpy as np

from src.ctr_features import CACHE, DATA, FEATURES, TZ, FeatureState, connection
from src.v2.train_rankers import Ranker, sha


def replay(db, cache, cutoff):
    saved = FeatureState.load(cache / 'state.npz')
    state = FeatureState(saved.categories.copy(), len(saved.un))
    arrays = {name: np.load(cache / f'train_{name}.npy', mmap_mode='r')
              for name in ('X', 'users', 'items', 'y')}
    first = db.execute("SELECT min(timestamp_ms) FROM clean_events WHERE scene='rec' AND split='train'").fetchone()[0]
    if first is None or first >= cutoff:
        raise ValueError('No training events before cutoff')
    day = datetime.fromtimestamp(first / 1000, TZ).replace(hour=0, minute=0, second=0, microsecond=0)
    cursor = 0
    reports = []
    while int(day.timestamp() * 1000) < cutoff:
        start = int(day.timestamp() * 1000)
        end = min(cutoff, int((day + timedelta(days=1)).timestamp() * 1000))
        rows = np.asarray(db.execute(
            "SELECT user_id,item_id,timestamp_ms,label FROM clean_events "
            "WHERE scene='rec' AND split='train' AND timestamp_ms>=? AND timestamp_ms<? ORDER BY id",
            (start, end)).fetchall(), dtype=np.int64).reshape(-1, 4)
        # Independent SQL check of the replay's input snapshot boundary.
        count, clicks = db.execute(
            "SELECT count(*),coalesce(sum(label),0) FROM clean_events "
            "WHERE scene='rec' AND split='train' AND timestamp_ms<?", (start,)).fetchone()
        prior_ok = bool(state.un.sum() == count and state.up.sum() == clicks)
        boundary_checks = []
        for index in np.linspace(0, max(0, len(rows) - 1), min(5, len(rows)), dtype=int):
            uid, iid = map(int, rows[index, :2])
            checks = []
            for field, value, n, p in [('user_id', uid, state.un[uid], state.up[uid]),
                                        ('item_id', iid, state.inum[iid], state.ip[iid])]:
                actual_n, actual_p = db.execute(
                    f"SELECT count(*),coalesce(sum(label),0) FROM clean_events "
                    f"WHERE scene='rec' AND split='train' AND timestamp_ms<? AND {field}=?",
                    (start, value)).fetchone()
                checks.append(bool(n == actual_n and p == actual_p))
            boundary_checks.append(all(checks))
        mismatch = np.zeros(len(FEATURES), dtype=np.int64)
        max_error = np.zeros(len(FEATURES))
        aligned = {name: True for name in ('users', 'items', 'y')}
        seen_sum = 0.
        search_sum = 0.
        # All feature comparisons finish before today's observations update state.
        for lo in range(0, len(rows), 100000):
            block = rows[lo:lo + 100000]
            sl = slice(cursor + lo, cursor + lo + len(block))
            if cursor + lo + len(block) > len(arrays['y']):
                raise ValueError('Training cache is shorter than database')
            x = state.features(block[:, 0], block[:, 1], block[:, 2])
            difference = np.abs(x - arrays['X'][sl])
            mismatch += np.count_nonzero(difference, axis=0)
            max_error = np.maximum(max_error, np.max(difference, axis=0))
            for name, col in [('users', 0), ('items', 1), ('y', 3)]:
                aligned[name] &= bool(np.array_equal(arrays[name][sl], block[:, col]))
            seen_sum += float(x[:, 10].sum())
            search_sum += float(x[:, 19].sum())
        reports.append({'date_beijing': day.date().isoformat(), 'rows': len(rows),
                        'clicks': int(rows[:, 3].sum()), 'prior_sql_totals_match': prior_ok,
                        'sampled_prior_user_item_sql_checks_pass': all(boundary_checks),
                        'alignment': aligned, 'feature_mismatch_counts': dict(zip(FEATURES, mismatch.tolist())),
                        'max_feature_abs_error': float(max_error.max()),
                        'seen_item_fraction': seen_sum / len(rows) if len(rows) else None,
                        'has_search_fraction': search_sum / len(rows) if len(rows) else None})
        if len(rows):
            state.update(rows[:, 0], rows[:, 1], rows[:, 3])
        search = np.asarray(db.execute(
            "SELECT user_id,item_id FROM clean_events WHERE scene='search' AND split='train' "
            "AND timestamp_ms>=? AND timestamp_ms<? AND label>0 ORDER BY id",
            (start, end)).fetchall(), dtype=np.int64).reshape(-1, 2)
        state.update_search(search[:, 0], search[:, 1])
        cursor += len(rows)
        print(f'Replayed {day.date()}: {len(rows)} rows, max error {max_error.max()}', flush=True)
        day += timedelta(days=1)
    final_state = {name: bool(np.array_equal(value, getattr(saved, name)))
                   for name, value in state.__dict__.items()}
    verified = cursor == len(arrays['y']) and all(final_state.values()) and all(
        r['prior_sql_totals_match'] and r['sampled_prior_user_item_sql_checks_pass']
        and all(r['alignment'].values()) and r['max_feature_abs_error'] == 0 for r in reports)
    return {'verified': verified, 'training_rows': cursor, 'days': reports,
            'final_state_exact_match': final_state,
            'scope': 'replay counters and cache order; shares feature formulas with original code; not a proof of every semantic assumption'}


def normalization_audit(checkpoint, cache):
    model = Ranker(checkpoint)
    indices = np.load(checkpoint.parent / 'sample_indices.npy')
    x = np.load(cache / 'train_X.npy', mmap_mode='r')
    expected = np.sort(np.random.default_rng(model.metadata['seed']).choice(
        len(x), model.metadata['sample_rows'], replace=False))
    selected = np.asarray(x[indices][:, model.columns], dtype=np.float32)
    mean = selected.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(selected.std(axis=0, dtype=np.float64).astype(np.float32), 1e-6)
    return {'sample_indices_match_seed': bool(np.array_equal(indices, expected)),
            'mean_max_abs_error': float(np.max(np.abs(mean - model.mean))),
            'std_max_abs_error': float(np.max(np.abs(std - model.std))),
            'sample_rows': len(indices), 'checkpoint_sha256': sha(checkpoint)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Choose a new audit directory')
    cutoff = json.loads((DATA / 'manifest.json').read_text(encoding='utf-8'))['cutoffs_ms'][0]
    db = connection()
    try:
        report = replay(db, CACHE, cutoff)
    finally:
        db.close()
    report['normalization'] = {
        name: normalization_audit(Path('artifacts/ranking_v3') / name / 'model.pt', CACHE)
        for name in ('coarse_lr', 'fine_dcn')}
    report.update({'test_used': False, 'diagnosis_only': True, 'source_sha256': sha(__file__),
                   'state_sha256': sha(CACHE / 'state.npz'),
                   'feature_schema': FEATURES, 'cutoff_ms': cutoff})
    args.out.mkdir(parents=True)
    (args.out / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'days'}, indent=2))
    if not report['verified'] or any(not r['sample_indices_match_seed'] or r['mean_max_abs_error'] or
                                     r['std_max_abs_error'] for r in report['normalization'].values()):
        raise RuntimeError('Training audit failed; inspect preserved report')


if __name__ == '__main__':
    main()
