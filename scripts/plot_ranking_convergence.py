"""Dependency-free scientific curve export from recorded epoch metrics."""
import argparse
import html
import json
from pathlib import Path


def chart(history, field, title):
    def value(record):
        result = record
        for key in field.split('.'):
            result = result[key]
        return float(result)
    values = [value(h) for h in history]
    low, high = min(values), max(values)
    pad = max((high-low)*.1, 1e-6)
    low -= pad
    high += pad
    def point(i, v):
        return 65+i*485/max(1, len(values)-1), 170-(v-low)*125/(high-low)
    points = ' '.join(f'{x:.2f},{y:.2f}' for x, y in (point(i, v) for i, v in enumerate(values)))
    circles = ''.join(f'<circle cx="{point(i,v)[0]:.2f}" cy="{point(i,v)[1]:.2f}" r="3">'
                      f'<title>epoch {i+1}: {v:.8f}</title></circle>' for i, v in enumerate(values))
    return (f'<svg viewBox="0 0 580 205" role="img" aria-label="{html.escape(title)}">'
            '<rect width="580" height="205" fill="#fff"/>'
            f'<text x="65" y="23" font-size="16">{html.escape(title)}</text>'
            '<path d="M65 40V175H555" fill="none" stroke="#999"/>'
            f'<text x="0" y="50" font-size="12">{high:.5f}</text>'
            f'<text x="0" y="174" font-size="12">{low:.5f}</text>'
            f'<text x="65" y="196" font-size="12">epoch 1</text>'
            f'<text x="510" y="196" font-size="12">{len(values)}</text>'
            f'<polyline points="{points}" fill="none" stroke="#2463a0" stroke-width="2"/>'
            f'<g fill="#2463a0">{circles}</g></svg>')


def render(study, output):
    if output.exists():
        raise FileExistsError('Preserve old curves')
    report = json.loads((study/'metrics.json').read_text(encoding='utf-8'))
    parts = ['<!doctype html><meta charset="utf-8"><title>Pointwise convergence</title>',
             '<style>body{font:15px system-ui;max-width:1240px;margin:32px auto;padding:0 20px;'
             'background:#f5f6f8;color:#18222f}.plots{display:grid;grid-template-columns:1fr 1fr;gap:12px}'
             'svg{width:100%;border:1px solid #ddd;border-radius:6px}table{border-collapse:collapse;'
             'width:100%;background:white}th,td{padding:7px;border:1px solid #ddd;text-align:right}'
             '.selected{background:#e3f2df}h2{margin-top:36px}</style>',
             '<h1>Pointwise BCE: epoch-wise convergence</h1>',
             '<p>Best epoch: minimum full validation LogLoss. Fixed training probe: 65,536 exposure rows. '
             'All curves use the reused development day, not an independent final test. '
             'Candidate NDCG uses the same fixed 256-user diagnostic subset, not model selection. '
             'Panels have independent y-axis ranges. Hover points for values.</p>']
    for name, record in report['runs'].items():
        train = record['training']
        h = train['history']
        best = train['selected_epoch']
        parts.append(f'<h2>{html.escape(name)} — best {best}, stopped {len(h)}</h2><div class="plots">')
        for field, title in [('valid.logloss', 'Validation LogLoss (lower is better)'),
                             ('train_probe_logloss', 'Fixed training probe LogLoss'),
                             ('valid.gauc', 'Validation user GAUC'),
                             ('candidate_diagnostic.ndcg_at20', '256-user candidate NDCG@20')]:
            parts.append(chart(h, field, title))
        parts.append('</div><details><summary>All epoch measurements</summary><table><tr>'
                     '<th>Epoch</th><th>LR</th><th>Train probe LL</th><th>Valid LL</th>'
                     '<th>AUC</th><th>GAUC</th><th>Timestamp NDCG</th><th>Candidate NDCG</th></tr>')
        for row in h:
            valid = row['valid']
            cells = [str(row['epoch']), f'{row["learning_rate"]:.7f}',
                     f'{row["train_probe_logloss"]:.7f}', f'{valid["logloss"]:.7f}',
                     f'{valid["auc"]:.7f}', f'{valid["gauc"]:.7f}',
                     f'{valid["same_timestamp"]["ndcg_at20"]:.7f}',
                     f'{row["candidate_diagnostic"]["ndcg_at20"]:.7f}']
            parts.append(('<tr class="selected">' if row['epoch'] == best else '<tr>')+
                         ''.join(f'<td>{c}</td>' for c in cells)+'</tr>')
        parts.append('</table></details>')
    output.write_text('\n'.join(parts), encoding='utf-8', newline='\n')
    print('Curves written to '+str(output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, default=Path('artifacts/ranking_convergence_20261022'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    render(args.study, args.out)
