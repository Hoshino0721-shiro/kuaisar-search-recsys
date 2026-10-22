"""Static multi-arm scientific curves with no external plotting dependency."""
import argparse
import html
import json
from pathlib import Path


COLORS = {'bce': '#2463a0', 'endpoint_aux': '#b86a11', 'pair_aux': '#26844b'}


def chart(groups, field, title):
    def value(row):
        result = row
        for key in field.split('.'):
            result = result[key]
        return float(result)
    values = [value(row) for rows in groups.values() for row in rows]
    low, high = min(values), max(values)
    pad = max((high-low)*.1, 1e-6)
    low -= pad
    high += pad
    last_epoch = max(rows[-1]['epoch'] for rows in groups.values())
    def point(epoch, v):
        return 70+epoch*480/max(1, last_epoch), 195-(v-low)*125/(high-low)
    pieces = [f'<svg viewBox="0 0 580 230" role="img" aria-label="{html.escape(title)}">',
              '<rect width="580" height="230" fill="white"/>',
              f'<text x="70" y="24" font-size="16">{html.escape(title)}</text>',
              '<path d="M70 65V200H555" fill="none" stroke="#999"/>',
              f'<text x="0" y="73" font-size="12">{high:.5f}</text>',
              f'<text x="0" y="199" font-size="12">{low:.5f}</text>',
              '<text x="70" y="222" font-size="12">epoch 0 (warm start)</text>',
              f'<text x="515" y="222" font-size="12">{last_epoch}</text>']
    for j, (key, rows) in enumerate(groups.items()):
        points = [point(row['epoch'], value(row)) for row in rows]
        coordinates = ' '.join(f'{x:.2f},{y:.2f}' for x, y in points)
        color = COLORS[key]
        pieces.extend([f'<text x="{70+j*165}" y="49" font-size="12" fill="{color}">{key}</text>',
                       f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2"/>'])
        for row, (x, y) in zip(rows, points):
            pieces.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.7" fill="{color}">'
                          f'<title>{key} epoch {row["epoch"]}: {value(row):.8f}; '
                          f'guardrail={row["guardrail_pass"]}</title></circle>')
    return ''.join(pieces)+'</svg>'


def render(study, output):
    if output.exists():
        raise FileExistsError('Preserve old curves')
    report = json.loads((study/'metrics.json').read_text(encoding='utf-8'))
    cfg = report['protocol']['config']
    parts = ['<!doctype html><meta charset="utf-8"><title>Exact-time pair fine-tuning</title>',
             '<style>body{font:15px system-ui;max-width:1250px;margin:30px auto;padding:0 20px;'
             'color:#18222f;background:#f5f6f8}.plots{display:grid;grid-template-columns:1fr 1fr;gap:12px}'
             'svg{width:100%;border:1px solid #ddd;border-radius:6px}table{border-collapse:collapse;'
             'width:100%;background:white}th,td{padding:7px;border:1px solid #ddd;text-align:right}'
             '.selected{background:#e3f2df}h2{margin-top:36px}</style>',
             '<h1>Exact timestamp pair fine-tuning: matched controls</h1>',
             '<p>All three arms in a seed share updates, full-exposure order, pair draws and learning rate. '
             'Selection: minimum validation pair LogLoss within exposure LogLoss/AUC guardrails; '
             'warm start (epoch 0) eligible. Candidate NDCG is diagnostic only (fixed 256 users). '
             'Same timestamp is not a verified request. Reused development day; not an online or blind test.</p>']
    for seed in cfg['seeds']:
        group = {key: report['runs'][f'{key}_s{seed}']['training'] for key in cfg['objectives']}
        histories = {key: value['history'] for key, value in group.items()}
        parts.append(f'<h2>Seed {seed}</h2><div class="plots">')
        for field, title in [('valid_pair.logloss', 'Validation pair LogLoss (lower is better)'),
                             ('valid.logloss', 'Validation exposure LogLoss'),
                             ('valid.gauc', 'Validation user GAUC'),
                             ('candidate_diagnostic.ndcg_at20', '256-user candidate NDCG@20')]:
            parts.append(chart(histories, field, title))
        parts.append('</div>')
        for key, train in group.items():
            best = train['selected_epoch']
            parts.append(f'<details><summary>{key}: selected {best}, stopped {train["epochs_completed"]}</summary>'
                         '<table><tr><th>Epoch</th><th>LR</th><th>Pair LL</th><th>Exposure LL</th>'
                         '<th>AUC</th><th>GAUC</th><th>Guardrail</th><th>Candidate NDCG</th></tr>')
            for row in train['history']:
                values = [str(row['epoch']), f'{row["learning_rate"]:.8f}',
                          f'{row["valid_pair"]["logloss"]:.7f}', f'{row["valid"]["logloss"]:.7f}',
                          f'{row["valid"]["auc"]:.7f}', f'{row["valid"]["gauc"]:.7f}',
                          str(row['guardrail_pass']), f'{row["candidate_diagnostic"]["ndcg_at20"]:.7f}']
                parts.append(('<tr class="selected">' if row['epoch'] == best else '<tr>')+
                             ''.join(f'<td>{v}</td>' for v in values)+'</tr>')
            parts.append('</table></details>')
    output.write_text('\n'.join(parts), encoding='utf-8', newline='\n')
    print('Fine-tuning curves written: '+str(output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, default=Path('artifacts/exact_pair_finetune_20261022/study'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    render(args.study, args.out)
