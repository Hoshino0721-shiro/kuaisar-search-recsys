"""Independently recount saved retrieval candidates and check checkpoint round-trip."""
import hashlib
import argparse
import json
import sqlite3
from collections import Counter,defaultdict
from pathlib import Path
import torch
from src.two_tower import TwoTower
from src.history_tower import HistoryTower


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',default='r2_two_tower_valid')
    args=parser.parse_args()
    out=Path('experiments')/args.run
    report=json.loads((out/'metrics.json').read_text())
    checksum=hashlib.sha256()
    with (out/'model.pt').open('rb') as h:
        for block in iter(lambda:h.read(1024*1024),b''):checksum.update(block)
    assert checksum.hexdigest()==report['checkpoint_sha256']
    # File is our own locally generated, checksum-verified checkpoint.
    saved=torch.load(out/'model.pt',map_location='cpu',weights_only=False)
    known=set(saved['user_ids']);catalog=set(saved['item_ids'])
    contextual='final_histories' in saved
    if contextual:
        model=HistoryTower(len(known),len(catalog),saved['state_dict']['categories'],saved['state_dict']['types'],use_attributes=saved['config'].get('use_attributes',True))
    else:
        model=TwoTower(len(known),len(catalog))
    model.load_state_dict(saved['state_dict'],strict=True)
    model.eval()
    with torch.inference_mode():
        if contextual:
            assert torch.isfinite(model.users(torch.arange(10),saved['final_histories'][:10])).all()
        else:
            assert torch.isfinite(model.users(torch.arange(10))).all()
        assert torch.isfinite(model.items(torch.arange(10))).all()
    db=sqlite3.connect(Path('data/processed/kuaisar_small_v1/dataset.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    positives=defaultdict(Counter);users=set()
    for u,i,y in db.execute("SELECT user_id,item_id,label FROM clean_events WHERE scene='rec' AND split='valid'"):
        users.add(u)
        if y:positives[u][i]+=1
    popular=json.loads(Path('experiments/r0_p0_valid/model_summary.json').read_text())['popular_top100']
    groups=defaultdict(Counter);unique=set();hits=Counter()
    macro=defaultdict(list)
    for line in (out/'candidates.jsonl').open():
        row=json.loads(line);u=row['user_id'];ranked=row['items']
        assert u not in unique and u in users
        unique.add(u)
        assert len(ranked)==len(set(ranked))==100 and set(ranked)<=catalog
        group='known_train_click_user' if u in known else 'no_train_click_user'
        if u not in known:assert ranked==popular
        g=groups[group];g['users']+=1;g['positive_events']+=sum(positives[u].values())
        for k in (50,100):
            selected=set(ranked[:k]);popset=set(popular[:k])
            h=sum(n for i,n in positives[u].items() if i in selected)
            g[f'hits_at_{k}']+=h;hits[k]+=h
            g[f'popular_hits_at_{k}']+=sum(n for i,n in positives[u].items() if i in popset)
            if positives[u]:macro[k].append(len(set(positives[u])&selected)/len(positives[u]))
    assert unique==users
    for k in (50,100):assert hits[k]==report['metrics'][str(k)]['hits']
    for g in groups.values():
        for k in (50,100):
            g[f'recall_at_{k}']=g[f'hits_at_{k}']/g['positive_events']
            g[f'popular_recall_at_{k}']=g[f'popular_hits_at_{k}']/g['positive_events']
    result={'checks':'passed','checkpoint_sha256':checksum.hexdigest(),'unique_users':len(unique),
            'groups':dict(groups),'macro_user_period_recall':{k:sum(v)/len(v) for k,v in macro.items()},
            'macro_definition':'mean over users with validation clicks of distinct relevant-item Recall in the whole validation period, not request-level Recall'}
    (out/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    db.close()


if __name__=='__main__':main()
