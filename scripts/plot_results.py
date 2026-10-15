"""Render final frozen metrics without training or changing configuration."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parents[1]
report=json.loads((root/'experiments/final_test/metrics.json').read_text())
ann=json.loads((root/'experiments/ann_valid/metrics.json').read_text())
fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
colors=['#94a3b8','#4c78a8','#159b8a']
ctr=report['ctr_exposure_metrics'];names=['constant','item_ctr','mlp_gate']
ax=axes[0,0];bars=ax.bar(['Constant','Item CTR','Gated MLP'],[ctr[n]['auc'] for n in names],color=colors)
ax.set(title='Recommendation exposure AUC (test)',ylim=(0,1),ylabel='AUC')
for bar in bars:ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.015,f'{bar.get_height():.4f}',ha='center')
ax=axes[0,1];ax.plot([0,1],[0,1],ls='--',color='#94a3b8',label='Ideal')
for name,color in [('item_ctr',colors[1]),('mlp_gate',colors[2])]:
    curve=[b for b in ctr[name]['reliability'] if b['rows']]
    ax.plot([b['predicted'] for b in curve],[b['observed'] for b in curve],marker='o',color=color,label=name)
ax.set(title='Probability reliability (10 bins, test)',xlabel='Mean predicted probability',ylabel='Observed click fraction',xlim=(0,1),ylim=(0,1));ax.legend(frameon=False)
ax=axes[1,0];records=report['recommendation_list_metrics'];keys=['popular','selected_retrieval','selected_ctr_pipeline']
bars=ax.bar(['Global popular','Recent popular','CTR pipeline'],[records[n]['recall_at_100']*100 for n in keys],color=colors)
ax.set(title='User-period positive-event Recall@100 (test)',ylabel='Recall (%)')
for bar in bars:ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.015,f'{bar.get_height():.3f}%',ha='center')
ax.set_ylim(0,max(bar.get_height() for bar in bars)*1.2)
ax=axes[1,1];points=ann['variants']
ax.plot([p['p95_ms'] for p in points],[p['ann_recall_at_100_vs_exact'] for p in points],marker='o',color=colors[1],label='IVF-flat')
for p in points:ax.annotate(str(p['nprobe']),(p['p95_ms'],p['ann_recall_at_100_vs_exact']),xytext=(4,-12),textcoords='offset points',fontsize=9)
ax.scatter([ann['exact_p95_ms']],[1],marker='*',s=120,color=colors[2],label='Exact GPU')
ax.axhline(.95,color='#94a3b8',ls='--',lw=1)
ax.set(title='ANN vs exact (256 validation queries)',xlabel='Synchronized P95 latency (ms)',ylabel='ANN overlap Recall@100',ylim=(.45,1.06));ax.legend(frameon=False,loc='lower right')
for ax in axes.ravel():
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('KuaiSAR-small: frozen offline results',fontsize=16)
out=root/'docs/figures';out.mkdir(exist_ok=True)
fig.savefig(out/'final_metrics.png',dpi=160)
plt.close(fig)
print(out/'final_metrics.png')
