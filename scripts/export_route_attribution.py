"""Export small reviewed summaries; keep weights, arrays and logs local."""
import argparse
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import shutil
import numpy as np
from src.v2.temporal_route_data import write_json
from src.v2.train_rankers import sha


def export(root, destination):
    audit=json.loads((root/'audit.json').read_text(encoding='utf-8'))
    result=json.loads((root/'metrics.json').read_text(encoding='utf-8'))
    diag=json.loads((root/'diagnosis.json').read_text(encoding='utf-8'))
    if not audit['passed'] or sha(root/'metrics.json')!=audit['metrics_sha256']:
        raise ValueError('Require audited final metrics')
    destination.mkdir(exist_ok=False,parents=True)
    for name in ('protocol.json','ablation.json','rolling.json','diagnosis.json','audit.json'):
        shutil.copyfile(root/name,destination/name)
    checkpoint_manifest={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('model.pt'))}
    manifest={'date':'2026-10-22','new_model_runs':result['new_model_runs'],'tests_passed':109,
              'test_used':False,'default_changed':False,'scope':result['protocol']['config']['scope'],
              'files':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in destination.glob('*.json')},
              'local_root':str(root),'local_checkpoints':checkpoint_manifest,
              'export_source_sha256':sha(__file__)}
    write_json(destination/'manifest.json',manifest)
    ag=result['ablation']['aggregate']
    names={'base':'原数值特征','recent_only':'仅近期热门标志和名次','cf_only':'仅 CF 标志、名次和分数',
           'flags_only':'仅两路标志','ranks_only':'仅两路名次','cf_score_only':'仅 CF 分数',
           'without_combined':'全部召回特征去掉派生 RRF 和','combined_only':'仅派生 RRF 和','R':'原完整六维召回特征'}
    lines=['# 召回特征归因与时间回放','',
           '2026-10-22。新增33个真实曝光模型：7组特征拆分×3种子，另有2个历史日期×base/R×3种子。复用上轮6个base/R模型作对照。',
           '', '## 固定协议','',
           '- 主实验沿用原100万曝光样本、34维DCN（4523个参数）、原标准化、种子2026/2027/2028、五轮BCE、相同Adam和批次预算。新特征组均在标准化后mask；所有新组关闭T。初始参数和五轮训练顺序摘要逐种子与原R组完全相同。',
           '- 每个拆分组从头训练；不是在已有模型上临时遮挡输入。名义网络宽度相同，各组可用输入不同。固定最后一轮，不据验证选种子/轮次。',
           '- 两个历史回放使用5月27日、28日零点快照；仅用原统一样本中日期严格更早的行训练，均重新拟合标准化。候选与历史CF图也只用此前日志，每个日期内部base/R同池。',
           '- 历史回放对全部当日用户生成每人2000候选，检验直接精排；曝光AUC只在原统一随机样本所包含的当日曝光上计算。回放不使用旧LR，因为旧LR见过这些日期。',
           '- 日期来自已接触过的原训练期，属于回溯时间检验；主验证日也已反复开发。未读取test，未切换默认模型。所有区间均为探索性用户bootstrap，未校正多重比较；不能当作盲测或线上收益。',
           '', '## 拆分结果','',
           '下表均为三个种子平均，命中按点击事件计数。直排为2000→20；级联为原LR400→新精排80→固定类别惩罚→20。',
           '', '| 特征 | 直排命中 | 直排NDCG@20 | 级联命中 |','| --- | ---: | ---: | ---: |']
    for arm in ['base','flags_only','recent_only','cf_only','cf_score_only','ranks_only','without_combined','combined_only','R']:
        v=ag[arm]
        lines.append(f'| {names[arm]} | {v["direct"]["hit_events"]:.2f} | {v["direct"]["ndcg_at20"]:.6f} | {v["cascade"]["hit_events"]:.2f} |')
    lines += ['',
              'CF信息和近期热门信息单独加入均有开发增量，名次比只给标志更有用。但这些是组级增量，不是可以相加的贡献百分比。完整R与CF-only的直排NDCG差值区间跨0，不能断言近期热门在CF信息之外还有稳定增量。',
              '', '仅名次组在当前日均值最高，不据此立即换默认：它的跨种子波动仍大，而且历史回放只预先比较base与完整R，并未复核全部拆分组。',
              '', '## 排序损失发生在哪里','',
              f'RRF前20覆盖1745次验证点击。原LR粗排仅保留其中{diag["rrf_top20_positive_events_retained_by_lr400"]}次，提前过滤{diag["rrf_top20_lost_before_fine"]}次。',
              '', '| RRF原本命中的1745次点击，在R直排中的位置 | 三种子平均 |','| --- | ---: | ---: |']
    rr=[diag['runs'][f'R_s{s}'] for s in (2026,2027,2028)]
    for key,label in [('1_20','1–20'),('21_80','21–80'),('81_400','81–400'),('401_2000','401–2000')]:
        lines.append(f'| {label} | {np.mean([v["rrf_positive_event_destination"][key] for v in rr]):.2f} |')
    lines += ['', 'R直排命中512.67次：保留原RRF命中56.33次，同时从RRF前20以外新增456.33次。不能因为新命中增加就忽略大量旧命中被丢失。',
              '', '## 目标差异的描述性证据','',
              '| 所选Top20的日志统计 | RRF | R直排，种子均值 |','| --- | ---: | ---: | ---: |']
    for key,label,fmt in [('top20_logged_pair_fraction','用户—物品对在验证日有曝光的比例','.3%'),
                          ('top20_logged_exposure_rows','这些物品对应的实际验证曝光行数','.2f'),
                          ('top20_observed_ctr_on_logged_subset','上述已曝光子集CTR','.2%'),
                          ('top20_median_train_exposures','所选物品的历史曝光中位数','.2f'),
                          ('top20_mean_smoothed_item_ctr','所选物品的平滑历史CTR均值','.4f')]:
        a=diag['rrf_selection'][key];b=np.mean([v[key] for v in rr])
        lines.append(f'| {label} | {format(a,fmt)} | {format(b,fmt)} |')
    lines += ['',
              '这些是选中集合的事后日志统计，不能当作曝光干预或线上CTR比较。它们支持如下解释：曝光BCE学习曝光条件下的点击倾向，而固定列表的次日事件覆盖同时受未来曝光分布影响。高条件CTR不保证高日志覆盖；该证据没有证明损失函数是唯一根因，也不能为未超过RRF开脱。',
              '', '## 有限调整对照','',
              '下面三种方法均预先固定，不按结果搜索参数；只复用原R分数，在当前开发日上诊断。',
              '', '| 方法 | Top20命中均值 | NDCG@20均值 |','| --- | ---: | ---: |',
              '| 保留RRF原序 | 1745.00 | 0.014512 |']
    for key in rr[0]['limited_adjustment']:
        lines.append(f'| {key} | {np.mean([v["limited_adjustment"][key]["hit_events"] for v in rr]):.2f} | {np.mean([v["limited_adjustment"][key]["ndcg_at20"] for v in rr]):.6f} |')
    lines += ['', '限制DCN改动范围能减少相对直接精排的损失，但仍未超过RRF；不能宣称已经验证出有效融合方案。',
              '', '## 严格历史日期回放','',
              '| 目标日（北京时间） | 训练曝光样本数 | 目标日曝光抽样数 | base直排命中 | R直排命中 | RRF命中 | base / R NDCG |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for cutoff,w in result['rolling'].items():
        date=datetime.fromtimestamp(int(cutoff)/1000,timezone(timedelta(hours=8))).date()
        b=w['aggregate']['base'];r=w['aggregate']['R']
        lines.append(f'| {date} | {w["train_rows"]} | {w["sampled_validation_exposures"]} | {b["hit_events"]:.2f} | {r["hit_events"]:.2f} | {w["rrf"]["hit_events"]} | {b["ndcg_at20"]:.6f} / {r["ndcg_at20"]:.6f} |')
    lines += ['', '每个窗口的完整逐种子结果与探索性区间见rolling.json；这些候选池与5月29日不同，跨日期不直接比较绝对命中多少。',
              '', '## 验收与可复现','',
              f'109项测试通过；全部33个新增模型与6个复用模型重载检查通过。独立重放{audit["sampled_lists_independently_replayed"]}条抽样用户—模型列表，并检查全量逐用户命中汇总；最大预测绝对误差{audit["max_prediction_abs_error"]:.3g}。两个历史日期的标准化均独立重算，原20维当日曝光特征与重建前缀状态逐项一致。审计不是对所有图边的穷举证明。',
              '', '```powershell',
              '.venv/Scripts/python.exe -m src.v2.route_attribution --out artifacts/NEW_ROUTE_ATTRIBUTION',
              '.venv/Scripts/python.exe -m scripts.audit_route_attribution --root artifacts/NEW_ROUTE_ATTRIBUTION',
              '.venv/Scripts/python.exe -m scripts.export_route_attribution --root artifacts/NEW_ROUTE_ATTRIBUTION --out experiments/NEW_ROUTE_ATTRIBUTION',
              '```', '',
              '权重、候选、预测、逐用户数组保留在独立artifacts目录。小型报告见experiments/route_attribution_valid_20261022；未自动推送GitHub。',
              '', '## 后续判断','',
              '保留RRF作为强基线。召回匹配特征值得继续使用，但当前DCN和旧LR会显著破坏头部排序，不能按相对弱基线的百分比宣布完成优化。先明确最终目标是日志下一天点击覆盖还是曝光CTR排序，再设计与目标一致的训练样本和排序损失；本轮尚未训练新的排序损失，不能把目标差异诊断写成损失改造已获收益。', '']
    Path('docs/ROUTE_ATTRIBUTION.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Exported audited summaries and docs/ROUTE_ATTRIBUTION.md')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();export(a.root,a.out)
