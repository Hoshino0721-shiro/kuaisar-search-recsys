# 实验记录

目前默认推荐为近期热门＋ItemCF 的 RRF 前20。属性/历史双塔、DCN 和 pairwise 已完成对照，保留为实验分支。汇总结论见 [实验总结](PROJECT_FINAL_V2.md)。

## 各轮对照

| 实验 | 主要设置 | 观察结果 | 报告 |
| --- | --- | --- | --- |
| ItemCF 索引 | 分块共现、TopK邻居、共同用户支持度 | 支持度至少2时，纯CF Recall@1000由3.4665%增至4.0107% | [召回消融](RECALL_ALLOCATION.md) |
| 召回融合 | 近期热门2000、CF1000，融合保留2000 | Recall@2000为9.7700%，近期热门对照9.4229% | [召回消融](RECALL_ALLOCATION.md) |
| 属性与ID | 固定100万曝光，24个模型 | 属性改善曝光预测；身份和历史池化未稳定改善列表 | [特征结果](DCN_FEATURE_RESULTS.md) |
| 初步pairwise | 真实曝光配对，30个模型 | 存在采样和校准影响，未替换默认方案 | [损失对照](PAIRWISE_RESULTS.md) |
| 时效与召回信息 | 同宽DCN，4组×3种子 | 召回信息使直排命中355.67→512.67，仍低于RRF | [四组对照](TEMPORAL_ROUTE_FACTORIAL.md) |
| 特征归因 | 21个拆分模型＋12个日期回放模型 | 名次比只给来源标志更有用；不同日期仍有种子波动 | [归因结果](ROUTE_ATTRIBUTION.md) |
| 数据规模 | 抽样、等步数重复、全量，共18模型 | 全量R直排536次，重复组547.67次，不能把全部收益归给新增数据 | [全量曝光](FULL_EXPOSURE_LOCAL.md) |
| 选轮与早停 | 6个全量模型 | R直排536→670.33，仍低于RRF的1745次 | [收敛检查](RANKING_CONVERGENCE.md) |
| 同时间戳微调 | 匹配BCE/端点辅助/pairwise，共9模型 | pairwise相对继续BCE有增量，但未超过微调前最佳权重 | [微调结果](EXACT_PAIR_FINETUNE.md) |
| 内容与兴趣头 | 全量曝光，12个safe模型 | 内容AUC为0.722349，冻结兴趣头没有额外收益 | [表示学习](REPRESENTATION_EXPERIMENT.md) |
| 属性/历史双塔 | 主窗口各3种子，较早窗口各1种子，共8模型 | 历史改善纯召回；固定预算融合未通过采用条件 | [双塔结果](TOWER_CLOSURE.md) |

命中数为各实验自身协议下的均值。不同轮次的数据状态、训练预算和选轮方式有差异，表中结果不能拼成一条单因素提升曲线。

## 发布内容

- [聚合结果](../experiments/closure_20261022/summary.json)、[训练曲线](../experiments/closure_20261022/curves.html)
- [独立审计](../experiments/closure_20261022/audit.json)、[142项回归记录](../experiments/closure_20261022/tests.json)
- [代码与产物摘要](../experiments/closure_20261022/release.json)

原始数据、SQLite、特征缓存、逐用户分数、模型与大索引保留在本地。复现所需步骤见 [运行说明](REPRODUCE_V2.md)。
