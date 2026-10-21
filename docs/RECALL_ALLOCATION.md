# 召回取量、融合与共同用户支持度验证

日期：2026-10-21。本轮复用已有 ItemCF 和排序权重，不重训业务神经模型；不使用测试标签。所有质量结果属于已经用于开发的同一验证日，不是独立测试或线上收益。

## 第一轮：候选截断与名额分配

旧配置每路至多取 1000、融合至多保留 2000，使用热门/近期热门/ItemCF 权重 1/2/1 的 RRF。近期热门单路基线直接取 2000，因此旧融合在进入融合前就截断了近期热门第 1001～2000 名。冷用户 CF 为空，旧配置两路热门去重后平均只有 1510 个候选。

本轮保持相同 951625 个训练正点击物品、训练截止时间、8495 个验证用户和 354626 个正点击事件，只调整各路供给、RRF 权重或唯一候选配额。配置在运行前列出 15 个方案，见 `configs/recall_allocation_experiments.json`。

实现：

- `route_limits` 为每路单独配置取量，未配置时沿用 `route_k`，0 表示关闭该路。关闭 ItemCF 时既不加载索引，也不构建旧矩阵。
- `src/v2/recall_fusion.py` 是级联与新评测共用的选择函数。RRF 仍是默认，常数 60、同分按 item_id 排序，旧配置的结果保持一致。
- 可选 quota 按顺序分配**新增且去重**的候选数量；空路、重叠造成的余量由 `fill_order` 补齐。不能把重复候选算作完成名额，也不能为冷用户伪造 CF 路。quota 由分配顺序决定，`route_weights` 只控制 RRF；关闭一路应使用 `route_limits=0`。
- 可用时间与显式排除在选择之前过滤。独立评测只读取训练索引和验证观察正例，不加载精排模型、不构造未曝光负标签。

| 方案 | 命中正事件@2000 | Recall@2000 | 相对近期热门新增 / 丢失事件 |
| --- | ---: | ---: | ---: |
| 旧 RRF、每路 1000 | 29530 | 8.3271% | 3319 / 7205 |
| 近期热门扩至 2000，其余旧权重 | 32572 | 9.1849% | 2333 / 3177 |
| 近期热门单路 2000 | 33416 | 9.4229% | 0 / 0 |
| 近期热门 + CF，RRF 权重 1/0.1 | 33332 | 9.3992% | 209 / 293 |
| 近期热门先取 1975，再补 25 个 CF 独有候选 | 33418 | 9.4234% | 124 / 122 |
| 近期热门先取 1900，再补 100 个 CF 独有候选 | 33364 | 9.4082% | 442 / 494 |
| 近期热门先取 1800，再补 200 个 CF 独有候选 | 33366 | 9.4088% | 890 / 940 |

结论：补齐近期热门供给修复了明确的候选损失，Recall 增加约 0.8578 个百分点，但原权重的多路融合仍低于单路近期热门。不能仅凭 CF 的独有命中，就判断它值得替换同数量的热门候选。

为了辅助检查，将用户按种子 2026 分成 5097 位选择用户和 3398 位检查用户，两个部分分别含 213043、141583 个正点击事件。选择部分按 Recall 选出 quota_cf_25：相对近期热门，选择部分净增 8 次命中，检查部分净减 6 次，全量仅净增 2 次。探索性用户 bootstrap 的全量差值区间约为 [-0.00734, 0.00914] 个百分点，包含 0。该配置不能宣称稳定超过近期热门，不作为已验证的个性化提升。

检查部分也来自此前看过的验证日，不能称为新的盲测或时间外泛化证据。所有方案的全量、用户分区、有/无训练点击分组，以及新增/丢失命中均保留在 `experiments/recall_allocation_valid_v1/metrics.json`，配对命中向量本地保存在同目录 `paired_hits.npz`。

报告生成后的评测脚本只补充了空正例分区的除零保护；本次真实选择/检查分区均有正例，结果不受该保护分支影响。生成报告时的原始脚本摘要在报告中，源码快照已发布在 `experiments/recall_allocation_valid_v1/source_snapshot.py`，仅用于核对历史报告来源；当前复现入口仍为 `src.v2.recall_allocation`。

## 第二轮：ItemCF 弱支持边

第一轮表明，扩大 CF 配额或调整 RRF 权重没有稳定超过近期热门。下一步在相同目录、全部训练去重历史、alpha=0.25、每物品 300 邻居的条件下，只把 `min_cooccurrence` 从 1 改为 2，检验单共同用户支持边的影响。

配置为 `configs/itemcf_index_support2.json`，两索引原始 CF 与原融合的对照为 `configs/itemcf_ablation_support.json`。支持度至少 2 只排除单用户支持边，不等于完全消除了目标用户自身共现贡献。

全量原始 CF 结果如下，热门兜底不计入纯 CF 路：

| CF 索引 | Recall@100 | Recall@1000 | 热门两路之外独有正事件 |
| --- | ---: | ---: | ---: |
| 原 alpha=0.25、L300、至少 1 个共同用户 | 0.6068% | 3.4665% | 3224 |
| 其余不变、至少 2 个共同用户 | **0.7721%** | **4.0107%** | **4459** |

Recall@1000 绝对增加约 0.5442 个百分点、相对提高约 15.7%；用户 bootstrap 的探索性差值区间约为 [0.5047, 0.5877] 个百分点。平均 CF 候选由 663.42 减至 579.40，收益不是来自返回更多候选。

过滤后的图有 20912618 条边、650446 个无邻居物品，CF 非空用户占全验证用户的比例从 70.75% 降至 69.26%。因此过滤的代价是稀疏物品与部分用户失去 CF 覆盖，需要其他路补充。邻居数组仍按固定宽度保存，索引仍约 2.14 GiB，不能把边数下降写成同等比例的磁盘节省。

原始报告：`experiments/itemcf_support_valid_all/metrics.json`。独立图抽查覆盖 512 个物品、24602 条边，验证共同用户支持度、分数公式、无自环和邻居预算，见同目录 `graph_audit.json`；抽查不是全图证明。

## 第三轮：更强 CF 的融合与采用结论

把第二轮新索引接入与第一轮相同的 15 个取量/融合方案，不扩大物品库、不更改训练截止时间。该轮是在第一轮分析后开展的开发跟进，不是对全部实验空间预注册的独立测试。配置见 `configs/recall_allocation_support_experiments.json`。

| 索引与融合 | 命中正事件@2000 | Recall@2000 |
| --- | ---: | ---: |
| 旧 CF + 旧取量/权重 | 29530 | 8.3271% |
| 旧 CF + 近期热门扩池，保留旧权重 | 32572 | 9.1849% |
| 新 CF + 旧取量/权重 | 30739 | 8.6680% |
| 新 CF + 近期热门扩池，保留旧权重 | 33596 | 9.4736% |
| 旧 CF + 近期热门 2000 / CF1000，RRF 1/1 | 33059 | 9.3222% |
| 近期热门单路 2000 | 33416 | 9.4229% |
| **新 CF + 近期热门 2000 / CF1000，RRF 1/1** | **34647** | **9.7700%** |

最终研究配置关闭全时热门路，取近期热门至多 2000、新 CF 至多 1000，以 1/1 RRF 去重后保留 2000。相对近期热门，新增 3903 次命中、丢失 2672 次，净增 1231 次；绝对增加约 0.3471 个百分点、相对提高约 3.68%。相对原始旧 CF 融合，净增 5117 次命中，Recall 相对提高约 17.33%。不能把这个总收益全部归因于支持度过滤或单个参数。

上述 1/1 方案由选择用户选出，在选择部分相对近期热门增加约 0.3600 个百分点，在内部检查部分增加约 0.3277 个百分点。探索性全量用户 bootstrap 差值区间约为 [0.2882, 0.4069] 个百分点，检查部分区间约为 [0.2336, 0.4159]。这些结果支持本开发窗口内的候选覆盖改善，仍有重复使用验证集和多次选择的限制，需要新的隔离时间窗口复核。

全部方案最终预算同为 2000；原方案实际平均 1818.81 个候选，新方案每用户补满 2000。对无训练点击的 2410 位用户，CF 为空，采用方案自然退回完整近期热门；对有训练点击用户，才检验 CF 替换热门的增量。

采用入口为 `configs/cascade_v2_recall_optimized.json`。原配置不覆盖，保留旧方案、单路控制和失败方案。此处采用仅指召回开发配置，不代表最终 Top20、生产上线或独立测试已经验收。粗排和精排沿用旧权重，尚需测量它们是否保住新增候选。

第三轮全量报告为 `experiments/recall_allocation_support_valid_v1/metrics.json`；配对命中向量保存在同目录 `paired_hits.npz`，原方案与新方案的配对比较见 `comparison.json`。

### 两路贡献的口径

当前路由实际供给为近期热门 Top2000 与 CF Top1000（不足时按实际返回），其独立命中分别为 33416、14223 次；共同覆盖 10320 次。融合前的并集覆盖 37319 次，其中近期热门独有 23096 次、CF 独有 3903 次。独立命中不能直接相加。

固定 2000 名额后，相对近期热门单路，新增 CF 独有命中 3903 次，同时丢掉近期热门命中 2672 次，净增 1231 次。移除 CF、将名额全部交回近期热门，命中从 34647 降至 33416；移除近期热门、只剩实际 CF 路，则降至 14223，少 20424 次。后一个对照只剩最多 1000 个 CF 候选且冷用户为空，不能解释成两个算法的同预算比较，也不能把两个移除差值相加当作独立贡献分摊。

### 第四轮：只加入全时热门的受控消融

第三轮的三路方案同时改变了其他路的相对权重，不能把差异全部归因于关闭全时热门。为直接回答该问题，固定支持度 2 的图、近期热门 Top2000、CF Top1000、两者 RRF 权重 1/1、融合预算 2000，仅加入全时热门 Top1000 并改变该路权重：

| 全时热门权重 | 命中正事件@2000 | Recall@2000 | 相对当前两路净变化 |
| --- | ---: | ---: | ---: |
| 0，当前两路 | 34647 | 9.7700% | 0 |
| 0.1 | 34529 | 9.7367% | -118 |
| 0.25 | 34342 | 9.6840% | -305 |
| 0.5 | 33889 | 9.5563% | -758 |
| 1 | 33058 | 9.3219% | -1589 |

这组受控结果支持在当前开发窗口和已测试的融合方式下关闭全时热门。加入一路不仅带来新物品，还会为重叠物品增加 RRF 得分，改变有限名额的分配；能命中部分正例并不等于有正的净增量。不将本结论推广为其他窗口、配额方式或所有权重下全时热门均无效。

配置为 `configs/recall_popular_ablation.json`；全量和用户分组结果为 `experiments/recall_popular_ablation_valid/metrics.json`，相对当前两路的配对差值见同目录 `attribution.json`。这是根据解释需求追加的开发消融，仍未使用测试标签，不构成独立时间窗口验证。

两轮融合报告均通过独立逐用户策略重放和命中汇总复核。通过实际 `Backend.retrieve` 入口对全量 8495 位用户重算，采用方案同样命中 34647 次；报告为第三轮目录中的 `audit.json`、`route_diagnostics.json`。真实 checkpoint 的单用户链路已完成 2000→400→80→20 推理，见 `integration.json`，这只验证接通，不验证 Top20 质量。全部 66 项测试通过，包括配额去重/补位、禁用 CF 不加载图、时间排除与空正例分组。

## 复现和接入

```powershell
# 不训练模型的第一轮全量配额实验，输出目录须不存在
./.venv/Scripts/python.exe -m src.v2.recall_allocation --out experiments/recall_allocation_reproduce

# 独立重算所有命中向量汇总，并逐用户重放对照和选中策略
./.venv/Scripts/python.exe scripts/audit_recall_allocation.py --report experiments/recall_allocation_reproduce

# 构建仅改变共同用户支持度的索引，并比较原始 CF
./.venv/Scripts/python.exe -m src.itemcf_index build --config configs/itemcf_index_support2.json --out artifacts/itemcf_support2_300
./.venv/Scripts/python.exe -m src.itemcf_evaluate --variants configs/itemcf_ablation_support.json --out experiments/itemcf_support_valid_all --benchmark-users 0

# 重跑新图的同一组融合方案及其独立复核
./.venv/Scripts/python.exe -m src.v2.recall_allocation --config configs/recall_allocation_support_experiments.json --out experiments/recall_allocation_support_reproduce
./.venv/Scripts/python.exe scripts/audit_recall_allocation.py --report experiments/recall_allocation_support_reproduce

# 采用配置接入现有排序权重，不暗示最终 Top20 已改善
./.venv/Scripts/python.exe -m src.v2.recommend --config configs/cascade_v2_recall_optimized.json --user-id 20235

# 保留扩池对照及近期热门候选控制
./.venv/Scripts/python.exe -m src.v2.recommend --config configs/cascade_v2_recall_expanded.json --user-id 20235
./.venv/Scripts/python.exe -m src.v2.recommend --config configs/cascade_v2_recent_control.json --user-id 20235
```

现有 `cascade_v2_itemcf_index.json` 保留旧方案以便对照。新增 expanded 配置修复取量但仍弱于近期热门；optimized 采用本轮更强 CF 与两路融合；recent_control 是候选覆盖的强对照。它们其后仍接原有粗排/精排，不能把候选 Recall 直接当作最终 Top20 指标。

下一步验收应同时检查原始 CF、固定预算候选的净增量和最终同 K 排序收益；物品库扩展与更实时历史更新另立实验，避免同时改动后无法归因。
