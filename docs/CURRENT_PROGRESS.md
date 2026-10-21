# 当前进度与发布产物

## GitHub 进展发布（2026-10-22）

本次同步排序诊断、DCN 特征增强、Pairwise 训练/评测/审计代码、复现脚本和结果报告。99 项测试在发布前重新通过；默认模型不变，没有新的最终测试或线上收益结论。

可直接查看 [24 组 DCN 特征结果](DCN_FEATURE_RESULTS.md)、[30 模型 Pairwise 结果](PAIRWISE_RESULTS.md) 和 [13 份聚合 JSON 及校验清单](../experiments/ranking_progress_2026_10_22/manifest.json)。不上传原始日志、SQLite、逐用户数组、增强权重或大索引，配置中的 artifacts 路径仍需本地重建；聚合 JSON 的来源摘要与发布文件摘要分别记录。

[下一阶段方向](NEXT_RANKING_DIRECTIONS.md) 仅为分析，LambdaRank、双通道 DIN、联合蒸馏尚未实施。同时间戳可配对性已做只读检查，作者 ID 0 的 padding 契约问题已发现但未修改旧模型，影响范围和边界均已披露。下面 2026-10-21 的“不推送”描述是当时未发布状态，本节记录后续用户授权的发布。

## 后续本机实施更新（2026-10-21）

最新：[Pairwise 训练对照](PAIRWISE_RESULTS.md) 已完成最终 30 模型、8495 用户同池比较和预设种子级联，99 项测试通过。数值 DCN 完整 BCE＋pairwise 的直排命中三种子平均 353.33→419.33，曝光 AUC 基本持平；预设单种子级联 433→441，尚无稳定全面收益，默认模型不变。配对采样对照、覆盖 mask 修正、时间匹配诊断及限制均保留，不读取最终测试标签、不推送 GitHub。

用户后续授权实施 DCN 新方案。本机已完成增强缓存、独立历史审计、24 组真实曝光训练和可选增强推理入口，见 [DCN 验证报告](DCN_FEATURE_RESULTS.md) 与 [复现说明](DCN_ENHANCEMENT_IMPLEMENTATION.md)。属性有小幅曝光预测增量，用户 ID/历史均值池化未稳定改善；没有自动替换默认模型、读取最终测试标签或推送 GitHub。新增大缓存和权重只在本机 artifacts 中。

新增验收：8495 用户同池对照、256 用户实际级联、用户分组/探索性配对 bootstrap、24 个 checkpoint 和旧冻结资产核验完成；93 项测试通过。按曝光 LogLoss 选中的模型在全量级联 Top20 命中 358 次，数值基线为 431 次，因此本轮不升级默认模型。

以下“本轮未训练”“71 项测试”等描述是 GitHub 同步时的旧发布快照，不是上述后续实施状态；新版双塔仍未真实训练，不能混用为 DCN 实验成果。

## GitHub 同步时的发布快照

更新：2026-10-21。当前 v2 是离线算法开发版本；没有线上部署或收益结论。本轮未训练真实神经模型，未使用测试标签选新方案。

## 已完成

- ItemCF：分块建图、TopK 邻居、时间/活跃度/热门度参数、不可变校验索引与查询。最近一轮将共同用户支持度阈值从 1 提至 2，纯 CF Recall@1000 从 3.4665% 提至 4.0107%；稀疏物品覆盖随之下降。
- 融合：独立路由取量、确定性 RRF、去重配额、空路补位与控制实验。当前开发配置为近期热门 Top2000 + 新 CF Top1000，等权 RRF 后保留 2000；在同一验证日 8495 用户、354626 正点击事件上命中 34647 次，Recall@2000 为 9.7700%，同期近期热门为 9.4229%。净增 1231 次命中；重复使用验证集，尚需独立时间窗口复核。
- 排序链路：已有独立 LR 粗排和原始 DCN 精排，沿用原权重；单用户已接通 2000→400→80→20。候选增益能否保留至最终 Top20 尚未验收。
- 双塔 v2：训练数据准备、属性/历史编码、批内对比训练、导出、精确向量检索、分组评测及可选级联接入。真实数据已准备 2860013 个正点击目标和 55990 个共享上下文；仅合成夹具完成训练测试，未训练真实双塔，未生成新模型指标或向量索引。
- 验证：71 项测试通过；独立召回重放、ItemCF 图抽查、双塔历史抽样重放的报告已发布。抽样审计不是全部数据正确性的证明。

详细过程见 [召回实验](RECALL_ALLOCATION.md)、[双塔 v2](DUAL_TOWER_V2.md)、[DCN 特征计划](DCN_FEATURE_PLAN.md)。DCN 特征增强仍为计划，尚未实现或训练。

## 已发布的权重与报告

| 文件 | 内容与状态 |
| --- | --- |
| `experiments/v2_coarse_lr/model.pt` | 已有 LR 粗排，3029 字节，8 个统计特征 |
| `experiments/v2_fine_dcn/model.pt` | 已有 DCN 精排，19621 字节，20 个统计特征 |
| `experiments/published_artifacts.json` | 文件字节数、SHA256、模型/特征来源摘要及大文件状态 |
| `experiments/itemcf_support_valid_all/index_metadata.json` | 当前 CF 索引版本、参数、各数组摘要，不包含索引本体 |
| `experiments/recall_allocation*_valid_v1/` | 召回取量/融合结果与审计，配对数组仍留本地 |
| `experiments/recall_popular_ablation_valid/` | 累计热门受控消融结果 |
| `experiments/tower_v2_preparation/` | 双塔真实数据准备摘要、抽样审计，不是训练效果 |

两份小型权重使用 `torch.load(..., weights_only=True)` 验证可读取，包含标准化参数、模型配置及参数张量，不包含逐用户/逐物品 embedding。它们在本轮发布前已经存在，本轮没有重训。

不上传原始数据、SQLite、特征状态和约 96 MB 双塔训练缓存。约 2.14 GiB 的 ItemCF 索引保留本地，由构建配置和已发布摘要定位。新版双塔没有真实权重；旧 v1 双塔权重本机缺失，不能仅凭历史报告恢复。

## 复现边界与入口

新克隆仅具备代码、报告与两个排序 checkpoint，不能直接启动依赖本地数据的完整推荐。先按 [环境说明](ENVIRONMENT.md) 和 [级联说明](CASCADE_V2.md) 准备相同 KuaiSAR-small 数据、SQLite 和 CTR 特征缓存。排序加载会检查特征状态摘要，发布权重不能替代特征状态，也不会绕过校验。

```powershell
python -m unittest discover -s tests -v

# 数据与特征状态准备好后；输出目录须不存在
python -m src.itemcf_index build --config configs/itemcf_index_support2.json --out artifacts/itemcf_support2_300
python -m src.v2.recommend --config configs/cascade_v2_recall_optimized.json --user-id 20235
```

双塔真实训练和消融步骤见 [双塔文档](DUAL_TOWER_V2.md)，需要单独显式执行。当前两路召回配置不会自动启用未训练的双塔。当前检索为精确点积，尚无新版 ANN 实现；冻结 query 也不等于实时行为更新服务。

下一步依次是：在选定机器训练并评测双塔、检查固定候选预算下的净增量、验证最终 Top20；然后开展 DCN 特征增强与独立时间窗口复核。当前代码与报告没有证明这些后续目标已经达成。
