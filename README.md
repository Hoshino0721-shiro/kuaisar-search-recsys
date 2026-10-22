# KuaiSAR 搜索与推荐离线实验系统

当前默认推荐采用 [近期热门＋个性化ItemCF，直接输出RRF前20](docs/RRF_MAIN.md)，不加载LR/DCN权重。用户按当前次日点击覆盖目标选择该方案；RRF本身完成两路融合排序。全量8495名验证用户的固定2000候选召回率为9.7700%，超过同期近期热门的9.4229%；神经排序尚未超过强基线，保留为实验链路，见 [多阶段算法改造](docs/CASCADE_V2.md)。当前没有新的独立最终验收，下文测试指标全部属于已冻结的v1。

截至 2026-10-22 的实现、实验、权重及复现边界见 [当前进度与发布产物](docs/CURRENT_PROGRESS.md)。

当天同步补充：[全量曝光训练代码、抽样覆盖检查与停止状态](docs/FULL_EXPOSURE_STATUS.md)。用户已要求停止实验；全量特征缓存准备完成，但没有全量模型训练/评测结论。发布的准备记录与已完成的45个时效/召回相关实验模型分开统计，不将计划当作成果。

ItemCF 新增[离线邻居索引与查询组件](docs/ITEMCF_INDEX.md)：分块建图、可配置活跃度/时间/热门度权重、TopK 邻居、不可变版本发布、校验与内存映射查询。组件及召回消融属于验证期开发结果，不沿用下方 v1 最终测试指标。

召回后续改进见 [取量、融合与共同用户支持度验证](docs/RECALL_ALLOCATION.md)：独立配置各路取量、RRF/去重配额对照、空路补位及至少 2 个共同用户的 ItemCF 支持度实验。指标只用于验证期开发；召回覆盖改善不等于最终 Top20 已改善。

[双塔召回 v2](docs/DUAL_TOWER_V2.md) 已补齐属性/历史双塔、训练、导出、精确检索和级联入口，真实缓存包含 286 万正点击、约 5.6 万共享上下文。按本机不训练的约束，真实模型尚未训练；新双塔没有效果指标，不沿用旧双塔或 ItemCF 的结果。

精排特征增强已实现并完成 24 组真实曝光训练，见 [验证结果](docs/DCN_FEATURE_RESULTS.md) 与 [实施及复现说明](docs/DCN_ENHANCEMENT_IMPLEMENTATION.md)：保留 Cross + Deep，比较属性、身份和推荐/搜索历史；属性有小幅曝光收益，历史池化没有稳定收益。增强推理使用独立实验配置，不修改默认模型，也不将验证结果称为新独立测试或线上收益。

随后完成 [Pairwise 训练对照](docs/PAIRWISE_RESULTS.md)：数值/属性 DCN 比较完整 BCE、配对端点 BCE、pairwise 与两类混合损失，最终 30 模型、8495 用户同池评测、99 项测试通过。数值 DCN 的完整曝光混合有小幅排序信号，但采样影响、曝光退化与级联瓶颈仍在，不切换默认推荐。协议及复现见 [训练说明](docs/PAIRWISE_TRAINING.md)。

新增 [时效×召回信息四组对照](docs/TEMPORAL_ROUTE_FACTORIAL.md)：保持同一 34 维 DCN、BCE、100 万曝光与 3 种子，使用严格日级前缀图。召回信息使直排 Top20 命中均值 355.67→512.67、全种子级联 405.67→506.33；曝光 AUC 基本不变，组合特征未全面超过单独召回信息。105 项测试和独立审计通过。这仍是同一开发验证日的结果，不替换默认模型。

后续完成 [召回特征归因与历史日期回放](docs/ROUTE_ATTRIBUTION.md)：新增33模型、109项测试、1056条抽样列表独立审计。CF和近期热门信息单独加入均有开发增量，名次比只给来源标志更有用；两个历史日期的完整R组均值也改善，但仍明显落后各日RRF。旧粗排与新精排都会丢失大量RRF头部命中，有限调整对照仍未超过原序。回放属于已接触训练期的严格前缀检验，不称独立盲测或线上收益。

2026-10-22 已整理发布 [13 份聚合 JSON 与 SHA256 清单](experiments/ranking_progress_2026_10_22/manifest.json)，不含逐用户记录或新权重。[下一阶段方向及已知问题](docs/NEXT_RANKING_DIRECTIONS.md) 明确区分可行性分析与已实现功能，包括尚未修正的 raw author 0 padding 处理。

排序优化从[固定候选诊断](docs/RANKING_V3.md)开始：同一改进 ItemCF 候选池比较 RRF、LR、DCN 与粗排预算，保存候选及逐用户指标，不使用测试标签。

已完成数据审计、时间切分、热门/ItemCF/双塔召回、多路融合、CTR 精排、搜索会话排序、类别多样性重排和近似检索实验。验证集选型后冻结代码与模型校验值，完成测试集评测和独立复核。

最终结果与限制见 [项目总报告](docs/FINAL_REPORT.md)，项目介绍见 [简历与面试材料](docs/RESUME_AND_INTERVIEW.md)。

## 测试集结果

| 任务 | 对照 | 冻结方案 |
| --- | ---: | ---: |
| 推荐 CTR AUC | 物品点击率 0.5699 | 门控 MLP 0.7091 |
| 推荐 CTR LogLoss | 0.6826 | 0.6226 |
| 推荐召回 Recall@100 | 全时热门 0.5628% | 训练末日热门 0.9160% |
| 搜索 NDCG@10，含点击会话 | 常数分数 0.4596 | 结构化特征排序 0.5050 |

双塔和当前多路精排链路未超过近期热门的召回效果；搜索特征与门控的增益很小。默认推荐使用验证选出的近期热门，完整链路保留用于实验。搜索模型未实现语义查询匹配，指标为离线结果。

## 运行

在项目根目录、已安装依赖的 Python 环境执行：

```powershell
# 完成结果存在时，仅核验，不重新训练或选择参数
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_pipeline.ps1 -VerifyOnly

# 默认推荐方案
python -m src.recommend --user-id 20235 --top-k 10

# 多路候选 -> CTR -> 重排 的完整实验链路
python -m src.recommend --user-id 20235 --top-k 10 --mode full_pipeline
```

完整构建命令为 `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_pipeline.ps1`，按结果文件跳过已完成阶段。新环境需提供 `data/raw/KuaiSAR.zip`（官方小版 MD5：`daea8cbf605db6bd5841740f0e4a12d9`）；运行环境见 [ENVIRONMENT.md](docs/ENVIRONMENT.md)。训练和数据准备默认拒绝覆盖已有结果，遇到中断按报错恢复对应阶段，不自动删除数据。

`configs/default.yaml` 是当前方案摘要；各阶段的实际参数以模块实现、CLI 参数与实验目录 config.json 为准，修改 YAML 不会自动重训或改变已冻结方案。结果图可用 `python scripts/plot_results.py` 重绘（额外需要 Matplotlib）。

## 数据与评测约定

- 数据库：`data/processed/kuaisar_small_v1/dataset.sqlite`，模型使用 `clean_events` 或等价的 `events.exclusion IS NULL`。
- 两场景共用北京时间切分：train `< 2023-05-29`，valid `[05-29,05-30)`，test `>= 05-30`。
- 序列双塔使用严格早于目标时刻的训练历史；CTR 统计使用此前完整日期；评测固定训练截止状态。
- 推荐 CTR 使用原始曝光日志的 click，搜索使用 click_cnt>0。不为未曝光召回候选编造 CTR 负标签。
- 推荐 Recall 为正点击事件加权；推荐列表 NDCG 是整个用户评测周期的观察正例指标。搜索按实际会话列表评测。

## 产物

| 路径 | 内容 |
| --- | --- |
| `data/processed/kuaisar_small_v1/manifest.json` | 数据切分、排除记录与完整性检查 |
| `experiments/final_test/freeze.json` | 测试前选择与代码/模型 SHA256 |
| `experiments/final_test/metrics.json` | 最终测试结果 |
| `experiments/final_test/audit.json` | 独立重算命中、概率指标与权重加载检查 |
| `experiments/r2_*` | 双塔、历史与属性消融及 checkpoint |
| `experiments/p_rank_valid` | CTR 模型、校准分桶与分组结果 |
| `experiments/pipeline_valid` | 融合与重排验证结果 |
| `experiments/search_valid` | 搜索排序验证模型和结果 |
| `experiments/ann_valid` | 自实现 IVF-flat 质量/延迟实验 |

已发布 v2 LR/DCN 两个小型 checkpoint（合计 22650 字节），它们是已有训练结果，本轮未重训；校验值见 [产物清单](experiments/published_artifacts.json)。原始日志、数据库、特征状态、训练缓存、向量/ItemCF 大索引仍由 .gitignore 排除，需要按文档重建。新版双塔尚未真实训练，无可发布权重。历史 v1 的 checkpoint 也不包含在本次发布中，其测试结论以冻结报告为准。

