# DCN 特征增强：实施与实验交付

最终曝光结果、取舍和产物索引见 [验证报告](DCN_FEATURE_RESULTS.md)。本轮最终训练为 8 配置 × 3 种子，共 24 组；历史父配置为 `item_only`。新增报告器 `enhanced_report.py` 对逐用户命中、权重/数据/源码摘要和旧冻结资产做独立一致性检查，补充按训练曝光活跃度分组及探索性用户 bootstrap，不读取最终测试标签。

## 对方案的评估

原计划值得执行：保留 Cross + Deep，通过属性、身份、历史的信息增量做对照，比同时换架构、样本与损失更容易归因。容量对照、冷 ID、时间可用性和固定候选评测是必要条件。

本次实施授权来自用户“评估新方案，可以的话实行到底”，取代计划中旧阶段“本机不训练”的备注。只在独立目录做离线训练与验证，没有线上部署、自动推送或读取最终测试标签。

## 实际实现

- 数据：`enhanced_data.py`，同一批种子 2026 的 100 万条真实曝光；候选字段和日级历史上下文分开保存，避免复制每条曝光的 60 个历史 token。
- 模型：`enhanced_model.py`，原版向量 Cross 两层 + Deep；属性/用户/物品 embedding；推荐和搜索分别做 masked mean，与候选共享物品表。不是 DIN，也不把 mean pooling 称为顺序建模。
- 实验：A、容量对照、B、C，各 3 个训练种子；C 若退化则执行仅用户/仅物品 ID 拆分；根据预先声明的平均验证 LogLoss 选择历史父配置，再运行 D/E 各 3 种子。实际父配置以最终报告为准。
- 训练：BCE、Adam 数值/网络参数，embedding 使用 SparseAdam、无 weight decay、同学习率 0.001；batch 4096、最多 5 轮，统一按最低验证 LogLoss 选 checkpoint。SparseAdam 是 CPU 稀疏表工程选择，不能假装 embedding 与原数值模型的优化器完全相同。
- 特征标准化只拟合公共训练样本；词表只拟合实际训练输入。属性假设静态，仍没有属性快照时间可验证。
- 推理：`enhanced_recommend.py` 是可选入口，保留原粗排和召回，新增结构化输入/权重校验；默认 `src.v2.recommend` 不变。
- 产物：训练曲线、各组曝光 AUC/GAUC/LogLoss/Brier/ECE、冷物品/低频/历史覆盖分组、权重摘要、同池列表和级联对照、CPU 推理耗时均单独记录。

## 实施中修正的两个输入问题

1. 全量训练日志词表不等于实际拟合样本词表。早期试跑发现约 3% 验证物品/作者有独立向量却没有实际更新。最终按实际拟合输入计频，并为每个模型保存实际 token 覆盖 mask，低频/未用 token 回退 UNK。
2. 只从推荐目标建共享词表，会使约 99% 的搜索历史 token 退化为 UNK。最终物品词表同时计入实际训练曝光的合法推荐/搜索历史，按目标行数加权计频，PAD 不入词表。没有用冻结 query、验证或测试历史扩词表。

原始身份、历史、类别和 train/valid 对齐都由数据缓存保存。模型 token 的 UNK=1 可训练，PAD=0 固定为零、既不参与均值分母也不接收梯度；没有验证新 ID 的未经训练独立向量。注意这不代表原始字段缺失约定均已正确处理：2026-10-22 发现 raw author 0 的官方含义是 padding，当前实现仍按普通作者映射，尚待新缓存与消融修正，见 [已知问题](NEXT_RANKING_DIRECTIONS.md)。历史产物不修改。

前两个试跑目录 `artifacts/dcn_enhancement_v1`、`artifacts/dcn_enhancement_v2` 保留以供追溯，不作为最终验收。最终词表/模型实验使用 `artifacts/dcn_enhancement_v3`；这些目录版本不是“DCN-v2/v3”架构名称。

## 固定召回与兼容性

采用新召回：近期热门 Top2000 + min_support=2 的 CF Top1000，等权 RRF 保留 2000。索引 `c5e5d6daaddf007507a5`；冻结用户 8495 人、每人 2000 候选，覆盖正点击 34647/354626=9.7700%。
候选文件 SHA256 为 `99abfbdb6e69d33264ae2b5c2a00031cf4edd8f984ef7e4c98054e861a73346e`，保存在 `artifacts/dcn_enhancement_v1/candidates` 并由全部最终模型复用；目录名不会改变候选内容。

远端发布 LR/DCN 权重的特征状态摘要与本机状态不同，被加载器拒绝。本次使用本机同状态的恢复权重作固定入口，配置为 `configs/dcn_enhancement_recall_local.json`，没有关闭校验；权重不参与候选融合，因此未改变所选召回策略。

## 复现与恢复

完整流水线（需已准备官方数据和数值缓存，CPU 也可运行）：

```powershell
.\scripts\setup_v2.ps1 -SkipInstall
.\scripts\run_dcn_enhancement.ps1 -RunName dcn_reproduce
```

可分别执行数据准备、独立审计、训练、候选冻结与评测。中断后的完整模型只有在数据、模型源码、样本、配置与 checkpoint 摘要验证后才允许复用；未完成模型保留原现场，在新目录重训：

```powershell
.\.venv\Scripts\python.exe -m src.v2.enhanced_train --cache artifacts/dcn_enhancement_v3/cache --out artifacts/dcn_recovered --reuse artifacts/dcn_enhancement_v3/models
```

评测每 500 用户刷写逐用户进度文件；最终汇总不存在时不能把部分结果当成完成报告。原数据、旧权重、V1 冻结资产和默认入口均不覆盖。

## 验收边界

这里完成的是验证期开发和可复现工程交付，不是新独立测试或线上收益验证。旧测试窗口已被查看，本次仍不读取它来调参。
若属性/身份/历史只改善曝光指标而没有改善最终列表，必须保留这一结论；若多种子或冷物品分组退化，也不能只发布最好的种子。是否采用增强模型要结合最终的同池与级联报告，不按“实现了更多特征”自动切换。
