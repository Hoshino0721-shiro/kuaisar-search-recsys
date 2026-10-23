# KuaiSAR 搜索与推荐离线实验系统

面向算法与工程实践的离线项目：数据审计与时间切分 → 多路召回 → 融合与排序 → 可复现实验和本地推荐。不是生产服务，没有线上 CTR/GMV 收益结论。

## 当前状态

- **默认推荐**：近期热门 Top2000＋改进 ItemCF Top1000，等权 RRF 后保留 2000、直接输出前 20；不加载 LR/DCN。
- **排序实验**：LR/DCN、曝光 BCE、pairwise、可学习 ID/属性/caption embedding 已训练并审计。更好的曝光预测没有转化为更好的周期点击覆盖，所以没有替换默认策略。
- **最后一轮已验收**：新版属性 / 属性＋推荐历史双塔共 8 个真实模型，完成三种子、固定预算及较早时间前缀复核；142 项测试和全量独立审计通过。未通过候选净增量及 Top20 稳定性采纳条件，保留 RRF，项目按 [结项报告](docs/PROJECT_FINAL_V2.md) 收尾，不继续追加模型分支。
- **历史 v1**：2026-10-15 已完成冻结测试，单独保存在 [v1 最终报告](docs/FINAL_REPORT.md)。不是新版的独立测试成绩。

## 已有结果及边界

| 任务 / 阶段 | 对照 | 当前证据 |
| --- | --- | --- |
| v2 召回，开发验证 | 近期热门 Recall@2000 9.4229% | 热门＋ItemCF 9.7700% |
| v2 曝光预测，三个种子均值 | 数值 R：AUC 0.713703、GAUC 0.607098 | 内容 embedding：AUC 0.722349、GAUC 0.623011 |
| v2 最终 Top20，周期观察命中 | RRF 1745 次 | 内容模型同池直排均值 535.33 次，未采用 |
| v2 最后双塔，三种子均值 | 原 RRF Recall@2000 9.7700%、Top20 命中 1745 | 加历史双塔：9.3326%、1746；候选下降且 Top20 种子方向不一致，未采用 |
| 历史 v1 冻结测试，曝光预测 | 物品 CTR：AUC 0.569893 | 门控 MLP：AUC 0.709079 |
| 历史 v1 冻结测试，搜索排序 | 常数分数正点击会话 NDCG@10 0.459557 | 结构化 HGB：0.505007 |

v2 推荐验证有 8495 位用户、354626 次正点击；Recall 按点击事件加权，NDCG 按用户周期观察相关物品计算。每位用户在训练截止点只生成一份列表，**不是请求级或无偏线上效果**。同时间戳组也不是已经确认的请求 ID。数值 R 与内容模型训练配置不同，不能将其差值归为纯 embedding 因果增益；完整对照见 [表示学习报告](docs/REPRESENTATION_EXPERIMENT.md)。

新版双塔已完成真实验收，但不是新的盲测成绩。历史测试已被查看，不重新包装为未接触的测试集；较早时间窗口是回溯复核。可提交的聚合结果、曲线及校验清单位于 [结项产物](experiments/closure_20261022/summary.json)，不含逐用户数组或大权重，尚未自动推送 GitHub。

## 本地运行

从项目根目录运行，使用 Python 3.12 项目环境，详见 [环境说明](docs/ENVIRONMENT.md)。

```powershell
# 已安装依赖后的环境与测试检查；不训练
.\scripts\setup_v2.ps1 -SkipInstall

# 当前默认推荐，需要先准备下述数据、CTR 状态和 ItemCF 索引
.\.venv\Scripts\python.exe -m src.v2.recommend --user-id 20235
```

新克隆**不自带原始数据、大缓存、ItemCF/向量索引及大模型权重**，不能直接运行完整推荐。小型 LR/DCN 历史权重不替代匹配的数据/特征状态。完整顺序、最后双塔实验和只读验收见 [复现与交付](docs/REPRODUCE_V2.md)。

```powershell
# 数据构建：先放入官方小版 ZIP；不覆盖已有数据
.\.venv\Scripts\python.exe -m src.prepare_data
.\.venv\Scripts\python.exe -m src.ctr_features
.\.venv\Scripts\python.exe -m src.itemcf_index build --config configs/itemcf_index_support2.json --out artifacts/itemcf_support2_300

# 最后一轮双塔实验：显式 CPU，最多三个四线程训练进程，不自动切换默认
.\.venv\Scripts\python.exe scripts/run_tower_closure.py
```

官方 KuaiSAR-small：`data/raw/KuaiSAR.zip`，488077651 字节，MD5 `daea8cbf605db6bd5841740f0e4a12d9`。旧 `.partial` 不参与计算。双塔回溯对照还需要较早前缀的 ItemCF 索引，见复现文档；不会拿全量图冒充历史图。

## 设计与复现约定

- 两场景共享北京时间切分：训练早于 2023-05-29；验证为 5 月 29 日；旧测试自 5 月 30 日起。
- 当前新增双塔历史按此前完整自然日冻结；验证不利用当日反馈更新用户。物品属性按静态元信息假设使用。
- 推荐 CTR 只用真实曝光 `click`；搜索使用实际会话与 `click_cnt>0`。批内对比物品不是曝光未点击，不编造 CTR 负标签。
- ItemCF 与向量索引不可变，保存 cutoff、配置、模型/数据 SHA256；加载拒绝不匹配的状态。
- 默认与实验链路明确分离。新版双塔为分块全库精确检索；历史 v1 自实现 IVF-flat 不代表新版 ANN/FAISS 已部署。

## 阅读入口

- [最终收尾协议](docs/CLOSURE_PLAN.md)、[双塔收尾](docs/TOWER_CLOSURE.md)、[复现与交付](docs/REPRODUCE_V2.md)
- [v2 结项报告](docs/PROJECT_FINAL_V2.md)、[训练曲线](experiments/closure_20261022/curves.html)、[冻结清单](experiments/closure_20261022/release.json)
- [当前进度与历史记录](docs/CURRENT_PROGRESS.md)、[默认 RRF](docs/RRF_MAIN.md)
- [ItemCF 索引工程](docs/ITEMCF_INDEX.md)、[召回融合对照](docs/RECALL_ALLOCATION.md)
- [收敛验收](docs/RANKING_CONVERGENCE.md)、[pairwise 微调](docs/EXACT_PAIR_FINETUNE.md)、[embedding 实验](docs/REPRESENTATION_EXPERIMENT.md)
- [简历与面试材料](docs/RESUME_AND_INTERVIEW.md)、[历史 v1 冻结报告](docs/FINAL_REPORT.md)

历史报告中的“未训练 / 未发布 / 测试数量”描述其当时快照；当前状态以上方摘要及收尾报告为准。旧实验、旧冻结清单保留，不覆盖或混合指标。
