# KuaiSAR 搜索与推荐

基于 KuaiSAR-small 的离线实验项目，包含数据处理、ItemCF、双塔召回、CTR 预估和多阶段排序。

项目主要比较两个问题：不同召回方法在固定候选预算下能覆盖多少点击，以及曝光样本上的预测改善能否转化为推荐列表的改善。当前默认使用近期热门与 ItemCF 的 RRF 融合；双塔和神经排序保留为实验对照。

## 推荐流程

```text
近期热门 Top2000 ─┐
                  ├─ RRF 融合 → 2000 候选 → Top20
ItemCF Top1000 ───┘

实验分支：属性 / 历史双塔；LR400 → DCN80 → 类别重排20
```

ItemCF 在离线阶段分块计算物品共现，每个物品保留有限邻居，查询时读取最近行为种子的邻居并累计分数。索引使用只读内存映射，记录数据截止时间和版本。

双塔采用批内对比学习，比较属性输入和增加推荐历史的版本；检索使用分块精确点积。排序部分比较统计特征、ID/属性与匿名 caption 表示，以及 BCE 和 pairwise 训练。

## 主要结果

| 实验 | 对照 | 结果 |
| --- | --- | --- |
| ItemCF 支持度过滤，Recall@1000 | 3.4665% | 4.0107% |
| 固定 2000 候选的召回融合，Recall@2000 | 近期热门 9.4229% | 近期热门＋ItemCF 9.7700% |
| 内容特征，曝光 AUC / GAUC | ID/属性 0.719850 / 0.619901 | 加 caption 0.722349 / 0.623011 |
| 推荐历史，纯双塔 Recall@1000 | 属性双塔 1.2954% | 属性＋历史 1.7866% |

后两项为三个随机种子的均值。主验证集有 8495 位用户、354626 次正点击，物品库为训练期点击过的 951625 个物品。

内容特征改善了曝光预测，但列表点击覆盖仍低于 RRF。历史双塔提高了单路召回，加入融合后 Recall@2000 为 9.3326%，低于原来的 9.7700%；Top20 平均命中 1746 次，对照为 1745 次，三个种子的变化方向不一致。因此默认配置没有启用双塔或 DCN。详细对照见 [实验总结](docs/PROJECT_FINAL_V2.md)。

评测在训练截止点为每个用户生成一份列表，与次日日志点击比较；结果属于开发验证及较早日期回放，不能直接解释为线上 CTR。旧版测试结果单独保存在 [v1 报告](docs/FINAL_REPORT.md)。

## 运行

环境为 Python 3.12、PyTorch 2.8 CPU。首次安装：

```powershell
.\scripts\setup_v2.ps1 -BasePython 'C:\path\to\Python312\python.exe'
```

下载官方 [KuaiSAR-small](https://kuaisar.github.io/)，将压缩包放在 `data/raw/KuaiSAR.zip`，然后准备数据和索引：

```powershell
.\.venv\Scripts\python.exe -m src.prepare_data
.\.venv\Scripts\python.exe -m src.ctr_features
.\.venv\Scripts\python.exe -m src.itemcf_index build --config configs/itemcf_index_support2.json --out artifacts/itemcf_support2_300
.\.venv\Scripts\python.exe scripts/demo_recommendation.py --user-id 20235
```

演示输出前 20 个物品、每路名次、RRF 分数和索引版本。原始数据、缓存、索引和实验权重需要在本地准备，仓库只保存代码、聚合报告和两份历史排序权重。安装脚本会运行依赖检查和单元测试；数据构建及训练命令见 [复现说明](docs/REPRODUCE_V2.md)。

## 目录

| 路径 | 内容 |
| --- | --- |
| `src/` | 数据处理、召回、排序及评测实现 |
| `src/v2/` | 索引化召回、DCN、内容表示和双塔实验 |
| `configs/` | 模型、数据与候选预算配置 |
| `scripts/` | 运行、演示和结果校验脚本 |
| `tests/` | 单元测试与小数据集成测试 |
| `experiments/` | 聚合指标、训练曲线与校验清单 |
| `docs/` | 方法说明和实验记录 |

## 文档

- [文档索引](docs/README.md)：按运行、方法、结果查找
- [ItemCF 实现](docs/ITEMCF_INDEX.md) · [召回消融](docs/RECALL_ALLOCATION.md)
- [内容表示](docs/REPRESENTATION_EXPERIMENT.md) · [双塔实验](docs/TOWER_CLOSURE.md)
- [项目描述](docs/RESUME_READY.md) · [面试问题](docs/RESUME_AND_INTERVIEW.md)
- [聚合结果](experiments/closure_20261022/summary.json) · [训练曲线](experiments/closure_20261022/curves.html) · [审计记录](experiments/closure_20261022/audit.json)

原实验代码与结果按 SHA256 保存；文档修订单独记录。只检查公开文件可运行 `python scripts/verify_repository.py`，无需数据或训练权重。
