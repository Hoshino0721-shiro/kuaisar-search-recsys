# 双塔召回 v2：实现、训练协议与接入

> 本页保留 2026-10-21 的原始实现/准备协议。2026-10-22 最后一轮真实 CPU 训练、8 模型评测与独立审计已完成，新增逐轮全库验证、早停、完整轮恢复和安全 embedding；当前状态及入口以 [TOWER_CLOSURE.md](TOWER_CLOSURE.md) 为准。下方“未训练”“固定三轮 CUDA”不是最新状态，不用来描述最终收尾实验。

日期：2026-10-21。状态：数据、模型、显式训练入口、导出、精确检索、评测及级联接入已实现；本机只准备真实数据并做小型合成夹具训练测试，没有训练真实数据模型，没有新的双塔效果指标。旧双塔的历史结果不能归给本实现。

## 要解决的问题

当前近期热门 + ItemCF 在开发验证日的 Recall@2000 为 9.7700%，但候选关系仍主要来自点击共现和共同热度。双塔用于学习用户历史与物品属性的向量匹配，并检验在固定候选预算下能否提供净增量。它不保证超过现有两路，也不因增加一路就直接成为默认方案。

旧 `src/two_tower.py`、`src/history_tower.py` 保留为历史基线；新实现位于 `src/v2/tower_data.py`、`tower_model.py`、`train_tower.py`、`tower_retrieval.py`。旧版只评测 Top100、含冷用户热门兜底，不能直接拿其指标与新纯双塔或 Top1000 比较。

## 模型与输入

```text
物品塔：ID(32) + 作者(16) + 一级类目(8) + 类型(4)
          → 拼接 60 → Linear128 → ReLU → Linear64 → L2 normalize

用户塔：用户ID(16)
       + 推荐历史的物品塔向量均值(64)
       + 可选搜索历史的物品塔向量均值(64)
       + 两种历史长度 log1p(2)
          → 拼接 146 → Linear128 → ReLU → Linear64 → L2 normalize

分数：用户向量与物品向量的点积
```

推荐和搜索历史共享物品塔，分别池化，PAD 不参与均值。当前使用平均池化，不称为行为顺序建模或 DIN。两塔没有候选相关的用户—物品交叉特征，物品向量可独立预计算。

PAD=0、UNK=1；ID、作者、类目、类型分别建表。训练可对用户/物品 ID 做 0.1 概率的 UNK dropout，降低只记忆 ID 的依赖。作者/类目/类型作为静态属性使用，数据没有其历史属性快照时间。

物品编码接口允许传入 UNK 物品 ID 与已知属性 token，因此具备只用属性编码冷 ID 的接口；**首轮导出的检索库仍严格限定相同 951625 个训练点击物品**。不把接口能力称为已经验证新物品冷启动；扩库需要独立审核物品可用时刻并重新评测。

## 样本与时间规则

训练目标仅使用推荐训练期的真实 click=1 事件，保留重复点击。推荐历史取此前完整北京时间自然日的最近 30 个去重点击；搜索历史取前一完整日最多 20 个去重点击，只保留 VIDEO/IMAGE_ATLAS 且在训练推荐点击词表中的种子。目标当天的推荐或搜索反馈不进入该日样本历史。

同一用户同一天共享上下文，不给每条曝光复制历史。训练按批读取内存映射缓存，未将全部样本和历史放进 GPU。验证推理使用训练截止点的冻结用户状态，不利用验证行为更新 query。

首轮时间协议与当前静态召回一致；分发/搜索会话时间不等于精确反馈到达时间，日级滞后是保守的历史可用性假设，不宣称恢复了逐请求反馈顺序。

## 真实准备结果，不是训练结果

缓存位置：`data/processed/kuaisar_tower_v2`，汇总：`experiments/tower_v2_preparation/summary.json`。

| 项目 | 数值 |
| --- | ---: |
| 推荐正点击训练事件 | 2860013 |
| 用户—日期上下文 | 55990 |
| 推荐训练点击用户 | 21568 |
| 推荐训练点击物品 | 951625 |
| 有推荐历史的训练目标事件 | 2124244 |
| 有可用搜索历史的训练目标事件 | 150711 |
| 冻结 query 用户 | 22170 |
| 冻结 query 中有推荐历史的用户 | 21568 |
| 冻结 query 中有可用搜索历史的用户 | 1876 |
| 完整准备缓存，包括词表和校验 | 约 96 MB |

训练期搜索视频/图集正点击种子记录为 313588 条，其中只有 18258 条（约 5.82%）落在当前推荐点击词表中。这是当前跨场景物品覆盖限制，不能把丢失的种子当作用户没有搜索兴趣。因此搜索历史默认关闭，作为消融项；以后应补充未知搜索物品的作者、类目或匿名文本表示，而不是把当前稀疏通道直接写成有效的搜索迁移。

真实缓存通过文件摘要核验，并按种子 2026 抽取 64 条训练目标、32 个冻结 query，用独立 SQL 时间前缀重放历史，全部一致。审计为同目录 `audit.json`；这是抽样正确性检查，不是模型效果或全行历史证明。

## 损失与负样本语义

一个批次先将目标物品 ID 去重，每个唯一物品编码一次，用户向量与这些候选计算 `dot / temperature`，以各自目标物品作为 softmax 类别。

- 同一个物品的重复目标不会互相充当负样本，也不因多列重复而额外放大。
- 不沿用旧模型的“同一用户在批次中的其他目标都视为正例”规则；不同时间上下文不能直接按用户 ID 合并目标。
- 可用历史中已经点击的物品从该行负样本中屏蔽，当前目标即使是重复点击也保留为正例；只使用当前上下文中已有的推荐/搜索历史，不查未来全量用户正例。
- 其他批内目标是采样对比对象，可能包含未观察的相关物品；不能解释为真实曝光未点击，更不能拿本损失拟合 CTR。
- 默认关闭 logQ 修正。可选项按训练目标频率减 logQ，属于去重候选后的近似修正，不宣称严格无偏；单独消融检验。
- 单一目标物品批次跳过并记录数量。两类优化器分别处理稀疏 embedding 与稠密投影。固定最后一轮 checkpoint，不在训练脚本中读取验证或测试标签选权重。

## 消融矩阵

共同种子 2026、相同训练事件、batch512、固定 3 轮、温度 0.1。默认设备为 CUDA；不可用时明确失败，不自动回退到真实全量 CPU 训练。

| 名称 | 配置 | 输入差异 |
| --- | --- | --- |
| T0 ID | `configs/two_tower_id_only.json` | 用户/物品 ID，无属性和历史输入 |
| T1 属性 | `configs/two_tower_attributes.json` | T0 + 作者、类目、类型 |
| T2 推荐历史，主配置 | `configs/two_tower_v2.json` | T1 + 推荐历史均值 |
| T3 搜索历史 | `configs/two_tower_search_history.json` | T2 + 搜索历史均值 |
| T4 logQ | `configs/two_tower_logq.json` | T2 + 近似频率修正 |

各配置保持同一负样本屏蔽规则，包括已有搜索历史的屏蔽，以隔离输入表示的差异；T3 测的是搜索表示的额外价值，不能据此证明全部搜索信息的价值。相邻输入开关不会改变投影层结构。真实训练、模型比较与参数选择均尚未执行。

## 导出、检索与接入边界

导出不可变索引：物品 ID/向量、冻结用户 ID/query 向量、query 是否有可用信号、文件校验、checkpoint 摘要、数据摘要及版本。不携带全部训练 embedding 表即可做本轮冻结检索；这不是在线新行为编码服务。

当前实现为分块全库精确点积，单块至多 32768 个物品、每批至多 64 个用户，分块保留 TopK，同分按原始 item_id 排序。负的余弦分数也保留有效排名，不按“分数小于 0”删除。没有实现 ANN，不能称为 FAISS 部署或生产延迟优化。先确定向量质量，随后再测 ANN 相对精确检索的损失和成本。

没有可用训练 ID 或已启用历史信号的用户，双塔路返回空，不偷偷混入热门；已有近期热门路承担兜底。未训练或缺少索引时明确报错，不构造随机向量替代。请求早于索引截止时间、文件校验失败或全局物品库与原管线不一致时拒绝接入。导出还核验训练时的模型源码摘要，避免同形状但不同计算语义的实现悄悄改变结果；版本涵盖物品与 query 的全部数组摘要。

纯双塔报告全量、有/无 query 分组的 Recall@100/1000、物品库覆盖上限、query 与物品库同时可达的正例数及覆盖范围。若提供 ItemCF 索引，还会重算当前近期热门+CF 和加入双塔后的同一 2000 预算结果，保存逐用户命中；没有 CF 索引时相关对照明确为缺失。

级联实验配置为 `configs/cascade_v2_tower_experiment.json`，以当前两路为基础增加 tower 路。原 `cascade_v2_recall_optimized.json` 不切换。接入后同时核验全量候选 Recall、逐阶段保留和最终同 K 排序质量，不能只依据训练 loss 或与弱热门对照的收益。

## 明确执行步骤

以下命令从项目根目录执行。Python 命令可在训练机器的相应虚拟环境中运行；本机当前只执行过准备和审计，没有执行真实训练。每个输出路径须不存在。

```powershell
# 本机已完成；其他机器从已准备的 SQLite 重新构建时使用新路径
./.venv/Scripts/python.exe -m src.v2.tower_data --out data/processed/kuaisar_tower_v2
./.venv/Scripts/python.exe scripts/audit_tower_data.py --prepared data/processed/kuaisar_tower_v2 --out experiments/tower_v2_preparation/audit.json

# 在选定训练机器上显式执行；默认 CUDA，不自动回退到 CPU
python -m src.v2.train_tower train --data data/processed/kuaisar_tower_v2 --config configs/two_tower_v2.json --out artifacts/tower_v2_training
python -m src.v2.train_tower export --data data/processed/kuaisar_tower_v2 --checkpoint artifacts/tower_v2_training/model.pt --out artifacts/tower_v2_index --device cuda

# 训练及导出只需要准备缓存、代码和依赖，可携带同一份缓存以保持摘要一致
# 把导出索引带回匹配原始数据快照的评测机器后，运行全库及当前强基线对照
python -m src.v2.tower_retrieval --index artifacts/tower_v2_index --itemcf-index artifacts/itemcf_support2_300 --out experiments/tower_v2_valid
python -m src.v2.recall_diagnostics --config configs/cascade_v2_tower_experiment.json --out experiments/tower_v2_routes.json
python -m src.v2.recommend --config configs/cascade_v2_tower_experiment.json --user-id 20235
```

双塔训练缓存、未来 checkpoint、向量索引均受现有 `.gitignore` 排除，留在本地或指定训练机器；本轮不存在真实双塔权重。仓库另行发布已有的小型 LR/DCN checkpoint，详见 `docs/CURRENT_PROGRESS.md`，不能将其当作双塔产物。训练与导出拒绝覆盖已有产物；失败时保留中间目录，不自动删除。

## 已验证与后续验收

全部 71 项测试通过。新增测试覆盖日级历史、当前日期/未来反馈排除、训练词表、PAD、属性冷 ID 接口、重复目标、已知历史负样本屏蔽、确定性检索、负分候选、索引校验、无信号用户空路，以及合成数据上的训练→导出→检索→当前 CF 对照→级联接入。

这些测试只验证实现正确性。后续仍需真实训练、多种子、其他时间窗口、作者/类目/历史消融、重复点击与新物品发现分组、固定预算净增量、检索成本和最终 Top20 验证。当前没有新的双塔 Recall，不能沿用旧双塔或 ItemCF 的指标。

参考：[Google 双塔检索架构](https://docs.cloud.google.com/architecture/implement-two-tower-retrieval-large-scale-candidate-generation)、[Meta Explore 多路召回](https://engineering.fb.com/2023/08/09/ml-applications/scaling-instagram-explore-recommendations-system/)。
