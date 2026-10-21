# Pairwise 训练对照

日期：2026-10-21。用户授权尝试训练方法变更；只做验证期实验，不读取最终测试标签、不改默认入口、不推送 GitHub。此处在查看新训练结果前固定规则，后续实测结果另附。

状态：最终 30 模型、多种子曝光/同池比较、预设单种子级联及完整性审计已完成。结果见 [实验报告](PAIRWISE_RESULTS.md)；其中后续 full_mixed 和输入覆盖修正均明确记录发生阶段，不将跟进实验伪装成最初预注册。

## 目的与配对契约

保留数值 DCN（A）和属性 DCN（B），不追加用户/物品 ID、历史或新架构。比较 pointwise BCE 与 `softplus(s_negative-s_positive)` 的排序目标。

推荐日志没有可恢复的推荐请求/展示列表 ID。本次按同一用户、固定 30 分钟时间桶建立近似上下文，不能声称同一请求、真实 slate 或相同意图。时间相近也不保证意图相同，点击反馈到达时间仍未知。

仅使用上一轮固定的 100 万条训练曝光；每个桶内先排除同一物品同时有 0/1 标签的记录，再将同标签重复物品折叠为最早时间记录，时间并列按缓存行号确定。每个点击物品均匀抽一个不同物品的真实未点击曝光，允许负例被多个正例抽中；不补跨用户、跨桶或未曝光负例。种子 2026。不将未点击解释为真实厌恶。

缓存时间由只读 SQLite 按原日级/id 顺序重放，完整核对 5752167 条训练记录的用户、物品和标签与旧缓存一致，才提取时间。训练截止与此前相同；验证配对只用于开发评估，不进入训练/词表/标准化。

配对审计：训练 205790 个用户时间桶中 113675 个可用，385577 对、17143 用户、603930 个不同曝光端点；剔除矛盾物品记录 38 条、折叠一致重复 154 条。验证 345061 对、7243 用户。训练时间间隔 P50/P95 为约 5.20/20.62 分钟，最大不足 30 分钟；相同时间戳配对 54127 个，不把它们解释为已知反馈先后或同请求。

## 固定训练协议

每个 A/B 架构各训练以下 4 组，3 个种子 2026/2027/2028，共 24 组：

| 组 | 输入样本与损失 | 用途 |
| --- | --- | --- |
| full_bce | 原固定 100 万条曝光，BCE | 完整曝光基线 |
| paired_bce | 同一份配对的全部正负端点，平衡 BCE | 控制配对筛选、端点重采样的影响 |
| pairwise | 同一配对，softplus(负分−正分) | 隔离排序损失变化 |
| mixed | 0.5×端点 BCE + 0.5×pairwise | 同时约束绝对点击与相对顺序 |

paired_bce/pairwise/mixed 使用完全相同的配对、初始化、批序列、步数，单批 4096 对；full_bce 单批 4096 曝光，步数更多，不宣称与配对组等计算预算。full_bce 与 paired_bce 的差异可揭示采样影响，paired_bce 与 pairwise/mixed 的差异才更接近损失本身的作用。

均最多 5 轮、学习率 0.001，数值网络 Adam、weight decay 1e-5，属性稀疏 embedding SparseAdam 无 weight decay。标准化和词表均复用同一固定训练样本；不利用验证配对扩词表。

所有组统一按固定验证配对的最低 softplus 损失选择轮次，相同值保留最早轮次。与上一轮按曝光 LogLoss 选型不同，因此不把两轮最佳 checkpoint 的差异全归因于 pairwise；本轮重训 full_bce 以保持统一选型规则。训练前固定混合权重 0.5，不遍历权重追验证分。

纯 pairwise 只约束分差，整体分数偏移不被识别；其 sigmoid 不能直接当成校准 CTR。曝光 LogLoss/Brier/ECE 仍报告作为诊断，不用它们悄悄替换预先声明的选型规则。平衡端点 BCE 的概率也不自动等于原流量 CTR。

## 验收

### 曝光采样退化后的受控补充

初轮数值模型显示 paired_bce 的全曝光 AUC 明显降低，而配对损失相近。于是，在第一次查看全量候选/级联结果之前，固定追加 `full_mixed`：保留原 100 万曝光 BCE，额外加 0.5 权重的配对损失；公式为 `0.5×完整曝光 BCE + 0.5×pairwise`，A/B 各 3 种子，共 6 个新模型。此分支不是最初预注册的组，明确属于观察训练/验证曝光结果后的跟进，不隐藏为初始设计。

每批 4096 原曝光，独立随机数生成器均匀有放回抽 4096 对，不影响 full_bce 的原曝光批顺序；每轮优化步数与 full_bce 一致，但前向处理的行数约 3 倍，且每轮配对抽样数不同于端点组。因此只说明保留完整 BCE 后的辅助排序方案，不能宣称所有组等算力。仍固定 0.5 权重、5 轮、同一验证配对选型；不遍历权重。额外级联对照预先加 A/B_full_mixed_s2026。

追加配置为 `configs/pairwise_fullmix_v1.json`；初步合并目录 `artifacts/pairwise_fullmix_v1/models` 通过校验复用首轮 24 个模型，新增 6 个，总计 30 个。首轮目录不修改。实际最终目录与修正见下文。

```powershell
.\.venv\Scripts\python.exe -m src.v2.pairwise_fullmix --out artifacts/pairwise_fullmix_v1/models
.\.venv\Scripts\python.exe -m src.v2.pairwise_evaluate --run artifacts/pairwise_fullmix_v1 --out artifacts/pairwise_fullmix_v1/evaluation
```

### 配对后属性覆盖修正

在首次全量候选评测前发现：端点组只使用约 60 万条不同曝光，原 100 万样本的属性覆盖 mask 不适用。11754 个作者 token 在原样本中至少 5 次、在实际重复配对端点中不足 5 次，其中 181 个从未进入配对；这会让低频/未训练独立向量进入推理。

已在 `pairwise_train.py` 按实际重复端点重新计频，对 paired_bce/pairwise/mixed 设置模型专用覆盖 mask，不足 5 次回退 UNK。全量曝光组、完整 BCE 混合组和无 embedding 的数值组不受影响。

使用 `pairwise_repair.py` 只重训受影响的 9 个 B 模型，校验权重后复用其余 21 个，产物写入 `artifacts/pairwise_final_v1/models`，原试跑目录均保留。最终候选评测只使用这个修正后的 30 模型集合。审计器会逐个验证 checkpoint 的覆盖 mask 等于该模型实际训练输入计频结果，不把错误试跑混入最终表。

新环境使用已修正的源码直接重建，不需要先制造旧错误再修复：

```powershell
.\scripts\setup_v2.ps1 -SkipInstall
.\scripts\run_pairwise.ps1 -RunName pairwise_reproduce
```

脚本拒绝覆盖，复用已存在的增强缓存、固定候选和匹配状态的粗排，不绕过摘要检查。每个阶段均可单独执行；最终目录为新 RunName 下的 `final`。实际本轮最终路径为 `artifacts/pairwise_final_v1`，配对原始缓存为 `artifacts/pairwise_v1/pairs`。

先报告固定曝光 AUC/GAUC/LogLoss/Brier、冷物品分组和验证配对损失/准确率。再在原冻结 8495 用户、每人 2000 候选上比较全部模型的直排 Top20，不给未曝光项补标签。

预先固定级联对照为 A_full_bce_s2026、B_full_bce_s2026、B_mixed_s2026、B_pairwise_s2026；同一 LR 粗排、400/80/20 预算和类别惩罚，同时提供去类别重排的 Top20，避免分数标度变化被误认为排序收益。纯 pairwise 分数的 sigmoid 标度不具 CTR 保证，相同类别惩罚不代表相同多样性约束强度。

列表是固定截止快照的用户周期观察正例代理，配对是近似时间上下文，两者都不是请求级线上收益。收益需跨种子并对照 paired_bce 才解释；无改善同样保留，不预设 pairwise 一定有效。不重复使用旧测试选模型。

## 入口与产物

配置：`configs/pairwise_v1.json`。产物：`artifacts/pairwise_v1/`，只新增、不覆盖已有目录。

```powershell
.\.venv\Scripts\python.exe -m src.v2.pairwise_data --out artifacts/<新目录>/pairs
.\.venv\Scripts\python.exe -m src.v2.pairwise_train --pairs artifacts/<新目录>/pairs --out artifacts/<新目录>/models
```

配对不等于原始请求；即使配对验证改善，也必须继续通过候选和级联检查，不能只报告容易变好的训练目标。
