# 当前主方案：两路召回后直接输出RRF前20

2026-10-22。用户决定按当前离线次日点击覆盖目标，直接使用RRF结果。`src.v2.recommend` 不指定配置时默认读取 `configs/recommend_rrf.json`。

当前流程为：近期热门2000、个性化ItemCF1000 → 等权RRF融合并取2000个候选 → 按同一顺序输出前20。每个用户的CF结果不同，因此融合列表仍然具有个性化；RRF本身已完成排序。

该模式不加载LR/DCN权重，不计算CTR排序特征，不执行类别或作者重排。沿用现有训练截止点、物品库、历史种子及索引版本校验。返回trace只包含真实执行的recall/output两步，`mode`为`offline_v2_rrf`，短候选显式报告shortfall，不补入未来物品。

历史级联配置未改写，显式传入原配置仍执行四阶段实验链路。后续模型需要在明确的目标和相同输出预算下超过RRF，再考虑替换主方案。当前选择只基于开发日志覆盖，不能据此宣称RRF的线上CTR优于神经排序。

## 使用

在仓库根目录且本地数据、特征快照和ItemCF索引已准备好时：

```powershell
# 默认主方案
.venv/Scripts/python.exe -m src.v2.recommend --user-id 20235

# 显式指定相同主方案
.venv/Scripts/python.exe -m src.v2.recommend --config configs/recommend_rrf.json --user-id 20235

# 保留的旧级联对照
.venv/Scripts/python.exe -m src.v2.recommend --config configs/cascade_v2_recall_optimized.json --user-id 20235

# 全部验证用户；必须指定尚不存在的报告路径
.venv/Scripts/python.exe -m src.v2.recommend --evaluate-valid --max-users 0 --out experiments/NEW_RRF_RUN/metrics.json
```

直接RRF没有粗排阶段，不能与`--compare-without-coarse`一起使用。新克隆仍需重建本地数据/索引；主方案不依赖排序checkpoint，但并非只克隆代码就能生成真实数据推荐。

本轮真实入口全用户验证报告位于 `experiments/rrf_default_valid_20261022/metrics.json`，复核摘要位于同目录 `audit.json`。使用的仍是反复开发的验证日，不读取最终测试，也不改变历史报告。

111项测试通过；实际默认入口对全部8495名验证用户复现Top20命中1745次、NDCG@20为0.01451187936143447。2000候选覆盖34647次正点击，候选顺序SHA256与此前冻结池完全一致。独立脚本重新读取验证SQLite标签和冻结候选，逐用户重算后全部一致：

```powershell
.venv/Scripts/python.exe -m scripts.audit_rrf_output --report experiments/rrf_default_valid_20261022/metrics.json
```
