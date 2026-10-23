# 默认推荐：近期热门与ItemCF

`src.v2.recommend` 默认读取 `configs/recommend_rrf.json`：

```text
近期热门2000＋ItemCF1000 → 等权RRF → 2000候选 → 前20
```

近期热门按前一天训练点击量排序，ItemCF根据用户最近行为产生个性化候选。融合分数为各路 `1/(60+rank)` 之和，同分按物品ID排序。两路共同支持或单路名次靠前的物品可能获得较高分数。

主验证集8495位用户中，前2000候选覆盖34647次正点击，前20覆盖1745次，周期NDCG@20为0.01451187936。候选顺序摘要与各轮固定池对照一致，见 [指标](../experiments/rrf_default_valid_20261022/metrics.json) 和 [重算记录](../experiments/rrf_default_valid_20261022/audit.json)。

## 运行

准备数据、特征状态和ItemCF索引后：

```powershell
.venv/Scripts/python.exe -m src.v2.recommend --user-id 20235
```

输出包含 `recall` 和 `output` 两步，不调用LR/DCN或类别重排。原实验链路可显式指定：

```powershell
.venv/Scripts/python.exe -m src.v2.recommend --config configs/cascade_v2_recall_optimized.json --user-id 20235
```

验证命令需指定新的输出文件：

```powershell
.venv/Scripts/python.exe -m src.v2.recommend --evaluate-valid --max-users 0 --out experiments/NEW_RRF_RUN/metrics.json
```

当前选择依据固定快照的次日点击覆盖。DCN和双塔的局部增益尚未稳定改善最终列表，详见 [实验总结](PROJECT_FINAL_V2.md)。
