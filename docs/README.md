# 文档索引

## 从这里开始

- [复现说明](REPRODUCE_V2.md)：环境、数据、索引和演示命令
- [实验总结](PROJECT_FINAL_V2.md)：结果与默认方案的选择
- [实验记录](CURRENT_PROGRESS.md)：各轮对照的简表

## 实现与实验

| 主题 | 方法与结果 |
| --- | --- |
| 数据 | [字段与时间契约](DATA_CONTRACT.md)、[初次审计](FIRST_DATA_AUDIT.md) |
| ItemCF | [索引实现](ITEMCF_INDEX.md)、[召回取量与支持度消融](RECALL_ALLOCATION.md) |
| 推荐输出 | [默认 RRF](RRF_MAIN.md)、[四阶段链路](CASCADE_V2.md) |
| 精排特征 | [属性/ID/历史](DCN_FEATURE_RESULTS.md)、[内容表示与兴趣头](REPRESENTATION_EXPERIMENT.md) |
| 训练方式 | [全量曝光](FULL_EXPOSURE_LOCAL.md)、[选轮与早停](RANKING_CONVERGENCE.md) |
| 排序目标 | [pairwise 初步对照](PAIRWISE_RESULTS.md)、[相同时间戳微调](EXACT_PAIR_FINETUNE.md) |
| 召回特征 | [时效与来源特征](TEMPORAL_ROUTE_FACTORIAL.md)、[拆分与日期回放](ROUTE_ATTRIBUTION.md) |
| 双塔 | [实现](DUAL_TOWER_V2.md)、[属性与历史对照](TOWER_CLOSURE.md) |

完整数字、曲线和摘要位于 [experiments/closure_20261022](../experiments/closure_20261022/summary.json)。

## 其他材料

- [环境](ENVIRONMENT.md)
- [剩余问题](NEXT_RANKING_DIRECTIONS.md)
- [项目描述](RESUME_READY.md)、[面试问题](RESUME_AND_INTERVIEW.md)
- [v1 冻结测试](FINAL_REPORT.md)

较早的计划、恢复记录和测试数量保留在原文件中；当前实现与结果以实验总结和对应报告为准。v1 使用独立测试集，v2 使用开发验证与历史日期回放，两版指标分别列出。
