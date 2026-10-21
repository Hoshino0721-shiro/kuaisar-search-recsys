# KuaiSAR 搜索与推荐离线实验系统

开发中的四阶段推荐 v2 见 [多阶段算法改造](docs/CASCADE_V2.md)：独立 LR 粗排、DCN 精排、约束重排与逐阶段诊断。v2 已完成首轮真实曝光训练和 256 名验证用户的开发评测，效果尚未超过近期热门，最终验收未完成。下文测试指标全部属于已冻结的 v1。

ItemCF 新增[离线邻居索引与查询组件](docs/ITEMCF_INDEX.md)：分块建图、可配置活跃度/时间/热门度权重、TopK 邻居、不可变版本发布、校验与内存映射查询。组件及召回消融属于验证期开发结果，不沿用下方 v1 最终测试指标。

下一步精排实验见 [DCN 特征增强计划](docs/DCN_FEATURE_PLAN.md)：保留 Cross + Deep，分组加入作者/类目、用户/物品 embedding，再验证推荐与搜索点击历史；包含容量对照、时间约束、冷 ID 处理和固定候选评测。本次仅确定计划，尚未实现或训练增强模型。

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

原始日志、数据库、模型、特征矩阵和候选大文件保存在本地并由 .gitignore 排除，未上传远端。历史实验说明保留在 docs；旧计划和审计中标注“待完成”的状态以最终报告为准。

