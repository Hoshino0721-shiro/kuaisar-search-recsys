# 可训练数据说明

数据库：`data/processed/kuaisar_small_v1/dataset.sqlite`

主视图：`clean_events`。字段包含场景、源行号、用户、物品、整数毫秒时间、二值 label、原始 click_cnt、搜索会话、关键词、来源、搜索结果类型、split，以及 `missing_item_features`。

时间切分使用北京时间半开区间：训练 `< 2023-05-29 00:00:00+08:00`，验证 `[2023-05-29, 2023-05-30)`，测试 `>= 2023-05-30`。

推荐标签为 `click`。搜索排序标签同时保留 `click_cnt`，二值排序标签为 `click_cnt > 0`；不要把搜索多点击计数当作推荐 CTR 标签。

默认排除重复源记录、重复事件键，以及 4 个会话内时间/关键词/来源不一致的搜索会话。原始记录仍在 `events` 表中，排除原因写在 `exclusion` 字段。

搜索物品特征使用 left join。缺失物品特征不删除，`missing_item_features=1`；搜索测试集的覆盖率和长尾指标需分组报告。

训练历史必须按目标事件时间严格过滤，只使用 `timestamp_ms < target timestamp` 的行为。当前数据库没有预计算用户历史，避免未来信息泄漏。
