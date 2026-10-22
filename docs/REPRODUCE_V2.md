# V2 复现与交付

本页只描述当前入口。`scripts/run_pipeline.ps1` 是历史 v1 全流程，会包含旧最终测试步骤，不作为新版开发训练入口。

## 1. 环境与数据

使用项目 `.venv` / Python 3.12 和 `requirements-v2-cpu.lock`。初始化见 [ENVIRONMENT.md](ENVIRONMENT.md)。

将官方小版 ZIP 放入 `data/raw/KuaiSAR.zip`，MD5 `daea8cbf605db6bd5841740f0e4a12d9`。不需要同时使用 v2 大版数据，也不使用 partial。以下命令在项目根目录执行；已有 ready 产物先检查，不覆盖重建。

```powershell
.\.venv\Scripts\python.exe -m src.prepare_data
.\.venv\Scripts\python.exe -m src.ctr_features
.\.venv\Scripts\python.exe -m src.itemcf_index build --config configs/itemcf_index_support2.json --out artifacts/itemcf_support2_300
.\.venv\Scripts\python.exe -m src.v2.recommend --user-id 20235
```

准备程序建立 SQLite 并统计原始切分情况，不等于使用 test 选择模型。CTR 特征训练只用此前完整自然日、验证使用冻结训练状态。新版默认推荐无需 LR/DCN 权重，但当前 Backend 需要上述 CTR `state.npz` 来验证物品库。

新克隆不包含数据、特征状态、2 GiB 量级 ItemCF 索引及新版双塔权重；不能声称即开即用。已发布的小型排序权重只有在特征状态摘要匹配时可用。

独立复现 v2 不需要找回 v1 未发布的大权重。完整旧归档存在时会核验全部冻结资产；归档不完整时只核验旧源码并明确报告缺失，不能声称完整 v1 归档已验证。新的 v2 数据、缓存与权重使用自己的摘要清单，不强迫新训练结果匹配旧权重。

## 2. 最后一轮双塔

```powershell
.\.venv\Scripts\python.exe scripts/run_tower_closure.py
```

此入口真实训练，不是 smoke test。最多三个四线程 CPU 子进程，没有后台可见窗口。主窗口两组各三个种子，较早窗口两组各一个种子；缓存与检查点拒绝覆盖，完整轮可恢复。只需训练时传 `--train-only`。使用新 `--root` 可创建另一套输出，不能混入旧权重。

主窗口 ItemCF 已由第 1 步构建。回溯对照需要 cutoff=1685203200000 的严格前缀图，本机复用：

`artifacts/full_exposure_local_20261022/reference/indexes/1685203200000`

新的独立环境中 `run_tower_closure.py` 会在评测前按相同配置、严格 SQL 前缀构建缺少的历史图并校验已有图；不会用主窗口图替代，也不自动跳过该项或填入结果。

训练/检索设计、选轮、采纳和停止标准见 [TOWER_CLOSURE.md](TOWER_CLOSURE.md)。模型未通过验收不会启用默认向量路。

## 3. 核验与演示

```powershell
# 不训练的环境/单元测试检查
.\scripts\setup_v2.ps1 -SkipInstall
# 默认推荐：历史快照，不是实时在线反馈
.\.venv\Scripts\python.exe -m src.v2.recommend --user-id 20235
# 简洁演示：前20个物品、每路名次、RRF分数、索引版本
.\.venv\Scripts\python.exe scripts/demo_recommendation.py --user-id 20235
# 训练/导出完成后显式查看双塔实验对照；不改默认
.\.venv\Scripts\python.exe scripts/demo_recommendation.py --user-id 20235 --with-tower
```

```powershell
# 全部单元/合成夹具回归，生成机器可读测试报告；不训练真实数据
.\.venv\Scripts\python.exe scripts/test_closure.py
# 全部实验完成后，独立重算/核验（首次执行，拒绝覆盖已有 audit）
.\.venv\Scripts\python.exe scripts/audit_tower_closure.py
.\.venv\Scripts\python.exe scripts/summarize_tower_closure.py
# 仅在结果与文档验收后冻结；不自动推送
.\.venv\Scripts\python.exe scripts/verify_closure.py --freeze
# 后续只检查，不训练、不重新选模型
.\.venv\Scripts\python.exe scripts/verify_closure.py
# 新克隆缺少本地大产物时，仅校验公开代码、文档及聚合报告
.\.venv\Scripts\python.exe scripts/verify_closure.py --public-only
```

本轮实验、142 项测试与独立审计已完成。交付目录为 `experiments/closure_20261022`，不包含逐用户 ID、数组或 embedding 权重。`--freeze` 是制作新交付版本的步骤；已有该版 `release.json` 时拒绝覆盖，日常仅使用不带 `--freeze` 的只读校验。新克隆缺少本地大产物时使用 `--public-only`。校验不会重新训练、访问最终测试标签或修改默认推荐。GitHub 尚需另行明确授权同步。

## 4. 旧版与新版不要混用

v1 冻结证据位于 `experiments/final_test` 与 [FINAL_REPORT.md](FINAL_REPORT.md)，保留原代码/资产摘要。v2 的验证、回溯、双塔和表示学习指标单独报告；不重新跑旧 test 来为新配置选型。最终文档会注明本地大产物路径与可公开聚合结果的复现边界。
