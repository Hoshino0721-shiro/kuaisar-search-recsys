# 复现说明

以下命令在仓库根目录运行。原始数据、特征缓存和大索引需要本地生成；默认RRF不依赖神经排序权重。

## 环境

推荐Python 3.12，已验证环境为3.12.14 / PyTorch 2.8.0+cpu。首次创建环境：

```powershell
.\scripts\setup_v2.ps1 -BasePython 'C:\path\to\Python312\python.exe'
```

脚本安装 `requirements-v2-cpu.lock`，执行 `pip check` 和单元测试。已有依赖时使用 `-SkipInstall`。历史v1环境见 [环境说明](ENVIRONMENT.md)。

## 数据和默认推荐

从 [KuaiSAR](https://kuaisar.github.io/) 下载small版，保存为 `data/raw/KuaiSAR.zip`。文件大小488077651字节，MD5为 `daea8cbf605db6bd5841740f0e4a12d9`。

```powershell
.\.venv\Scripts\python.exe -m src.prepare_data
.\.venv\Scripts\python.exe -m src.ctr_features
.\.venv\Scripts\python.exe -m src.itemcf_index build --config configs/itemcf_index_support2.json --out artifacts/itemcf_support2_300
.\.venv\Scripts\python.exe scripts/demo_recommendation.py --user-id 20235
```

数据构建拒绝覆盖已有产物。推荐统计只使用此前完整自然日，验证使用训练末快照。演示显示最终物品、召回来源名次和RRF分数。

直接查看完整trace，或运行原LR/DCN级联：

```powershell
.\.venv\Scripts\python.exe -m src.v2.recommend --user-id 20235
.\.venv\Scripts\python.exe -m src.v2.recommend --config configs/cascade_v2_recall_optimized.json --user-id 20235
```

级联需要与当前特征状态匹配的排序权重。新环境可通过 `scripts/run_v2.ps1` 生成运行配置；具体说明见 [四阶段链路](CASCADE_V2.md)。

## 训练双塔对照

```powershell
.\.venv\Scripts\python.exe scripts/run_tower_closure.py
```

该命令启动真实训练。主窗口属性/历史组各3个种子，较早窗口各1个种子，最多同时启动3个四线程CPU进程。训练参数、选轮和停止规则见 [双塔实验](TOWER_CLOSURE.md)。

回溯评测需要截止2023-05-28的ItemCF图；脚本校验已有索引，在缺少时按严格时间前缀构建。输出目录拒绝覆盖已有完成产物，可用 `--root` 指定新目录，用 `--train-only` 只执行训练；完整轮状态支持恢复。

训练并导出后，查看默认方案与增加双塔的对照：

```powershell
.\.venv\Scripts\python.exe scripts/demo_recommendation.py --user-id 20235 --with-tower
```

## 检查结果

只检查仓库公开文件，无需数据和模型：

```powershell
.\.venv\Scripts\python.exe scripts/verify_repository.py
```

它核对原发布清单中的代码、配置和报告，以及单独记录的文档修订。修订前文档保存在 `experiments/documentation_20261023/frozen/`，原发布清单和实验结果未改写。

生成自己的训练结果后，可分别执行：

```powershell
.\.venv\Scripts\python.exe scripts/test_closure.py
.\.venv\Scripts\python.exe scripts/audit_tower_closure.py --root artifacts/YOUR_RUN
.\.venv\Scripts\python.exe scripts/summarize_tower_closure.py --root artifacts/YOUR_RUN
```

`test_closure.py` 运行单元和小数据夹具，不训练真实数据。审计与汇总读取对应目录的已完成实验，输出路径已有文件时会拒绝覆盖；参数以各脚本 `--help` 为准。

`scripts/verify_closure.py` 保留原始冻结快照的校验方式，`--freeze` 只用于新发布目录，已有release时不可重写。历史v1流程 `scripts/run_pipeline.ps1` 包含旧测试步骤；新版实验使用本页入口。
