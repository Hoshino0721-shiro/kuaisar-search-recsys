# 运行环境与恢复

## V2 / 索引化 ItemCF 环境

新版要求 Python 3.10+，推荐使用 Python 3.12 和独立的项目 `.venv`。
本机已验证 Python 3.12.14、PyTorch 2.8.0+cpu 和完整锁定依赖：2026-10-22 发布前 `pip check` 通过，99 项单元测试全部通过。
下面的初始化脚本安装 `requirements-v2-cpu.lock` 中的 CPU 依赖，检查依赖一致性并运行完整单元测试；不训练模型、不重建真实数据索引，也不改动全局 Python / CUDA 环境。

```powershell
# 首次创建时，传入本机 Python 3.12 可执行文件的路径。
.\scripts\setup_v2.ps1 -BasePython 'C:\path\to\Python312\python.exe'
# 已安装依赖后，仅检查环境并运行测试：
.\scripts\setup_v2.ps1 -SkipInstall
# 直接运行模块时也应显式使用项目解释器：
.\.venv\Scripts\python.exe -m src.itemcf_index --help
```

脚本将临时文件放在忽略提交的 `artifacts/environment_tmp` 中，并在结束时恢复当前进程的 TEMP / TMP。
`run_itemcf.ps1` 和 `run_v2.ps1` 默认使用此 `.venv`；这两个入口会执行真实索引构建或训练，不是环境检查命令。
CPU 环境用于新版复现与兼容性检查，不代表复现旧版 CUDA 模型的逐位数值或延迟。

## V1 历史验证环境

已验证环境：Windows、Python 3.9、NumPy 1.26.4、SciPy 1.13.1、scikit-learn 1.6.1、joblib 1.4.2、threadpoolctl 3.5.0、PyTorch 2.8.0.dev20250611+cu128，RTX 5070（约 11.94 GiB 显存）。requirements 中 Torch 为 API 兼容范围，不表示所有版本组合都已验证。

当前机器已具备依赖，无需再安装。迁移时安装 requirements 中依赖并选择适合硬件的 PyTorch 构建。CPU 可以运行模型代码，训练和检索时间会改变；CUDA 同分排名与逐位数值不保证跨版本完全一致。

从项目根目录执行模块。核心依赖只有 NumPy、SciPy、scikit-learn、joblib、threadpoolctl、PyTorch；SQLite 使用 Python 标准库。近似检索为本项目 IVF-flat 实验实现，没有安装 FAISS。

原始 ZIP 约 488 MB，SQLite 约 2.6 GB，特征、模型和实验会继续占用若干 GB，建议预留至少 10 GB。源 ZIP 保持原样，不要求解压。

数据构建若中断，不要把缺少 ready manifest 的部分数据库用于训练。代码拒绝覆盖已有数据库；保留故障文件供检查，在新的明确输出目录恢复数据构建或由人工决定清理，脚本不会自行删除。

`scripts/run_pipeline.ps1` 按完成文件跳过阶段，`-VerifyOnly` 只运行测试、最终冻结检查和本地推理示例。

CTR 训练曾在汇总时遇到 NumPy 整数 JSON 序列化问题；修复后使用已保存的同批模型预测恢复汇总，没有重新选择样本或模型。恢复路径保留为 `python -m src.ranking --summarize`。原训练计时未恢复，汇总中 training_seconds 为 null，不填写推测值。

SGD Logistic Regression 基线达到 15 次迭代上限时出现收敛警告，是迭代预算受限的弱基线；最终模型按验证集 LogLoss 选择为门控 MLP。Windows 物理 CPU 核心探测警告不影响已完成结果，运行脚本设置 LOKY_MAX_CPU_COUNT=8。

冻结后 src 与选中模型及状态文件的 SHA256 会被核验。修改这些文件会触发拒绝复核，需要新建版本和验证协议；不要覆盖 freeze.json 来追求测试分数。
