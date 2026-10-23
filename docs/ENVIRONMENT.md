# 运行环境

## 当前版本

已验证Python 3.12.14、PyTorch 2.8.0+cpu，完整依赖见 `requirements-v2-cpu.lock`。SQLite使用Python标准库，主要计算依赖NumPy、SciPy、scikit-learn和PyTorch。

```powershell
# 首次安装
.\scripts\setup_v2.ps1 -BasePython 'C:\path\to\Python312\python.exe'
# 已安装依赖，检查环境并运行单元测试
.\scripts\setup_v2.ps1 -SkipInstall
```

脚本使用项目 `.venv`，不修改全局Python或CUDA配置，临时目录位于 `artifacts/environment_tmp`。新版代码要求Python 3.10以上，复现建议使用3.12。

原始ZIP约488MB，SQLite约2.6GB。ItemCF索引约2.14GiB，训练缓存、预测数组和权重会继续占用空间；只运行默认推荐与复现全部对照所需磁盘不同。数据与运行顺序见 [复现说明](REPRODUCE_V2.md)。

当前检索使用精确计算：ItemCF读取离线邻居，双塔分块计算点积。新版未使用FAISS。

## 历史v1环境

v1报告使用过Windows、Python 3.9、NumPy 1.26.4、SciPy 1.13.1、scikit-learn 1.6.1及PyTorch 2.8.0.dev20250611+cu128，硬件为RTX 5070。该环境的逐位数值和延迟不作为当前CPU版本的保证。

历史训练曾在JSON汇总时遇到NumPy整数序列化问题，后续从已有预测恢复结果，未重新选择模型；对应训练耗时缺失项仍为null。v1完整记录见 [最终报告](FINAL_REPORT.md)。
