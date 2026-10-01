# pyml-workbench

`pyml-workbench` 是面向表格和序列数据的 Python 包和中文 Windows 桌面工作台，提供单模型流程与多数据集/多模型验证搜索队列。Python 导入名为 `pyml_workbench`。当前模型目录有 78 个可运行条目：71 个 scikit-learn 传统模型、3 个 HMM 和 4 个深度学习模型。HMM 使用 `sequence` extra；深度模型使用 `deep` extra。

本 README 适用于 **0.4.2（2026-10-02）**：提供 Windows EXE、整体中文指南和应用内离线阅读入口。发布内容、验证范围和已知限制见[版本记录](version.md)。

首次使用请看[整体指南](docs/overall-guide.md)。主窗口顶部点击“整体指南”，或使用“帮助 → 整体指南”/F1，可离线阅读、跳转章节和搜索；指南随 Python 包和 EXE 一起分发。

## 安装与启动

Windows EXE 分发位于 `dist/PYML-Workbench`：双击 `PYML-Workbench.exe`。复制或解压时必须保留整个文件夹，包括 `PYML-Worker.exe` 和 `_internal`，不能只复制主 EXE；使用者不需要另装 Python。Windows 打包与验证说明见[EXE 指南](docs/exe-guide.md)。

需要 64 位 Windows、Python 3.12 和项目锁文件。打开 PowerShell，在项目根目录运行：

```powershell
uv sync --locked --no-editable --no-dev --extra desktop --extra search
.\run_pyml_workbench.bat
```

启动器调用项目 `.venv` 中的 Python，不会安装依赖。也可直接启动：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m pyml_workbench
```

安装为 non-editable：uv 会构建并安装本地包到虚拟环境，而不是从源码目录导入。基础依赖包括 `openpyxl`，供 XLSX 读取和默认结果导出使用；`desktop` extra 提供 Qt 界面、指标图、基础图表和旧版 XLS 输入。完整搜索后端由 `search` extra 提供；只安装基础依赖仍可使用 Grid、随机和退火搜索，TPE 与遗传搜索需要 Optuna/pymoo。要启用 HMM 或深度模型，可分别追加 `--extra sequence` 或 `--extra deep`。修改包源码后需再次同步；开发者命令见[开发指南](docs/development.md)。`-X utf8` 用于让中文路径、日志和 JSON 在 Windows 上使用一致的 UTF-8 编码。

## 当前交付范围

- 导入 CSV、XLSX、XLS 表格，预览内容并选择特征列、目标列和工作表；“分析”菜单可查看完整数据概况，包括列类型、缺失数和描述统计。
- 对已加载表格、单任务结果或已完成批量任务打开基础图表选择器。可用图表按所选数据列与通过绘图清单核验的缓存输出决定，支持直方图、类别计数、散点/相关性、分类混淆矩阵、回归实际值/预测值与残差、聚类、已有降维坐标和 HMM 状态后验图；图表可另存为 PNG 或 SVG。单任务结果缓存需要自己的绘图清单；批量 winner 清单核验 train/validation，最终会话的独立清单用于 test。旧缓存按各自来源停用缺少清单的分区，绘图入口不会自动重训或测试。
- H01/H02/H03 用于 HMM 序列建模；N01/N02 是表格神经网络分类/回归器，N04/N06 是基于滑动窗口的 LSTM/GRU 回归器。序列配置可指定分组、时间顺序、观测列、窗口和预测跨度；模型按训练中确定的早停 epoch 在训练+验证数据上重拟合。
- `SplitConfig.seed` 固定数据划分；HMM 和深度模型的 `random_state` 可显式设为 0 到 `2**32-1` 的整数控制初始化，未提供或设为 `None` 时回退到划分 seed。
- 运行单模型的训练与验证；查看验证指标后，显式冻结当前配置，再执行一次最终测试。
- 从主窗口顶部的“批量搜索队列”配置多个数据集和模型组合，按组合预检、持久记录与恢复；可用 1 或 2 个 worker，每个组合最多 50 次真实拟合、250 个 proposals 和 20 分钟活动时间。搜索只使用训练/验证数据，候选参数以验证目标评分；冻结 validation winner 后，在训练+验证数据上重拟合并执行一次最终测试。
- 导出指标、逐行结果、配置和可重载的拟合模型；新数据推理只开放模型真实支持的操作。
- Python API 支持以 `prepare_experiment → freeze_experiment → evaluate_test` 分开控制验证和最终测试，也提供自动完成一次测试的 `run_experiment` 便捷接口。
- Python 批量 API 提供 `run_batch` 一步预检/入队/运行入口，以及 `load_search_result → freeze_search_winner → finalize_frozen_search` 的恢复与显式最终化流程。
- 在“文件”菜单选择“导出诊断包…”可将当前 GUI 会话或指定 `error_id` 导出为脱敏 ZIP；保存位置由用户选择，程序不会上传或覆盖已有文件。

详细单任务和基础图表操作见[用户指南](docs/user-guide.md)，批量队列、搜索配置和批量结果图见[批量搜索指南](docs/batch-guide.md)，序列配置见[序列建模指南](docs/sequence-guide.md)，深度模型见[深度学习指南](docs/deep-learning-guide.md)，Python 接口见[API 指南](docs/api-guide.md)，78 项模型目录见[模型与参数索引](docs/model-index.md)。[扩展模型合成示例](examples/synthetic_extended.py)覆盖 HMM 与 N06 的验证、冻结、测试、导出和推理。运行时和许可证来源见[第三方组件说明](docs/third-party.md)。版本状态与未覆盖范围见 [`version.md`](version.md)。

本仓库的自动化检查可这样运行：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -X utf8 examples\synthetic_classification.py
.\.venv\Scripts\python.exe -X utf8 examples\synthetic_dimensionality_reduction.py
.\.venv\Scripts\python.exe -X utf8 examples\synthetic_batch.py
.\.venv\Scripts\python.exe -X utf8 examples\synthetic_extended.py
```

示例只创建临时的合成数据，不读取项目夹具或用户文件。当前格式验证覆盖 CSV 与 XLSX；虽提供 XLS 读取路径，本期没有用实际 XLS 样本验证它。
