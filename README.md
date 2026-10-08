# pyml-workbench

`pyml-workbench` 是面向表格和序列数据的 Python 包和中文 Windows 桌面工作台，提供单模型流程与多数据集/多模型验证搜索队列。Python 导入名为 `pyml_workbench`。当前目录有 84 个可运行模型：71 个 scikit-learn 模型、6 个可选提升树模型、3 个 HMM 和 4 个深度学习模型。提升树模型各自使用独立 extra；HMM 使用 `sequence` extra，深度模型使用 `deep` extra。

本 README 记录 **0.4.4 断点恢复版（2026-10-07）**：在原 0.4.4 功能上加入单任务与批量进度、设置和结果的跨启动恢复，以及安全退出；中断的计算步骤整步重新开始。标准版继续不打包提升树 SDK。发布内容、验证范围和已知限制见[版本记录](version.md)。

首次使用请看[操作指南](docs/overall-guide.md)。主窗口底部点击“操作指南”，或使用“帮助 → 操作指南”/F1，可离线阅读、跳转章节和搜索；指南随 Python 包和 EXE 一起分发。

带截图的[PDF 操作指南](output/pdf/PYML-Workbench-0.4.4-图文指南.pdf)包含 13 章、真实界面截图、可点击目录、章节书签和页码，适合离线阅读或打印。配套[合成演示数据](docs/assets/pdf-guide/demo-data.csv)可用于练习分类流程；截图中的指标仅说明操作。

## 安装与启动

后续 Windows 发行支持两种 PyInstaller profile：`standard` 保留桌面向导、71 个 scikit-learn 模型及 Grid/随机/退火搜索；`full` 额外包含 HMM、PyTorch/skorch 深度模型、TPE 与遗传搜索。profile 依赖清单见 `packaging/profiles.json`。在 Python 3.12.14 打包环境安装对应 `desktop` 或全部 extras 及 `packaging/build-requirements.txt` 后，运行 `python packaging/build_release.py --profile standard` 或 `--profile full`；可加 `--work-dir <绝对的新临时目录>` 将 PyInstaller 中间文件放在发行输出以外，并可加 `--dry-run` 查看命令。新输出目录与 ZIP 带有 profile 名称，保留既有 0.4.2 发行目录。实际文件大小以各次 `release-result.json` 为准；未构建的 profile 不代表已测量体积变化。

0.4.4 标准版 Windows EXE 可从 [ZIP 发行包](dist/PYML-Workbench-0.4.4-windows-x64-standard-ui-polish-20261008.zip) 获取，程序目录位于 `dist/ui-polish-20261008/PYML-Workbench-standard`：双击 `PYML-Workbench.exe`。本机也可使用 `F:/桌面/程序/APP/PY-ML 机器学习工作台.exe` 入口。新版包含操作指南、悬停菜单、GitHub Issue 联系入口和适配 Windows 缩放的启动动画。原有发行保留。复制或解压时必须保留整个文件夹，包括 `PYML-Worker.exe` 和 `_internal`，不能只复制主 EXE；使用者不需要另装 Python。Windows 打包与验证说明见[EXE 指南](docs/exe-guide.md)。

单任务和批量搜索会在用户数据目录保存基本设置与已完成的进度、结果；发现记录时可选择恢复或新建。安全退出会停止当前计算并自动保存此前完成的步骤；例如 500 次迭代中途退出，恢复时这一整步从第 1 次重新计算。恢复已完成的批次只加载结果；选择新建且设置有效时，会开始独立的新批次。

需要 64 位 Windows、Python 3.12 和项目锁文件。打开 PowerShell，在项目根目录运行：

```powershell
uv sync --locked --no-editable --no-dev --extra desktop --extra search
.\run_pyml_workbench.bat
```

启动器调用项目 `.venv` 中的 Python，不会安装依赖。也可直接启动：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m pyml_workbench
```

安装为 non-editable：uv 会构建并安装本地包到虚拟环境，而不是从源码目录导入。基础依赖包括 `openpyxl`，供 XLSX 读取和默认结果导出使用；`desktop` extra 提供 Qt 界面、指标图、基础图表和旧版 XLS 输入。完整搜索后端由 `search` extra 提供；只安装基础依赖仍可使用 Grid、随机和退火搜索，TPE 与遗传搜索需要 Optuna/pymoo。提升树库不会随基础、`standard` 或 `full` 配置自动安装；需要时，在同步命令中分别追加 `--extra xgboost`、`--extra lightgbm` 或 `--extra catboost`。应用启动时只检查可选包是否可发现，缺包时不会自动安装，并会说明对应 extra。要启用 HMM 或深度模型，可分别追加 `--extra sequence` 或 `--extra deep`。修改包源码后需再次同步；开发者命令见[开发指南](docs/development.md)。`-X utf8` 用于让中文路径、日志和 JSON 在 Windows 上使用一致的 UTF-8 编码。

## 当前交付范围

- 导入 CSV、XLSX、XLS 表格，预览内容并选择特征列、目标列和工作表；“分析”菜单可查看完整数据概况，包括列类型、缺失数和描述统计。
- 对已加载表格、单任务结果或已完成批量任务打开基础图表选择器。可用图表按所选数据列与通过绘图清单核验的缓存输出决定，支持直方图、类别计数、散点/相关性、分类混淆矩阵、回归实际值/预测值与残差、聚类、已有降维坐标和 HMM 状态后验图；图表可另存为 PNG 或 SVG。单任务结果缓存需要自己的绘图清单；批量 winner 清单核验 train/validation，最终会话的独立清单用于 test。旧缓存按各自来源停用缺少清单的分区，绘图入口不会自动重训或测试。
- C25–C27 与 R24–R26 分别提供 XGBoost、LightGBM、CatBoost 的分类和回归模型。每个库有单独的 `xgboost`、`lightgbm`、`catboost` extra，模型使用 CPU 和单数值线程作为默认值。没有对应库时，模型状态会显示缺少的 extra，训练不会自动安装。
- 新训练的分类缓存可在有真实类别分数时绘制逐类 ROC 和精确率-召回率曲线；缓存标明概率或非概率决策分数、类别顺序及行位置。多分类决策分数只在 OvR 语义可确认时启用。支持的监督模型可绘制 `feature_importances_` 或有正负方向的 `coef_`，特征名来自实际拟合的变换器输出。旧缓存保留混淆矩阵等基础图；没有分数的旧缓存会说明需重新训练才能绘 ROC/PR。
- 批量队列新增同条件模型指标图，只比较 `_comparison_group` 判定身份完全匹配、任务完成且有成功 winner 分数的记录；数据哈希、实际划分哈希、特征/目标、指标、方向和评分分区不一致时不会放在同一图中。
- H01/H02/H03 用于 HMM 序列建模；N01/N02 是表格神经网络分类/回归器，N04/N06 是基于滑动窗口的 LSTM/GRU 回归器。序列配置可指定分组、时间顺序、观测列、窗口和预测跨度；模型按训练中确定的早停 epoch 在训练+验证数据上重拟合。
- `SplitConfig.seed` 固定数据划分；HMM 和深度模型的 `random_state` 可显式设为 0 到 `2**32-1` 的整数控制初始化，未提供或设为 `None` 时回退到划分 seed。
- 运行单模型的训练与验证；查看验证指标后，显式冻结当前配置，再执行一次最终测试。
- 从主窗口底部的“批量搜索队列”配置多个数据集和模型组合，按组合预检、持久记录与恢复；可用 1 或 2 个 worker，每个组合最多 50 次真实拟合、250 个 proposals 和 20 分钟活动时间。搜索只使用训练/验证数据，候选参数以验证目标评分；冻结 validation winner 后，在训练+验证数据上重拟合并执行一次最终测试。
- 导出指标、逐行结果、配置和可重载的拟合模型；新数据推理只开放模型真实支持的操作。
- Python API 支持以 `prepare_experiment → freeze_experiment → evaluate_test` 分开控制验证和最终测试，也提供自动完成一次测试的 `run_experiment` 便捷接口。
- Python 批量 API 提供 `run_batch` 一步预检/入队/运行入口，以及 `load_search_result → freeze_search_winner → finalize_frozen_search` 的恢复与显式最终化流程。
- 在“文件”菜单选择“导出诊断包…”可将当前 GUI 会话或指定 `error_id` 导出为脱敏 ZIP；保存位置由用户选择，程序不会上传或覆盖已有文件。
- 在“文件”菜单选择“打开统一操作日志…”可查看持续追加的 UTF-8 应用操作记录；源码运行时文件位于项目目录，打包运行时位于 EXE 目录。

详细单任务和基础图表操作见[用户指南](docs/user-guide.md)，批量队列、搜索配置和批量结果图见[批量搜索指南](docs/batch-guide.md)，序列配置见[序列建模指南](docs/sequence-guide.md)，深度模型见[深度学习指南](docs/deep-learning-guide.md)，Python 接口见[API 指南](docs/api-guide.md)，84 项模型目录见[模型与参数索引](docs/model-index.md)。[扩展模型合成示例](examples/synthetic_extended.py)覆盖 HMM 与 N06 的验证、冻结、测试、导出和推理。运行时和许可证来源见[第三方组件说明](docs/third-party.md)。版本状态与未覆盖范围见 [`version.md`](version.md)。

本仓库的自动化检查可这样运行：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -X utf8 examples\synthetic_classification.py
.\.venv\Scripts\python.exe -X utf8 examples\synthetic_dimensionality_reduction.py
.\.venv\Scripts\python.exe -X utf8 examples\synthetic_batch.py
.\.venv\Scripts\python.exe -X utf8 examples\synthetic_extended.py
```

示例只创建临时的合成数据，不读取项目夹具或用户文件。当前格式验证覆盖 CSV、XLSX 和本轮使用 xlwt 生成的合成 OLE/BIFF8 `.xls` 样本；经应用导入核对两个工作表、六个字段、样本值及类型。这不代表对所有 XLS 文件的完整兼容保证。
