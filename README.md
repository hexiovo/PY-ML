# pyml-workbench

“格式 → 数据转换”提供 HyperEEG 和 PYfNIRs 入口，可在统一的“插件设置…”页同时配置两类插件。首次启动时两个入口默认置灰，只有对应插件目录保存成功且通过清单检查后才会启用；设置保存在同一用户配置中并在重启后恢复。路径失效时对应入口重新置灰并显示原因。两套转换器默认只需文件/分组表和数据目录，其余参数位于可选设置，参考表格可在窗口中编辑、选文件、增删行列并保存。插件代码位于独立仓库 [PY-ML_support_trans](https://github.com/hexiovo/PY-ML_support_trans)。

`pyml-workbench` 是面向表格和序列数据的 Python 包和中文 Windows 桌面工作台，提供单模型流程与多数据集/多模型验证搜索队列。Python 导入名为 `pyml_workbench`。当前目录有 84 个可运行模型：71 个 scikit-learn 模型、6 个可选提升树模型、3 个 HMM 和 4 个深度学习模型。提升树模型各自使用独立 extra；HMM 使用 `sequence` extra，深度模型使用 `deep` extra。

本 README 记录 **0.4.5（2026-10-26 GitHub 发布修复）**：本轮增加统一多插件设置、旧配置迁移、保存后立即更新转换入口状态，并在启动时恢复有效配置；此前 0.4.4 版本加入单任务与批量进度、设置和结果的跨启动恢复，以及安全退出，中断的计算步骤整步重新开始。标准版继续不打包提升树 SDK。发布内容、验证范围和已知限制见[版本记录](version.md)。

GitHub 页面中的 **Code → Download ZIP** 下载的是源码；Windows 用户请下载 [Release 程序包](https://github.com/hexiovo/PY-ML/releases/tag/v0.4.5)。启动后，在“格式 → 数据转换”中可看到 HyperEEG、PYfNIRs 和“插件设置…”。转换器是独立插件，请从 [转换插件仓库](https://github.com/hexiovo/PY-ML_support_trans) 获取对应发行目录，在“插件设置…”中分别选择包含 `pyml-plugin.json` 的目录并保存；有效配置会启用对应入口，并在重启后保留。

首次使用请看[操作指南](docs/overall-guide.md)。主窗口底部点击“操作指南”，或使用“帮助 → 操作指南”/F1，可离线阅读、跳转章节和搜索；指南随 Python 包和 EXE 一起分发。

“帮助 → 检查运行环境…”可检查当前程序的 Python、依赖版本、实际加载结果、后台程序和必要资源，区分必需组件与可选算法。支持停止、重新检查和复制报告；报告提供修复建议，检查在独立进程中执行。

带截图的[PDF 操作指南](output/pdf/PYML-Workbench-0.4.5-图文指南.pdf)包含 13 章、真实界面截图、可点击目录、章节书签和页码，适合离线阅读或打印。配套[合成演示数据](docs/assets/pdf-guide/demo-data.csv)可用于练习分类流程；截图中的指标仅说明操作。

## 安装与启动

Windows 发行支持三个 PyInstaller profile：`standard` 保留桌面向导、71 个 scikit-learn 模型及 Grid/随机/退火搜索；`deep` 在此基础上加入 PyTorch/skorch 深度模型；`full` 额外包含 HMM、TPE 与遗传搜索。三个 profile 都不包含 XGBoost、LightGBM 或 CatBoost SDK。依赖清单见 `packaging/profiles.json`。在 Python 3.12.14 打包环境安装对应 extras 与 `packaging/build-requirements.txt` 后，运行 `python packaging/build_release.py --profile standard`、`--profile deep` 或 `--profile full`；可加 `--work-dir <绝对的新临时目录>` 将 PyInstaller 中间文件放在发行输出以外，并可加 `--dry-run` 查看命令。新输出目录与 ZIP 带有 profile 名称，保留既有发行目录。实际文件大小以各次 `release-result.json` 为准；未构建的 profile 不代表已测量体积变化。

0.4.5 标准深度版 Windows EXE 可从 [GitHub Release 下载](https://github.com/hexiovo/PY-ML/releases/tag/v0.4.5)，或直接下载 [Windows ZIP 发行包](https://github.com/hexiovo/PY-ML/releases/download/v0.4.5/PYML-Workbench-0.4.5-windows-x64-deep.zip)。解压后的程序目录为 `PYML-Workbench-deep`：双击 `PYML-Workbench.exe`。本机 `F:/桌面/程序/APP/PY-ML 机器学习工作台.exe` 入口指向该标准深度版，包含 CPU PyTorch/skorch；CUDA 尚未验证。新版包含操作指南、悬停菜单、运行环境检查、GitHub Issue 联系入口和适配 Windows 缩放的启动动画。2026-10-11 修正任务栏默认图标：主窗口显示前登记图标来源、名称、重新启动命令及独立身份。复制或解压时必须保留整个文件夹，包括 `PYML-Worker.exe` 和 `_internal`，不能只复制主 EXE；使用者不需要另装 Python。Windows 打包与验证说明见[EXE 指南](docs/exe-guide.md)。

单任务和批量搜索会在用户数据目录保存基本设置与已完成的进度、结果；发现记录时可选择恢复或新建。安全退出会停止当前计算并自动保存此前完成的步骤；例如 500 次迭代中途退出，恢复时这一整步从第 1 次重新计算。恢复已完成的批次只加载结果；选择新建且设置有效时，会开始独立的新批次。

主向导多模型验证默认每个算法最多 10 个候选模型、30 个 proposals 和 10 分钟，运行 1 个 worker；候选上限可提高到 50。候选数不是底层拟合调用数：K 折下每个候选通常训练 K 次，用户冻结参数后还会额外在完整开发集拟合一次。候选数、proposals 或时限任一预算先到即停止，取消也可提前结束。

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
- 从主窗口底部的“批量搜索队列”配置多个数据集和模型组合，按组合预检、持久记录与恢复；新任务默认每个组合最多 10 个候选评估、30 个 proposals 和 10 分钟活动时间，运行 1 个 worker；可提高到 50 个候选评估、250 个 proposals、20 分钟和 2 个 worker。主向导与批量队列的搜索预算都按候选评估次数计：每个有效、未命中缓存并开始评估的候选计一次，评估失败也占一次；无效提案和缓存命中不占。该计数不等于底层模型拟合次数：留出法下每个候选通常拟合一次，K 折下每个候选通常进行 K 次折内训练。搜索只使用训练/验证数据，候选参数以验证目标评分；冻结 validation winner 后，还会在完整开发集上重拟合并执行一次最终测试，这两步不计入搜索候选预算。
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
