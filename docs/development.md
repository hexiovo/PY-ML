# 开发指南

## 环境

项目要求 Python 3.12，base 与桌面依赖由 `uv.lock` 固定。请从仓库根目录同步依赖：

```powershell
uv sync --locked --no-editable --no-dev --extra desktop --extra search --extra sequence --extra deep
```

这里有意使用 `--no-editable`：本地包构建安装到 `.venv`，避免 Windows 中文路径和非 UTF-8 locale 下导入源码路径问题。修改 `src/pyml_workbench` 后重新运行同步；在已有环境中若需强制替换已安装本地发行包，可追加 `--reinstall-package pyml-workbench`。运行 GUI、示例与测试时用 `.venv\Scripts\python.exe -X utf8`。

## 检查命令

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -X utf8 examples\synthetic_classification.py
.\.venv\Scripts\python.exe -X utf8 examples\synthetic_dimensionality_reduction.py
.\.venv\Scripts\python.exe -X utf8 examples\synthetic_batch.py
.\.venv\Scripts\python.exe -X utf8 examples\synthetic_extended.py
```

测试使用 unittest。示例在系统临时目录内生成 CSV 和结果，不依赖项目维护用 fixtures，不会改写用户数据。不要把一次性验证脚本或产生的模型/CSV留在源码目录；完成后清理 RUN/tmp 中已登记的临时文件。

## 包结构

- `config.py` 定义数据、划分和单实验配置契约。
- `data.py` 实现 CSV/Excel 读取、预览和显式特征/目标校验。
- `model_catalog.json`、`catalog.py`、`sequence_models.py` 和 `parameters.py` 提供 71 个 scikit-learn 模型及 7 个 HMM/deep 扩展模型的目录、构造和参数 schema。
- `sequence.py` 与 `owned_snapshot.py` 构建并验证分组序列计划、窗口映射和持久 receipt；`extended_experiment.py` 提供 HMM/deep 拟合、评分、refit、导出、加载和推理后端。
- `preprocessing.py` 构建在训练分区拟合的特征预处理；`experiment.py` 将传统与扩展模型接入验证、配置冻结、一次性测试、导出和重载流程。
- `search_space.py`、`objectives.py`、`search.py` 和 `selection.py` 提供类型化搜索空间、任务目标、五种搜索适配器与 winner 冻结/最终化。
- `history.py`、`batch.py` 负责 SQLite 队列、预算记账、缓存与恢复；`batch_gui.py` 提供批量配置和比较窗口。
- `gui.py` 提供 PySide6 单任务工作台及“批量搜索队列”入口，`_worker.py` 在独立 Python worker 进程中执行动作；`__main__.py` 是 `python -m pyml_workbench` 的入口。

## 可选搜索后端

NumPy、SciPy、pandas、scikit-learn、threadpoolctl 和 openpyxl 构成 base；openpyxl 用于 XLSX 读取和默认结果导出。桌面 UI、指标图和旧版 XLS 输入由 `desktop` extra 提供。TPE 与遗传搜索分别在执行前按需检查 Optuna 与 pymoo，并由 `search` extra 安装。Grid、随机与退火不需要 Optuna/pymoo。HMM 与深度模型分别由 `sequence` 与 `deep` extras 提供。桌面完整方法可用环境通过 `uv sync --locked --no-editable --no-dev --extra desktop --extra search --extra sequence --extra deep` 安装；缺少可选包时 GUI/API 应在拟合前显示包名和安装命令，不能把未安装的方法报告为可运行。

新增模型时必须沿用获批任务目录，确认上游 estimator、输入条件、真实能力和参数 schema；不要把目录可见性当作功能实现。运行时能力以 estimator 实例检测为准。更新参数索引时从实际 `list_models()`、`model_capabilities()`、`parameter_schema()` 和 `model_catalog.json` 生成，不手工编造默认值。

## 工作流边界

### PDF 图文指南维护

`docs/overall-guide.md` 是正文来源；实际 UI 截图、合成练习 CSV 与来源/哈希位于 `docs/assets/pdf-guide`。截图采集使用 Qt 工作台与合成数据，执行 C01 单任务与三候选 Grid 搜索；设置 `PYTHONPATH=src` 时捕获当前源码 UI，否则捕获已安装包。运行前需要完整桌面环境，使用新的演示目录避免混入旧任务：

```powershell
$env:PYTHONPATH=(Resolve-Path src).Path
.\.venv\Scripts\python.exe -X utf8 -B packaging\capture_pdf_guide.py --work tmp\pdfs\new-demo
Remove-Item Env:PYTHONPATH
```

只刷新当前向导页面时追加 `--wizard-only`：采集数据、预处理、算法/评价、资源、真实运行监控和结果导出页面；使用两个分类模型各完成一次 validation Grid 拟合，并把新截图合并到既有 `screenshots.json`，保留未受界面改动影响的截图。此模式要求已有截图清单和相同的确定性演示数据；每次传入新的 `--work` 目录，避免复用旧结果。

PDF 构建使用独立文档 Python 环境中的 `reportlab`、`Pillow` 和 `pypdf`，以及 Windows 微软雅黑字体，不增加应用运行依赖：

```powershell
python -X utf8 -B packaging\build_pdf_guide.py
pdftoppm -r 120 -png output\pdf\PYML-Workbench-0.4.2-图文指南.pdf tmp\pdfs\page
```

最终 PDF 与结构检查摘要保存在 `output/pdf`；构建后必须查看渲染页面，检查中文、英文代码和表格、图题图注、截图及页码，再清理 `tmp/pdfs`。更新图文指南时同步 README 与 `version.md`。发行脚本会把当前版本 PDF 复制到新 profile 目录的 `output/pdf`，使发行包 README 的相对链接可用；旧发行目录保持原样。

Windows EXE 的入口、spec 与开发构建依赖位于 `packaging`。构建使用独立 Python 3.12.14；项目环境仍可按原 Python 3.12 方式运行。重建与 frozen 检查详见 [EXE 指南](exe-guide.md)。

### Windows 发行 profile

`packaging/profiles.json` 定义发行依赖、模型目录段和 PyInstaller 收集范围。`standard` 收集桌面界面与 71 个 scikit-learn 模型，排除 hmmlearn、torch/skorch、Optuna 和 pymoo；`full` 通过 `sequence`、`deep`、`search` extras 纳入这些扩展。标准版仍包含 Grid、随机与退火搜索逻辑。

在隔离的 Python 3.12.14 环境中，按要构建的 profile 安装项目 extras 和打包工具：

```powershell
uv sync --locked --no-editable --no-dev --extra desktop
python -m pip install -r packaging/build-requirements.txt
python packaging/build_release.py --profile standard --dry-run
python packaging/build_release.py --profile standard
```

构建完整扩展版时，在同步命令上再加 `--extra search --extra sequence --extra deep`，并将两个构建命令中的 `standard` 改为 `full`。构建器检查 Python 必须为 3.12.14；它把 profile 传给 spec，并输出到各自目录和带 profile 名称的 ZIP。若当前 profile 的目录、ZIP 或 PyInstaller 工作目录已存在，构建会在 PyInstaller 启动前退出并保留原文件。可给 `build_release.py` 传入绝对的 `--work-dir`，将临时构建中间文件放到发行输出以外。每次实际构建生成 `delivery/exe-<version>/<profile>-release-result.json`，记录目录和 ZIP 的实测字节数；没有构建记录时不要声称已实测缩小。

- 一次运行使用固定的 60/20/20 划分；预处理只在训练分区拟合。
- 验证结果用于检查当前配置；测试数据只在 `freeze_experiment` 之后通过 `evaluate_test` 使用一次。
- 批量搜索每个组合最多 50 次真实拟合、250 个 proposals 和 20 分钟活动时间；最多并行 2 个 worker，每个数值线程数为 1。搜索只读取训练/验证分区，winner 必须先显式冻结，再用训练+验证数据重拟合并测试一次。
- 固定手动参数也可作为一次候选运行：空 `fields` 的搜索空间只接受 Grid；每个 fixed 和基础模型参数都经 estimator 参数/类型校验，随机/TPE/遗传/退火的空变量空间必须以清楚错误拒绝。
- 保存模型必须包含预处理和 estimator，并保留能力信息。对于未提供新数据 `predict`/`transform` 的算法，报告不可用能力而不是替代拟合或声称支持。
- 只新增由当前获批范围要求的依赖。HMM/deep 可选依赖不得进入 base；它们的 lazy import 应保证仅安装 base 时包根目录、传统模型和 API 可导入。
- 改动文档或功能时更新根目录 `version.md` 中的交付状态和验证范围；独立 S02 验收完成前不得标记 PASS。
