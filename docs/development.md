# 开发指南

## 环境与启动

项目使用 Python 3.12 和 `uv.lock` 管理依赖。Windows 源码开发环境可在仓库根目录同步桌面界面、搜索、序列、深度学习和各可选模型依赖：

```powershell
uv sync --locked --no-editable --no-dev --extra desktop --extra search --extra sequence --extra deep --extra manifold --extra forecasting --extra xgboost --extra lightgbm --extra catboost
```

`--no-editable` 会将当前项目构建并安装到 `.venv`。修改包源码后重新同步；需要显式替换已安装的本地发行包时，可使用 `uv sync --locked --no-editable --no-dev --reinstall-package pyml-workbench` 并附上所需 extras。运行桌面程序：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m pyml_workbench
```

Windows 用户也可使用仓库根目录的 `run_pyml_workbench.bat`。桌面程序的安装与更新方式见[下载与安装指南](exe-guide.md)。

## 项目结构

- `config.py` 定义数据、划分、交叉验证和单实验配置契约；`data.py` 负责 CSV/Excel 读取、预览和列选择校验。
- `catalog.py`、`model_catalog.json`、`sequence_models.py` 和 `parameters.py` 提供模型目录、构造器与参数 schema。
- `preprocessing.py` 构建训练内填补、缩放、编码及可选降维；`experiment.py` 负责单模型准备、冻结、最终测试和工件导出。
- `cross_validation.py`、`cv_experiment.py` 和 `validation.py` 提供独立 K 折路径以及留出流程上的折评估。
- `sequence.py` 与 `extended_experiment.py` 负责 HMM 和窗口模型的序列计划、训练、验证、重拟合和推理。
- `search_space.py`、`objectives.py`、`search.py`、`selection.py` 和 `batch.py` 提供参数搜索、队列、暂停恢复、winner 冻结和最终化。
- `gui.py`、`batch_gui.py` 和 `evaluation_ui.py` 提供桌面界面；`_worker.py` 供应用内部后台进程调用，不是面向用户的通用命令行。
- `plugins.py`、`plugin_dialog.py`、`environment_check.py` 与 `environment_install.py` 管理可选数据转换插件和用户级依赖检查/安装。
- `docs/overall-guide.md` 是桌面端离线操作指南的正文；Markdown 文档是随发行包提供的规范资料。

## 可选依赖

基础包包含表格机器学习和数据处理依赖。`desktop` 提供 Qt 界面、绘图和旧版 XLS 读取；`search` 提供 Optuna TPE 与 pymoo 遗传搜索；`sequence` 提供 HMM；`deep` 提供 PyTorch/skorch；`manifold` 与 `forecasting` 分别提供 UMAP 和统计时序预测；`xgboost`、`lightgbm`、`catboost` 提供各自的提升树模型。版本以 `uv.lock` 为准。缺少可选依赖时，模型目录可仍然显示该条目，但运行能力由真实环境检查决定。

## 检查与示例

项目测试使用 `unittest`。需要验证本地改动时，在目标环境运行：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```

合成示例可用于查看公开 API 的调用方式；它们不证明模型适用于真实研究数据：

```powershell
.\.venv\Scripts\python.exe -X utf8 examples\synthetic_classification.py
.\.venv\Scripts\python.exe -X utf8 examples\synthetic_dimensionality_reduction.py
.\.venv\Scripts\python.exe -X utf8 examples\synthetic_batch.py
.\.venv\Scripts\python.exe -X utf8 examples\synthetic_extended.py
```

## 配置与模型文档

新增模型或配置字段时，应先更新真实运行时契约，再更新[Python API 指南](api-guide.md)、[模型索引](model-index.md)、[批量搜索指南](batch-guide.md)和[PYML 实验配置技能](../skills/pyml-study-config/SKILL.md)。用目标 checkout 的公开 API、数据划分和工件回执核对文档；模型目录数量、参数构造或文档示例不能替代真实拟合验证。

Windows Release 的依赖收集、安装脚本、启动脚本和归档由 `packaging/` 下的发布工具维护。发行版包含完整运行目录、离线 Markdown 操作指南和技能文件；详细下载与安装步骤见[Windows 指南](exe-guide.md)。
