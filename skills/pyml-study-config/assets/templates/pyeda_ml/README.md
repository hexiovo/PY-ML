# PYEDA 一站式转换、验证与报告

将 `pyeda_ml_config.example.json` 复制到自己的分析目录后填写来源。所有步骤使用同一个已安装 PYML 的 Python 3.12 环境。先在该环境安装本目录 `requirements.txt` 中的报告依赖。

```powershell
python -m pip install -r requirements.txt
python -X utf8 pyeda_ml_pipeline.py --config <config.json> --validate-only
python -X utf8 pyeda_ml_pipeline.py --config <config.json>
```

无参数打开图形界面。Windows 的一键入口使用当前仓库的虚拟环境或 `PYML_PYTHON` 选择的解释器。VBS 可接收配置文件路径作为唯一参数。每次运行会建立新输出目录。运行时记录缺少的可选算法及其锁定版本，按需要在同一隔离环境安装后重新运行。

- `validation` 为默认模式，可比较多个结局、方法与轮次，只保存训练和开发集验证。普通分类保存源标签及其文本编码映射。IOC 必须明确指定事件表和连接键，保留匹配率及排除数。
- 时序预测必须显式提供可核验的 `machine_learning.time_column`。窗口回归依赖已登记的记录表、事件顺序和 PYML 的原生序列计划。
- `final` 要求单一结局、一个明确模型、一轮，并设置 `confirm_final_holdout: true`。仅在研究方案已冻结且待测试数据独立预留后使用。配置示例中的多方法运行属于开发验证。
- `feature_columns` 只接受转换字典内的特征。通过 `valid_mask.csv` 核对样本键及有限值。全部缺失的特征由 `all_missing_feature_policy` 明确处理。
- 可在 `study` 中提供 `title`、`author`、`affiliation`、`research_question` 和 `methods`。仅填写本研究有依据的信息。生成 DOCX、PDF、Markdown、图表、结果明细和证据哈希。报告页的视觉验收状态由研究者核对。

运行目录包含验证或最终结果、`method-matrix.csv`、`oneclick-summary.json`、报告目录和 `integrity.json`。只有所有计划步骤和报告成功完成时 CLI 返回 0。部分完成、报告失败或全失败会保留已生成文件及状态明细，并返回非零值。
