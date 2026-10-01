# Python API 指南

本指南对应 **0.4.0（2026-10-02）**。基础图表 API 仅消费调用方提供的不可变来源数据；payload 构造不读文件、不启动模型操作，Matplotlib 仅在渲染时导入。

从 `pyml_workbench` 导入公开配置、数据、目录和实验接口。安装并启动虚拟环境后，运行示例：

```powershell
.\.venv\Scripts\python.exe -X utf8 examples\synthetic_classification.py
.\.venv\Scripts\python.exe -X utf8 examples\synthetic_dimensionality_reduction.py
.\.venv\Scripts\python.exe -X utf8 examples\synthetic_batch.py
```

分类示例生成临时合成 CSV，依次调用 `prepare_experiment`、`freeze_experiment` 和 `evaluate_test`，导出模型，再通过能力检查重载并预测。降维示例用临时合成数据运行 t-SNE，展示模型不支持 `transform` 时验证/测试指标的 `not_supported` 状态。

## 准备配置与读取数据

核心数据配置由 `DatasetConfig`、`SplitConfig` 和 `ExperimentConfig` 组成。`DatasetConfig` 接受 CSV/XLSX/XLS 路径、可选工作表名、目标列名和特征列；监督任务必须指定目标列。`SplitConfig` 默认使用 seed 42 与固定的 60/20/20 划分。`ExperimentConfig` 再指定任务、模型 ID、参数、可选序列配置和输出目录。

```python
from pyml_workbench import DatasetConfig, ExperimentConfig, SequenceConfig, SplitConfig

config = ExperimentConfig(
    dataset=DatasetConfig(
        source_path="synthetic.csv",
        target_column="target",
        feature_columns=("signal", "category"),
    ),
    task="classification",
    model_id="C01",
    parameters={"max_iter": 500},
    split=SplitConfig(seed=42),
    output_dir="run-output",
)
```

`ExperimentConfig.sequence` 可选接收 `SequenceConfig`，用于 HMM 和窗口回归模型。未设置分组/时间列时，源文件行顺序表示一条序列；指定分组后会将整组分配到同一个 train、validation 或 test 分区，指定时间列后会在每组内稳定排序。`preview_dataset(path, rows=20)` 返回样本、列名/类型、缺失数和 Excel 工作表；`load_dataset(path)` 返回完整的 `LoadedDataset`。`select_features_target(dataset, task=..., target_column=..., feature_columns=...)` 返回列副本并拒绝缺失监督目标。读取函数不会写入原始文件。

## 显式控制验证与最终测试

需要检查验证结果并在看到结果后确认配置时，使用三阶段 API：

```python
from pyml_workbench import (
    evaluate_test,
    freeze_experiment,
    prepare_experiment,
)

session = prepare_experiment(config)
print(session.metrics["validation"])

# 可将当前 GUI/API 中的配置传入，要求它与训练快照完全一致。
freeze_experiment(session, expected_config=config)
result = evaluate_test(session)
print(result.metrics.get("test"))
print(result.audit["test_evaluation_count"])
```

- `prepare_experiment(config, *, on_event=None) -> ExperimentSession` 验证配置、建立共同划分、只在训练分区拟合预处理和 estimator，并计算训练/验证结果。它不计算测试指标，也不让测试结果参与选择。
- `freeze_experiment(session, expected_config=None) -> ExperimentSession` 校验训练配置快照和 SHA-256；如果传入 `expected_config`，也必须与拟合配置相同。它记录对当前模型/参数快照的显式批准。
- `evaluate_test(session, *, on_event=None, export_artifacts=True) -> ExperimentResult` 只接受已冻结会话。测试评估至多执行一次；成功结果会缓存并在重复 API 读取时返回同一结果，不会重测。失败或中断会把会话标记为不可重试。需要新尝试时，应重新准备一个新会话并重新冻结。
- 如配置了 `output_dir`，默认在最终评估时写出配置、指标、逐行结果、模型和 manifest。`export_artifacts=False` 可在调用中关闭文件写入。

结果包含 `config`、`metrics`、`audit`、`results`（pandas DataFrame）、拟合 `model` 和 `artifact_paths`。审计字段记录 split 索引、配置哈希、事件顺序及测试计数。

## HMM 与窗口回归配置

扩展模型沿用相同的准备、冻结和最终测试生命周期。H01/H02/H03 任务名为 `sequence_modeling`，不使用监督目标列；N04/N06 是需要目标列的时序回归模型。序列列只参与分组、排序或构造输入窗口，不会作为观测特征自动加入模型。示例使用分组和时间戳构造 Gaussian HMM 配置：

`SplitConfig.seed` 决定数据划分。HMM 和深度模型的 `random_state` 可显式设为 0 到 `2**32-1` 的整数作为初始化 seed；未提供或为 `None` 时使用 `SplitConfig.seed`。实际使用的 seed 会写入拟合元数据和最终 refit provenance。批量搜索的 `random_state` 不作为搜索字段；它属于 job 配置，winner refit 沿用同一个有效 seed。

```python
from pyml_workbench import DatasetConfig, ExperimentConfig, SequenceConfig

hmm_config = ExperimentConfig(
    dataset=DatasetConfig(
        source_path="observations.csv",
        feature_columns=("sensor_a", "sensor_b"),
    ),
    task="sequence_modeling",
    model_id="H01",
    parameters={"n_components": 3, "n_iter": 100},
    sequence=SequenceConfig(
        group_column="device_id",
        time_column="timestamp",
        order_mode="time",
        observation_columns=("sensor_a", "sensor_b"),
    ),
)
session = prepare_experiment(hmm_config)
print(session.metrics["validation"])
freeze_experiment(session, expected_config=hmm_config)
result = evaluate_test(session)
```

HMM 搜索默认最大化 `log_likelihood_per_observation`；H03 从训练分区冻结类别编码表，验证中出现未见符号会被拒绝，测试阶段的未见符号会在最终计算时报告错误。N04/N06 通过 `SequenceConfig(window=..., horizon=...)` 构造每组内的滑动窗口；同一窗口映射会用于训练、验证、重拟合和测试。

`SequenceConfig`、`SequencePlan` 和 `SequenceError` 在包根目录公开导出。准备函数可自动建立并持有计划；若手动预构建 `DatasetSnapshot`，则须同时传入与它验证绑定的 `SequencePlan`。搜索队列会将计划与快照一起持久化，暂停恢复不重新读取源文件。

H01/H02/H03 需安装 `pyml-workbench[sequence]`，N01/N02/N04/N06 需安装 `pyml-workbench[deep]`。缺少对应可选依赖时，模型仍保留在目录中，但实际准备或拟合会给出明确的 extra 安装提示。

## 基础图表模块

桌面端在“分析”菜单中提供完整数据概况和基础图表选择器；批量窗口可为已完成 job 绘制缓存结果。结果来源由应用按 session/job 所有权、分区、结果行绘图清单和持久回执校验。单任务 session 需要自己的绘图清单；批量 winner 清单核验 train/validation，final-session 独立清单与测试回执共同核验 test。若批量 winner 清单有效但旧 final-session 缺少独立清单，train/validation 仍可绘制，test 不可用；缺 winner 清单则该 job 没有结果图来源。应用不会为了绘图自动补清单、重新训练、搜索或测试。用户操作与图表类型见[用户指南](user-guide.md#数据概况与基础图表)和[批量搜索指南](batch-guide.md#桌面操作入口)。

需要在 Python 中直接构造不可变的图表 payload 时，可从 `pyml_workbench.plotting` 导入类型与函数；这些符号不从 `pyml_workbench` 根目录重导出。payload builder 不读取文件、不调用模型，也不导入 Matplotlib。Matplotlib 只在调用 renderer 时导入：

```python
from pyml_workbench.plotting import (
    PlotColumn, PlotKind, PlotSource, available_plot_specs,
    build_plot_payload, render_plot_payload,
)

source = PlotSource(
    schema_version=1,
    source_kind="loaded_eda",
    source_id="example",
    owner_kind="loaded_dataset",
    owner_id="example",
    partition="all",
    row_positions=(0, 1, 2),
    columns=(PlotColumn("data.0", "signal", "float64", (1.0, 2.0, 3.0)),),
    partition_count=3,
)
spec = next(spec for spec in available_plot_specs(source) if spec.kind == PlotKind.HISTOGRAM)
payload = build_plot_payload(source, spec)
figure = render_plot_payload(payload)  # Figure; 可调用 figure.savefig(...)
```

`PlotSource` 的列值按 `row_positions` 对齐。`available_plot_specs` 只返回该来源具有必要字段的 allowlist 图表；类别身份保持类型区分，非有限数值会按图表规则排除。`render_plot_payload(payload, figure=...)` 可渲染到调用方提供的 `matplotlib.figure.Figure`，不依赖 pyplot 的当前图状态。桌面图窗由 `desktop` extra 提供，并可保存当前图为 PNG 或 SVG。

## 一步完成的便捷接口

```python
from pyml_workbench import run_experiment

result = run_experiment(config)
```

`run_experiment(config, *, on_event=None)` 等价于 `prepare_experiment` 后调用 `finalize_experiment`；后者自动冻结并评估测试集一次。它适合明确要自动完成全部一次性流程的调用方，不会先返回一个供检查的验证会话。若调用方需要根据验证结果再确认模型，应使用三阶段 API。便捷调用返回 `ExperimentResult`，不返回会话供第二次测试。

## 持久批量搜索与最终化

批量搜索 API 从包根目录公开导入。`run_batch` 先预检并在 SQLite 历史中创建各个数据/模型组合，然后按 `max_workers`（仅 1 或 2）运行；它返回持久历史路径、job 标识和各组合结果。对需要在多个调用之间暂停、恢复或重启的工作，可分开使用入队、运行和恢复接口：

```python
from pyml_workbench import (
    DatasetConfig, ExperimentConfig, ObjectiveSpec, SearchSpace, SearchSpec,
    SplitConfig, finalize_frozen_search, freeze_search_winner,
    get_job_summary, load_search_result, run_batch,
)

search_config = ExperimentConfig(
    dataset=DatasetConfig("synthetic.csv", target_column="target", feature_columns=("signal", "category")),
    task="classification",
    model_id="C01",
    parameters={"max_iter": 500},
    split=SplitConfig(seed=42),
)
space = SearchSpace(fields={"C": {
    "type": "real", "low": 0.1, "high": 10.0, "log": True,
    "values": [0.1, 1.0, 10.0],
}})
spec = SearchSpec(
    method="grid", space=space, objective=ObjectiveSpec(),
    max_fits=3, max_proposals=15, timeout_seconds=1200, seed=42,
)

batch = run_batch(
    "batch/history.sqlite3", "batch/artifacts",
    [(search_config, spec)], max_workers=1,
)
job_id = batch["jobs"][0]["job_id"]
print(batch["results"][0]["status"])
search_result = load_search_result(batch["history_path"], job_id)
if search_result is not None and search_result.winner is not None:
    selection = freeze_search_winner(batch["history_path"], job_id)
    final = finalize_frozen_search(
        batch["history_path"], job_id, output_dir=f"batch/exports/{job_id}"
    )
    print(final.to_dict()["final_test"], get_job_summary(batch["history_path"], job_id)["test_permission"])
```

`create_search_job(history_path, artifact_root, config, spec)` creates one preflighted job; `create_search_jobs` accepts a list of `(config, spec)` pairs. `run_search_job` and `run_search_jobs` execute queued work, with at most two parallel worker slots. `load_search_result` reconstructs the typed result from the SQLite records and local snapshot/session artifacts. `freeze_search_winner` records an explicit validation-only selection; `finalize_frozen_search` refits on the combined 80% train+validation partition and consumes the job's one final-test permission. A completed search without a scoreable validation winner cannot be frozen. A failed or interrupted final test is recorded as consumed and cannot be retried from that job.

For a single manual parameter configuration, use an empty variable space with fixed values and `method="grid"`; the empty Grid produces one candidate and one actual fit. The row's `ExperimentConfig.parameters` are retained, then `SearchSpace.fixed` overrides matching keys. Estimator parameter names and types are checked before the fit, and this validation search still does not read test data:

```python
manual_spec = SearchSpec(
    method="grid",
    space=SearchSpace(fields={}, fixed={"max_iter": 500}),
    max_fits=1,
)
```

Grid, random, and annealing use the base scientific stack. TPE requires the optional Optuna package and genetic search requires optional pymoo; install `pyml-workbench[search]` or use `uv sync --extra search`. HMM and deep models additionally require their respective `sequence` and `deep` extras. If an optional model dependency is absent, its catalog entry remains listed and the attempted run reports the missing extra before fitting. Each job is capped at 50 actual fits, 250 proposals, and 20 active minutes; the batch has one worker by default and supports at most two, each limited to one numeric thread.

## 模型目录、保存与能力

- `list_models(task=None, include_deferred=False)` 默认返回 78 个可运行目录条目，其中包括 71 个传统 estimator 和 7 个扩展模型。传任务名可筛选；`include_deferred=True` 只会额外列出未来添加的未实现条目。
- `parameter_schema(model_id)` 返回模型参数名、当前 schema 默认值和类型。`build_estimator(model_id, parameters=None, seed=42)` 用于传统 scikit-learn estimator；HMM 和神经网络须通过 `prepare_experiment` 或搜索/批量 API 构建。
- 拟合模型 `FittedModel` 保留预处理器、特征列、配置快照和方法能力。`load_model(path)` 接受模型文件或输出目录。`predict(model, values)` 与 `transform(model, values)` 尊重拟合后的方法能力和特征列。
- LOF (A03) 默认 `novelty=False`，其新数据 `predict`、`decision_function` 和 `score_samples` 不可用；要为新行开启这些方法，拟合时设置 `novelty=True`。SVC (C07) 默认 `probability=False`，需要概率输出时在拟合前开启。

其他公开函数包括 `get_model`、`model_capabilities`、`load_dataset`、`preview_dataset`、`select_features_target` 和错误类型 `ConfigError`、`DatasetError`、`MissingTargetError`、`SplitError`、`ExperimentError`、`ArtifactError`、`UnsupportedOperationError`。扩展模型的配置字段见[序列建模指南](sequence-guide.md)和[深度学习指南](deep-learning-guide.md)；完整 78 项目录见[模型索引](model-index.md)。
