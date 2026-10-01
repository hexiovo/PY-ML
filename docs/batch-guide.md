# 批量参数搜索与最终评估

本文介绍如何为多份表格数据运行参数搜索、比较验证集结果，并在选定 winner（验证集表现最佳的候选）后执行一次最终评估。批量搜索的数据会保存在独立的 SQLite 历史库和会话文件中，便于查看、暂停、恢复和审计。另可将 `train_exploratory` 用作训练集探索评分；这种结果不进入验证榜，也不能进入冻结 winner 或最终测试流程。

## 安装可选能力

基础安装提供网格、随机和退火搜索；TPE 需要 Optuna，遗传搜索需要 pymoo，桌面窗口需要 PySide6 等 `desktop` 组件。要在本地启用桌面界面和五种搜索方法，仓库开发环境使用：

```powershell
uv sync --locked --no-editable --no-dev --extra desktop --extra search
```

如果只运行网格、随机或退火搜索，可不安装 `search` extra；选择 TPE 或遗传搜索时，缺少相应包会在预检时提示通过 `uv sync --extra search` 或 `pip install pyml-workbench[search]` 安装。

## 桌面操作入口

在 PY-ML 主窗口顶部点击 **“批量搜索队列”**。如果主窗口已加载有效的单任务配置，批量窗口会预填当前数据与模型；否则会打开空配置。你可以编辑 JSON、添加当前配置或选择多个表格文件，并为每个模型提供搜索空间。

批量窗口提供搜索方法、每个模型/数据组合的拟合数、提案数、活动时限和并行 worker 数设置。表格按数据源内容哈希、工作簿 sheet、数据划分哈希与种子、任务、实际特征/目标/异常目标列，以及完整 ObjectiveSpec（指标、方向、评分分区和设置）生成分组并排序；只有分组键完全相同的任务才放在一起比较验证分数。暂停、恢复、取消、冻结 winner、最终重拟合/测试和导出 CSV 都在该窗口完成。完成的 `train_exploratory` 任务可点击“导出训练探索 winner”导出已有训练 session；该操作不重拟合、不进入验证榜，也不会读取验证集或测试集。训练探索任务不能冻结或执行最终测试。

窗口默认将历史与会话保存在系统临时目录下的 `pyml_workbench_batch`；如需长期保留，请在“批次目录”中选一个稳定目录。历史 SQLite 文件、数据快照、试验会话、冻结选择和最终会话会写到批次目录的子目录中。

任务状态为 `completed` 后，可选中该任务并点击“绘制结果图…”。选择器只读取所选 job 的自有快照、winner session、最终结果缓存和回执，并按 train、validation、test 分区展示通过绘图完整性清单核验的数据。winner session 的独立清单核验 train/validation；若缺失或不匹配，该 job 的所有结果图来源均不可用，即使原始快照收据仍存在。test 还要求最终重拟合与最终测试成功、持久测试回执有效，以及 final-session 自己的清单核验通过；如果 winner 清单有效但较早版本的 final-session 没有该清单，train/validation 仍可从 winner session 绘制，只有 test 不可用。每张图只使用一个任务分区，不会把不同分区混在同一张图中。可用图表取决于当前分区实际缓存的特征和输出；图窗非模态，可将当前图保存为 PNG 或 SVG。上述限制不会禁用旧任务原有的查看、恢复或导出流程。绘图入口不会自动补清单或重新搜索、重拟合、测试；需要新图时请按正常流程手动创建新 job。单任务和已加载原表的图表入口见[用户指南](user-guide.md#数据概况与基础图表)。

## 搜索和评估经过哪些步骤

1. **预检和固定快照。** 每个数据集与模型组成独立搜索任务。系统检查表格、目标列、模型/任务兼容性、参数空间和评分目标，并保存数据内容哈希、固定的数据划分及其哈希。入队时会把已校验的数据快照原子写入该任务自己的 artifact 目录并保存校验收据；worker 和恢复任务始终使用这份快照，不重新读取源文件。入队后源文件被修改或删除都不会改变搜索数据；快照缺失、损坏或与历史收据不符时，任务会在首次 fit 前失败，拟合计数保持为 0。预检阶段不拟合模型。
2. **只用训练集和验证集搜索。** 当前审批的数据划分是训练 60%、验证 20%、测试 20%。默认在 validation 上评分；显式选择 `train_exploratory` 时只在训练行上评分。每个有效且未命中缓存的参数候选实际拟合一次，再在指定评分分区计算目标指标；无效候选和重复缓存不会进入真实拟合。搜索期间不读取测试行。
3. **记历史并比较候选。** 每次提案、实际拟合、失败、缓存命中、暂停/恢复和搜索结束都会写入 SQLite。验证榜只比较同一数据内容、任务、目标指标/方向、评分分区和划分种子的 validation 任务；换了数据、指标、分区或种子，分数就不属于同一个比较组。榜单按目标方向排序，同分共享 competition rank（例如 1、1、3）；训练探索、无分数或非有限分数不排名。导出的对比 CSV 也包含 `group_rank`。`train_exploratory` 分数单独标记为训练探索，不进入验证榜。目标指标不受当前模型支持的候选会保留为不可评分记录，不进入榜单。
4. **显式冻结验证 winner。** 搜索结束后，先执行“冻结所选验证 winner”。冻结会固定 winner 参数、试验 ID、搜索 fingerprint 和数据/划分身份，不会读取测试集。
5. **最终重拟合并测试一次。** 冻结后，系统用训练集与验证集合并后的 80% 数据重新拟合 winner，再评估保留的 20% 测试集。测试许可和结果会持久化；对已完成任务再次点击最终评估会读取同一缓存，不会重拟合或重测。若重拟合或测试中断/失败，系统不会自动重试被消耗的测试许可。

`selection_validation` 是选 winner 时的验证分数，`final_refit` 描述 80% 数据上的最终重拟合，`final_test` 是最终测试结果。三者用途不同，不能用最终测试分数重新挑 winner，也不能把它当作搜索榜排名依据。

## 训练探索结果导出

`ObjectiveSpec(split="train_exploratory")` 只在训练分区评分。分数代表训练集探索表现，不是独立验证估计，因此不能冻结成 validation winner，也不能执行最终重拟合/测试。搜索完成且存在可评分 winner 后，可通过公开 API `export_training_exploration(history_path, job_id, output_dir=...)`，或在桌面窗口点击“导出训练探索 winner”，导出搜索时已经持久化的获胜训练 session、结果表、指标和 manifest。导出会检查重载后的模型配置与 winner 一致；它不重新拟合、不重排候选、不读取验证/测试分区。相同目录中的同一导出再次执行时会复用已保存的导出产物，不改变搜索 fit 计数或测试许可。也可调用 worker 命令 `batch-export-exploration` 导出同一已完成队列任务；D10 批量整合验证确认 worker 与公开 API 复用同一获胜 session，重复 API 导出返回 `cached`，fit/proposal 计数和测试许可均不变，manifest 标记 `validation_ranked=false` 且不含测试指标。

## 目标指标

| 任务 | 默认评分 | 方向 | 说明 |
| --- | --- | --- | --- |
| 分类 | balanced accuracy（平衡准确率） | 越高越好 | 按类别平均召回率，适合类别数不均衡的情况。 |
| 回归 | RMSE（均方根误差） | 越低越好 | 误差以目标列原单位表达。 |
| 聚类 | silhouette（轮廓系数） | 越高越好 | 在验证数据上计算。预测为 `-1` 的噪声点不参与轮廓系数；先计算有效覆盖率，默认至少为 0.8，且去除噪声后至少要有两个簇。条件不满足时，该候选不可评分。 |
| 降维 | trustworthiness（可信度） | 越高越好 | 只在模型支持所需嵌入操作时评分；不支持时不进入排名。 |
| 异常检测 | balanced accuracy | 越高越好 | 自动搜索必须提供独立的真实标签列，映射后评分分区同时含 `-1` 异常和 `+1` 正常样本。没有独立标签，或评分分区缺少任一类别时，预检会阻止搜索/评分；模型自己的异常预测不能代替真实标签。 |

聚类覆盖率下限可通过 `ObjectiveSpec(coverage_min=...)` 调整到 0.8–1.0。异常检测可以通过 `ObjectiveSpec(objective_labels_column=..., label_mapping=...)` 指定标签列和映射；标签列不得混入特征。使用 `train_exploratory` 作为评分分区时，分数只表示训练集上的探索结果，不代表独立验证表现。

## 搜索空间怎么填写

每个搜索空间由 `fields`（待搜索参数）和可选的 `fixed`（本次固定参数）构成。通常至少指定一个待搜索字段；如果只是用给定配置拟合一次，允许 `method="grid"` 且 `fields` 为空，此时会对唯一配置做一次搜索拟合。其他四种方法要求至少一个待搜索字段。参数名必须是所选模型实际支持的 estimator 参数。可以通过 `parameter_schema(model_id)` 查看模型参数名与默认值；不能把一种模型的参数空间直接套给另一种模型。

每个 `fields` 项使用以下三种类型之一：

- `real`：有限实数范围，要求 `low < high`；可选 `log: true`，但下界必须大于 0。
- `integer`：整数范围，要求 `low`、`high` 都是整数且 `low < high`。
- `choice`：从非空 `values` 列表中选择；列表项只能是 JSON 标量（字符串、数字、布尔或 `null`）。

`values` 用于声明离散候选；网格搜索要求每个待搜索参数都有显式 `values`。对实数和整数，列表值还必须处于对应上下界之内。`when` 可以声明条件参数何时生效：它将父参数映射到允许启用当前参数的值列表。无效的候选值或未知的 estimator 参数会被预检或候选校验拦截；条件未满足时，该参数不会传给模型。

工作线程会把 estimator 的 `n_jobs` 固定为 1，以限制每个 worker 的并行线程数；`n_jobs` 与 `random_state` 不能作为搜索变量。固定值可以写在 `fixed`，其值必须符合模型参数要求。网格、随机、TPE 和遗传方法适合按类型设置参数；退火方法有额外限制（见下表）。如果只想用基础配置拟合一次，可使用 `SearchSpec(method="grid", space=SearchSpace.from_dict({"fields": {}, "fixed": {}}))`；空字段配合随机、TPE、遗传或退火方法会被拒绝并提示固定配置只能走 `grid`。这一个搜索 fit 仍只用于训练/验证评分，搜索阶段的 test 评估次数为 0；冻结后才执行一次最终 test。

示例空间：

```python
space = SearchSpace.from_dict({
    "fields": {
        "C": {
            "type": "real",
            "low": 0.1,
            "high": 10.0,
            "log": True,
            "values": [0.1, 1.0, 10.0],
        }
    },
    "fixed": {},
})
```

此处只是空间格式示例，不代表所有模型都支持 `C`，也不代表这些范围适合你的数据。每个模型都应使用自己的实际参数和有依据的范围。

## 五种搜索方法

所有方法共享同一个类型化搜索空间与预算。每个批次组合默认最多 50 次真实搜索拟合、250 个提案和 20 分钟活动时间；可以设置更低上限，不能超过这些上限。GUI 中并行 worker 为 1 或 2，每个 worker 固定使用一个线程。

| 方法 | 参数如何取值 | 适用边界 |
| --- | --- | --- |
| `grid` | 枚举每个字段 `values` 的笛卡尔积。 | 可复现、便于检查候选；组合很多时会较快耗尽拟合预算。每个字段都必须给出有限的 `values`。 |
| `random` | choice 从 `values` 选；integer 在闭区间抽样；real 在上下界均匀抽样，`log` 参数按对数尺度抽样。 | 适合范围较大而不需要完整枚举的情况。除 choice 列表外，数值字段的 `values` 不限制随机抽样。 |
| `tpe` | Optuna TPE 按 choice 列表、integer 范围或 real 范围提出候选；支持 `when` 条件参数。 | 适合逐步利用已有试验结果的混合参数空间。数值搜索依据边界（及 `log`），不是枚举 `values`。 |
| `genetic` | pymoo 混合变量遗传算法使用 choice 选项、integer 范围和 real 范围。 | 适合混合类型参数；数值 `values` 不会变成离散网格。一个无效或重复提案仍消耗提案数，但重复缓存不消耗真实拟合数。 |
| `annealing` | SciPy dual annealing 连续优化 real 参数；`log` 参数先在对数范围优化。 | 至少需要一个 real 搜索变量。integer/choice 搜索字段必须同时在基础 `config.parameters` 中提供值，作为本次退火运行的固定值；需要搜索离散选择时请改用 grid、random、TPE 或 genetic。 |

## 预算、停止原因和恢复

预算按**每个数据集/模型搜索任务**计算，不是整个批次共享的一笔总预算。拟合计数只统计搜索阶段真正进入模型拟合的次数；失败的实际拟合仍消耗一次预算。相同搜索 fingerprint 下的重复参数命中缓存，会记录为提案/缓存事件，但不增加真实拟合计数。仅参数不合法、尚未开始 estimator fit 的提案不消耗真实拟合预算。

活动时长计入搜索运行时间；用户暂停后，等待恢复的时间不计入 20 分钟活动时长。暂停会在安全检查点生效，当前正在进行的试验可能先结束。主要停止原因如下：

| 原因 | 含义 |
| --- | --- |
| `space_exhausted` | 有限搜索空间已枚举完，例如网格候选已经耗尽。 |
| `fit_limit` | 已达到真实搜索拟合次数上限。 |
| `proposal_limit` | 已达到提案数上限；无效或重复提案也会占提案数。 |
| `time_limit` | 已达到活动时间上限，暂停等待时间已扣除。 |
| `cancelled` | 用户取消任务后，在安全检查点停止。 |

达到预算时若已有可评分候选，搜索结果仍可完成并产生 winner；若没有任何可评分候选，会显示无有效试验。某次真实拟合异常会记为失败记录并消耗一次 fit；只要预算与空间尚有余量，后续候选仍可继续。

历史 SQLite 保存任务状态、配置、数据/划分摘要、预算计数、逐次提案与 fit/cache 记录、事件和最终测试许可。会话模型、快照和冻结/最终选择保存在其 artifact 目录。恢复时会核对 fingerprint、计数和本地 artifact；fingerprint 会绑定模型配置、搜索方法/空间、目标、随机种子及数据/划分快照。数据、配置、空间或目标发生变化，或关键 artifact 缺失/身份不匹配时，系统会拒绝把旧历史接到新搜索上。

在 GUI 中从历史表选中任务，再点击“暂停所选任务”“恢复所选任务”或“取消所选任务”。也可在 Python API 中调用 `request_job_control(history_path, job_id, "pause" | "resume" | "cancel")`。取消是终止操作；暂停任务在恢复后从持久化的试验记录继续，并保留已消耗的拟合预算。

## Python API 示例

下例展示一份 CSV、一个模型的基本调用。`run_batch` 会先预检并创建任务，再运行队列；要在运行后继续处理已保存历史，可改用 `create_search_jobs`、`run_search_jobs`、`load_search_result` 等分步 API。

```python
from pyml_workbench import (
    DatasetConfig,
    ExperimentConfig,
    ObjectiveSpec,
    SearchSpace,
    SearchSpec,
    SplitConfig,
    run_batch,
    freeze_search_winner,
    finalize_frozen_search,
    load_search_result,
)

config = ExperimentConfig(
    dataset=DatasetConfig(
        source_path="data.csv",
        target_column="target",
        feature_columns=("x1", "x2"),
    ),
    task="classification",
    model_id="C01",
    split=SplitConfig(seed=42),
)
spec = SearchSpec(
    method="grid",
    space=SearchSpace.from_dict({
        "fields": {
            "C": {
                "type": "real",
                "low": 0.1,
                "high": 10.0,
                "log": True,
                "values": [0.1, 1.0, 10.0],
            }
        }
    }),
    objective=ObjectiveSpec(),  # classification 默认 balanced_accuracy，越高越好
    max_fits=3,
    max_proposals=15,
    timeout_seconds=1200,
    seed=42,
)

batch = run_batch(
    "runs/history.sqlite3",
    "runs/artifacts",
    [(config, spec)],
    max_workers=1,
)
job_id = batch["jobs"][0]["job_id"]
result = load_search_result(batch["history_path"], job_id)
if result is None or result.winner is None:
    raise RuntimeError("没有可冻结的验证集 winner")
freeze_search_winner(batch["history_path"], job_id)
final = finalize_frozen_search(batch["history_path"], job_id)
print(final.to_dict())  # 包含 selection_validation、final_refit、final_test 和 test 计数
```

`max_fits`、`max_proposals` 和 `timeout_seconds` 在这个示例里是用户主动设置的较低预算；上限分别为 50、250 和 1200 秒。若同一已完成任务再次调用 `finalize_frozen_search`，它会恢复已保存的最终会话及 test 结果。

训练探索任务使用相同的创建与运行流程，只需将评分分区设为 `train_exploratory`；完成后从既有 session 导出，不调用 freeze 或 finalize：

```python
from pyml_workbench import (
    ObjectiveSpec,
    SearchSpec,
    SearchSpace,
    create_search_job,
    export_training_exploration,
    run_search_job,
)

exploration_spec = SearchSpec(
    method="grid",
    space=SearchSpace.from_dict({
        "fields": {},
        "fixed": {"n_components": 2, "perplexity": 15.0, "max_iter": 250},
    }),
    objective=ObjectiveSpec(split="train_exploratory"),
    max_fits=1,
    max_proposals=1,
    timeout_seconds=120,
    seed=42,
)
job = create_search_job("runs/history.sqlite3", "runs/artifacts", config, exploration_spec)
result = run_search_job("runs/history.sqlite3", job["job_id"])
if result.winner is None:
    raise RuntimeError("没有可导出的训练探索 winner")
artifacts = export_training_exploration(
    "runs/history.sqlite3", job["job_id"], output_dir="runs/exports/d10-exploration"
)
print(artifacts["artifact_paths"])
```

此示例沿用前文定义的 D10 `config`。导出读取 batch job 已保存的 winner session；不会新增 fit，也不会发放测试许可。

## 模拟数据演示

运行 [`examples/synthetic_batch.py`](../examples/synthetic_batch.py) 可生成两份文件名和列名均明确标为“模拟”的小型分类 CSV，并让 C01 与 C02 各自搜索两个参数候选。示例会建立真实 SQLite 历史与模型会话，冻结每个搜索的 validation winner，最终 refit/test 一次，再从已保存结果恢复并确认 test 计数仍为 1。

```powershell
.\.venv\Scripts\python.exe -X utf8 examples\synthetic_batch.py
```

默认输出位于系统临时目录，运行结束会自动清理。若要保留演示 CSV、历史和会话，请传入一个新建或空目录：

```powershell
.\.venv\Scripts\python.exe -X utf8 examples\synthetic_batch.py --output .\batch-demo-output
```

示例不会读取或更改项目原始数据。模拟分数仅用于演示队列工作流，不代表真实模型质量。




