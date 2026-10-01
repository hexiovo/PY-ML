# 序列模型指南

本指南介绍 HMM 序列建模和 N04/N06 窗口回归。HMM 依赖 `sequence` extra；窗口回归使用 `deep` extra。两类模型都能在 CPU 上运行。本项目将深度模型固定为 CPU 设备；当前没有经过验证的 GPU 运行路径。

在项目虚拟环境中安装所需 extra：

```powershell
uv sync --locked --no-editable --no-dev --extra sequence --extra deep
```

如果只用 HMM，可只安装 `--extra sequence`；只用 LSTM/GRU，可只安装 `--extra deep`。

## 选择序列顺序和划分

在 `ExperimentConfig.sequence` 中设置 `SequenceConfig`。`group_column` 标识互相独立的序列；`time_column` 标识组内时间；时间列必须搭配 `order_mode="time"`。组、时间和目标列都是结构列，不能同时作为输入特征。

```python
from pyml_workbench import DatasetConfig, ExperimentConfig, SplitConfig
from pyml_workbench.sequence import SequenceConfig

sequence = SequenceConfig(
    group_column="machine_id",
    time_column="recorded_at",
    order_mode="time",
    observation_columns=("temperature",),  # HMM 使用的观测列
)

config = ExperimentConfig(
    dataset=DatasetConfig(
        source_path="measurements.csv",
        feature_columns=("temperature",),
    ),
    task="sequence_modeling",
    model_id="H01",
    parameters={"n_components": 3, "n_iter": 50},
    split=SplitConfig(seed=42),
    sequence=sequence,
)
```

`order_mode="time"` 会按每组时间排序；组内重复、缺失或无效时间会被拒绝。选择 `order_mode="row"` 时，源表行顺序就是序列顺序；没有 `time_column` 时请确认文件中的行顺序确实代表时间。若提供 `group_column`，同一组不会被拆进不同集合；整组按确定顺序分配到训练、验证、测试，以接近 60/20/20 的行比例。未提供组列时，所有行属于一条序列，按源表行顺序连续切成训练、验证、测试。窗口只在切分完成后生成，因此不会用前一集合的尾部作为后一集合的上下文。

## H01、H02、H03 的观测和长度

- **H01** 使用 GaussianHMM，适合连续数值观测。观测矩阵形状为 `(N, D)`，其中 `N` 是拼接后的观测行数，`D` 是观测特征数。
- **H02** 使用 GMMHMM，同样接收连续数值观测矩阵；它为每个隐藏状态使用高斯混合发射分布。
- **H03** 使用 CategoricalHMM，只接受一列离散观测。训练数据中首次出现的类别构成冻结字母表，训练和验证都按它编码。

HMM 可选的 `algorithm` 为 `viterbi` 或 `map`，`implementation` 为 `log` 或 `scaling`，`verbose` 必须是布尔值。用于初始化和拟合更新的 `init_params` 与 `params` 只接受各模型支持的字母：H01 为 `stmc`、H02 为 `stmcw`、H03 为 `ste`；空字符串也可用于关闭相应步骤。这些值会在构造 estimator 前校验。

HMM 的每次拟合、评分、解码和后验概率计算都会收到对应分区的 `lengths`。它是一维正整数列表，每个数表示一个完整序列的长度，且长度总和必须等于观测行数。例如，两个设备各有 30 条记录时，训练观测形状可能是 `(60, 1)`，`lengths` 为 `[30, 30]`。模型据此在每台设备边界重新开始序列转移，不会把设备末尾接到下一台设备开头。HMM 参数 `random_state` 默认是 `None`；省略或显式设为 `None` 时使用 `SplitConfig.seed`，显式非布尔整数 `0` 到 `2**32-1` 会覆盖该回退值。生效 seed 会记录在训练元数据和 refit provenance 中，并在 train+validation fresh refit 时校验和复用。模型 seed 不改变已冻结的组/行划分。

HMM 的隐藏状态是拟合出来的潜在状态编号，例如 `0、1、2`。它们不是原始类别或有序等级；编号大小没有业务含义，不同拟合之间的状态编号也可能置换。可检查训练和验证分区上的状态片段、每行后验概率与观测分布，再结合领域知识解释状态。

验证指标 `log_likelihood_per_observation` 是每条观测的对数似然。重载模型后的序列推理通过 `pyml_workbench.extended_experiment.predict_extended` 调用，必须同时提供观测和长度：

```python
import numpy as np
from pyml_workbench import load_model
from pyml_workbench.extended_experiment import predict_extended

model = load_model("artifacts/hmm/model.joblib")
observations = np.asarray([[19.8], [20.1], [20.4], [20.0]], dtype=np.float32)
states = predict_extended(
    model,
    {"observations": observations, "lengths": [4]},
)
posterior = predict_extended(
    model,
    {"observations": observations, "lengths": [4]},
    operation="posterior",
)
```

HMM 输入映射必须恰好包含 `observations` 和 `lengths`。可用的操作是 `predict`（隐藏状态路径）、`decode`、`posterior`（形状 `(N, 状态数)`）和 `log_likelihood`（每条观测的平均对数似然）。H03 推理接受原始类别值，helper 按已冻结的训练字母表编码；未知类别会报错。

H03 验证观测如果不在训练字母表中，会在训练/验证准备阶段被拒绝。测试观测中的未知类别会留到显式最终测试时再检查；该错误发生在测试调用中，核心会将这次测试尝试标记为已进入最终阶段，同一会话不能重试。预测新数据时同样会拒绝训练字母表之外的类别。H01/H02 的测试原始值也延迟到最终测试转换和校验。

## N04/N06 窗口与预测位置

N04 是 LSTM 回归，N06 是 GRU 回归。它们读取数值特征窗口，并预测该窗口之后一个时点的目标。`SequenceConfig.window` 和 `horizon` 默认分别为 `10` 和 `1`。对 0 起始位置 `0..`，窗口 `0..9` 的目标位置是 `10`；下一条窗口 `1..10` 的目标是 `11`。一般情况下，窗口来源行从 `s` 到 `s + window - 1`，目标位置为 `s + window - 1 + horizon`。长度为 `L` 的组或切分段最多产生 `max(0, L - window - horizon + 1)` 个窗口。

N06 的配置会把回归目标与输入特征分开：

```python
from pyml_workbench import DatasetConfig, ExperimentConfig, SplitConfig
from pyml_workbench.sequence import SequenceConfig

config = ExperimentConfig(
    dataset=DatasetConfig(
        source_path="measurements.csv",
        target_column="next_temperature",
        feature_columns=("temperature", "pressure"),
    ),
    task="regression",
    model_id="N06",  # GRU；N04 使用 LSTM
    split=SplitConfig(seed=42),
    sequence=SequenceConfig(
        group_column="machine_id",
        time_column="recorded_at",
        order_mode="time",
        window=10,
        horizon=1,
    ),
)
```

下面是一个由实际 `SequencePlan` 生成的计数例子：20 组、每组 30 行，共 600 行，按组切分得到训练/验证/测试 12/4/4 组。设置 `window=10, horizon=1` 后，各分区仍只有自己的组和行：

| 分区 | 组数 | 源行数 | 窗口数 |
| --- | ---: | ---: | ---: |
| 训练 | 12 | 360 | 240 |
| 验证 | 4 | 120 | 80 |
| 测试 | 4 | 120 | 80 |

每组 30 行可形成 `30 - 10 - 1 + 1 = 20` 个窗口。第一条训练窗口的来源位置是 `[0, 1, ..., 9]`，目标位置是 `10`。组大小不等时，分区行数和窗口数会相应变化；请使用会话的 `session.sequence_plan` 查看实际计划，不要把上表示例当成每份数据的固定计数。

准备 N04/N06 时，目标列必须指定，输入特征列必须是数值且不得包括组、时间或目标列。每个分区内每个组至少需要 `window + horizon` 行。最终 bundle 的 recurrent 推理形状为 `(B, L, F)`：`B` 是窗口批数，`L` 必须等于训练时的窗口长度，`F` 是原始数值特征数。

```python
from pyml_workbench import load_model
from pyml_workbench.extended_experiment import predict_extended

gru = load_model("artifacts/gru/model.joblib")
new_windows = ...  # float 数组，形状 (B, 10, F)，内容仍是原始特征值
future_targets = predict_extended(gru, new_windows)
```

## 验证、冻结和最终测试

扩展模型使用与其他模型相同的核心流程：`prepare_experiment(config)` 只拟合训练数据并给出验证结果；检查验证结果后调用 `freeze_experiment(session, expected_config=config)`；最后调用 `finalize_experiment(session)` 或 `evaluate_test(session)`，在冻结后评估测试分区一次。模型配置应在最终测试之前确定，不能比较最终测试指标后再选择模型。

## 序列计划、会话和模型文件

`DatasetSnapshot` 和 `ExperimentSession` 是核心训练/最终测试流程持有的内部状态：快照包含选定的原始特征、目标、划分和来源指纹；会话还持有拟合器、验证结果、生命周期状态和最终测试记录。序列会话中的 `SequencePlan` 还保留所需序列观测、窗口、目标与行位置映射，其中可能含有未检查的测试原始值。

需要持久保存队列/恢复所用的序列计划时，可使用 `save_owned_sequence_plan(plan, path)` 和 `load_owned_sequence_plan(path, receipt, ...)`（从 `pyml_workbench.sequence` 导入）。receipt 把文件 SHA-256 与 plan manifest 绑定；加载时还可传 `expected_manifest`、`config` 和 `snapshot` 核验身份。保存的计划包括序列数据和映射，应按数据文件的访问规则管理；它不是删去了原始数据的模型包，也不会替代原始数据授权。

设置 `ExperimentConfig.output_dir` 后，最终结果导出 `model.joblib`、指标与逐行结果；通过 `load_model()` 加载推理模型。扩展模型的 CPU bundle 保存已拟合预处理、模型状态、配置和训练类别/字母表等推理所需信息，不保存原始行、序列计划、验证/测试数据或优化器状态。把 `model.joblib` 作为推理模型分享时，仍须遵守训练数据和模型工件的访问策略。

本指南中 HMM 与 recurrent 示例均针对 CPU 路径；深度实现显式设置 `device="cpu"`。尚未记录 GPU 支持验证结果。
