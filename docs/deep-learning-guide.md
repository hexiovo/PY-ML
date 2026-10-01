# 深度学习模型指南

当前深度模型由 PyTorch 和 skorch 实现，需要安装 `deep` extra。在项目虚拟环境中运行：

```powershell
uv sync --locked --no-editable --no-dev --extra deep
```

实现目前固定使用 CPU，并通过验证数据进行 early stopping；本文只描述已经过实际运行检查的 CPU 路径，不声称 GPU 已支持或已验证。

## 模型和输入

| ID | 架构 | 任务 | 每次输入 |
| --- | --- | --- | --- |
| N01 | 多层感知机（MLP） | 分类 | 表格特征 `(B, F)` |
| N02 | 多层感知机（MLP） | 回归 | 表格特征 `(B, F)` |
| N04 | 长短期记忆网络（LSTM） | 序列回归 | 固定窗口 `(B, L, F)` |
| N06 | 门控循环单元网络（GRU） | 序列回归 | 固定窗口 `(B, L, F)` |

`B` 是批次数，`F` 是特征数；对于 N04/N06，`L` 是 `SequenceConfig.window`。窗口先完成分组/时间排序和 60/20/20 划分，再在各自分区内生成。上下文行和预测目标行都必须属于同一组、同一分区。详细行位置示例见[序列模型指南](sequence-guide.md)。

N01/N02 使用明确配置的特征和目标列。例如：

```python
from pyml_workbench import DatasetConfig, ExperimentConfig, SplitConfig

config = ExperimentConfig(
    dataset=DatasetConfig(
        source_path="training.csv",
        target_column="label",
        feature_columns=("temperature", "pressure"),
    ),
    task="classification",
    model_id="N01",
    parameters={
        "hidden_size": 32,
        "num_layers": 1,
        "batch_size": 32,
        "learning_rate": 0.001,
        "max_epochs": 30,
        "patience": 5,
    },
    split=SplitConfig(seed=42),
)
```

N04/N06 需要回归目标列及数值输入特征，并在 `sequence=SequenceConfig(...)` 中指定组/时间顺序、窗口长度和 horizon。标准参数字段为 `hidden_size`、`num_layers`、`batch_size`、`learning_rate`、`weight_decay`、`max_epochs`、`patience` 和 `random_state`；`max_epochs` 上限为 100。N01/N02 的 `num_layers` 表示隐藏的 Linear+ReLU 层数，`1` 保持单个隐藏 Linear 的原有结构，输出层单独计算；N04/N06 的 `num_layers` 控制循环层数。没有传入的字段采用实现默认值。

## CPU 训练和验证选 epoch

`prepare_experiment(config)` 返回已拟合训练分区的会话，并计算训练/验证结果。预处理器只在允许的训练源行上拟合；验证集只用于验证评分和 early stopping；准备阶段不计算测试指标。深度实现的 early stopping 配置为：

- 监控 `valid_loss`，以更小为优。
- 使用绝对变化阈值 `0`，因此仅严格下降才算改进。
- 设置 `load_best=True`，训练结束后恢复验证 loss 最低的权重；相同最低值保留首次出现的 epoch。
- `selected_epochs` 和 `best_epoch` 记录这组已恢复权重对应的 epoch。

拟合后，可读取 `session.extended_training_metadata` 中的 `selected_epochs`、`best_epoch`、`best_weight_policy` 和 JSON `refit_provenance`。这些字段描述被验证选出的模型状态，便于审核训练和最终 refit 使用的轮数。

同一元数据中的 `training_curves` 保存 skorch 每个实际 epoch 的纯 JSON `epochs`、`train_loss` 和 `validation_loss`。训练阶段有验证集时，`validation_available` 为 `true`；fresh train+validation refit 仅训练，不生成验证 loss，因此记录 `validation_loss: null` 和 `validation_available: false`。

单模型 API 中，`freeze_experiment(session, expected_config=config)` 冻结当前已拟合模型；紧接着的 `finalize_experiment(session)` 会测试这个模型一次。它不会重新训练。持久搜索的流程不同：`freeze_search_winner` 根据验证结果显式冻结赢家，`finalize_frozen_search` 会创建新模型，在原训练+验证数据上 fresh refit，然后才消费一次最终测试。对 N01/N02/N04/N06，refit 使用冻结 provenance 记录的 `selected_epochs`，新建网络和优化器，并按固定 epoch 拟合训练+验证数据，不再使用验证或测试来重新选择轮数。配置参数不会被改写来暗中塞入 epoch。

`SplitConfig(seed=...)` 决定划分。deep 参数 `random_state` 若设为 0 到 `2**32-1` 的整数，就作为深度拟合 seed；未提供或设为 `None` 时，拟合 seed 回退到 `SplitConfig.seed`。布尔值、负数、浮点数和超范围整数都会在拟合前拒绝。实际使用的 seed 会保存在 `session.extended_training_metadata["effective_training_seed"]` 和 `session.refit_provenance["effective_training_seed"]` 中；它不会写回参数，也不会改变配置哈希或分区。early stopping 的 `selected_epochs`、`best_epoch`、`best_weight_policy` 与 seed 一起用于复现与 fresh refit。

拟合时实现串行化 PyTorch 的全局随机状态和线程设置，再恢复调用方原有状态；模型运行设备固定为 CPU。小型合成工作流示例可运行：

```powershell
.\.venv\Scripts\python.exe -X utf8 examples\synthetic_extended.py
```

它分别展示 HMM 和 N06 GRU 的验证、冻结、最终测试、导出、加载与新数据推理。示例在临时目录创建数据，不读取真实数据，也不按最终测试指标选择模型。

## 训练类别和未知类别

N01 的类别编码只由训练分区目标值建立，并保存在模型中。验证标签必须能映射到该训练类别集；验证中新类别会在准备阶段报错。测试标签中的未知类别会在显式最终测试时才检查，不能提前用测试标签扩展映射。冻结模型上的新数据推理也只接受训练时的类别集合。

H03 的观测字母表同样只由训练观测建立；它不是目标类别映射。H03 验证观测必须属于训练字母表；测试中的新观测会在最终测试时被报告，推理时也会拒绝。N04/N06 的回归目标只要求在训练和验证阶段是有限数值；测试目标在最终测试时检查。

核心在进入最终测试前标记会话已冻结，并最多调用一次测试计算。若测试期间遇到未知类别或无效数值，同一会话不能跳过错误并再次尝试；应修复数据后用新会话重新拟合和冻结。

## 新数据推理

最终 artifact 的 `model.joblib` 可使用公共 `load_model(path)` 加载。N01/N02/N04/N06 的扩展协议通过 `pyml_workbench.extended_experiment.predict_extended` 执行：

```python
import numpy as np
from pyml_workbench import load_model
from pyml_workbench.extended_experiment import predict_extended

model = load_model("artifacts/gru/model.joblib")
windows = np.zeros((2, 10, 3), dtype=np.float32)  # 2 个窗口，10 个时点，3 个原始数值特征
predictions = predict_extended(model, windows)
print(predictions.shape)  # (2,)
```

这里的 `L=10` 只是示例，必须替换成拟合时配置的 `window`；`F=3` 必须与模型的输入特征数相同。传入窗口内容应是原始数值特征，bundle 中的预处理器会按训练状态转换。窗口预测不接收目标值。N01/N02 接收 `(B,F)` 表格特征，N01 还支持 `operation="predict_proba"`。

HMM 使用相同的 `predict_extended` 函数，但传入映射 `{"observations": array, "lengths": [...]}`；连续 HMM 的观测为 `(N,D)`，H03 则使用一列原始类别值。长度必须是正整数且总和等于观测行数。HMM 的详细状态与分段语义见[序列模型指南](sequence-guide.md)。

## 会话、计划和推理 bundle 的数据边界

`DatasetSnapshot` 和实验会话服务于训练、验证和最终测试：它们持有源路径、已选原始特征/目标、分区位置、验证指标和生命周期状态。N04/N06 的 `SequencePlan` 还持有窗口值、目标值以及源行/目标行位置映射。保存并恢复 owned plan 时，`save_owned_sequence_plan` 写出 plan 和带哈希的 receipt，`load_owned_sequence_plan` 验证文件哈希、manifest 和传入身份。计划是可恢复的训练数据资产，可能包含测试原始值，请按原始数据权限保护。

最终模型 bundle 保存推理所需的预处理状态、估计器或 CPU 权重、训练类别映射以及配置指纹，不包含原始数据行、`DatasetSnapshot`、验证/测试窗口或优化器状态。设置 `ExperimentConfig.output_dir` 后，显式最终测试会同时导出模型、指标和逐行结果；`load_model()` 可加载模型 artifact。不要把内部会话或 owned plan 当作经过脱敏的推理模型文件。
