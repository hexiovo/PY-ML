# 指标、算法与验证扩展（2026-10-27）

本轮源码目录从 84 项扩展到 **111 项**。当前 Python 环境可运行全部可选依赖；本轮完整扩展版 EXE 已单独构建，历史 Windows EXE 保留。基础安装提供 90 项，其余按 extra 启用。模型目录和参数见 [模型索引](model-index.md)。

## 新增模型

| ID | 模型 | 依赖 |
|---|---|---|
| C28 / R27 | Dummy 分类 / 回归基线 | base |
| C29 / R28 | Voting 分类 / 回归 | base |
| C30 / R29 | Stacking 分类 / 回归 | base |
| C31 / R30 | Bagging 分类 / 回归 | base |
| R31 / R32 | BayesianRidge / ARDRegression | base |
| R33 | KernelRidge | base |
| R34 / R35 | RANSAC / Theil–Sen | base |
| R36 | PLSRegression | base |
| K11 / K12 | GaussianMixture / BayesianGaussianMixture | base |
| K13 | HDBSCAN | base |
| D11 | UMAP | manifold |
| D12 | Autoencoder 降维 | deep（PyTorch） |
| T01 / T02 | 朴素 / 季节朴素预测 | base |
| T03 | 指数平滑 | forecasting |
| T04 / T05 | ARIMA / SARIMA | forecasting |
| N07 / N08 / N09 | 1D CNN / TCN / Transformer 窗口回归 | deep |

Voting/Stacking 的高级参数支持 JSON，例如：

```json
{"members":["C01","C16"],"member_parameters":{"C16":{"n_estimators":50}},"weights":[1,2]}
```

Stacking 使用 `cv` 控制内部折数，`final_model` 指定最终模型；`weights` 仅用于 Voting。融合成员限于相同任务的普通表格模型，不允许嵌套融合、序列网络或传统预测器。每个成员的预处理放在各自 pipeline 内，Stacking 内部训练折独立拟合预处理。

HDBSCAN 没有原生新样本 `predict`，因此只有训练探索标签和指标，不能冻结成独立验证 winner，也不会用重训代替验证/测试预测。UMAP 默认支持新样本 `transform`；某些参数（例如 densMAP）会限制此能力，上游错误会如实报告。

## 评价指标

- 分类：原有准确率、平衡准确率和宏平均 Precision/Recall/F1，加权 Precision/Recall/F1、MCC、Cohen's Kappa、ROC-AUC、AP、梯形积分 PR-AUC、Log Loss、Brier，以及逐类 Precision/Recall/F1/Specificity/Support。
- 回归：R²、MAE、RMSE、MSE、中位绝对误差、MAPE、MSLE/RMSLE、分位数损失和 sMAPE。有顺序的目标序列额外提供训练基线 MASE 和按预测步长的 MAE/RMSE。
- 聚类：轮廓系数、CH、DB；有独立参考标签时提供 ARI、NMI、AMI。外部指标包含噪声行，`-1` 作为一种预测标签。
- 异常检测：有独立标签时提供 Balanced Accuracy、异常 Precision/Recall/F1，以及 ROC-AUC/AP。异常标签使用 `-1`，正常使用 `+1`，分数方向取上游正常度分数的负值。

ROC-AUC/AP/PR-AUC 可以使用真实概率或语义明确的决策分数。Log Loss/Brier 只接受真实概率。二分类以拟合 `class_order[1]` 为正类；多分类使用逐类 OvR 宏平均。AP 和梯形积分 PR-AUC 是不同指标。Brier 定义为各类别概率平方误差之和的样本平均；二分类值为常见单正类 Brier 的两倍。

MAPE 使用比例单位；含零或机器精度附近目标时返回 null。sMAPE 使用 0–200 的百分比单位，双方均为零时该项贡献为零。MSLE/RMSLE 要求真实值和预测值非负。分位数损失默认使用 0.5；支持的分位数模型使用实际拟合分位数。MASE 分母仅从训练目标的朴素或季节朴素误差计算，按组分段，不跨组作差；常量或不足的训练序列返回 null。缺少类别、概率、参考标签或不符合定义域时，相应指标为 null，并在 `metric_notes` 中说明原因。

## 界面配置与搜索目标

“算法与评价”页的搜索目标下拉框可选择与任务兼容的指标；最大化/最小化方向自动确定。勾选的展示指标用于图表与导出。单任务结果 JSON 包含逐类结果、指标说明和交叉验证明细；指标图支持交叉验证均值。

模型参数下方的“划分与评价 JSON”默认 `{}`，可配置：

```json
{
  "split":{"strategy":"group","group_column":"subject_id"},
  "cross_validation":{"folds":5,"strategy":"group","gap":0},
  "reference_labels_column":"reference_label"
}
```

这三个顶层字段均可省略。分类/回归一般不需要 `reference_labels_column`；它专供聚类或异常检测的独立标签。分组、时间和参考标签列会从自动特征选择中排除。异常搜索若源标签不是 `-1/+1`，在高级批量 `objective.label_mapping` 中显式映射。

`split.strategy` 支持：

- `random`：默认原有分层分类 / 随机回归 60/20/20 划分，保持旧配置摘要兼容。
- `group`：整组划分，组不跨训练/验证/测试；至少 5 组，比例按组近似 60/20/20。
- `time`：按 `time_column` 递增划分，未指定时按源行顺序；指定时间列必须无缺失、无重复。

`cross_validation` 支持 2–10 折，`strategy` 为 `auto`、`stratified`、`kfold`、`group` 或 `time`。`auto` 沿用分组/时间隔离，普通分类用分层折，回归用 KFold。滚动折可用 `gap` 留出间隔。分组/滚动验证必须配合相同类型的外层划分。

折评估在原训练+验证的 80% 开发分区内执行，每折独立拟合预处理；最终 20% 测试集保持保留。启用后，搜索使用折均值，记录标准差和逐折行位置；单个训练/验证模型的结果仍单独保留。每个候选消耗“1 次主模型 + 折数”的实际拟合预算，启动前必须有足够额度。模型内部的树、Bagging 成员和 Stacking 内部拟合属于一个模型训练操作；预算计数覆盖主模型及显式外层验证折。中断时沿用整候选重做的恢复策略，旧尝试保留在事件记录中。

折评估支持普通表格分类/回归及 T01–T05。HMM 和窗口神经模型使用已有的 SequencePlan 分区、验证与早停流程，不能配置普通表格折评估。不同划分或折设置的任务不会放入同一批量比较组。

## 传统时序与窗口深度模型

T01–T05 是单序列、单变量、固定起点预测器。目标列是历史序列；特征值不作为外生变量，输入未来表格的行数代表未来连续步数。支持只有一列目标的 CSV。必须使用时间划分，例如：

```json
{"split":{"strategy":"time","time_column":"timestamp"}}
```

T02 默认季节长度 `seasonal_period=12`。T03 可用 `trend="add"`、`seasonal="add"`、`seasonal_period=12`；T04 的 `order` 和 T05 的 `seasonal_order` 用 JSON 数组指定。模型只拟合训练目标，预测验证/测试时不追加其真实值。单任务模型若仍只拟合了 60% 训练数据，最终测试会跳过验证区间后，从同一训练起点预测；批量 winner 则按既有流程先在训练+验证上重拟合。输出报告真实的预测 lead。预测模型每项任务处理一个序列，多序列请拆为独立任务。

N07/N08/N09 延用 N04/N06 的窗口配置，例如 `{"window":10,"horizon":3}`，分别提供卷积、因果扩张卷积和含位置编码的 Transformer 编码器。当前每次配置预测一个固定 horizon；按步长报告该 horizon 的误差，不将同一个配置伪称为多步输出。Autoencoder 通过 `n_components` 选择瓶颈维度，以训练集重建损失学习，再对新样本输出瓶颈特征。

## Python 配置示例

```python
from pyml_workbench import DatasetConfig, ExperimentConfig, EvaluationConfig, ObjectiveSpec, SplitConfig

config = ExperimentConfig(
    dataset=DatasetConfig("data.csv", target_column="y", feature_columns=("x1", "x2")),
    task="classification",
    model_id="C30",
    evaluation=EvaluationConfig(folds=5),
)
objective = ObjectiveSpec(metric="log_loss")  # 自动选择 min
```

高级批量数据项可填写 `split`、`evaluation`、`reference_labels_column` 和 `objective`，其余队列、冻结和最终测试操作沿用原流程。

依赖同步（保留环境中其它工具）：

```powershell
uv sync --locked --inexact --no-editable --no-dev --extra desktop --extra search --extra sequence --extra deep --extra manifold --extra forecasting --extra xgboost --extra lightgbm --extra catboost
```

`manifold` 锁定 umap-learn 0.5.12，`forecasting` 锁定 statsmodels 0.15.0；未安装时界面显示对应 extra，应用不会自动安装。当前 `standard` 构建配置提供 90 个基础模型，`full` 配置包含全部可选扩展；本轮源码修改本身不代表已重新构建或验证 EXE。

## 本轮 EXE

完整解压 `dist/PYML-Workbench-0.4.4-windows-x64-full-models111-20261027.zip` 后运行 `PYML-Workbench.exe`。本机新目录为 `dist/model-expansion-20261027/PYML-Workbench-full`，包含本页列出的可选依赖；旧安装目录不变。发行与验证范围见[EXE 指南](exe-guide.md)。
