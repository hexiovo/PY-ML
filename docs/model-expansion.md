# 模型、指标与验证方式

PY-ML 当前模型目录包含 **111 个条目**。条目数量不表示所有模型都已在全部依赖、参数和数据条件下完成拟合验证。可用能力取决于所选模型、输入结构和运行环境；参数目录见[模型索引](model-index.md)，Windows 部署见[下载与安装指南](exe-guide.md)。

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

## 指标与搜索目标

分类、回归、聚类、降维和异常检测可用的指标取决于模型实际提供的预测或表示能力及指标定义域。Log Loss、Brier 等概率指标要求真实概率；ROC-AUC/AP 可使用概率或语义明确的决策分数。二分类正类和多分类平均方式须按结果中的类别顺序解释。缺少概率、类别或独立参考标签时，相应指标可能为空，并附原因。逐项定义见下文和结果文件中的 `metric_notes`。

搜索目标应与研究任务相符。默认方向由指标决定；异常检测自动搜索必须提供独立真实标签和明确映射，不能把模型自身异常预测当作真值。预测指标用于模型选择和预测性能报告，不能代替研究假设的推断检验。

## 数据划分与交叉验证

原生 `ExperimentConfig` 用 `split` 表示外层数据划分，用 `evaluation` 表示留出流程上附加的交叉验证；独立参考标签列是 `dataset.reference_labels_column`。`DatasetConfig` 不提供任意数据转换图、编码/分隔符或统计分析字段。外部步骤必须保存在独立的数据准备和统计计划中。

留出比例固定为 60/20/20。`split.strategy` 支持 `random`、`holdout`、`group`、`time` 和兼容的 `kfold` 路径：

- `random` / `holdout`：随机留出，分类按类别分层，回归按行随机。
- `group`：按整组划分，至少 5 个非缺失组；组大小不同会使行比例近似而非精确。分组列不得成为特征。
- `time`：按唯一、非缺失时间升序划分；若未提供时间列，则使用源行顺序。外层 group 与 time 不能同时启用。
- `kfold`：普通表格分类/回归的独立 K 折路线，`split.n_splits` 为 2–10，先留出 20% 独立测试，再在其余开发集上做折验证。不能同时设置 `evaluation`；HMM、窗口序列、传统时序预测及无监督任务不支持此路线。

`evaluation` 可配置 `folds`（2–10）、`strategy`（`auto`、`stratified`、`kfold`、`group`、`time`）和非负整数 `gap`。它在原训练+验证的 80% 开发分区中评估，最终 test 保留。`auto` 沿用外层 group/time 结构；普通分类用分层折，回归用 KFold。Group/time 折需要匹配的外层分区。时间 `gap` 只适用于相应的折验证，不代表最终测试的 purge/embargo。每折的填补、缩放、编码和可选降维均只在该折训练行拟合。

两种 CV 路线的预算口径不同：独立 `split.strategy="kfold"` 中 `max_fits` 按候选计数，底层每个候选会在多个折上拟合；`evaluation` 路线中 `max_fits` 按模型拟合次数计数，每个候选通常消耗 1 次主拟合加折数。冻结后的重拟合和唯一最终测试另行执行。请在批量搜索的预算说明中核对当前设置，不能把候选上限直接称为底层模型拟合总数。

## 原生预处理和降维

普通表格数值列使用训练数据拟合的缺失值填补与缩放；类别列使用训练数据拟合的填补和 one-hot 编码。CategoricalNB 还会进行训练内分箱。学习参数仅从允许的训练分区估计；折验证时每折重新拟合。

原生可选训练前降维字段为旧版 `pca_components`，或成对出现的 `reduction_method` 与 `reduction_components`。方法包括 PCA、TruncatedSVD、FastICA、Gaussian Random Projection 和 Kernel PCA。降维接在普通特征预处理后，按主训练分区或 CV 训练折拟合。有效秩、样本数和模型兼容条件由运行时校验；如需单位换算、reshape/merge、量表合成、目标变换、SMOTE、目标编码等，必须用独立脚本在划分约束下实现。降维不能与要求非负输入的 MultinomialNB/ComplementNB、CategoricalNB 或部分序列/原始特征模型任意组合。

## 传统时序与窗口深度模型

T01–T05 是单序列、单变量、固定起点预测器。目标列是历史序列；特征值不作为外生变量，输入未来表格的行数代表未来连续步数。支持只有一列目标的 CSV。必须使用时间划分，例如：

```json
{"split":{"strategy":"time","time_column":"timestamp"}}
```

T02 默认季节长度 `seasonal_period=12`。T03 可用 `trend="add"`、`seasonal="add"`、`seasonal_period=12`；T04 的 `order` 和 T05 的 `seasonal_order` 用 JSON 数组指定。模型只拟合训练目标，预测验证/测试时不追加其真实值。单任务模型若仍只拟合了 60% 训练数据，最终测试会跳过验证区间后，从同一训练起点预测；批量 winner 则按既有流程先在训练+验证上重拟合。输出报告真实的预测 lead。预测模型每项任务处理一个序列，多序列请拆为独立任务。

N07/N08/N09 延用 N04/N06 的窗口配置，例如 `{"window":10,"horizon":3}`，分别提供卷积、因果扩张卷积和含位置编码的 Transformer 编码器。当前每次配置预测一个固定 horizon；按步长报告该 horizon 的误差，不将同一个配置伪称为多步输出。Autoencoder 通过 `n_components` 选择瓶颈维度，以训练集重建损失学习，再对新样本输出瓶颈特征。

## Python 配置示例

```python
from pyml_workbench import DatasetConfig, ExperimentConfig, EvaluationConfig, ObjectiveSpec

config = ExperimentConfig(
    dataset=DatasetConfig("data.csv", target_column="y", feature_columns=("x1", "x2")),
    task="classification",
    model_id="C01",
    evaluation=EvaluationConfig(folds=5),
)
objective = ObjectiveSpec(metric="balanced_accuracy")  # 自动选择 max
```

`reference_labels_column` 属于 `DatasetConfig`，用于聚类或异常检测的独立参考标签。搜索配置另用 `ObjectiveSpec`。批量入队、冻结和最终测试步骤见[批量搜索指南](batch-guide.md)；不要把外部预处理或统计计划字段加进原生实验配置。

依赖同步（保留环境中其它工具）：

```powershell
uv sync --locked --inexact --no-editable --no-dev --extra desktop --extra search --extra sequence --extra deep --extra manifold --extra forecasting --extra xgboost --extra lightgbm --extra catboost
```

`manifold` 使用 umap-learn，`forecasting` 使用 statsmodels；可选依赖未安装或不可用时，模型条目仍可见，运行前会报告对应原因。Windows 程序的下载入口是 [GitHub 最新 Release](https://github.com/hexiovo/PY-ML/releases/latest)。
