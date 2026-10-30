# 当前 PYML 契约与证据边界

本文件是随技能附带的起草参考，不替代目标环境核验。使用前运行 `scripts/inspect_contract.py --project <PYML 项目路径>`，确认导入模块位于该项目的 `src/pyml_workbench`，并记录包版本、解释器和源码哈希。契约快照只表示生成时检查过的 checkout；同版本号不等于源码或依赖相同。

## 配置对象

当前 `ExperimentConfig` 原生顶层字段：

- 必需：`dataset`、`task`、`model_id`
- 可选：`parameters`、`split`、`output_dir`、`sequence`、`evaluation`、`pca_components`、`reduction_method`、`reduction_components`

没有原生的任意 `preprocessing` 转换图，也没有推断统计配置字段。`ExperimentConfig.from_dict` 只读取它认识的顶层键，因此可能静默忽略未知顶层字段；本技能的 checker 必须先严格拒绝未知键，再调用真实解析器。嵌套 dataclass 会拒绝不认识的字段。研究侧计划和脚本必须保存在单独文件，不能塞进 `experiment.json`。

`DatasetConfig` 的字段为 `source_path`、`sheet_name`、`target_column`、`feature_columns`、`reference_labels_column`。输入格式为 CSV/XLSX/XLS。当前对象没有编码、分隔符、小数点、日期格式、合并、ID、单位转换或任意变换字段；CSV 读取细节须按目标版本的读取 API 实查。监督任务须给目标列；缺失目标会阻止拟合。结构列、目标列和独立参考标签不得混作特征。

`SplitConfig` 字段为 `seed`、`train_fraction`、`validation_fraction`、`test_fraction`、`strategy`、`group_column`、`time_column`、`n_splits`。留出比例固定为 60/20/20。策略为 `random`、`holdout`、`kfold`、`group`、`time`；`n_splits` 为 2–10。

- `random` / `holdout`：普通随机留出；分类按类别分层，回归随机划分。
- `group`：按组整体划分。至少需要五个非缺失组，行比例近似而非精确；分类要求各分区均有全部类别。`group_column` 只能和该策略一起用。
- `time`：按唯一、非缺失时间值排序划分；未指定 `time_column` 时使用源行顺序。`time_column` 只能和该策略一起用。
- 外层 group 与 time 不能同时启用。
- `kfold`：直接使用独立的 K 折验证路径，只支持普通表格分类/回归；保留 20% 最终测试，折在其余开发数据上进行。它不能与 `evaluation` 同时设置，也不支持 HMM、窗口序列、传统预测模型或无监督任务。

`EvaluationConfig` 字段为 `folds`（2–10）、`strategy`（`auto`、`stratified`、`kfold`、`group`、`time`）和非负整数 `gap`。它将额外折验证附加到留出流程，折只使用 train+validation 开发分区，最终 test 保留。每折须在该折训练行重新拟合学习型预处理。`group` / `time` evaluation 必须配合相应的外层 split；`auto` 会沿用 group/time 结构，普通分类选择分层折，普通回归选择 KFold。时间 gap 只影响折验证，不是最终 holdout purge/embargo。

两种折验证的搜索预算口径不同，应在参数台账中分别说明：`evaluation` 附加折验证按每候选一次主拟合加折数扣真实拟合预算；兼容路径 `split.strategy="kfold"` 的 `max_fits` 按候选计数，底层每折仍会调用多次拟合，不能把候选数说成底层 estimator 拟合总数。冻结后的重拟合和唯一最终测试另行记录。

## 原生预处理与限制

普通表格输入会经过当前原生预处理：数值缺失用中位数填补，再按模型使用标准缩放或相应的非负缩放；类别值转字符串、用训练类别众数填补、one-hot 编码，并忽略拟合后出现的未知类别。CategoricalNB 另外在训练输入上进行分箱编码。具体实现须以目标 checkout 的 `preprocessing.py`、模型和快照为准。

当前原生训练前降维字段有两种表示：旧字段 `pca_components`，或新字段 `reduction_method` 与 `reduction_components`（必须成对）。新方法包括 `pca`、`truncated_svd`、`fast_ica`、`gaussian_random_projection`、`kernel_pca`。降维接在特征填补、缩放和类别编码之后；每个主训练分区或 CV 训练折单独拟合，再用于对应验证、测试或预测数据。维数、行数、有效秩、符号范围和模型兼容限制由当前源码检查。降维不是任意转换图；不能借它表达 log、Box-Cox、单位换算、目标变换、特征筛选、重采样或其他研究步骤。

当前支持不了的转换（例如 long/wide reshape、merge、语义量表合成、单位换算、目标变换、目标编码、重采样和信号特征生成）应由独立、可审阅的 `prepare_data.py` 完成。若研究需要推断统计，必须另交 `statistics.plan.json` 和可实际运行的 `run_statistics.py`。两者都有清楚输入输出契约并通过合成检查前，不能把方案标为可执行。

## 生命周期、搜索与模型

普通留出与 `evaluation` 附加折验证走 `prepare_experiment` → 检查 train/validation → `freeze_experiment` → `evaluate_test`；折只在开发数据中形成，且逐折重拟合预处理。直接 `split.strategy="kfold"` 是另一条兼容路径：`prepare_experiment` 返回 OOF 折评估会话，不保留最终模型，`freeze_experiment` 会拒绝该会话。选定配置后，单任务可调用 `pyml_workbench.cv_experiment.refit_cv_session`，它在开发数据重拟合并返回已冻结会话，再由调用方按批准流程调用 `evaluate_test`；不要把该 refit 混作 OOF 评估的一部分。`finalize_experiment` 是自动重拟合并进入最终测试的便捷路径，只能在明确批准后调用。批量搜索应通过已验证的 winner 冻结与最终化 API 完成选择。两条路径重复读取成功结果会返回缓存；失败或中断后的同一会话不可重测。`run_experiment` 会自动走完整最终化流程，不应用于只检查配置的任务。

N01/N02 的 early-stopping 训练不支持直接 `split.strategy="kfold"` 兼容路径；不要通过省略训练过程、手工伪造折结果或调用不匹配的 API 绕过限制。具体模型与模式支持仍须在目标源码和 API 中核实。

HMM 与 N04/N06/N07/N08/N09 使用 `SequenceConfig` / `SequencePlan` 描述组、时间排序或窗口，并要求外层 `split.strategy` 为 `random` 或 `holdout`；不能据此声称支持普通 group/time 外层划分。T01–T05 传统预测模型必须使用 time split。遇到研究设计需要同时满足模型与分组/时间隔离、而当前接口没有已验证路径时，停止并标 `blocked`。

`SearchSpec` 字段为 `method`、`space`、`objective`、`max_fits`、`timeout_seconds`、`max_proposals`、`seed`；方法为 grid、genetic、annealing、random、tpe。`SearchSpace` 只接受 `fields` 和 `fixed`；字段项支持 `type`、`low`、`high`、`values`、`log`、`when`。`ObjectiveSpec` 字段为 `metric`、`direction`、`split`、`objective_labels_column`、`label_mapping`、`coverage_min`。实际可用指标与参数应从当前源码 API 检查，不根据旧快照猜测。模型目录总数、smoke 测试或构造器解析不代表全部 111 项都完成拟合，也不证明研究数据适用。

预测指标和 CV 折均值/标准差是模型选择与性能评估资料，不是独立样本推断检验。PYML 不会自动生成研究假设对应的效应量、置信区间、多重比较校正或推断统计报告。

桌面程序由 `python -m pyml_workbench` 启动。内部 worker 参数不是面向研究人员的通用配置 CLI。使用公开 Python API 或真实 GUI；不要宣传不存在的 `validate-config`、`plan` 命令。

## 快照与重新核实

快照中的 dataclass、模型清单、参数 schema 和源码 SHA-256 是起草参考。使用当前项目显式导出后，检查 import 路径和源码哈希；缺依赖、文件或导入错误时如实报告 `NOT VERIFIED`。配置解析、模型参数构造、输入列头预览、合成数据行为、真实数据检查、推断统计执行是不同证据层级，不能相互替代。`check_bundle.py` 不拟合模型、不查看测试指标、不验证研究设计，也不授予 `ready_for_execution`。
