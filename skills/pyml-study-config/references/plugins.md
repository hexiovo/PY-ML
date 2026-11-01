# HyperEEG 与 PYfNIRs 转换插件

本流程将已有科研特征整理成 PYML 输入。先识别来源软件、输入类型与每行含义。插件可读取文件不代表支持其中全部结果；不能把原始信号、统计结果表或任意张量直接当成逐样本训练矩阵。转换不代替信号预处理、训练或最终统计检验。

## 获取与连接：主试操作卡

从 [转换插件 Releases](https://github.com/hexiovo/PY-ML_support_trans/releases) 获取对应来源的发行包。执行时核对实际资产、版本、平台与说明，记录下载地址和文件 SHA256；仅在发布方提供哈希且比对通过时称为发布方校验通过。技能包不含插件 EXE，PYML 依赖安装器也不下载插件或 MATLAB。

交付主试前，将下列步骤中的文件名、目录、参数替换为实际核实值：

1. 完整解压对应插件，保留 EXE、`_internal` 和 `pyml-plugin.json`；选择直接包含清单的目录，不选 ZIP 或上一级目录。
2. 在 PYML 0.4.6 打开“格式 → 数据转换 → 插件设置…”，分别选择 HyperEEG、PYfNIRs 目录。EXE 版不需 Python；Python 入口指定插件依赖环境的 `python.exe` 或 `pythonw.exe`。
3. 保存后，从“格式 → 数据转换 → HyperEEG”或“PYfNIRs”启动。未启用时保存检测提示，核对清单和路径，不修改来源标识强行通过。
4. 先对一个代表文件检查或预览，核对身份、特征和目标，再执行已确定范围的批量转换，输出放入新的运行目录。
5. 按本指南验收输出，再导入 PYML，显式选择特征、目标及划分字段。

目标版本菜单优先于旧文档。上游曾使用“插件 → 设置”或“PYfNIRs 设置…”；不能让主试在 0.4.6 中寻找旧入口。只有真实存在的按钮和已核实的命令才能写进操作卡。

## 接入契约与交付文件

PYML 0.4.6 检查插件清单 `schema_version=1`、`id="data-conversion"`、非空 `name/version` 和 `entrypoint`。入口 `kind` 为 `executable` 或 `python`，`path` 必须是目录内实际 `.exe` 或 `.py` 的相对路径。PYfNIRs 必须声明 `source_project="PYfNIRs"`；HyperEEG 使用 `source_project="HyperEEG"`，并兼容旧版未声明来源的清单。此清单 schema 1 与主程序本地插件注册表 schema 2 不同。

主程序检测只检查清单与启动文件，不验证数据、MATLAB、科学参数或运行依赖。菜单只启动转换器，不自动传入实验设计、`experiment.json` 或转换参数。

`conversion.plan.json` 是研究侧记录，不直接传给插件或 PYML 配置解析器。至少包含：

| 项目 | 必须记录 |
|---|---|
| 来源与环境 | enabled、来源软件、插件版本、下载地址与哈希、清单、入口、工作目录、Python/MATLAB 实际路径 |
| 输入契约 | 文件/工作表/原生路径、每行含义、观察/主体/配对/窗口 ID、完整连接键、特征坐标、数值列、单位 |
| 科学选择 | 真实标签和协变量来源、用途、纳入排除规则、支持能力与适用条件 |
| 转换设置 | 插件原生参数文件、契约依据、输出格式、合并与重名策略、命令或逐项 GUI 设置 |
| 交接与验收 | 预期输出、行数/键/值/缺失检查、实际结果、失败项、未运行项、PYML 列映射和当前状态 |

需要时交付 `hypereeg-options.json` 或 `pyfnirs-selection.json`，字段必须由目标版本真实接口确认。未知事实写 missing；预期输出路径与实际存在的输入文件分别标记。插件参数、`preprocessing.plan.json`、`experiment.json`、`statistics.plan.json` 各自遵循其接口，不能混传。

## HyperEEG：特征表转换

官方说明覆盖 `FeatureResult.longData` MAT、正式特征 XLSX/清单、单导特征/频带功率/连接/网络 CSV，以及明确映射的标量 MATLAB table。任意张量、含向量单元的表、原始 BDF/RDS/IDS 不可根据形状猜测展平；先在来源软件导出有真实身份和坐标的特征表。

MAT 读取需要有许可的本机 MATLAB。核对 MATLAB 路径和表路径；有多个表时不能任取第一个。外部标签表用两边真实存在的完整键连接，要求右表键唯一、每个样本匹配且已有元数据无冲突。禁止按行号、文件名含义或排序推断标签。

从当前 `hypereeg_converter.core.Options` 核对字段与默认值后生成 `hypereeg-options.json`。公开映射字段包括 `sample_columns`、`coordinate_columns`、`value_columns`；其他设置须查实后填写。以下仅演示语法：

```json
{"sample_columns":["subject_id","session"],"coordinate_columns":["channel_index"],"value_columns":["rms"]}
```

若真实数据含多个片段/窗口，样本键须补齐区分列，不能直接套用示例。每项映射都要能追溯到真实字段、单位与研究定义。

官方命令形态如下，主试版替换所有占位符；PowerShell 调用带引号的程序路径时前置 `&`：

```text
"<插件EXE>" --input "<输入路径>" --output "<新输出目录>" --options "<hypereeg-options.json>"
```

源码方式在 `hypereeg` 工作目录用插件环境的 Python 运行 `launch.py`，参数同上。执行前确认实际版本支持这些参数。窗口版可能不输出终端日志，启动成功不等于转换完成，必须读输出摘要。

核对成功目录中的 `dataset.csv`、`valid_mask.csv`、`feature_dictionary.json`、`source_tables/`、`provenance.json`，以及本轮 `batch_summary.json`。PYML 选择 `feature__` 数值列与有依据的 `target__` 目标；保留 `sample_id`、`meta__` 身份用于追踪和分组。掩码只表示数值有限性，不证明源数据通过科学质量审查。检查逐文件和合并状态，不把部分成功写成全批完成。

## PYfNIRs：Study 与明确支持的来源

输入包括规范 Study MAT 或符合 `pyfnirs.matlab-export/1` 的 `study_export.json`。原生 MATLAB table MAT 需要外部 MATLAB；已验证的中间 JSON 可直接读取。`results.xlsx` 的统计结果不能充当逐样本训练数据。

先检查输入和目标插件 `capabilities.json`。`study-defined` 只搬运规范 Study 已登记的 `FeatureDefinitions`、`Values`，不表示所有 raw Results 都可转换。其他适配器的能力 ID、轴选择、支持条件和参数以实际注册表与检查结果为准；unverified/unsupported 项不得强行启用。

`pyfnirs-selection.json` 按当前契约明确 `source_adapter_id`、`identity`、`feature_selections`、`targets`、`covariates`、`output_format`、`merge_mode`、`collision_policy`。每项特征明确 `capability_id`、`feature_id`、`source_path`、`axis_selection`、`summary_parameters`。身份路径与 FeatureID 来自输入检查；Group、Condition 不自动作为目标，不复制合成示例 ID 到真实配置。

官方源码 CLI 形态如下，使用插件依赖环境，并在可导入插件模块的目录运行；不要将模块子命令直接当作发行 EXE 的参数：

```text
"<插件Python>" -X utf8 -B -m pyfnirs_converter.cli inspect "<输入文件>"
"<插件Python>" -X utf8 -B -m pyfnirs_converter.cli preview "<输入文件>" --selection "<pyfnirs-selection.json>"
"<插件Python>" -X utf8 -B -m pyfnirs_converter.cli convert "<输入目录>" --selection "<pyfnirs-selection.json>" --output-dir "<新输出目录>"
```

按需显式加 `--recursive`；原生 MAT 用 `--matlab` 指定 MATLAB。仅有 EXE 且 CLI 未核实时，交付 GUI 操作：选择目录 → 检查输入 → 选择适配器、身份、特征、目标及协变量 → 预览转换 → 选择格式、合并和重名策略 → 启动批处理。未核实的 GUI JSON 导入按钮不能写入说明。

CSV 输出核对 `FeatureMatrix.csv`、`Samples.csv`、`FeatureDictionary.csv`、`Provenance.json`；XLSX 核对对应工作表。PYML 导入 FeatureMatrix，XLSX 明确 `sheet_name="FeatureMatrix"`。掩码和缺失原因在 Provenance 中，按观察与特征顺序对应。若所需分组字段仅在 Samples 中，先按真实 ObservationID 做受验证的键连接，保持行数和顺序，再将新表设为 `dataset.source_path`。

## 一键转换入口的进度要求

当用户要求点击运行的一站式转换时，除真实可用入口和原生参数外，必须交付可见实时进度，遵守SKILL.md“一键程序与实时进度”及operator.md验收要求。MATLAB导出、源哈希检查、逐表转换、数据写入、逐值/掩码校验和摘要保存各阶段均显示状态；已处理计数覆盖未选择、跳过、不支持和失败分支。外部进程可在后台运行，但其进度需转发到可见窗口，至少每5秒显示进度或耗时等待提示。

转换范围必须在启动和终态中明确。保留逐项结果与进度日志/快照，显示“支持范围内完成”不能替代全特征覆盖；范围缩小、原生不可用、未映射、失败和取消分别记录，不静默合并为成功。不得为显示进度而重算特征、推断标签或重新启动模型最终测试。


对本轮实际见到的 `Artifact.payload.periodicPeaks`，必须保留 `peak_index` 等真实峰坐标。当前插件若只映射channel_index，不能把同通道多峰当作数据重复后静默跳过或平均；先明确记为“峰坐标未映射”，待补齐有依据的坐标再转换。真正的重复样本×特征冲突继续失败。

## 内存与错误分类核查

采用SKILL.md“大型原生产物与内存边界”：原生MAT解码每产物一次，复用已加载对象导出多表；载入前按展开尺寸估算准入，源表逐张释放，输出矩阵设规模上限。复杂MCOS/table可能仍需完整原生解码，不能声称通用流式解码已经实现。

资源拒绝、OOM、I/O和数据冲突必须停止并向外报告，不能被“格式不支持”的catch吞掉。限定到实际确认的不支持列类型/布局才可继续并记不支持。主界面为固定UI（阶段、已用时间、预计剩余），细节日志保留在用户指定输出目录。必须区分未选择、未映射、原生不可用与失败，并说明本轮覆盖范围。

## 数据验收与后续统计

- 核对样本数、独立单位数、重复键、特征顺序、单位、条件/窗口/配对身份；对已知记录逐值比对，保留数值精度和缺失原因，不补零。
- 标签按 ID 对齐。监督目标缺失、连接冲突或未匹配时停止；合并前比较特征定义、单位、节点/通道顺序及参数指纹，不能只比较列名。
- 保存失败、跳过、取消和未处理状态，取消后的部分输出不代表完整研究样本。保留来源哈希、参数、输出字典和摘要。
- 同一主体、配对、相关窗口的划分尊重真实独立单位；源特征若在全体样本上共同拟合，格式转换无法消除泄漏，须回到来源处理解决。
- 验收后重新检查原生配置，显式选择特征与目标。ID 不自动成为特征，协变量按研究用途决定。最终检验继续遵循 `statistics.md` 的独立单位、重复测量和多重比较规则。
- 分别记录清单通过、启动成功、预览通过、转换完成和数据验收通过。只检查文档或合成示例时，不标为真实数据已验证；缺依赖、定义或验证时保留 blocked/pending_real_data。

## 核对依据

2026-11-01 核对了 PYML 0.4.6 的 `plugins.py`、`plugin_dialog.py`、`gui.py`，以及 [HyperEEG 使用说明](https://github.com/hexiovo/PY-ML_support_trans/blob/master/docs/HyperEEG使用说明.md)、[PYfNIRs 使用说明](https://github.com/hexiovo/PY-ML_support_trans/blob/master/docs/PYfNIRs转换器使用说明.md)、[PYfNIRs 数据契约](https://github.com/hexiovo/PY-ML_support_trans/blob/master/docs/PYfNIRs数据契约.md)。这些资料提供配置依据，不是具体研究的运行证据；使用时仍须复核实际安装版本。
