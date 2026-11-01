# PYML便携模板

本目录不含固定机器路径、个人身份、原始EEG、原生运行清单或业务日志。程序、报告示例和CMD可以复制到新位置；请保留相对目录结构。

## 文件

- `一键HyperEEG转换.cmd` / `一键转换.cmd`：同一转换入口的两个名称。启动后选择Python、工作目录、输出目录、插件、引用式结果包、MATLAB和HyperEEG工程。
- `一键生成APA报告模板.cmd`：选择报告Python、工作目录和输出目录，生成可编辑的空白APA Word/Markdown模板；不运行模型。
- `apa_report/apa_report_template.docx` / `.pdf` / `.md`：已生成的填写模板，包含方法、结果、讨论及子章节、指标/图表占位、真实参考文献与完整方法差异附录。PDF供预览，Word可编辑。
- `apa_report/apa_report_example.docx` / `.pdf` / `.md`：保留数值、图表和引用的匿名化完整历史接口示例。人工目标、输入契约例外、未通过及优化失稳均保留；不能作为新实验结果。
- `apa_report/figures/`：示例中14幅图的PNG/PDF；`figure_data/`是重绘所需的匿名汇总和固定示例输出，不含源矩阵、主体映射、原始特征名称或本机指针。
- `apa_report/plot_model_comparison.py`：由保存的汇总重绘全部示例图，不拟合模型。分类逐模型accuracy/AUC、回归误差、预测—目标、特征排序、PCA和覆盖图均有独立输出。
- `hypereeg_converter/`：完整转换调度、固定进度UI、内存准入调用、流式累积和原生表桥接代码；`settings.example.json`的机器字段为空。
- `validation-summary.json`：模板级实际验证范围；不把模板启动检查称为新的原生数据转换成功。

## 环境与启动

CMD优先使用环境变量`PYML_PYTHON`，其次使用系统Python launcher或PATH中的Python启动标准库配置窗口。窗口中的Python解释器决定实际执行环境，可以选择各自已有的虚拟环境。转换环境需要插件源码及其依赖和PySide6；报告环境需要python-docx，图表重绘需要NumPy、pandas和Matplotlib。参考依赖清单分别位于子目录；插件/MATLAB不随模板分发，也不自动安装。

首次选择的工作目录和输出目录必须在SKILL外；个人配置只写到工作目录的`convert.local.json`或`report.local.json`，程序副本放在其`apps/`下。直接修改SKILL中的`settings.example.json`不会启动分析，不能向公共模板粘贴个人路径。

转换界面原位置更新阶段、已用时间和当前步骤ETA，详细路径/逐项状态保留在用户输出。ETA没有有效速度时显示“正在估计”。模板生成窗口显示阶段和已用时间，未知ETA不编造。转换失败、取消、支持范围内完成分开处理。

## 转换支持边界

此转换程序针对`结果清单.json`中`storage_mode=linked_native_results`的HyperEEG引用式包，要求保留的原生目录、`特征结果索引.mat`、带源码的HyperEEG转换插件，以及包含`FeatureArtifactMemoryEstimate`的HyperEEG工程。只声明当前桥接支持的signal/spectrum/nonlinear标量记录表；其他方法族或不支持的坐标单独记录，不能说上游全部产物都已转换。

MATLAB每个Artifact只加载一次，多表复用；先做内存准入，不绕过预算。source ID、数值与有限性掩码保留；峰表缺少真实`peak_index`映射时明确不支持。原始张量不猜轴展平，身份/标签不从文件名推断。若使用别的格式，按SKILL的插件契约另行核实，不能直接套本模板。

## 报告复用

空白模板中的`[[...]]`须用新研究事实填充；未知项目保留缺失，不沿用示例的18/49行、696维、3,260槽位或分数。示例使用人工目标/顺序并包含专用夹具，不能解释为真实脑电预测效度。报告模板不是“任意结果目录自动分析器”；结果目录到完整APA报告仍由PYML SKILL核查实验设计、保存指标和引用后完成。

```text
python apa_report/build_apa_template.py --output <报告输出目录>
python apa_report/plot_model_comparison.py --output <重绘输出目录>
python launch.py convert --config <用户侧配置文件> --validate-only
```

最后一条只核验配置契约，不加载MAT、不转换、不训练。私有路径可以出现在用户侧来源台账和执行日志中，但不得回写本目录或提交Git。

## 来源与验证

转换实现来自已验证的一站式转换程序，模板改动限于配置、相对入口和可迁移启动。`export_loaded_table.m`由公开转换插件桥接实现调整为接受已载入Artifact，保留标量校验和17位数值序列化；插件来源见SKILL的`references/plugins.md`。本轮未重跑真实原生MAT，也未重跑机器学习或最终test。匿名示例保留原有拟合指标，另行执行路径/元数据扫描、图表重绘和文档渲染检查。
