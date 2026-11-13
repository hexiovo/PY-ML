# PYEDA 特征转换与质控

使用 [PYEDA 转换插件](https://github.com/hexiovo/PY-ML_support_trans/tree/master/pyeda) 读取已完成的结果包。检查 `manifest.json`、`summary.json`、`intake/analysis_settings.json`、`intake/feature_definitions.csv` 和选定的特征来源。参与者任务角色表使用 `participant_task_role`，暴露汇总表使用 `participant_exposure`，压缩记录文件使用 `records`。后者默认只保留 `status=valid` 的窗口。登记哈希不一致时停止并核对来源。

默认只导出 `sample_id`、来源要求的身份键、可用的质控状态和所有登记的 `feature__*`。按原样保留来源角色、任务与窗口信息，建模时据实验设计确定独立分组。需要转换结果列时，操作员可显式启用该选项，并根据研究方案确认标签来源和任务类型。

用 `valid_mask.csv` 核对每列有限值，检查样本键、行序、全缺失列和单位。公开示例位于 [PYML 的 examples/pyeda](https://github.com/hexiovo/PY-ML/tree/master/examples/pyeda)，可合成输入或处理本地结果包。示例把非有限值改为空值，保留全缺失特征并在质控摘要标记。先根据独立单位划分数据，再在训练范围内拟合必要的填补或缩放。

发布公开示例前，检查代码、配置、源清单、文档、归档和示例输出。使用合成身份和合成特征，排除真实路径、运行编号、标签、身份、日志与数据。将用户的预填配置及实际结果存放在私人工作区。
