# PY-ML 工作台

PY-ML 是面向表格和序列数据的中文机器学习工作台，提供单任务训练、批量搜索、数据转换、结果导出和 Python API。它包含 111 个模型目录条目，支持分类、回归、聚类、降维、异常检测、时序预测、HMM 和深度学习任务。模型条目数描述目录范围，不代表本版已对每个模型完成独立的全流程拟合验证；各模型的能力和依赖见[模型与参数索引](docs/model-index.md)。

## 下载与启动

Windows 用户请从 [GitHub 最新 Release](https://github.com/hexiovo/PY-ML/releases/latest) 下载完整 Windows x64 程序包。GitHub 页面的 **Code → Download ZIP** 是源码包，不是可直接运行的程序。

1. 将 ZIP 完整解压到一个有写入权限的文件夹；路径可包含中文。
2. 双击 `Start.cmd` 启动工作台。程序包已包含 Python 运行环境，无需另装 Python。
3. 如需安装到当前 Windows 用户目录，双击 `Install.cmd`。安装使用版本化的 `%LOCALAPPDATA%` 目录，不需要管理员权限。

运行时请保留整个程序目录，包括 `PYML-Workbench.exe`、`PYML-Worker.exe`、`_internal`、文档和 `skills`；不要只复制主程序或单独移动 `_internal`。Windows 压缩包可直接运行，也可用安装脚本部署。Release 页面还提供独立的 PYML 技能 ZIP、单文件便携版、Python wheel/sdist 和 SHA-256 校验文件。

## 初次使用

打开“帮助 → 操作指南”或按 **F1**，按页面步骤导入 CSV/Excel、选择目标列和特征、设置数据划分与模型，再运行训练和验证。确认验证结果后，冻结配置并按一次性流程执行最终测试。批量搜索、序列与深度模型的详细步骤见对应指南。

工作台支持表格数据的训练前转换与降维、验证与交叉验证、批量模型搜索和结果导出。模型选择与预处理应只使用训练/开发数据；最终测试应留到方案冻结后。模型运行能力取决于数据类型、模型限制和实际环境，不能把预测指标直接解释为统计推断结论。

### 数据转换插件

“格式 → 数据转换”提供 HyperEEG 与 PYfNIRs 转换入口，并通过“插件设置…”管理插件路径。首次打开时入口默认不可用；选择包含 `pyml-plugin.json` 的插件目录并保存，程序检查通过后启用对应入口。插件设置在重启后保留，路径失效时入口会停用并显示原因。

转换器独立于主程序发布，请从[转换插件仓库](https://github.com/hexiovo/PY-ML_support_trans)获取相应发行目录。主程序包不包含这两个转换器的实现。

### 运行环境

“帮助 → 检查运行环境…”会检查 Python、依赖、程序文件和必要资源，并区分必需组件与可选算法。需要补充组件时可用“一键安装”：安装在当前用户的专用依赖目录，只有后台加载检查通过后才切换生效；安装结束会再次检查，失败时保留已有环境。安装完成后按提示重启工作台。

## 使用 AI 生成实验配置

程序包内含 `skills/pyml-study-config/` 完整技能文件夹；Release 另提供独立 ZIP 和单文件便携版。该技能是给 AI 助手使用的工作说明，会帮助整理实验信息、生成 PYML 配置，并在需要时要求提供独立的数据准备脚本与统计脚本。它不会替研究者补猜单位、被试身份或统计单位，也不会因为配置生成而自动运行完整分析。

在支持技能目录的 AI 中，可把 `pyml-study-config` 文件夹复制到该 AI 的技能目录后在新会话中启用。其他 AI 可上传完整技能 ZIP，或直接提供单文件便携版。便携版包含技能正文与全部参考资料，但不包含可执行辅助脚本；没有文件或运行工具的 AI 只能给出完整文件内容和准确命令，不能声称已验证。

可用下面的提示开始：

> 请使用 PYML study-config 技能，把我的实验设计整理成可检查的 PYML 配置和操作包。先区分已知、建议和缺失信息；关键事实不足时集中询问。只依据当前源码契约声明可执行性，不要猜单位、身份或统计单位，不要自动读取最终测试结果。

## 文档

- [操作指南](docs/overall-guide.md)：从导入数据到导出结果的完整流程。
- [用户指南](docs/user-guide.md)：单任务操作与常用设置。
- [批量搜索指南](docs/batch-guide.md)：多数据集搜索、恢复和结果比较。
- [序列建模指南](docs/sequence-guide.md)和[深度学习指南](docs/deep-learning-guide.md)：序列、窗口模型与深度模型的要求。
- [Windows 下载与安装](docs/exe-guide.md)：便携运行、用户级安装和常见问题。
- [Python API](docs/api-guide.md)、[模型索引](docs/model-index.md)与[第三方组件说明](docs/third-party.md)：开发与模型参考。
- [PYML 实验配置技能](skills/pyml-study-config/SKILL.md)：面向 AI 助手的配置和分析工作流。
- [版本记录](version.md)：本版主要变化。

## Python 开发环境

源码用户需要 64 位 Windows、Python 3.12 和 uv。在仓库根目录打开 PowerShell，同步项目依赖并启动：

```powershell
uv sync --locked --no-editable --no-dev --extra desktop --extra search --extra sequence --extra deep --extra manifold --extra forecasting --extra xgboost --extra lightgbm --extra catboost
.\run_pyml_workbench.bat
```

开发构建和 API 说明见[开发指南](docs/development.md)与[Python API 指南](docs/api-guide.md)。
