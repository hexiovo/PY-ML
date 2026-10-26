# Windows EXE 使用与重建

## 使用

从 [GitHub Release](https://github.com/hexiovo/PY-ML/releases/tag/v0.4.5) 下载 `PYML-Workbench-0.4.5-windows-x64-deep.zip`，解压到本地文件夹，双击 `PYML-Workbench-deep/PYML-Workbench.exe`。2026-10-26 发布修复构建位于 `dist/github-release-20261026/PYML-Workbench-deep`；本机通用 APP 入口 `F:/桌面/程序/APP/PY-ML 机器学习工作台.exe` 当前指向该目录，包含 CPU PyTorch/skorch。程序包含 Python 和相应 profile 的运行依赖，目标电脑无需安装 Python。

启动时显示按当前屏幕 DPI 绘制的提示和动画加载条，主窗口准备好后自动关闭。底部青绿色“操作指南”按钮、帮助菜单和 F1 打开同一离线指南，命令以代码块显示。顶部菜单与嵌套子菜单支持悬停展开；不可用项置灰。点击“帮助 → 联系（GitHub Issue）”打开项目的新建 Issue 页面。

此版本包含单任务与批量断点恢复和安全退出。已完成步骤与设置自动保存在用户数据目录；启动后由用户选择恢复或新建。被中断的步骤会从头重做，不从内部迭代继续。

通过“帮助 → 检查运行环境…”可在独立进程中核对依赖版本、实际加载及必要资源。首次打开自动检查并逐项显示结果，支持停止、重新检查和复制报告。当前可选 `standard`、`deep` 与 `full` profile：`deep` 包含标准版和 PyTorch/skorch 深度模型，但不包含 HMM、TPE、遗传搜索或提升树 SDK；`full` 还包括 HMM 与两种搜索扩展。缺少的可选算法单独标注；必需组件异常时提示重新完整解压发行包。

启动时先显示“正在开启程序中…”，初始化完成后提示会自动关闭并进入主窗口。首次启动耗时受磁盘和系统扫描影响，看到提示后请稍候。

底部“操作指南”按钮、帮助菜单或 F1 可打开离线指南，支持目录跳转和搜索；同文档位于 `docs/overall-guide.md`。

首次解压后启动可能较慢：本机首次复制启动测得约 38 秒，后续约 3 秒。请等待主窗口出现；维护启动检查允许 60 秒，不把原 30 秒时限当成应用启动失败。

必须保留 `PYML-Worker.exe`、`_internal` 和其子文件；不能只复制主 EXE。`standard` 包含 Qt 与 71 个 scikit-learn 模型；`deep` 额外包含 PyTorch/skorch 深度模型；`full` 额外包含 HMM、TPE 与遗传搜索。所有 profile 都不包含提升树 SDK。结果、日志、设置按既有应用规则保存，不写入 `_internal`。

使用方法继续见[用户指南](user-guide.md)。程序出现错误时复制错误编号，或通过“文件 → 导出诊断包…”保存本地诊断材料。

转换入口位于“格式 → 数据转换”，包含 HyperEEG、PYfNIRs 及“插件设置…”。首次启动两项转换默认置灰；在统一设置页分别选择对应转换器中包含 `pyml-plugin.json` 的目录并保存，通过检查的入口立即启用，重启后保留。转换器需从 [独立插件仓库](https://github.com/hexiovo/PY-ML_support_trans) 获取，主程序不包含两个转换器的运行文件。GitHub 的源码 ZIP 不能代替此 Windows 发行包。

## 重建

构建使用 **Python 3.12.14 x64**；不要用 3.12.0，其字节码 `code.replace` 缺陷会导致 SciPy 在冻结后导入失败。已有项目环境不必替换，可下载独立构建运行时并通过 `PYTHONPATH` 复用同一 Python 3.12 ABI 的已安装依赖。构建工具只用于开发，不是应用运行依赖。使用 non-editable 包安装以保证发行元数据匹配源码版本。

```powershell
uv sync --python 3.12.14 --extra desktop --extra deep --no-editable
uv pip install --python .venv/Scripts/python.exe -r packaging/build-requirements.txt hatchling
uv pip install --python .venv/Scripts/python.exe --no-deps --no-build-isolation .
uv run --python 3.12.14 packaging/build_release.py --profile deep --work-dir <绝对的新临时目录> --dry-run
```

入口和 spec 收集动态算法模块、搜索/HMM/deep 后端、模型目录、发行元数据、Qt 平台插件和 Matplotlib 图形后端。主程序使用 windowed 子系统，后台使用 console 子系统以保留 QProcess 管道；两者共享同一 `_internal`。运行时钩子保持计算核心先于 Qt 的导入顺序；计算后台跳过 Qt 初始化。

## 验证范围

2026-10-11 的 0.4.5 插件设置版：GUI/Worker 的 54 项内嵌模块与当前源码比对通过，包含插件设置和 CSV 身份字段读取；deep profile 的排除项检查通过。实际主 EXE 在无 Python 路径、仅系统 PATH 和外部中文临时目录下启动并显示主窗口，冻结 Worker 返回结构化错误事件。没有重复源码插件测试、模型训练或真实转换器端到端检查。记录位于 `delivery/exe-0.4.5`。

2026-10-07 断点恢复标准版：GUI/worker 的 11 个应用模块各自与当前源码比对，22 项检查通过；两个 EXE 均确认排除了 profile 中的 9 个可选包。实际 GUI EXE 在中文临时目录和仅系统 PATH 下启动；源码 Qt 控制器驱动实际 worker 完成单任务训练、冻结、一次测试、导出和一次批量拟合；另执行了实际 frozen C19 训练。检查使用隔离用户数据目录，没有重跑全部模型或此前已通过的 500 次迭代恢复验证。记录位于 `delivery/exe-0.4.4/checkpoint-20261007`。

0.4.2 记录位于 `delivery/exe-0.4.2`，包括指南/相关 Qt 回归、资源一致性、最小窗口预览、GUI/worker 嵌入代码与源码比对、实际主 EXE 的 F1 离线指南入口，以及 ZIP 文件校验和外部中文路径解压启动。首次启动检查超时及诊断记录也保留，详情见 `version.md`。下面的计算生命周期记录来自 0.4.1，0.4.2 未重跑未受影响的全部模型。

实际验证记录位于 `delivery/exe-0.4.1`：主 EXE 在无 Python 路径、仅系统 PATH 和外部中文工作目录中启动并显示中文主窗口；开发 Qt 控制器驱动实际 frozen worker 完成单任务训练、冻结、一次测试、自动导出与批量搜索。C01、H01、N01、N06 的实际 frozen 生命周期、缓存测试和重载检查共 26 个子步骤通过；最终构建对受影响的 GUI、五种搜索和错误日志关联作了复验，没有重复那 26 个未受影响步骤。详细失败、更正和验证范围见根目录 `version.md`。

本次在 Windows 11 x64 本机验证，未在干净虚拟机或其他 Windows 版本验证，未重新拟合全部 78 个模型。程序使用 Windows 自带的 ICU，排除了构建机其他工具链的同名冲突 DLL。深度模型使用 CPU。

维护用检查命令（控制器需要项目开发环境；最终用户运行 EXE 不需要）：

```powershell
.\.venv\Scripts\python.exe -X utf8 -B packaging/smoke_worker.py dist/PYML-Workbench delivery/exe-0.4.1/worker-smoke
.\.venv\Scripts\python.exe -X utf8 -B packaging/smoke_gui.py dist/PYML-Workbench delivery/exe-0.4.1/gui-smoke
```

最终 ZIP 附整个程序目录及文档/许可文本，目录内 `SHA256SUMS.txt` 校验每个文件，`dist/SHA256SUMS.txt` 校验主/后台 EXE 和 ZIP。运行 `packaging/build_release.py` 会先构建，再对 GUI/Worker 的内嵌源码和 profile 排除项执行强制检查，通过后才调用 `packaging/release.py` 生成 ZIP；该脚本拒绝覆盖同名 ZIP。
