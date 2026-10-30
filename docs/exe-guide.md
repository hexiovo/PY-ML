# Windows 版下载、运行与安装

## 下载和启动

从 [GitHub 最新 Release](https://github.com/hexiovo/PY-ML/releases/latest) 下载 Windows x64 完整程序包。Release 页面里的 **Code → Download ZIP** 是源码，不是 Windows 程序。

1. 把完整 ZIP 解压到一个有写入权限的目录；路径可以包含中文。
2. 双击 `Start.cmd` 启动工作台。
3. 若希望安装到当前 Windows 用户的应用目录，双击 `Install.cmd`。它会安装到版本化的 `%LOCALAPPDATA%` 目录，不需要管理员权限。

程序包自带所需的 Python 运行环境。运行时要保留完整目录结构，包括 `PYML-Workbench.exe`、`PYML-Worker.exe`、`_internal`、`skills` 和文档；不要只复制主程序，也不要单独移动后台程序或 `_internal`。便携运行可直接保留解压目录；用户级安装由 `Install.cmd` 管理。

每个 Release 附有 SHA-256 校验文件。下载后可在 PowerShell 中用以下命令核对 ZIP，文件名按实际下载的附件填写：

```powershell
Get-FileHash .\<发行包文件名>.zip -Algorithm SHA256
```

将结果与 Release 提供的校验值比较。校验文件中的散列值只标识对应附件；如果文件名或散列不一致，请重新下载。

## 检查运行环境

打开“帮助 → 检查运行环境…”，工作台会检查 Python、依赖版本、导入结果、后台程序和所需资源，并区分必需组件与可选算法。检查过程不会开始训练或转换数据。

缺少 Python 依赖时，可在检查窗口使用“一键安装”。安装使用当前 Windows 用户的专用目录；只有后台加载校验通过后，新环境才会启用。安装结束会再次检查。成功后按提示重启工作台；安装或校验失败时，现有可用环境会保留。

如果检查报告指出程序文件或 `_internal` 缺失，请重新完整解压 Windows 发行包；一键安装只处理依赖，不会补回被移动或删除的程序文件。HyperEEG、PYfNIRs 等转换插件独立发布，也不由依赖安装器代为下载。

## 配置数据转换插件

HyperEEG 与 PYfNIRs 可在“格式 → 数据转换”中使用。选择“插件设置…”后，分别指定对应插件的发行目录。目录须包含 `pyml-plugin.json`；保存后，工作台检查插件清单并启用通过检查的入口。未配置或路径失效时，相关入口保持不可用；有效设置会在重新启动后恢复。

两个转换器不包含在 PY-ML 主程序包中。请从[转换插件仓库](https://github.com/hexiovo/PY-ML_support_trans)下载相应发行文件。插件设置只登记本机插件目录，不会下载或安装转换器。

## 使用实验配置技能

完整程序包的 `skills/pyml-study-config/` 目录包含技能入口、参考资料、检查脚本和合成示例。Release 还提供单独的技能 ZIP 和单文件便携版。

- 在兼容 Codex 技能的 AI 中，将 `pyml-study-config` 文件夹复制到 `%USERPROFILE%\.codex\skills\` 下，之后在新的 AI 会话中调用该技能。
- 在支持上传文件的其他 AI 中，可上传完整技能 ZIP；便携 Markdown 适合只能接收单文件的场景。
- 完整 ZIP 内带有辅助脚本；便携 Markdown 包含指令和参考资料，但不带脚本。没有文件或运行工具的 AI 不能声称已经生成并验证了可执行文件。

技能可以把实验设计整理成配置、数据准备脚本和统计分析脚本。它会标明未知信息和实际检查范围，不替研究者猜测测量单位、独立统计单位或科学假设，也不会因生成配置而自动运行完整分析。

## 继续阅读

- [完整操作指南](overall-guide.md)
- [单任务用户指南](user-guide.md)
- [批量搜索指南](batch-guide.md)
- [PYML 实验配置技能](../skills/pyml-study-config/SKILL.md)
- [开发指南](development.md)
