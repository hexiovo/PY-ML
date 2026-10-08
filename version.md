# 版本记录

## 2026-10-08 — 修正开屏动画居中位置（0.4.4）

- 复现原 APP 入口开屏一直位于物理坐标 `(0, 0)`。定位到 PyInstaller SplashWriter 按字符数保存 UTF-8 脚本长度，中文导致末尾几条指令被截断，居中和置顶指令没有执行。
- 将完整启动脚本以 ASCII 十六进制载荷打包，Tcl 在运行时解码 UTF-8 执行；保证归档字符长度等于字节长度，保留中文、按 DPI 绘制、动画和当前屏幕居中计算。产品版本保持 0.4.4。
- 最终标准版已更新到 APP 入口实际调用的 `dist/ui-polish-20261008/PYML-Workbench-standard`。本机屏幕为 2560×1440、192 DPI（200% 缩放），启动窗口矩形为 `(721, 470, 1839, 969)`；从首次可见到主窗口出现共 59 次采样，中心误差均为横向 0、纵向 -0.5 物理像素。主窗口出现后提示关闭，工作台与 APP 入口正常退出，退出码为 0。
- 原包脚本声明长度为 4312 字节，实际 UTF-8 长度为 4390 字节，回归检查确认居中指令位于被截断部分；新 ASCII 载荷经过 Tcl 解码后保留中文、居中、置顶和最后的 raise 指令。最终 GUI/worker 22 项源码比对及 standard 排除检查通过，原有机器学习模块未改动。
- 构建、编码回归、实际窗口坐标和截图证据位于 `delivery/splash-center-20261008`。刷新同名标准版 ZIP 与 SHA-256 校验清单，保留原主 EXE 备份；临时检查脚本在完成后清理，产品版本保持 0.4.4。

## 2026-10-08 — 操作指南、悬停菜单与清晰启动动画（0.4.4）

- 主窗口、帮助菜单和指南窗口统一使用“操作指南”，底部入口改为青绿色。阅读器给 Markdown 代码块设置独立浅灰底色、内边距和等宽字体，保留命令复制、搜索和窄窗口换行。
- 顶部文件、分析、帮助菜单支持悬停展开，子菜单沿用 Qt 悬停展开与键盘导航；可用项目选中为蓝底白字，不可用项目置灰且不可点击。帮助增加“联系（GitHub Issue）”，仅点击后打开 `https://github.com/hexiovo/PY-ML/issues/new`。
- 菜单状态跟随数据和运行状态更新：没有加载数据、路径已更改、没有输入特征或模型缺依赖时禁用保存配置；运行中禁用加载配置和最近文件；不存在自定义日志目录时禁用恢复默认目录，诊断会话不可用时禁用诊断导出。
- 原启动画面在创建 Tk 窗口前缺少 DPI 声明，文字又被固化为位图。新 GUI EXE 在 bootloader 创建窗口前启用 PerMonitorV2，标题、图形和动画加载条根据屏幕 DPI 实时绘制，状态文字与主窗口出现后关闭的机制保留。
- Qt 定向检查通过：顶层与嵌套菜单悬停、联系悬停不跳转而点击打开正确 URL、无数据/有数据/运行中菜单状态、启动命令代码块、窄窗换行、13 章目录与章节跳转。实际 APP 入口已指向 `dist/ui-polish-20261008/PYML-Workbench-standard`；本机 200% 缩放（192 DPI）下的启动画面已截图核对，DPI awareness 为 per-monitor，主窗口出现后启动提示关闭，F1 指南与关闭退出码 0 检查通过。
- 最终冻结 GUI/worker 的 22 个模块与当前源码逐项一致，standard 的 9 项排除检查通过。更新后的实际 APP 入口在仅系统 PATH、隔离设置目录下启动，192 DPI、启动提示自动关闭、F1 指南和退出码 0 均通过；最终重建后首次复测约 1.218 秒显示提示、26.937 秒显示主窗口，前一构建的后续启动约 3.6 秒，启动时间受系统扫描、磁盘缓存与负载影响。
- 标准版发行目录为 `dist/ui-polish-20261008/PYML-Workbench-standard`，ZIP 为 `dist/PYML-Workbench-0.4.4-windows-x64-standard-ui-polish-20261008.zip`；发布脚本刷新文档、生成逐文件 SHA-256 清单及 ZIP。构建与验证记录位于 `delivery/ui-polish-20261008`。保留此前发行与原 APP 入口备份，本轮临时检查脚本在收尾删除；本轮没有重跑全部模型拟合或重建既有 PDF。
- 按用户要求配置 GitHub 仓库 `https://github.com/hexiovo/PY-ML.git`，使用 master 同步当前 0.4.4 源码、资源和版本记录。

## 2026-10-08 — APP 单 EXE 入口与桌面清理（0.4.4）

- 按用户修正，桌面的“PY-ML 工作台 0.4.4”程序目录和“PY-ML 机器学习工作台 0.4.4”快捷方式已移入回收站，桌面不再保留这两项。
- `F:/桌面/程序/APP` 仅新增 `PY-ML 机器学习工作台.exe` 启动入口（49,664 bytes），其源码为 `packaging/app_launcher.cs`。利用本机已有 .NET Framework 编译为带图标的 x64 窗口程序，定位 APP 同级 PY-ML 项目中的标准版运行目录；依赖保留在 `dist/startup-20261008/PYML-Workbench-standard`，不把完整包放入 APP。
- 入口在运行目录缺失时提供中文错误提示，正常启动不显示控制台，并随实际工作台关闭退出。实际 APP 入口在隔离设置、仅系统 PATH 下验证通过：启动提示可见，主窗口约 4.26 秒显示，工作台关闭后入口退出码为 0。此入口依赖项目发行目录，并非自包含便携包。
- 复核确认两个桌面路径已不存在，APP 原有其他文件保留；验证记录位于 `delivery/startup-20261008/app-entry-check/result.json`。

## 2026-10-08 — EXE 启动提示与标准版重建（0.4.4）

- 定位原 EXE 的无提示等待：打包入口在窗口创建前加载机器学习、数据与图形组件；本机源码导入测量约 14.26 秒，原 GUI EXE 没有 bootloader splash 资源。
- 增加 bootloader 阶段的“正在开启程序中…”启动画面，先于 Python 业务模块加载显示，置于其他窗口前面，并显示加载组件、准备界面的状态文字；主窗口实际绘制后自动关闭，入口异常退出也会关闭。worker 不创建或连接启动画面，保留 core-before-Qt 导入约定。
- 首轮实际截图发现 PyInstaller 对含空格的字体名生成了无效 Tcl 命令，提示呈空白窗口；将字体参数正确加 Tcl 花括号后重建并完成实际屏幕截图核对。截图验证采用物理像素区域，避免 Windows 缩放下的坐标虚拟化及透明 Tk 窗口的 PrintWindow 黑图。
- 新 EXE 包含此前的顶部单行六步导航、底部工具栏、标题行删除与五张图文指南图片；指南初始尺寸进一步限制在当前屏幕可用范围内，适配本机 Windows 缩放。
- 使用现有独立 CPython 3.12.14、PyInstaller 6.22.3 环境重建 `standard` profile，发行目录为 `dist/startup-20261008/PYML-Workbench-standard`。加入启动画面的 Tcl/Tk 运行库及许可，产品版本保持 0.4.4。
- 最终冻结文件通过 GUI/worker 22 项源码字节码比对及 standard 排除检查；GUI 存在 Splash 资源，worker 不存在。实际 GUI 在仅系统 PATH、外部中文临时目录、隔离设置下启动：本轮首次最终产物测得约 1.046 秒显示提示、12.609 秒显示主窗口；提示自动关闭，F1 指南正常，五张打包图片与源码一致。耗时受系统扫描、磁盘缓存与负载影响，后续启动可能更快。此前同轮源码 Qt 控制器连接真实 frozen worker 的 C01 训练/冻结/一次最终测试/导出、批量一次拟合检查通过，未重跑全部模型。
- 新 ZIP 为 `dist/PYML-Workbench-0.4.4-windows-x64-standard-startup-20261008.zip`；构建、启动预览、源码核对与集成验证记录保存在 `delivery/startup-20261008/`。此前发行和桌面快捷方式备份保留；既有 PDF 指南未重建。
- 本机完整程序部署至 `C:/Users/Lenovo/Desktop/PY-ML 工作台 0.4.4`，原桌面“PY-ML 机器学习工作台 0.4.4”快捷方式更新至该目录的主 EXE，图标也复制到程序目录。快捷方式经 COM 重读，EXE 与构建目录 SHA-256 一致；实际桌面 EXE 在隔离设置下复测约 0.719 秒显示提示、11.532 秒显示主窗口，F1 指南与提示自动关闭通过。
- 发行 ZIP 及完整性清单排除运行时可变的统一操作日志与锁文件；启动验证后重新生成包，避免把验证日志作为发行文件或导致正常运行后的日志追加被误判成文件损坏。临时启动验证脚本已在收尾清理，保留有用的截图与报告。

## 2026-10-08 — 指南图文界面与主向导布局调整（0.4.4）

- 删除主窗口内容区的“PY-ML 机器学习工作台”标题行和说明；当前步骤标题右侧将六个步骤入口排成单行，保留当前步骤高亮、可访问名称及原有跳转条件。
- “整体指南”“批量搜索队列”“复制错误详情”和状态提示移至底部，与“上一步 / 下一步”同一行；顶部空间直接用于当前步骤与内容。
- 离线指南采用浅蓝标题区、独立搜索栏、章节目录和白色阅读卡片，加入五张操作图片。其中主窗口、划分与算法配置为本次界面截图，验证图与批量设置沿用已存在的合成示例截图；图片按阅读区域等比缩放。表格及代码可在窄窗口换行，章节跳转定位至标题附近，并处理 Qt 图片缩放后正文布局失效的问题。
- 图文指南与图片统一随 wheel 和 PyInstaller 数据资源分发，同步 README、用户指南和 EXE 指南中的按钮位置说明。
- 定向验证通过：主窗口 1060×720 / 1420×930 下的顶部单行导航、底部按钮范围与标题删除；六步 Qt 鼠标点击切换；指南 760×520 / 1160×820、13 章目录、双向及循环搜索、五张图片读取与宽度缩放、菜单再次打开同一窗口。构建 wheel 后从独立临时目录加载包并重复验证，文档和五张图片与源码资源一致；未启动训练。报告与预览位于 `delivery/ui-20261008/`，临时验证脚本和验证用 wheel 已清理。
- 本次更新源码启动入口 `run_pyml_workbench.bat` 使用的界面，产品版本保持 0.4.4；未重建现有 EXE 或既有 PDF 指南。

## 0.4.4 — 向导与批量配置用户体验改进（2026-10-07）

- 主向导增加六步紧凑导航，提供中文可访问名称、当前页高亮和键盘焦点提示；步骤跳转保留数据前置条件和运行期间的既有限制，运行与结果页仍可查看空状态。
- 将同一任务选择控件移至首页数据配置区，并增加数据加载、任务、特征数量和目标列状态摘要；随路径、数据、任务、目标与特征变化刷新。手工编辑数据路径后提示重新加载。监督任务仍要求目标列，非监督任务显示目标不适用；摘要只提示“可继续配置”，不代表训练预检通过。
- 批量添加文件时，空配置按空列表处理；非空无效数据 JSON 显示中文原因并停止添加，不覆盖原编辑内容。添加当前单任务配置也会先验证非空数据 JSON，避免错误内容被覆盖；搜索空间编辑逻辑未改。
- 界面验证：本轮未运行 QTest 或界面截图；在用户随后明确要求重建 EXE 后完成 standard 发行构建，但未启动冻结产物（见下）。
- **后续 standard EXE 重建（2026-10-07）**：用户追加“重建exe”后，使用 CPython 3.12.14、PyInstaller 6.22.3 和 `standard` profile 按 0.4.4 源码执行 `packaging/build_release.py`，exit 0；产品版本号保持 0.4.4。
  - 主 EXE：`dist/rebuild-20261007-01a11647/PYML-Workbench-standard/PYML-Workbench.exe`，SHA-256 `f883222a01a7a74f95a074dbb975f8f2c84718ffbda55caaea6b97d9a8e12153`；worker：`dist/rebuild-20261007-01a11647/PYML-Workbench-standard/PYML-Worker.exe`，SHA-256 `c9e8831424df04010c6cfa9e8e98b059791258962b6b9143c1c7658b7731c07b`。
  - ZIP：`dist/PYML-Workbench-0.4.4-windows-x64-standard-rebuild-20261007-01a11647.zip`，166,852,597 bytes，SHA-256 `fd38cdbb65d24c1c81c9d1baff14a4509a48cbcaa4349839150ec98d94da80be`；发行目录为 358,994,882 bytes、1,814 files。新报告保存在 `delivery/exe-0.4.4/rebuild-20261007-01a11647/standard-release-result.json`，latest 报告为 `delivery/exe-0.4.4/standard-release-result.json`；旧 checkpoint 报告继续保留在 `delivery/exe-0.4.4/checkpoint-20261007/`。`dist/SHA256SUMS.txt` 保留其他路径并登记本次新产物。
  - 静态核对确认两个 EXE 为 Windows x64 PE，ZIP 含 1,814 项及预期主程序、worker 和依赖。未启动 EXE 或 worker，未运行 QTest、未截图；构建成功不代表 UI/运行验证。构建与环境日志：`H:/AIcode/HexiPlan/runs/pyml-ui-ux-20261007-01a11647/logs/exe-rebuild-20261007-01a11647.log`。
- **桌面入口与图标更新（2026-10-07）**：将 `C:\Users\Lenovo\Desktop\PY-ML 机器学习工作台.lnk` 指向本次 0.4.4 standard 主 EXE，并在同一桌面重命名为 `C:\Users\Lenovo\Desktop\PY-ML 机器学习工作台 0.4.4.lnk`。参数保持空，WorkingDirectory 为完整发行目录，Description 为“PY-ML 机器学习工作台 0.4.4 · 标准版”，IconLocation 为 `F:\桌面\程序\PY-ML\assets\icons\workbench-desktop.ico,0`。
  - 新图标的可复用源与预览位于 `assets/icons/workbench-desktop.svg`、`workbench-desktop-256.png`、`workbench-desktop-512.png` 和 `workbench-desktop-small-preview.png`；多尺寸 ICO 含 16/24/32/48/64/128/256。冰青色 M 节点标记在实际 16 px 预览中仍可辨。原 LNK 在修改前原样备份至 `H:/AIcode/HexiPlan/runs/pyml-ui-ux-20261007-01a11647/backups/r3-desktop/PY-ML 机器学习工作台.lnk`，备份 SHA-256 `d1dc8d225bd51d28d91a747250bfb012d2bbfb7054d98aad5599594fb0eafcac`；原字段与COM重读结果见同目录 `shortcut-before.json` 及 `logs/r3-desktop-icon.log`。目标 EXE/图标路径均存在，更新后的快捷方式字段已静态重读确认；未启动 EXE/worker，未运行 QTest 或截图。

## 2026-10-07 — 0.4.4 断点恢复版 Windows EXE 重建与本机安装

- 按当前源码重建 standard 发行，包含单任务、批量断点保存与恢复、安全退出，以及中断步骤整步重做。保留原 0.4.4 发行目录；本次构建使用独立的 `dist/releases/0.4.4-checkpoint-20261007/PYML-Workbench-standard`。
- 首轮打包后的 profile 检查发现 scikit-learn 的梯度提升工具可选导入了构建环境中的 XGBoost、LightGBM、CatBoost，导致标准包意外收集这些 SDK。已为 standard 显式排除三个 SDK，按原标准版范围重新构建；首轮构建/烟测与导入链证据保留在本轮记录中。
- 已使用 CPython 3.12.14 和 PyInstaller 6.22.3 完成构建。GUI 与 worker 的 11 个应用模块逐一与当前源码字节码比对，22 项检查通过，包括 checkpoints、preferences、batch_gui、batch、history、search、selection 与 _worker。
- 重建后的两个 EXE 均通过 standard 的 9 项排除包检查，包含三个提升树 SDK 的排除保护；实际 frozen C19 HistGradientBoostingClassifier 完成 5 次训练迭代，未调用最终测试。该定向检查最初在启动 worker 前遇到辅助脚本的环境变量大小写问题，修正辅助脚本后通过，未修改模型实现。
- 本次实际 GUI EXE 在外部中文临时目录、仅系统 PATH 下启动并显示窗口；源码 Qt 控制器驱动实际 frozen worker 完成训练、冻结、一次最终测试、自动导出和一次批量搜索，均通过。检查的 APPDATA、LOCALAPPDATA、TEMP 与图形缓存目录隔离在本轮 evidence 下，没有使用用户现有断点目录；未重跑全部模型或原 500 次迭代恢复验证。
- 发布 ZIP 为 `dist/PYML-Workbench-0.4.4-windows-x64-standard-checkpoint-20261007.zip`，包含 GUI、worker、共享依赖、说明、指南和许可证。本机快捷方式指向本次发行目录，原有发行保留。构建与验证记录位于 `delivery/exe-0.4.4/checkpoint-20261007`。

## 2026-10-05 — 单任务与批量断点恢复（最终独立复核 PASS；源码启动交付，0.4.4 EXE 未重建）

- 本轮为 PY-ML 加入单任务和批量搜索的跨启动恢复能力。单任务状态与设置保存在稳定的用户数据目录；checkpoint 使用原子替换和 SHA-256 完整性校验。启动时提供“恢复/新建”选择，恢复由用户明确触发，不自动开始拟合、搜索或测试。
- 单任务恢复复用已完成的训练验证、冻结配置和最终测试结果；运行配置与源数据/快照/划分身份需匹配。安全退出会停止活动 worker 并保留之前完成的阶段；只对确认安全中断、worker 已停止且没有完整测试回执的最终测试步骤授予一次整步重做。普通测试失败与手动取消继续禁止重试，已有完整测试结果优先复用。
- 批量计划、SQLite 历史和候选快照使用稳定默认位置。安全中断的 active trial 按原 proposal 和参数从整步重做且不重复占拟合预算；已提交测试结果但缺少外层完成回执时补回执并复用。候选、配置、selection 与完成回执仍需绑定同一任务身份。
- 本轮不改变拟合算法、结果计算或模型选择规则；final-test 仍保持单次评估语义，安全恢复仅适用于明确安全退出且缺少完整回执的中断步骤。最终独立整合复核已通过，具体范围与限制见本节 SUPPLEMENT-02 记录和 RUN 证据。
- SUPPLEMENT-02 补齐有效已完成批次的 Restore / New / Later。Restore 显示已保存设置、历史和结果并标为完成，不启动 worker、fit 或 final-test；Later 不改旧记录；New 创建独立 sibling root。五项新增 GUI 回归全部通过。有效设置下 New 只执行一次新 search fit、不启动 final-test；缺少数据/搜索空间时只建立新目录并请求设置，不启动 worker。
- 独立 QProcess 场景使用真实 C21 MLPClassifier 的 500 次 fit，在第 250 次安全中断并确认 worker 已停；新进程将未完成步骤从头拟合至 500 次，配置、数据、划分和 seed 指纹一致，完整拟合一次返回、部分结果不保存，fit 调用共两次且 final-test 调用为零。原 100 个回归用例、8 个 Qt 场景、14 项错误标记拒绝矩阵和 live-PID 检查复用有效证据；SUPPLEMENT-02 的新增 5 项回归与三个真实场景通过，未重跑整套无关测试。
- 已完成批次保护核验确认原历史 schema 和逻辑记录、消费权限、非数据库文件及持久 session 字节保持不变。SQLite 全文件字节不能声称不变：现有初始化会执行 PRAGMA user_version=4，导致 header offsets 27 和 95 改变；未发现 schema、记录或 header 之后 payload 的变化。
- 两项 SUPPLEMENT-02 harness 误断言已保留原始日志并按实际情况修正：一个夹具不符合 SearchSpec，另一个把 SQLite header 的初始化写入误判成历史记录变化；均未改业务代码，不计为业务修复。累计业务修复轮次为 1；原 QCloseEvent 夹具修正单独记账，不清除累计历史。
- Astra 最终覆盖检查确认本任务已要求的功能均有证据覆盖，没有新的实质遗漏。交付限于源码启动入口 `run_pyml_workbench.bat`；项目版本仍为 0.4.4，本轮未重建或声称更新 `dist` 中的 EXE。

- 2026-10-05 最终验收：HexiPlan 步骤 s1-checkpoint-recovery 已依据本任务真实“无需审核”授权映射最终报告哈希 f6da5b01da5dceaeff540b3799432670c52c59e2459ce16971be4790a64ffea5，workflow 状态为 complete；固定验收报告见 H:/AIcode/HexiPlan/runs/pyml-checkpoint-20261004-01/reports/step_acceptance-v1/report.html。收尾清理删除 2 个未被引用的临时脚本，5 个受引用保护的副本保留；5 个本轮 Python 3.11 缓存因安全策略在 PowerShell 启动前拒绝而仍保留，详见 RUN evidence/final-cleanup.json。

## 0.4.4 — 依赖提示、六个提升树模型验证与 standard 发行物（2026-10-04，独立复核定向文档修正）

- 新增统一 UTF-8 操作日志与任务计时：GUI 和 worker 共享程序固定路径的追加日志，覆盖安全的界面交互、对话框结果、应用生命周期和 worker 开始/结束；计时使用单调时钟并明确显示估算依据。主菜单“文件 → 打开统一操作日志…”可查看记录；源码默认使用项目目录，冻结版使用 EXE 目录。

- 运行时依赖提示覆盖 HMM、深度学习与三个提升树 SDK。XGBoost、LightGBM、CatBoost 继续由各自 extra 提供，不进入基础依赖或 standard/full PyInstaller profile；缺包时说明安装项，不自动安装。标准 profile 的模型目录与依赖可用性核验显示 71 个 scikit-learn 模型可用；冻结烟测只覆盖 C01 单任务和一次 batch 拟合，不代表 71 个模型全部通过端到端训练。
- 在项目 Python 3.12.0 环境用 CPU 单线程，通过 `run_experiment → model.joblib → load_model/predict` 验证 C25–C27 与 R24–R26。分类覆盖二分类与非连续多类别标签；模型重载前后预测一致，概率列顺序与类别标签核对通过，各行概率和为 1（浮点误差范围内）。证据：`H:/AIcode/HexiPlan/runs/pyml-completion-20261004-01/evidence/sdk-application-validation.json`。
- CatBoost 分类器原生 `(n, 1)` 预测现由应用边界规范为 `(n,)`；多输出形状仍保留。实际 worker 推理的 CatBoost 分类与回归 CSV 均仅输出 `prediction` 列；单测和实际命令行结果见 `sdk-postfix-inference-shapes.json`。
- 以 xlwt 生成合成 OLE/BIFF8 `.xls` 样本，经 `load_dataset` 和 `preview_dataset` 核对两个工作表、六个字段、样本值、布尔值和日期类型。该结果只覆盖此样本，不代表对所有 XLS 文件的兼容保证。记录：`real-biff8-xls-import.json`。
- README 与 `docs/overall-guide.md` 更新至 0.4.4，说明可选 SDK、六个提升树模型、standard/full 包体范围、ROC/PR 与特征解释图及旧缓存重新训练要求。PDF 指南为 36 页、13 章、19 张截图；本轮刷新 9 张向导/应用内指南截图，保留 10 张仍适用的截图。独立复核发现 XLS 验证范围描述过时后，本轮仅重建文档 PDF 并复用 19 张截图；受影响的第 8 页已用 Poppler 渲染检查。更新后 PDF SHA-256 为 `e8b3f1d78cef37fb620dcdfc8855486500de5742a1aaf9c2fc2dd20ace7ee5f7`。结构与页面检查见 `output/pdf/PDF-guide-checks.json`，截图来源与哈希见 `docs/assets/pdf-guide/screenshots.json`。
- r1 文档更正将指南与 README 的 XLS 描述限定到本轮合成 BIFF8 样本：应用实际导入并预览了两个工作表、六个字段、样本值、布尔值和日期类型，不据此保证所有 XLS 文件。冻结目录中的 `docs/overall-guide.md` 与 `_internal/pyml_workbench/resources/overall-guide.md` 是独立数据文件；仅同步资源和发布文档并刷新 manifest/ZIP，不重建 PyInstaller EXE。最终修正版文档刷新包发布在规范路径 `dist/PYML-Workbench-0.4.4-windows-x64-standard.zip`。历史 docfix-r1 ZIP（SHA-256 `ae62d68a1566dcb0a2370aed9fed93e843ef4f4cd15805b975849e328f5c11dc`）保留在 `dist/PYML-Workbench-0.4.4-windows-x64-standard-docfix-r1.zip`，已由最终 doc-only 标准包取代；原始 ZIP（SHA-256 `c0fa3182fccae860443781f1bed245af40662862ee61032ce007710fe024ac78`）已归档至 `H:/AIcode/HexiPlan/runs/pyml-completion-20261004-01/evidence/pre-docfix-r1/PYML-Workbench-0.4.4-windows-x64-standard.zip`。首次独立 r1 记录、原 PDF 和旧 PDF 检查哈希保留在 RUN evidence；本轮更正记录见 `evidence/S1-documentation-correction-r1.json`。
- 已用 CPython 3.12.14、PyInstaller 6.22.3 构建 standard onedir 发行包：`dist/releases/0.4.4/PYML-Workbench-standard`，含 1,814 个文件；唯一当前标准 ZIP 路径为 `dist/PYML-Workbench-0.4.4-windows-x64-standard.zip`。压缩包内包含 GUI、独立 worker、离线指南、文档和许可证。主程序 SHA-256 为 `219d9af5f7f6b31d131f83e30e37024830947d04a339f63f9d9d62a00484d8ad`，worker 为 `3d6ccef88f1892559a873dbcc582b6603a000428052565064476a243b5bd39ba`；ZIP 实际大小与哈希见 `dist/SHA256SUMS.txt`、`delivery/exe-0.4.4/standard-release-result-publication-final.json` 与 RUN `evidence/standard-0.4.4-public-standard-archive-final.json`。本次仅为纠正公开交付路径的 doc-only 刷新：Sol R2 已复核 docfix-r1 内容；未重建 EXE、未修改 PDF、未重跑业务验证，最终 ZIP 只更新版本记录/manifest。
- standard 构建目录的 84 个目录模型中，运行环境实际可用 71 个 scikit-learn 模型；6 个提升树模型及 3 个 HMM、4 个深度模型因 SDK/extra 缺失而不可用。GUI 与 worker 的冻结 PYZ 均未包含 xgboost、lightgbm、catboost、hmmlearn、skorch、torch、optuna、pymoo 或 moocore；标准 profile 的可用数及模块排除检查见 RUN `evidence/standard-0.4.4-profile-and-archive-validation.json`。
- 冻结 GUI 从外部 Unicode 工作目录、移除 Python 路径并限制为 Windows 系统 PATH 后可见启动；F1 可打开与源指南一致的离线指南。GUI QProcess 实际调用冻结 worker 完成 C01 训练、冻结、一次最终测试和导出，并完成一个 batch 搜索拟合。烟测是代表性运行，不表示全部模型均已拟合。结果见 RUN `evidence/standard-0.4.4-frozen-source/`、`evidence/standard-0.4.4-frozen-smoke/` 与 `evidence/standard-0.4.4-smoke-harness-reconciliation.json`；首次烟测日志保留了旧模型数断言失败，脚本现从模型目录动态计算目录总数。
- 本轮验证使用小型合成数据，不处理用户业务数据。旧版 0.4.2 EXE/PDF 保留；当前截图源目录已更新到 0.4.4，旧 PDF 中仍保有原版本的截图快照。
- 2026-10-04 操作日志修复轮 1 补充：延迟扫描动态控件树并校验 Qt wrapper 生命周期；列表/表格的鼠标与键盘交互只记录安全的行列索引、选择摘要和勾选状态，不记录单元格内容；向导与批量队列按 completed/failed/cancelled outcome 区分计时和结束日志，部分失败或取消时仍保留可用验证结果。当前补丁冻结前未追加测试；独立复测范围限于受影响的 GUI、batch 状态与交互路径。
- 2026-10-04 操作日志修复轮 2：`ChildAdded` 不再捕获事件 QObject；合并为延迟扫描应用的完整顶层窗口树，避免保留构造期的 Shiboken base wrapper。Space 键的补充复测纠正为夹具误报（测试项 `currentRow=-1`，按键没有切换目标），实现保持不变。最终受影响复测待完成。
- 2026-10-04 批量取消补充：worker 仅在持久任务状态明确为 `cancelled` 时把 pre-dispatch `DispatchStopped` 映射为取消 outcome；已排队取消且没有搜索结果时也直接返回取消。暂停、已完成/失败缺少结果及其他异常继续报告失败，不改变搜索计算、派发或暂停规则。Sol 的定向复测待完成。
- 2026-10-04 计时进度作用域补充：只有当前 `train` / `wizard_batch` 计时动作且 fit 计数有效时，才将向导计数写入当前 `RunTiming`；freeze/test/export/inspect/inference 保留本阶段独立估算，不借用前一阶段已完成 fit 数。全局结果计数及 elapsed/finish 算法不变。Sol 定向复测待完成。

## 0.4.3 — 可选提升树模型与结果解释图（2026-10-04，独立整合复核通过并注明限制）

- 模型目录新增 C25–C27 / R24–R26：XGBoost、LightGBM、CatBoost 分类与回归器。三个库分别由 `xgboost`、`lightgbm`、`catboost` extra 提供，不进入基础依赖或 `standard` / `full` PyInstaller profile。模型发现不导入可选库；缺包会说明 extra，不会自动安装。训练默认为 CPU，数值线程数限制为 1，并映射每个库的 seed/thread 参数。
- XGBoost 分类器把原始标签持久映射为连续整数，并在预测与分数列中还原原标签。新分类 session 只缓存估计器实际产生的概率或决策分数，并绑定类别顺序、分数列映射和源行位置；多分类决策分数仅在确认 OvR 语义后用于曲线。ROC/PR 可按类别绘制；单类别评估分区会说明不可用原因。
- 模型解释图只使用真实 `feature_importances_` 或 `coef_`，名称从拟合后的预处理器读取，包含变换和 one-hot 展开的特征；系数图保留正负方向及类别行对应关系。PNG/SVG 继续通过现有 `PlotDialog` 导出。同组批量指标图复用 `_comparison_group`，限定 completed 且成功 winner 的严格同数据/划分/字段/任务/指标/方向/分区记录。
- 绘图清单 schema v2 纳入分数和解释量摘要；既有 v1 清单仍可用于原有混淆矩阵、回归和其他基础图。旧 session 没有真实分数或解释量时不会生成新曲线/解释图，并会说明需重新训练。
- 依赖约束取 2026-10-03 PyPI 当日发布：`xgboost==3.4.1`、`lightgbm==4.7.0`、`catboost==1.2.10`。当日 Windows wheel 约为 46.7 MiB、1.3 MiB、95.6 MiB，未计运行依赖。XGBoost 与 LightGBM 的 PyPI 元数据声明 Python 3.12 Trove classifier；CatBoost 未声明该 classifier，但发布中存在 cp312 Windows wheel。以上是发行元数据证据，不是本项目的运行兼容性验证；三库均未安装，本次没有执行真实训练、序列化重载或推理。许可证依据上游仓库：XGBoost Apache-2.0、LightGBM MIT、CatBoost Apache-2.0（[XGBoost](https://github.com/dmlc/xgboost/blob/master/LICENSE)、[LightGBM](https://github.com/microsoft/LightGBM/blob/master/LICENSE)、[CatBoost](https://github.com/catboost/catboost/blob/master/LICENSE)）。
- 2026-10-04 首轮定向修复补齐 CatBoost 空 `get_params()` 时的参数合同、别名冲突与默认项；多分类 SVC/NuSVC 的成对系数不再冒充逐类系数，单行系数图会在实际图例显示类别/输出名称；绘图 manifest v1 只保留已摘要保护的基础图数据，隐藏未受摘要保护的新分数和解释量。独立 Sol 复核四项受影响检查全部通过；执行环境另以 Python 3.12.0、NumPy 2.5.3、pandas 3.0.6、scikit-learn 1.9.1、Matplotlib 3.11.2、PySide6 6.11.2 完成五项窄测试。记录见 `H:/AIcode/HexiPlan/runs/pyml-algorithms-plots-20261003-01/evidence/s1-final-independent-verdict-20261004.json`（SHA-256 `ba634523f8f59cf4dcacdd6b179180bed52dca37445dd0d0fb34d57fd7378bf94`）、`s1-independent-round1-targeted-review-20261004.json`（SHA-256 `553270717f197f73c093127b0276b6c97e25f2f4f8b958f5e564fcffeda24a4a`）及窄测日志 `s1-round1-targeted-tests-20261004-retry.log`（SHA-256 `491b5040dca59d621117dcae9cd3d3c7c2e63734787f3cb95b21b5345d23ab60`）。此前已通过的 22/22 和不受影响检查复用，没有重跑。
- 复核结论为 `PASS_WITH_EXPLICIT_LIMITATIONS`，本次修复计为 round 1；CatBoost 验证使用显式-only 参数合同替身，三个外部提升树库都未安装，真实 SDK 训练、序列化重载与推理尚未验证。GUI 检查使用 offscreen 环境；本次没有重建现有 0.4.2 EXE 和 PDF，它们仍是原交付快照。项目代码版本仍为 0.4.3。
- 2026-10-04 S1 最终验收已按固定 document 框架生成并依据本任务真实持续授权映射：`H:/AIcode/HexiPlan/runs/pyml-algorithms-plots-20261003-01/reports/s1-final-acceptance-v1/report.html`，内容 SHA-256 `a37af91047aeb11f7725d2a6cacd5e75da6902e9c8b2036bc2b53ea651b37597`；官方 workflow phase=complete，三个工作包均记录 PASS。README 源码启动入口为 `.\.venv\Scripts\python.exe -X utf8 -m pyml_workbench`。
- 最终清理移除了本次首轮修复生成的临时测试方法及附加断言，保留原有产品用例与其它工作区改动；RUN tmp 的受控清理记录为零个待删文件。真实修复轮次为 1；因原始失败 verdict 未先写入状态机，workflow repair_rounds 计数保留 0，真实轮次已在验证证据与 G5 记录，没有手改状态或伪造历史。


## 2026-10-03 — 图文指南、分步向导与标准发行包（完成）
- G5 最终验收已按固定 acceptance 模板批准，结论为 PASS_WITH_EXPLICIT_LIMITATIONS；报告 content SHA-256 为 ca8207adfbd746e7c9a50da595be8d115321cf0527bb2ae81dbedff8431f06fa，workflow 已完成。最终 ZIP 仍保留打包时快照，SHA-256 为 83a76fabbe1fb8fb700de72b15544635b4fe607fdd0506b77c41e598efd6b1fd。

- 新增 `packaging/profiles.json` 和 profile 化 PyInstaller 构建入口：`standard` 含桌面能力与 71 个 scikit-learn 模型；`full` 使用 `search`、`sequence`、`deep` extras。0.4.1 完整版只作为历史包体积基线。
- `docs/overall-guide.md` 的 6 张表加入连续编号和居中标题；PDF 正文字号由 10 pt 增至 11 pt，截图页标题与图注居中。最终 PDF 为 13 章、36 页、19 张截图，带可点击目录、章节书签、页码和嵌入中文字体。26 个目录链接、13 个章节书签、19 个图片页通过结构检查；Poppler 以 120 dpi 检查全部页面，并细看第 7、8、11、20、24、36 页，未见空白页、图注跨页、裁切或表格溢出。检查记录：`output/pdf/PDF-guide-checks.json`。
- 标准发行包目录 409,569,720 bytes、`_internal` 350,978,614 bytes（1,727 个文件）；最终 ZIP 188,866,613 bytes，SHA-256 `83a76fabbe1fb8fb700de72b15544635b4fe607fdd0506b77c41e598efd6b1fd`。ZIP CRC、2,020 条清单哈希和 PDF 嵌入哈希均通过。主 EXE 与 worker EXE 哈希分别为 `530ffcf11da3033d23c27d2eb1d7b193db3aebb3ad6a0b064daebb2c023e0305` 和 `66ca5190355d0c791f7b1e5f96d5c52567c0cb671f3463361b9a6696ba00e8d0`；主窗口 5.4 秒内显示并正常退出。
- 标准版嵌入的 PYZ 与构建目录逐字节一致；`gui`、`runtime`、`_worker`、`search`、`sequence_models`、`extended_experiment` 的冻结代码与当前源码逐项核对通过。GUI 源码 SHA-256 `3cf7f0f5139c6f197764d7e36e05f37ff42606675828015f0149b2023a0d92aa`。独立 Sol 使用标准冻结 worker 完成 90 行 R01/R02 回归，各模型 1 次拟合/提案并输出 r2、mae、rmse；每个模型使用 18 个 validation 样本，final-test 计数为 0。
- 0.4.1 full 的目录、`_internal`、ZIP 分别为 1,192,241,139、1,069,512,383、452,044,881 bytes。0.4.2 standard 包更小，但两者 profile 功能不同，此对比不表示同功能包体缩小。standard 的真实构建记录、最终文档归档和独立评审分别见 `H:/AIcode/HexiPlan/runs/pyml-enhancements-20261003-01/evidence/standard-release.json`、`standard-release-final.json` 和 `final-integration-review.json`。
- 本机仅有 CPU 版 PyTorch，未做 CUDA 硬件训练；90 行小样本没有观察到模型拟合重叠，也未测得并行提速基准。没有重跑全部模型，也没有在干净 Windows 虚拟机上做安装验收。
- 源码 version.md 在最终独立验收后更新。为保留已核验的最终归档，最终 ZIP 没有因这次版本记录更新而再次压缩；ZIP 内 version.md 是打包时快照，结案状态以最终 G5 报告和上述独立评审证据为准。

## 文档更新 — PDF 图文整体指南（2026-10-02）

- 新增 `output/pdf/PYML-Workbench-0.4.2-图文指南.pdf`：完整 13 章、30 页、14 张操作截图，包含可点击目录、章节书签、页码和嵌入中文字体；主界面与批量队列采用 3 张横向截图页。
- 截图来自本机 non-editable 安装的 0.4.2 Qt 应用，使用 offscreen 平台直接抓取实际窗口。180 行合成数据真实完成 C01 训练、冻结、一次最终测试与导出；Grid 搜索的 SQLite 记录核实 3 次拟合和 3 次提案。HMM 图片仅展示配置。保留截图哈希、来源记录、配套 CSV 和维护脚本。
- 整体 Markdown 指南明确单任务先“冻结模型设置”，再“执行最终测试（一次）”；README 加入 PDF 链接，开发指南记录重建方法。
- PDF 已渲染检查全部页面，核验 13 个目录跳转目标、13 个书签和 14 个截图页，无空白页、图注跨页或正文表格溢出。检查摘要与文件哈希位于 `output/pdf/PDF-guide-checks.json`。
- 本次是 0.4.2 文档补充，不修改计算代码或运行依赖，不重建既有 EXE、ZIP、wheel/sdist，也不将旧发行快照描述为已包含本次 PDF。未重跑全部模型测试。

## 仓库管理 — Git 初始提交（2026-10-02）

- 在项目根目录初始化本地 Git 仓库，使用 `master` 分支，保存当前 0.4.2 源码、指南、回归测试、打包脚本、依赖锁文件和发行校验清单。
- 新增 `.gitignore` 与 `.gitattributes`：虚拟环境、构建缓存、EXE/压缩包、运行诊断和本地实验数据保留在本地；文本按类型统一换行，Windows 启动脚本采用 CRLF。
- 本次为仓库管理记录；现有 0.4.2 发行件及其校验/验收记录沿用已交付快照。

## 0.4.2 — 整体指南与应用内离线帮助（2026-10-02）

- 新增整体中文指南，覆盖启动、布局 A、数据/列选择、单任务验证/测试、五种批量搜索、HMM/深度模型、图表、配置/历史、重载推理、日志排错和 Python API。
- 主窗口顶部新增“整体指南”按钮；帮助菜单与 F1 使用同一入口。非模态阅读窗口支持 13 章目录跳转、前后搜索与循环查找，重复打开复用窗口。
- `docs/overall-guide.md` 为唯一内容来源，构建时作为资源加入 Python wheel 和 EXE，断网和外部工作目录均可阅读。缺失资源显示恢复方法，可在修复后重新打开。
- 不增加运行依赖，不改变模型训练、搜索或一次性测试规则。更新项目/锁文件版本，保留既有版本发行件和验证记录。
- 指南 6 项 Qt 回归在源码与 non-editable 安装各通过；另有后台路由/空审计/参数/错误响应 6 项相关回归通过。回车曾同时触发对话框默认的“上一个”按钮，现禁用搜索/关闭按钮的 autoDefault，并以真实键盘事件回归覆盖。指南模块直接导入时也保持 dateutil-before-Qt 的既有兼容顺序。
- 安装资源与文档逐字一致；1060×720 主窗口按钮可见，13 章目录与正文离屏预览检查通过。实际 GUI/worker EXE 的 6 项嵌入模块与当前源码编译结果一致，包含回车修正。主 EXE 在清除 Python 路径、系统 PATH 和外部中文目录下，原生 F1 进入实际离线指南通过；真实按钮及搜索/目录内容由 Qt 测试另行覆盖。
- 原 30 秒启动检查两次未捕获窗口，初次构建及 ZIP 解压检查的失败记录均保留。独立新副本测量实际在 38.094 秒显示主窗口，同一目录后续启动为约 3 秒；确认原检查时限不足，改为 60 秒，ZIP 子进程时限为 90 秒。底层首次启动耗时的原因未完全定位；没有修改业务启动逻辑，指南已说明等待时间。诊断用 EXE 留在外部审计目录，不进入分发。
- Windows 完整目录与 ZIP、Python wheel/sdist、日志与校验记录位于 `dist` 和 `delivery/exe-0.4.2`。本轮未重跑全部模型拟合或历史完整 suite，计算核心沿用 0.4.1 已记录检查；新分发的复制启动和完整 ZIP 校验由发布脚本记录。

## 0.4.2 — Windows standard-profile final artifact (2026-10-03)

- PyInstaller 6.22.3 with Python 3.12.14 produced the standard-profile folder once. This final delivery revision synchronizes the current README, docs directory, and this version record, then regenerates the folder manifest and a separately named ZIP; it does not rebuild the EXEs or rerun worker tests.
- The frozen main EXE and worker remain SHA-256 530ffcf11da3033d23c27d2eb1d7b193db3aebb3ad6a0b064daebb2c023e0305 and 66ca5190355d0c791f7b1e5f96d5c52567c0cb671f3463361b9a6696ba00e8d0. The GUI source hash remains 3cf7f0f5139c6f197764d7e36e05f37ff42606675828015f0149b2023a0d92aa; actual embedded PYZ code comparison passed.
- Initial standard measurements were 409,568,128 bytes for the folder, 350,978,614 bytes across 1,727 _internal files, and 188,865,210 bytes for its preliminary ZIP. The 0.4.1 full-profile baseline measured 1,192,241,139 bytes for the folder, 1,069,512,383 bytes across 4,235 _internal files, and 452,044,881 bytes for ZIP. The 65.65%, 67.18%, and 58.22% differences are a profile comparison, not a same-feature-set shrink claim. Final revision measurements are in standard-release-final.json.
- Independent frozen-worker review passed R01/R02 on 90 synthetic rows with the UI configured for two workers: one fit/proposal per model, r2/MAE/RMSE, 36 validation prediction rows, zero test evaluations, four export formats, three PNGs, and two model reload predictions. The short fits completed serially; no parallel speedup is claimed.
- The preliminary ZIP is preserved. The final ZIP and its SHA-256 are recorded in delivery/exe-0.4.2/standard-release-final-result.json, dist/SHA256SUMS.txt, and H:/AIcode/HexiPlan/runs/pyml-enhancements-20261003-01/evidence/standard-release-final.json; this file does not embed the final ZIP's own hash.
- Limitations: GPU performance was not tested and parallel speedup was not benchmarked. The final evidence records final package size, ZIP hash/CRC, manifest verification, and reused build/startup evidence.
## 0.4.1 — Windows EXE 分发（2026-10-02，frozen 功能检查通过）

- 新增 PyInstaller 文件夹分发：主界面 `PYML-Workbench.exe` 与共享依赖的 `PYML-Worker.exe`。后台保留 UTF-8 JSON stdout/stderr 协议；Python 启动方式继续可用。
- 新增维护用 spec、入口、构建依赖、后台路由测试和 EXE 指南；业务依赖版本不变。离线 `uv lock` 因缺少 NumPy 解析缓存失败，锁文件仅同步本地包 0.4.1 版本字段，所有第三方版本、哈希保持原值。
- 修复训练和冻结阶段合法 `audit: null` 响应被 GUI 误拒绝的问题；只在匹配 train/trained 或 freeze/frozen 且内置整数测试次数为 0 时接受空审计，test/export 仍要求对象审计。增加正负例回归。
- 构建采用独立 Python 3.12.14。原项目 Python 3.12.0 的 `code.replace` 缺陷已通过 SciPy 编译代码最小复现定位；3.12.14 下同一复现通过。曾尝试的 six 虚拟模块补丁已移除。Qt 使用 Windows 系统 ICU；排除构建主机 PDF 工具链中导出符号不兼容的同名 ICU DLL，避免主界面启动失败。补齐 pymoo 的 moocore 发行元数据。
- 定向回归共 7 项通过：后台路由 3 项、既有 Qt 输出/批量 3 项、空审计正负例 1 项。实际 frozen worker 的 C01/H01/N01/N06 共 26 个生命周期子步骤在 Python 3.12.14 初次构建上通过，随后该组合脚本因遗传搜索缺元数据整体退出失败；不能将该初次组合描述为整体 PASS。最终构建补齐元数据并纳入 GUI 修复后，五种搜索与诊断日志关联重新检查通过；同一计算核心未重跑那 26 个未受影响步骤。
- 最终主 EXE 在清除 Python 环境路径、PATH 仅保留 Windows 系统目录、工作目录位于外部中文临时路径时显示真实中文主窗口。Qt 集成控制器使用开发环境，驱动实际 frozen worker 完成 train/freeze/test/自动导出和批量搜索；它与主 EXE 启动检查属于两个如实记录的验证范围。
- 发行件、ZIP 和 SHA256SUMS 位于 `dist`，外部交付记录位于 `delivery/exe-0.4.1`。旧版 0.3.0/0.4.0 分发和原 HexiPlan 验收保持不变；本次后续按普通流程执行。
- 本次在 Windows 11 x64 本机验证，未在干净虚拟机或其他 Windows 版本验证，未重新拟合全部 78 项模型；深度模型继续使用 CPU。构建/启动失败记录保留用于后续更正，未重跑历史完整 suite。

## 0.3.0 — S03 序列与深度模型集成（2026-10-01，独立整步复核 PASS，已接受）

本版本把 3 个 HMM 与 4 个深度模型加入可运行目录，使活动模型总数达到 78 项。S01 与 S02 的独立整步验证均已由工作流登记为 PASS，两个步骤均已接受；证据及哈希见下文对应版本段。S03 的 A/B/C/D 四包均已登记 PASS，安装后完整测试为 114/114，reviewer 的 D 技术检查为 PASS；正式 whole-S03 verdict 已于 2026-10-01 登记 PASS（23 项 required criteria 全 PASS，CUDA criterion 非 required）。S03 已通过 G5 步骤接受，workflow phase 为 `complete`；G5 报告 content SHA-256 为 `6476a22d4b209ce1dcc98172bd59043333c8a8df41ac3980c67b61dd4717a871`。

- **模型与依赖**：H01/H02/H03 为 Gaussian、Gaussian-mixture 与 categorical HMM，依赖 `sequence` extra；N01/N02/N04/N06 为 PyTorch/skorch 分类、回归与 LSTM/GRU 窗口回归，依赖 `deep` extra。可选依赖缺失时，目录仍可浏览，拟合阶段会给出对应 extra 提示。深度模型固定使用 CPU。
- **单任务与 API**：单任务流程显式冻结并测试同一个已完成训练/验证的模型；批量 winner 流程先冻结验证选择，再在训练+验证数据上 fresh refit 后测试一次。模型包可导出、加载并对新输入推理；深度 fresh refit 使用验证期冻结的 selected epoch 与训练 seed。
- **序列数据**：`SequenceConfig`/`SequencePlan` 描述分组、时间顺序、HMM 观测和窗口映射。批量 job 持有带 receipt 的快照/计划，恢复不重新读取源文件。测试覆盖 HMM 计划在暂停、源文件删除后恢复，以及 HMM/N06 winner 的 freeze、80% refit、一次最终测试和缓存读取。
- **桌面路径**：Qt 回归驱动单任务 H01 的真实 QProcess 训练、冻结、测试和导出；批量对话框回归驱动 H01 与 N06 的真实请求、搜索、冻结、最终重拟合和测试 worker，并核对持久 test permission 与 cached 结果。
- **CSV 序列推理与报告**：HMM CSV 推理按冻结的 group/time 配置输出源行、组、隐藏状态和各状态后验；N04/N06 按预测目标行输出预测与窗口来源映射。worker 将来源映射编码为 CSV JSON 字段；GUI 回归对 H01/N04/N06 导出包执行真实 QProcess 加载和 CSV 推理。单任务新增序列划分页，显示真实 train/validation/test 源行、组、窗口计数与各自比例；有效 JSON 中的 group/time 列自动从特征中排除。
- **GUI 用户指南同步（2026-10-01）**：`docs/user-guide.md` 已说明单任务序列 JSON 编辑器、group/time 结构列自动排除、实际序列划分比例、单任务与批量曲线、重载 HMM/N04/N06 模型后的 CSV 推理及源行/窗口映射。界面行为由安装后 GUI 回归覆盖；S03 定稿仍不包含 XLS 专项夹具或 GPU 验证。
- **文档与示例**：新增序列模型与深度学习指南，并提供合成扩展模型示例；示例 smoke 和当前来源/哈希由 `S03-seed-docs-example-supplement-20261001.json` 补充记录（SHA-256 `e9a413a6d56d49b68555d4bae4b8ddeaa9faf4b7e8c54be6d291f324db6a9e08`）。示例 direct lifecycle 不替代批量 winner fresh-refit/cache 验证。
- **S03 定向修复（2026-10-01）**：N01/N02 的 `num_layers` 现在控制隐藏 Linear+ReLU 层数且保留单层结构；HMM 随机种子支持显式值与 split-seed 回退，refit 在 train+validation 联合分区评分并校验参数；skorch 实际 epoch history 写入 JSON 训练元数据。源加载定向检查分别覆盖 helper、N01/N02 导出重载预测、N06 curves 元数据和 HMM fit 前参数拒绝；HMM seed/refit 组合调用有一个独立 guard 因缺少 `patch` 导入而退出 1，修复导入后该 guard 单独通过，组合调用不记为整体通过。
- **最终整合验证**：序列适配器 4 项 source unittest、单任务结构列/划分摘要 QProcess 回归、H01/N04/N06 重载包 CSV 推理 worker 回归均通过。匹配当前源码哈希的 non-editable 安装包含 25 个包文件，均与源码一致；安装后完整 suite 为 114/114，通过时间 74.704 秒。wheel 与 sdist 的默认依赖隔离安装 smoke 均确认 78 个模型可见、XLSX 导出成功且可选依赖未安装。最终发行物检查确认 wheel/sdist 包内 25 个包文件均与源码一致，sdist 包含文档、示例、测试且没有工作区临时文件。独立进程读取 N06 已完成的 winner/test 缓存并加载导出模型成功；cached=true、actual_fit_count=1、test_evaluation_count=1，未重新拟合。
- **独立整步复核**：D 技术审查与正式 whole-S03 verdict 均为 PASS，required criteria 为 23/23、repair rounds 为 0。正式 verdict 见 `evidence/S03-independent-whole-step-verdict-20261001.json`（SHA-256 `a0402f74cfa2240eb29bde47edbea5a083dbb0dcd9f0098e8edeb06e63a92ae1`）；文档/发行物补充审计见 `evidence/S03-independent-documentation-release-audit-20261001.json`（SHA-256 `8b73d65adbdfa1b2de3289fe31d347c7ad0e83b0a3a107138240185821904223`）。S03 已完成 G5 步骤接受；报告见 `reports/step-S03-acceptance-v1/report.json`（content SHA-256 `6476a22d4b209ce1dcc98172bd59043333c8a8df41ac3980c67b61dd4717a871`），工作流状态为 complete。

## 0.4.0 — 本机诊断、数据概况与可信结果绘图（2026-10-02）

0.4.0 汇总已接受的 S01 本机诊断功能，以及 S02 数据概况、可信缓存绘图和 v0.4.0 分发文件。项目与锁文件版本均为 0.4.0；本次没有增加运行时依赖。S02 获批范围见 `reports/g4-s02-step-plan-v1/report.json`（content SHA-256 `9f75fdbccede22a556656a60e7d64336ec8acd51b8fadf69c5a8c417381fb5ee`）。

### 功能范围

- **诊断与错误追踪**：worker 与 GUI 以同一 error ID 关联 traceback 和本机日志；用户可从文件菜单导出当前会话或指定错误编号的脱敏诊断 ZIP。导出限于白名单、最大 20 MiB、本机保存，不上传或覆盖已有文件。S01 G5 报告 content SHA-256 为 `ae7738c6d5ec2f4998fb3f4394d9b609580f149e389e27a800061fb3c4d6a6c5`。
- **完整数据概况**：主窗口按当前完整 `LoadedDataset` 显示行列数、逐列 dtype、缺失数和 `describe()` 结果；概况不启动模型操作或修改输入。
- **基础绘图**：用户可从已加载表格绘制直方图、类别计数、数值散点与相关性图，也可从单任务或批量结果选择一个来源分区，查看分类混淆矩阵、回归实际值/预测值与残差、已有聚类/降维坐标，以及 HMM 隐藏状态和后验概率。绘图清单限定来源与字段；对话框按需加载 Matplotlib，并能将所选图保存为 PNG 或 SVG。
- **来源与测试分区门禁**：主应用先复制不可变绘图来源；结果图只消费缓存，不重新读源表、训练、搜索、预测、冻结配置或执行测试。test 仅在本次最终测试成功、来源所有权和凭据有效且计数为内置 `int` 严格等于 1 时开放；`bool`、浮点值和无效计数均关闭该分区。未涉及的 batch winner/session 零计数行为保持不变。
- **时间与状态语义**：序列图使用稳定的源行位置、组与窗口映射；显示时间统一为 UTC。HMM 状态表示潜在状态，不表示真实类别。

### 验证与发行记录

- S02 whole-step r1 verdict 为 FAIL，唯一必需失败是测试计数把 `True` 或 `1.0` 当成 1；该不可变原件保留在 `evidence/S02-independent-whole-step-verdict-r1-20261002.json`（SHA-256 `79283d8a45f06003bb3f7d65d225c6e857b86ef25d12bff89ee60973e7116c07`）。formal repair round 1 后，Sol 的独立 r2 verdict 为 7/7 required PASS，哈希审计绑定 51 项；其他六项沿用未受影响的 r1 PASS，计数门在源码和已安装 sdist 各通过 14 个正负例。r2 原件见 `evidence/S02-independent-whole-step-verdict-r2-20261002.json`（SHA-256 `54fb69579b965434a20e2e2290b3b2223cd3b2a2dc2584cef1e1b7a54dc558f4`）。
- 修复后的源码定向测试为 GUI/CACHE 10/10、batch CACHE 1/1；受影响的计数门在 wheel 与 sdist 的非 editable 安装上分别通过。final 文档只改 README、用户/API 指南和本文件，没有改变 Sol 复核过的 Python 源；最终 wheel/sdist 的 32 个包文件均逐文件与源码核对，项目 `.venv` 安装后的元数据、安装路径与来源也另行记录。
- 最终发行件的实际 SHA-256 只记录在 `dist/SHA256SUMS.txt` 和本 RUN 的外部证据 `evidence/S02-AC-final-docs-package-result-20261002.json`；本文件不嵌入自身所述分发文件的 SHA。旧版 v0.3.0 wheel/sdist 保留在 `dist`，对应哈希由同一校验清单记录。
- repair/final 文档发行没有重跑 48 项 package suite、18 项 PLOT、19 项 search/selection 或 28 项相关整合套件；本轮依赖 r2 的独立计数复验、修复后的受影响门、final 发行件安装和 32/32 来源核对，不把历史 suite 结果描述为本轮新跑。

### 已知限制

- 旧结果缓存若缺少来源绘图清单或所需 sidecar，其相应图表分区不可用；程序不会自动补写清单、重读原始数据、重训模型或重测。按正常流程创建新 session/job 后，可使用其有效缓存。
- D10 t-SNE 不支持 `transform`，因此不能生成新样本或 test 分区的投影图；没有对应输出的 estimator 不会伪造逐行结果。
- HMM 图只展示隐藏状态及后验概率；时间轴按 UTC 展示。XLS 读取路径存在，但本版本没有使用真实 XLS 样本验证；CSV 与 XLSX 已有样本验证。

## 0.2.0 — S02 批量搜索与持久队列（独立整步复核 PASS，已接受，2026-10-01）

本版本实现了 S02 批量搜索功能。独立 workflow verdict 为 PASS，步骤状态已接受；证据为 `evidence/S02-workflow-verdict-r1.json`，SHA-256 `c57c4b35d50d39fddddff157ccb4ea1be059f7f9481210bd103ab36775f314e6`。

- 批量入口：主窗口顶部“批量搜索队列”；桌面窗口可配置多个数据集/模型组合，并读取 SQLite 历史、暂停/恢复/取消、冻结 validation winner、导出对比 CSV。
- Python API：`run_batch` 提供预检、入队、运行的一步入口；也可使用 `create_search_jobs`、`run_search_jobs`、`load_search_result`、`freeze_search_winner` 和 `finalize_frozen_search` 控制持久化生命周期。
- 每组合上限为 50 次实际搜索拟合、250 个 proposals、20 分钟活动时间；默认 1 个并行 worker，最多 2 个，每 worker 数值线程固定为 1。
- 搜索只使用训练/验证分区。固定参数 Grid 可将空字段空间解释成单一候选并执行一次实际拟合；validation winner 必须明确冻结，最终在训练+验证（80%）数据上重拟合，再访问一次测试分区。
- 每个 job 入队时原子持久化自己的数据快照和完整性收据；执行、暂停恢复均使用该快照。源文件随后修改或删除不会改变排队数据，快照缺失、损坏或与配置/历史身份不符时会在 fit 前失败且不消耗拟合次数。
- GUI 验证榜和对比 CSV 按目标方向排列同组有限 validation 分数，同分共享 competition rank；训练探索和无分数任务不排名。训练探索导出可由 worker 或公开 API 从同一完成 job 复用现有 winner session，重复导出返回缓存且不增加 fit/proposal 或测试许可。
- base 含 NumPy/SciPy/pandas/scikit-learn/threadpoolctl；`desktop` extra 含桌面依赖；`search` extra 含 Optuna 与 pymoo。Grid、随机和退火不需要 `search` extra，TPE 和遗传搜索需各自依赖已安装。
- 当前统一安装的非 editable wheel 完成整包验证：unittest 66/66 通过、`uv lock --check` 通过；D10 同一完成 batch job 的 worker/API 导出、双 worker 并行、暂停恢复、取消、失败继续和进程崩溃恢复均通过。上述原记录写于独立复核前；S02 后续正式 workflow verdict 为 PASS，步骤已接受。
- 最新补充证据：`H:/AIcode/HexiPlan/runs/py-ml-20260930-01/evidence/S02-BD-package-completion-r1-result-20261001.json`、`H:/AIcode/HexiPlan/runs/py-ml-20260930-01/evidence/S02-BD-training-export-r1-result-20261001.json` 及日志 `H:/AIcode/HexiPlan/runs/py-ml-20260930-01/logs/S02-BD-training-export-r1-20261001.log`。原 A+C manifest 未改写。

发布 0.2.0 时仍未交付：3 个 HMM、4 个深度模型与 S03 扩展。此处的交付时点说明不改变 S02 后续已登记的独立 PASS 与接受状态。

## 0.1.0 — S01 首期包完成，独立整步复核 PASS，已接受（2026-10-01）

本文件记录 0.1.0 当时的交付范围和包级执行证据。S01 的独立 workflow verdict 为 PASS，步骤状态已接受；证据为 `evidence/S01-workflow-verdict-evidence.json`，SHA-256 `c7bbddc562c79c4338a540f6cdb13282b24d17d0db410a2aa30b49a22ea8172b`。

- **P01 PASS**：PEP 621 元数据、`src/pyml_workbench` 包结构、base 与 desktop 锁定依赖、71 个活动 scikit-learn 模型注册，以及 7 个明确未实现的 HMM/deep 条目。
- **P02 PASS**：CSV/XLSX 数据读取、预览、明确特征/目标选择、只在训练分区拟合的预处理、固定 60/20/20 seed 划分、验证后冻结再测、指标/逐行输出和可重载模型。71/71 活动模型的适用条件 smoke 与 LOF `novelty=True` 留出预测记录见 `RUN/evidence/P02-result-20260930.json` 和 `P02-71-model-outcomes.csv`。
- **P03 PASS**：PySide6 单任务工作台和 QProcess worker；已有流程记录显示训练期间 1,226 次 UI timer tick、最终测试计数 1、导出模型重载后预测 5 行；中文和 1280×720 offscreen 布局检查零缺字警告。证据见 `RUN/evidence/P03-result-20260930.json` 与 `P03-ui-preview-cjk.png`。
- **P04 包级检查**：2026-10-01 完成 README、用户指南、API 指南、71 项模型/参数索引、第三方来源/许可证指南、开发指南和两个合成示例。最终维护 suite 22/22 通过。监督示例完成显式 `prepare_experiment → freeze_experiment → evaluate_test`、单次测试、导出/重载/预测；t-SNE 示例显示训练 trustworthiness，并如实标注 validation/test 不支持 `transform`、没有 held-out 分数。offscreen UI 检查确认错误路径和缺失目标有可见错误，任务未启动，源 CSV SHA-256 不变；应用模块入口在 offscreen 环境启动后保持运行 3 秒。证据见 `RUN/evidence/P04-result-20261001.json`。
- **S01 独立审查定向修复（2026-10-01）**：GUI 将准备阶段的参数快照与用户实际冻结分开审计；仅当 `not_supported`、能力保护事件、零评估计数和对应缺失能力相互吻合时接受不评分的最终结果，并保留后续导出；常用浮点参数以精确文本显示和解析，支持科学计数法并拒绝非有限值。新增 4 项 GUI 回归通过；这些修改随后由独立整步 PASS 覆盖。
- **S01 独立复验第二轮定向修复（2026-10-01）**：最终结果继续绑定 session ID、session 路径与冻结配置摘要；对零计数能力保护结果，以训练阶段已绑定的能力映射为准，并拒绝响应能力声明与该映射冲突的结果。针对伪造能力声明新增负向回归；随后独立最终复核通过。

当前包包含 71 个活动传统模型（分类 24、回归 23、聚类 10、降维 10、异常检测 4），CSV/XLSX 输入路径已有样本验证。XLS 虽有 xlrd 读取实现，本期没有实际 XLS 样本验证。合成模型 smoke 证明方法可执行，不代表预测质量或科学结论；t-SNE 没有新样本 `transform`，LOF 默认 `novelty=False`，SVC 默认 `probability=False`。

**0.1.0 发布时尚未交付**：3 个 HMM、4 个深度模型、S02 批量队列/参数搜索及 S03 内容。S01 独立审查当时尚未完成，后续 workflow verdict 已 PASS 并接受；该历史状态不适用于 0.2.0/0.3.0。
