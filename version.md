# 版本记录

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
