# 第三方组件、上游来源与许可证

本项目声明表格机器学习基础依赖、可选搜索与 HMM/deep 后端，以及桌面组件。XLSX 读写属于基础包功能；旧版 XLS 输入属于桌面 extra。下表版本来自本项目锁定环境；许可证栏概述各上游发行包的主要许可证和安装元数据。重分发时还应保留各发行包附带的版权与许可证文本，尤其是包含第三方二进制组件的科学计算发行包。

| 组件 | 锁定版本 | 用途 | 主要许可证 | 上游来源 |
|---|---:|---|---|---|
| NumPy | 2.5.3 | 数组与数值计算（base） | BSD-3-Clause、0BSD、MIT、Zlib、CC0-1.0 组件声明 | [项目](https://numpy.org/) · [许可](https://numpy.org/doc/stable/license.html) |
| SciPy | 1.18.1 | 科学计算（base） | BSD 条款；发行包可能包含具有独立许可文本的二进制组件 | [项目](https://scipy.org/) · [许可](https://docs.scipy.org/doc/scipy/reference/dev/license.html) |
| pandas | 3.0.6 | 表格读写与数据帧（base） | BSD 3-Clause | [项目](https://pandas.pydata.org/) · [许可](https://pandas.pydata.org/about/) |
| scikit-learn | 1.9.1 | 71 个活动传统 estimator（base） | BSD-3-Clause | [项目与源码](https://github.com/scikit-learn/scikit-learn) · [许可与引用](https://scikit-learn.org/stable/about.html#citing-scikit-learn) |
| threadpoolctl | 3.7.0 | 将搜索和最终重拟合的数值线程限制为每 worker 1 个（base） | BSD-3-Clause | [项目与源码](https://github.com/joblib/threadpoolctl) · [许可](https://github.com/joblib/threadpoolctl/blob/master/LICENSE) |
| hmmlearn | 0.3.3 | H01/H02/H03 隐马尔可夫模型（`sequence` extra） | BSD-3-Clause | [项目与源码](https://github.com/hmmlearn/hmmlearn) · [许可证](https://github.com/hmmlearn/hmmlearn/blob/main/LICENSE) |
| PyTorch | 2.14.1 | N01/N02/N04/N06 深度模型张量、网络与训练（`deep` extra；当前实现固定 CPU） | BSD-3-Clause | [项目](https://pytorch.org/) · [许可证](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| skorch | 1.4.0 | PyTorch 与 scikit-learn 风格训练接口的适配层（`deep` extra） | BSD-3-Clause | [项目与文档](https://skorch.readthedocs.io/) · [许可证](https://github.com/skorch-dev/skorch/blob/master/LICENSE) |
| Optuna | 5.0.0 | TPE 参数搜索（`search` extra） | MIT | [项目](https://optuna.org/) · [源码与许可](https://github.com/optuna/optuna) |
| pymoo | 0.6.2 | 混合变量遗传搜索（`search` extra） | Apache-2.0 | [项目](https://pymoo.org/) · [源码与许可](https://github.com/anyoptimization/pymoo) |
| PySide6 | 6.11.2 | Qt 桌面界面（desktop） | LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only | [项目](https://wiki.qt.io/Qt_for_Python) · [开源许可说明](https://www.qt.io/licensing/open-source-lgpl-obligations) |
| Matplotlib | 3.11.2 | QtAgg 指标图、基础图表渲染及 PNG/SVG 导出（desktop） | Matplotlib License Agreement | [项目](https://matplotlib.org/) · [许可](https://matplotlib.org/stable/project/license.html) |
| openpyxl | 3.1.5 | XLSX 输入/输出及默认结果导出（base） | MIT | [项目](https://openpyxl.readthedocs.io/) · [许可](https://openpyxl.readthedocs.io/en/stable/license.html) |
| xlrd | 2.0.2 | XLS 输入（desktop） | BSD | [项目与源码](https://github.com/python-excel/xlrd) · [许可](https://xlrd.readthedocs.io/en/latest/) |

包构建后端为 Hatchling（`pyproject.toml` 的 build-system 要求 `hatchling>=1.27,<2`）；它是构建依赖，不是运行时依赖。版本解析以仓库 `uv.lock` 为准。Optuna 与 pymoo 只在使用 `search` extra 时安装；Grid、随机与 SciPy 退火不要求该 extra。hmmlearn、PyTorch 与 skorch 分别只在安装 `sequence`/`deep` extras 时安装，不属于基础运行依赖。

## 算法上游

传统模型目录中的每个条目记录 scikit-learn estimator 的完整导入名和官方 API 链接；HMM 与深度模型记录各自上游库和安装 extra，见[模型与参数索引](model-index.md)。参数默认值由实际 estimator `get_params(deep=True)` 或扩展模型 `parameter_schema()` 暴露；模型 smoke 结果证明的是对应模拟任务下的执行，不代表模型准确率、科学有效性或任意参数组合都适用。当前深度实现显式使用 CPU，尚未验证 GPU 运行路径。

许可证信息用于说明当前直接依赖来源，不构成法律意见。具体重分发义务以随对应锁定发行包提供的完整许可证、版权声明和 NOTICE 为准；依赖升级后应重新核对其元数据和许可证文件。
