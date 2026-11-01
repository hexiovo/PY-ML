# [[研究标题]]

[[作者]]
[[单位]]
[[日期]]

填写说明：所有双中括号内容须用本轮可核实事实替换；不适用项目写明原因。匿名示例仅供版式和表达参考。

# 方法

## 研究目的与实验设计

本研究考察[[研究问题及目标含义]]。实验采用[[组间/被试内/混合设计、条件和重复测量结构]]；独立统计单位为[[被试/其他已确认单位]]。样本来源、纳入与排除标准为[[依据及实际人数]]。未知信息应标为缺失，不从文件名推断。

## 数据与变量

数据表[[编号]]包含[[记录数]]行和[[原始特征数]]个特征，缺失单元数为[[数量]]。每行代表[[整段/片段/试次等]]；身份、目标与特征分别由[[可核实来源]]确定。目标为[[真实条件/已授权人工夹具]]；若为夹具，所有分数仅验证接口。正文使用来源编号，个人路径保留在研究侧来源台账，不写入公共模板。

## 划分、预处理与降维

数据按[[被试组/时间/其他合理规则]]划分为[[训练/验证/最终测试范围]]。填补、缩放、特征选择和降维仅在训练分区拟合，并在内部交叉验证中重新拟合。堆叠内层OOF同样尊重[[组边界]]（Pedregosa et al., 2011; Varoquaux et al., 2017）。

实际降维为[[未执行/方法]]，输入[[p]]维、输出[[k]]维，拟合范围为[[范围]]。未选择列不应称为学习型降维；全部特征与算法输入契约例外分别说明。

## 模型、差异与预算

使用[[实际模型及版本]]。线性、近邻、条件分布、树集成及神经网络的目标与假设不同；平均集成与逐阶段提升的训练依赖也不同（Breiman, 2001; Pedregosa et al., 2011）。搜索规则、候选预算和停止条件预先指定（Bergstra & Bengio, 2012）。完整算法差异见附录，未使用的方法标为不适用，不能沿用示例PASS。

[[按实际方法保留以下引用，未使用则删除对应句与文末条目。]] 外部提升树实现遵循各自接口与参数约束（XGBoost developers, n.d.; LightGBM developers, n.d.; CatBoost developers, n.d.）。深度模型及其适配分别依据PyTorch和skorch契约（PyTorch contributors, n.d.; Skorch developers, n.d.）。HMM与单变量预测使用各自观测和历史定义（Hmmlearn developers, n.d.; Statsmodels developers, n.d.）。TPE与通用评价接口分别见相应文档（Optuna contributors, n.d.; Scikit-learn developers, n.d.）；t-SNE和UMAP表示目标及假设不同（van der Maaten & Hinton, 2008; UMAP developers, n.d.）。

外层重复[[R]]次；这不等于[[内层折数]]折或max_iter。随机种子、训练迭代、深度epoch及所有输入契约见配置台账。HMM按发射分布、序列边界和有限似然核验；人工时序或符号不冒充真实时间或观测。

## 指标、解释与接续

分类报告accuracy、balanced accuracy、macro F1及适用AUC，正类/多分类平均口径为[[定义]]。回归报告MAE、RMSE、适用R²及目标域相关偏差。无监督和HMM按任务输出，不强行填写分类指标。重复种子的SD不等于独立样本置信区间。

特征排序使用[[系数/原生解释/验证集置换及定义]]，包括[[全部列数]]列，保留负值和不可估计原因。阶段状态仅记录true/false；日志、错误与代码/数据/环境指纹分别保存。最终测试状态为[[实际调用次数及是否冻结]]，不能因模板填充而重跑最终test。

# 结果

## 实际覆盖与数据范围

本轮预定[[流程数]]条方法×目标×来源流程，各[[次数]]次，共[[槽位数]]项；通过[[数量]]，未通过[[数量]]，未执行[[数量]]。PASS只说明相应检查通过，不证明收敛、质量或科学效度。

**表 1** *数据与实际执行范围*

来源编号 | 行×原始列 | 独立单位 | 实际模型输入 | 目标/状态
--- | --- | --- | --- | ---
[[来源]] | [[n×p]] | [[单位数]] | [[维数及例外]] | [[真实/人工、状态]]

注。不同来源、目标和评价单位分开，未映射与未选择分别记录。

## 分类结果与特征排序

预先指定的[[模型]]在[[来源/目标]]上的accuracy为[[M]]（SD = [[SD]]），AUC为[[M]]（SD = [[SD]]）。若AUC不可估计，应报告类别/分数缺失等实际原因。

**表 2** *逐模型分类验证指标*

模型 | 有效次数 | accuracy M (SD) | AUC M (SD) | macro F1
--- | --- | --- | --- | ---
[[模型]] | [[n]] | [[值]] | [[值或原因]] | [[值]]

注。写明正类、多分类OvR/OvO及macro/weighted；失败模型也保留。

**图 1** *各模型准确率与AUC对比*

[[插入本轮实际结果绘制的图；不得直接沿用匿名示例的数值或图形]]

注。同一目标、来源和划分口径。误差条须注明SD或适用区间；参考线含义明确。

**表 3** *特征排序*

排名 | 匿名特征ID | 排序量 M (SD) | 有效次数
--- | --- | --- | ---
[[序号]] | [[ID]] | [[值]] | [[n]]

注。完整排序和可授权的原始特征映射另存；排序不等于因果或生理显著性。

**图 2** *固定模型的特征排序*

[[插入本轮实际结果绘制的图；不得直接沿用匿名示例的数值或图形]]

注。注明展示范围、完整排序位置和负值处理；不按最好分数挑示例。

## 回归、计数、窗口及预测结果

**表 4** *任务特定验证误差*

目标/来源 | 模型 | 有效次数 | MAE | RMSE | R²/原因
--- | --- | --- | --- | --- | ---
[[目标]] | [[模型]] | [[n]] | [[值]] | [[值]] | [[值或原因]]

注。跨任务不混排；少于2个观测或常数目标时解释R²不可估计。巨大有限误差保留，使用科学计数法。

**图 3** *回归误差及预测—目标对比*

[[插入本轮实际结果绘制的图；不得直接沿用匿名示例的数值或图形]]

注。明确指标单位、评价样本数和坐标变换；必要时对数或对称对数刻度，但不能隐藏极端值。

## HMM、无监督与失败诊断

HMM的[[训练/验证似然、观测数与发射维数]]为[[结果]]，不同观测空间的似然不作总榜。聚类、异常与降维的[[标签数量/输出维数/其他适用输出]]为[[结果]]；描述性全量拟合不等于独立验证。

**表 5** *未通过类型与处理*

方法/阶段 | 错误类别 | 实际原因 | 处理及剩余限制
--- | --- | --- | ---
[[方法]] | [[程序/接口/参数/数值/依赖/资源]] | [[证据]] | [[修复或限制]]

注。未定义指标、警告和拟合失败分别记录；不以删方法、挑种子或强制补概率制造PASS。

# 简短讨论

## 主要结果及设计含义

针对[[研究问题]]，结果支持[[证据可支持的范围]]。与[[实验设计及独立单位]]结合解释；不把人工目标、窗口数量或重复种子当作真实效应或新增独立样本。

## 限制及后续工作

本轮限制包括[[样本量、高维、目标来源、优化预算、缺失、依赖及验证限制]]。后续需[[真实标签/独立数据/预先冻结协议等]]。引用须与实际使用的方法对应，删除未使用方法的文献，而非机械保留全部。

# 参考文献

Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. Journal of Machine Learning Research, 13, 281–305. https://jmlr.org/papers/v13/bergstra12a.html

Breiman, L. (2001). Random forests. Machine Learning, 45, 5–32. https://doi.org/10.1023/A:1010933404324

CatBoost developers. (n.d.). How training is performed. CatBoost documentation. Retrieved November 1, 2026, from https://catboost.ai/docs/en/concepts/algorithm-main-stages

Hmmlearn developers. (n.d.). Tutorial. Hmmlearn documentation. Retrieved November 1, 2026, from https://hmmlearn.readthedocs.io/en/stable/tutorial.html

LightGBM developers. (n.d.). Features. LightGBM documentation. Retrieved November 1, 2026, from https://lightgbm.readthedocs.io/en/stable/Features.html

Optuna contributors. (n.d.). optuna.samplers.TPESampler. Optuna documentation. Retrieved November 1, 2026, from https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html

PyTorch contributors. (n.d.). torch.nn. PyTorch 2.14 documentation. Retrieved November 1, 2026, from https://docs.pytorch.org/docs/2.14/nn.html

Scikit-learn developers. (n.d.). User guide. Scikit-learn documentation. Retrieved November 1, 2026, from https://scikit-learn.org/stable/user_guide.html

Skorch developers. (n.d.). NeuralNetRegressor. Skorch documentation. Retrieved November 1, 2026, from https://skorch.readthedocs.io/en/stable/regressor.html

Statsmodels developers. (n.d.). Time series analysis tsa. Statsmodels documentation. Retrieved November 1, 2026, from https://www.statsmodels.org/stable/tsa.html

UMAP developers. (n.d.). How UMAP works. UMAP documentation. Retrieved November 1, 2026, from https://umap-learn.readthedocs.io/en/latest/how_umap_works.html

van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. Journal of Machine Learning Research, 9, 2579–2605. https://jmlr.org/papers/v9/vandermaaten08a.html

Varoquaux, G., Raamana, P. R., Engemann, D. A., Hoyos-Idrobo, A., Schwartz, Y., & Thirion, B. (2017). Assessing and tuning brain decoders: Cross-validation, caveats, and guidelines. NeuroImage, 145, 166–179. https://doi.org/10.1016/j.neuroimage.2016.10.038

XGBoost developers. (n.d.). Introduction to boosted trees. XGBoost documentation. Retrieved November 1, 2026, from https://xgboost.readthedocs.io/en/stable/tutorials/model.html

# 附录：算法差异与本轮配置

## 模型

**表 A1** *模型差异*

ID与名称 | 机制差异 | 本轮配置与实际状态
--- | --- | ---
model:C01；分类 · LogisticRegression | 以逻辑损失拟合线性类别边界；可给出类别概率，区别于最小二乘分类。 | [[使用/不适用、参数、输入及实际PASS]]
model:C02；分类 · RidgeClassifier | 将类别编码后作L2惩罚最小二乘；线性边界，通常无原生概率输出。 | [[使用/不适用、参数、输入及实际PASS]]
model:C03；分类 · SGDClassifier | 随机梯度优化线性分类目标；具体损失决定间隔或概率含义，学习率影响收敛。 | [[使用/不适用、参数、输入及实际PASS]]
model:C04；分类 · PassiveAggressiveClassifier | 对满足间隔的样本保持参数，对违例样本作受约束更新；适合在线线性分类。 | [[使用/不适用、参数、输入及实际PASS]]
model:C05；分类 · Perceptron | 依据误分类更新线性权重；不同于逻辑回归的概率损失，无概率模型。 | [[使用/不适用、参数、输入及实际PASS]]
model:C06；分类 · LinearSVC | 优化线性大间隔分类目标；区别于核SVC，不显式构造非线性核边界。 | [[使用/不适用、参数、输入及实际PASS]]
model:C07；分类 · SVC | 核支持向量分类，可学习非线性边界；C与核参数共同控制复杂度。 | [[使用/不适用、参数、输入及实际PASS]]
model:C08；分类 · KNeighborsClassifier | 由近邻类别投票；局部、非参数，距离尺度及邻居数决定结果。 | [[使用/不适用、参数、输入及实际PASS]]
model:C09；分类 · GaussianNB | 假定给定类别后特征独立且各特征服从高斯分布；适于连续输入。 | [[使用/不适用、参数、输入及实际PASS]]
model:C10；分类 · MultinomialNB | 基于非负计数/频率的多项式朴素贝叶斯；数值转为非负不自动满足计数解释。 | [[使用/不适用、参数、输入及实际PASS]]
model:C11；分类 · BernoulliNB | 建模二值特征出现与未出现；与多项式模型的计数解释不同。 | [[使用/不适用、参数、输入及实际PASS]]
model:C12；分类 · ComplementNB | 用类别补集统计量构建权重；面向非负、常见于不平衡文本计数。 | [[使用/不适用、参数、输入及实际PASS]]
model:C13；分类 · CategoricalNB | 按每个特征的离散类别概率建模；需要整数类别编码及合法取值范围。 | [[使用/不适用、参数、输入及实际PASS]]
model:C14；分类 · DecisionTreeClassifier | 递归选择特征阈值以降低节点不纯度/损失；可解释路径，但深树容易过拟合。 | [[使用/不适用、参数、输入及实际PASS]]
model:C15；分类 · ExtraTreeClassifier | 在候选特征上随机选阈值形成单树；比常规最优阈值树引入更多随机性。 | [[使用/不适用、参数、输入及实际PASS]]
model:C16；分类 · RandomForestClassifier | 聚合随机特征子集下训练的多棵树；主要以平均/投票降低单树方差。 | [[使用/不适用、参数、输入及实际PASS]]
model:C17；分类 · ExtraTreesClassifier | 聚合随机阈值树；与随机森林的区别主要在切分阈值随机化及抽样默认策略。 | [[使用/不适用、参数、输入及实际PASS]]
model:C18；分类 · GradientBoostingClassifier | 按当前损失梯度逐阶段增加树；依赖前序模型，区别于独立树的平均。 | [[使用/不适用、参数、输入及实际PASS]]
model:C19；分类 · HistGradientBoostingClassifier | 先将连续特征分箱再进行梯度提升；区别于逐候选值精确搜索的树提升。 | [[使用/不适用、参数、输入及实际PASS]]
model:C20；分类 · AdaBoostClassifier | 逐阶段调整样本权重或基学习器权重；困难/异常样本可产生较大影响。 | [[使用/不适用、参数、输入及实际PASS]]
model:C21；分类 · MLPClassifier | 以多层前馈非线性网络拟合目标；分类与回归损失不同，需控制缩放与训练轮数。 | [[使用/不适用、参数、输入及实际PASS]]
model:C22；分类 · LinearDiscriminantAnalysis | 假定类别条件高斯且共享协方差，得到线性判别；监督投影维度至多类别数减一。 | [[使用/不适用、参数、输入及实际PASS]]
model:C23；分类 · QuadraticDiscriminantAnalysis | 每类估计独立协方差而形成二次边界；小样本下协方差估计更不稳定。 | [[使用/不适用、参数、输入及实际PASS]]
model:C24；分类 · GaussianProcessClassifier | 用核定义潜在函数先验并作分类近似推断；区别于仅寻找间隔的SVM。 | [[使用/不适用、参数、输入及实际PASS]]
model:R01；回归 · LinearRegression | 以残差平方和拟合条件均值；无正则惩罚，受共线性和离群点影响。 | [[使用/不适用、参数、输入及实际PASS]]
model:R02；回归 · Ridge | 平方损失加L2惩罚；收缩相关特征系数，一般不使其精确归零。 | [[使用/不适用、参数、输入及实际PASS]]
model:R03；回归 · Lasso | 平方损失加L1惩罚；可产生稀疏系数，尺度与正则强度影响选择。 | [[使用/不适用、参数、输入及实际PASS]]
model:R04；回归 · ElasticNet | 结合L1与L2惩罚；在稀疏性与相关特征共同收缩之间折中。 | [[使用/不适用、参数、输入及实际PASS]]
model:R05；回归 · SGDRegressor | 用随机梯度求解线性回归目标；区别在求解及可选损失，不自动产生非线性关系。 | [[使用/不适用、参数、输入及实际PASS]]
model:R06；回归 · PassiveAggressiveRegressor | 对超出容忍区间的误差作在线受约束更新；依赖尺度与正则控制。 | [[使用/不适用、参数、输入及实际PASS]]
model:R07；回归 · HuberRegressor | 小残差采用平方损失，大残差采用线性损失；降低大残差的影响。 | [[使用/不适用、参数、输入及实际PASS]]
model:R08；回归 · QuantileRegressor | 最小化分位数损失，估计指定条件分位数；不能直接解释为条件均值。 | [[使用/不适用、参数、输入及实际PASS]]
model:R09；回归 · LinearSVR | 线性支持向量回归，在epsilon容忍区间外惩罚误差；不同于普通平方损失。 | [[使用/不适用、参数、输入及实际PASS]]
model:R10；回归 · SVR | 用核构建非线性epsilon支持向量回归；距离尺度与核宽度重要。 | [[使用/不适用、参数、输入及实际PASS]]
model:R11；回归 · KNeighborsRegressor | 以近邻目标的局部平均/加权平均预测；缺少全局参数方程。 | [[使用/不适用、参数、输入及实际PASS]]
model:R12；回归 · DecisionTreeRegressor | 递归选择特征阈值以降低节点不纯度/损失；可解释路径，但深树容易过拟合。 | [[使用/不适用、参数、输入及实际PASS]]
model:R13；回归 · ExtraTreeRegressor | 在候选特征上随机选阈值形成单树；比常规最优阈值树引入更多随机性。 | [[使用/不适用、参数、输入及实际PASS]]
model:R14；回归 · RandomForestRegressor | 聚合随机特征子集下训练的多棵树；主要以平均/投票降低单树方差。 | [[使用/不适用、参数、输入及实际PASS]]
model:R15；回归 · ExtraTreesRegressor | 聚合随机阈值树；与随机森林的区别主要在切分阈值随机化及抽样默认策略。 | [[使用/不适用、参数、输入及实际PASS]]
model:R16；回归 · GradientBoostingRegressor | 按当前损失梯度逐阶段增加树；依赖前序模型，区别于独立树的平均。 | [[使用/不适用、参数、输入及实际PASS]]
model:R17；回归 · HistGradientBoostingRegressor | 先将连续特征分箱再进行梯度提升；区别于逐候选值精确搜索的树提升。 | [[使用/不适用、参数、输入及实际PASS]]
model:R18；回归 · AdaBoostRegressor | 逐阶段调整样本权重或基学习器权重；困难/异常样本可产生较大影响。 | [[使用/不适用、参数、输入及实际PASS]]
model:R19；回归 · MLPRegressor | 以多层前馈非线性网络拟合目标；分类与回归损失不同，需控制缩放与训练轮数。 | [[使用/不适用、参数、输入及实际PASS]]
model:R20；回归 · GaussianProcessRegressor | 核定义函数先验并进行高斯过程回归；区间来自模型假设，不是接口PASS的含义。 | [[使用/不适用、参数、输入及实际PASS]]
model:R21；回归 · PoissonRegressor | 使用Poisson偏差与对数链接；目标非负，科研解释需要适当计数/均值方差结构。 | [[使用/不适用、参数、输入及实际PASS]]
model:R22；回归 · GammaRegressor | Gamma偏差与对数链接；目标严格为正，区别于Poisson的方差函数。 | [[使用/不适用、参数、输入及实际PASS]]
model:R23；回归 · TweedieRegressor | 用power指定均值—方差幂关系；默认power与实际目标尺度共同决定含义。 | [[使用/不适用、参数、输入及实际PASS]]
model:K01；聚类 · KMeans | 最小化样本到质心的平方距离；需预设簇数，偏好欧氏紧凑簇。 | [[使用/不适用、参数、输入及实际PASS]]
model:K02；聚类 · MiniBatchKMeans | 用小批量近似更新质心；牺牲部分精确优化换取可扩展性。 | [[使用/不适用、参数、输入及实际PASS]]
model:K03；聚类 · BisectingKMeans | 递归二分簇形成层次划分；区别于KMeans直接同时更新全部质心。 | [[使用/不适用、参数、输入及实际PASS]]
model:K04；聚类 · AffinityPropagation | 通过相似度消息传递选择代表点；簇数由偏好参数和数据共同决定。 | [[使用/不适用、参数、输入及实际PASS]]
model:K05；聚类 · MeanShift | 沿核密度梯度寻找模态；不预设簇数，带宽决定合并尺度。 | [[使用/不适用、参数、输入及实际PASS]]
model:K06；聚类 · Birch | 以聚类特征树压缩样本再聚类；阈值决定子簇粒度与内存。 | [[使用/不适用、参数、输入及实际PASS]]
model:K07；聚类 · SpectralClustering | 先对相似图作谱嵌入再划分；可处理非凸结构，通常无原生新样本预测。 | [[使用/不适用、参数、输入及实际PASS]]
model:K08；聚类 · AgglomerativeClustering | 从单点逐级合并形成层次结构；链接准则决定簇间距离，无原生新样本预测。 | [[使用/不适用、参数、输入及实际PASS]]
model:K09；聚类 · DBSCAN | 用固定邻域密度连接核心点并标记噪声；无需预设簇数，密度尺度需适配。 | [[使用/不适用、参数、输入及实际PASS]]
model:K10；聚类 · OPTICS | 构造可达距离排序表达多尺度密度结构；区别于单一邻域尺度的DBSCAN。 | [[使用/不适用、参数、输入及实际PASS]]
model:D01；降维 · PCA | 中心化后寻找最大方差正交方向；不使用目标标签，不等同于生理信号源分离。 | [[使用/不适用、参数、输入及实际PASS]]
model:D02；降维 · IncrementalPCA | 分批近似估计主成分；区别于一次性完整矩阵分解。 | [[使用/不适用、参数、输入及实际PASS]]
model:D03；降维 · KernelPCA | 在核空间提取主成分；非线性映射依赖核设置，默认核不一定非线性。 | [[使用/不适用、参数、输入及实际PASS]]
model:D04；降维 · TruncatedSVD | 对未强制中心化的矩阵作截断奇异值分解；适合稀疏表示，区别于中心化PCA。 | [[使用/不适用、参数、输入及实际PASS]]
model:D05；降维 · FastICA | 通过非高斯性等准则寻找统计独立成分；独立性假设强于不相关性。 | [[使用/不适用、参数、输入及实际PASS]]
model:D06；降维 · NMF | 在非负约束下分解为基与系数；加性成分与PCA有符号正交成分不同。 | [[使用/不适用、参数、输入及实际PASS]]
model:D07；降维 · FactorAnalysis | 用潜在线性因子和特征特异噪声解释协方差；区别于PCA直接最大化总方差。 | [[使用/不适用、参数、输入及实际PASS]]
model:D08；降维 · Isomap | 以近邻图最短路径近似流形测地距离后嵌入；图连通性影响表示。 | [[使用/不适用、参数、输入及实际PASS]]
model:D09；降维 · LocallyLinearEmbedding | 保留局部线性重建权重生成嵌入；对邻居数和局部几何敏感。 | [[使用/不适用、参数、输入及实际PASS]]
model:D10；降维 · TSNE | 匹配高低维邻域概率以可视化局部结构；本适配器无新样本transform，不能据图推断条件效应。 | [[使用/不适用、参数、输入及实际PASS]]
model:A01；异常检测 · IsolationForest | 随机切分隔离样本，以平均路径长度表征异常；阈值依赖污染比例设置。 | [[使用/不适用、参数、输入及实际PASS]]
model:A02；异常检测 · OneClassSVM | 在核空间估计正常样本支持域；nu与核宽度决定边界和异常比例约束。 | [[使用/不适用、参数、输入及实际PASS]]
model:A03；异常检测 · LocalOutlierFactor | 将局部密度与邻域密度比较；默认novelty=False用于已拟合样本，不能混作新样本检测。 | [[使用/不适用、参数、输入及实际PASS]]
model:A04；异常检测 · EllipticEnvelope | 用鲁棒位置/协方差拟合椭圆正常域；适用于近似高斯椭圆分布。 | [[使用/不适用、参数、输入及实际PASS]]
model:C25；分类 · XGBoost | 正则化梯度提升树实现；其优化与树控制参数不同于随机森林的独立树平均。 | [[使用/不适用、参数、输入及实际PASS]]
model:C26；分类 · LightGBM | 直方图提升树、通常按叶扩展；分箱和叶复杂度控制区别于其他提升实现。 | [[使用/不适用、参数、输入及实际PASS]]
model:C27；分类 · CatBoost | 支持类别特征处理及有序提升等机制的提升树库；本次数值夹具未验证类别优势。 | [[使用/不适用、参数、输入及实际PASS]]
model:R24；回归 · XGBoost | 正则化梯度提升树实现；其优化与树控制参数不同于随机森林的独立树平均。 | [[使用/不适用、参数、输入及实际PASS]]
model:R25；回归 · LightGBM | 直方图提升树、通常按叶扩展；分箱和叶复杂度控制区别于其他提升实现。 | [[使用/不适用、参数、输入及实际PASS]]
model:R26；回归 · CatBoost | 支持类别特征处理及有序提升等机制的提升树库；本次数值夹具未验证类别优势。 | [[使用/不适用、参数、输入及实际PASS]]
model:C28；分类 · DummyClassifier | 忽略输入关联，按预设常数/频率策略预测；提供基线而非学习信号。 | [[使用/不适用、参数、输入及实际PASS]]
model:C29；分类 · VotingClassifier | 组合多个基学习器的投票或数值平均；不训练负责组合的第二层模型。 | [[使用/不适用、参数、输入及实际PASS]]
model:C30；分类 · StackingClassifier | 用基学习器的交叉验证输出训练第二层模型；区别于直接投票，需防止元学习泄漏。 | [[使用/不适用、参数、输入及实际PASS]]
model:C31；分类 · BaggingClassifier | 在重采样子集上拟合基学习器并聚合；区别于逐阶段纠正前序误差的提升。 | [[使用/不适用、参数、输入及实际PASS]]
model:R27；回归 · DummyRegressor | 以均值等固定策略输出目标；衡量复杂方法是否超越简单基线需另行有效评价。 | [[使用/不适用、参数、输入及实际PASS]]
model:R28；回归 · VotingRegressor | 组合多个基学习器的投票或数值平均；不训练负责组合的第二层模型。 | [[使用/不适用、参数、输入及实际PASS]]
model:R29；回归 · StackingRegressor | 用基学习器的交叉验证输出训练第二层模型；区别于直接投票，需防止元学习泄漏。 | [[使用/不适用、参数、输入及实际PASS]]
model:R30；回归 · BaggingRegressor | 在重采样子集上拟合基学习器并聚合；区别于逐阶段纠正前序误差的提升。 | [[使用/不适用、参数、输入及实际PASS]]
model:R31；回归 · BayesianRidge | 在贝叶斯线性回归中估计共同权重精度和噪声精度；区别于固定惩罚的Ridge。 | [[使用/不适用、参数、输入及实际PASS]]
model:R32；回归 · ARDRegression | 为各特征估计独立精度并可剪除权重；比共同精度的BayesianRidge更具稀疏性。 | [[使用/不适用、参数、输入及实际PASS]]
model:R33；回归 · KernelRidge | 以核技巧求解L2正则最小二乘；与SVR的epsilon损失不同，通常不稀疏。 | [[使用/不适用、参数、输入及实际PASS]]
model:R34；回归 · RANSACRegressor | 反复抽样识别内点后拟合；鲁棒性依赖残差阈值和足够的内点比例。 | [[使用/不适用、参数、输入及实际PASS]]
model:R35；回归 · TheilSenRegressor | 聚合子样本回归估计以获得鲁棒斜率；计算规模和高维鲁棒性有限制。 | [[使用/不适用、参数、输入及实际PASS]]
model:R36；回归 · PLSRegression | 提取与目标协变的潜成分再回归；有监督，区别于只保留输入方差的PCA。 | [[使用/不适用、参数、输入及实际PASS]]
model:K11；聚类 · GaussianMixture | 以高斯混合分布拟合样本并提供软归属概率；簇形状受协方差结构制约。 | [[使用/不适用、参数、输入及实际PASS]]
model:K12；聚类 · BayesianGaussianMixture | 以变分贝叶斯权重先验约束混合成分；可降低冗余成分权重。 | [[使用/不适用、参数、输入及实际PASS]]
model:K13；聚类 · HDBSCAN | 基于密度层次提取稳定簇，可标记噪声；区别于固定epsilon的DBSCAN，无原生新样本预测。 | [[使用/不适用、参数、输入及实际PASS]]
model:D11；降维 · UMAP | 构建模糊近邻图并优化低维图表示；与t-SNE目标不同，可支持已拟合映射的新样本变换。 | [[使用/不适用、参数、输入及实际PASS]]
model:D12；降维 · Autoencoder | 通过瓶颈编码重建输入；非线性表示需训练，重建成功不证明保留研究相关信息。 | [[使用/不适用、参数、输入及实际PASS]]
model:T01；回归 · NaiveForecast | 朴素预测：未来均取训练末值；提供单变量固定起点基线，不学习周期。 | [[使用/不适用、参数、输入及实际PASS]]
model:T02；回归 · SeasonalNaiveForecast | 季节朴素预测：重复训练末段季节周期；依赖正确给定周期，本次为4。 | [[使用/不适用、参数、输入及实际PASS]]
model:T03；回归 · ExponentialSmoothing | 指数平滑：对历史值递减加权估计水平；本次未设置趋势或季节项，不称为完整季节模型。 | [[使用/不适用、参数、输入及实际PASS]]
model:T04；回归 · ARIMA | ARIMA：结合自回归、差分与移动平均；本次(1,0,0)，无差分和MA项。 | [[使用/不适用、参数、输入及实际PASS]]
model:T05；回归 · SARIMA | SARIMA：在ARIMA上增加季节结构；本次非季节(1,0,0)、季节(1,0,0,4)。 | [[使用/不适用、参数、输入及实际PASS]]
model:N07；回归 · 1D CNN | 一维CNN：以局部时间卷积提取窗口模式；感受野由卷积结构决定。 | [[使用/不适用、参数、输入及实际PASS]]
model:N08；回归 · TCN | TCN：采用因果/膨胀时间卷积建模窗口依赖；区别于递归逐步状态更新。 | [[使用/不适用、参数、输入及实际PASS]]
model:N09；回归 · Transformer | Transformer：以注意力组合窗口内信息；时间位置编码及掩码决定顺序/可见范围。 | [[使用/不适用、参数、输入及实际PASS]]
model:H01；序列/HMM · GaussianHMM | Gaussian HMM：离散隐状态转移与单高斯发射；需明确每段序列长度。 | [[使用/不适用、参数、输入及实际PASS]]
model:H02；序列/HMM · GMMHMM | GMM HMM：每个隐状态使用高斯混合发射；比单高斯HMM增加状态内分布复杂度。 | [[使用/不适用、参数、输入及实际PASS]]
model:H03；序列/HMM · CategoricalHMM | Categorical HMM：发射为离散符号概率；与连续高斯发射的数据契约不同。 | [[使用/不适用、参数、输入及实际PASS]]
model:N01；分类 · Sequential | Sequential分类网络：前馈非线性映射输出类别；由本包模块与skorch封装组成。 | [[使用/不适用、参数、输入及实际PASS]]
model:N02；回归 · Sequential | Sequential回归网络：前馈映射输出连续目标；与N01的目标和损失契约不同。 | [[使用/不适用、参数、输入及实际PASS]]
model:N04；序列回归 · LSTM module | LSTM：显式细胞状态及门控保留历史信息；以固定窗口作下一步回归，保留序列边界。 | [[使用/不适用、参数、输入及实际PASS]]
model:N06；序列回归 · GRU module | GRU：用更新/重置门调节隐状态；与LSTM相比省去独立细胞状态。 | [[使用/不适用、参数、输入及实际PASS]]

注。基于已核实的方法清单供选择；当前版本、适用条件和来源需在本轮复核。完整官方链接随algorithm-reference.csv提供。

## 搜索

**表 A2** *搜索差异*

ID与名称 | 机制差异 | 本轮配置与实际状态
--- | --- | ---
search:grid；grid | 网格搜索：按离散候选组合枚举；本次只搜索C的两个预设取值。 | [[使用/不适用、参数、输入及实际PASS]]
search:random；random | 随机搜索：随机采样候选；覆盖依赖预算和采样分布，本次最多2次候选拟合。 | [[使用/不适用、参数、输入及实际PASS]]
search:genetic；genetic | 遗传搜索：通过种群选择、交叉和变异提出候选；两次拟合预算仅验证执行路径。 | [[使用/不适用、参数、输入及实际PASS]]
search:annealing；annealing | 退火搜索：温度控制接受较差候选的概率；本次C为[0.25,1.0]连续范围。 | [[使用/不适用、参数、输入及实际PASS]]
search:tpe；tpe | TPE搜索：利用较好/较差候选的参数密度提出新候选；极小预算不检验长期优化优势。 | [[使用/不适用、参数、输入及实际PASS]]

注。基于已核实的方法清单供选择；当前版本、适用条件和来源需在本轮复核。完整官方链接随algorithm-reference.csv提供。

## 训练前降维

**表 A3** *训练前降维差异*

ID与名称 | 机制差异 | 本轮配置与实际状态
--- | --- | ---
reduction:pca；pca | 中心化后寻找最大方差正交方向；不使用目标标签，不等同于生理信号源分离。 本次作为C01训练前的二维转换，而非独立降维任务。 | [[使用/不适用、参数、输入及实际PASS]]
reduction:truncated_svd；truncated_svd | 对未强制中心化的矩阵作截断奇异值分解；适合稀疏表示，区别于中心化PCA。 本次作为C01训练前的二维转换，而非独立降维任务。 | [[使用/不适用、参数、输入及实际PASS]]
reduction:fast_ica；fast_ica | 通过非高斯性等准则寻找统计独立成分；独立性假设强于不相关性。 本次作为C01训练前的二维转换，而非独立降维任务。 | [[使用/不适用、参数、输入及实际PASS]]
reduction:gaussian_random_projection；gaussian_random_projection | 高斯随机投影：不按目标或方差优化方向，用随机矩阵压缩维度；区别于数据学习的PCA。 本次作为C01训练前的二维转换，而非独立降维任务。 | [[使用/不适用、参数、输入及实际PASS]]
reduction:kernel_pca；kernel_pca | 在核空间提取主成分；非线性映射依赖核设置，默认核不一定非线性。 本次作为C01训练前的二维转换，而非独立降维任务。 | [[使用/不适用、参数、输入及实际PASS]]

注。基于已核实的方法清单供选择；当前版本、适用条件和来源需在本轮复核。完整官方链接随algorithm-reference.csv提供。
