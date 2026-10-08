"""Common and advanced parameter descriptions for registered estimators."""
from __future__ import annotations

from typing import Any

from .catalog import build_estimator, estimator_parameter_names
from .sequence_models import EXTENDED_MODEL_IDS, validate_extended_parameters

_COMMON_LABELS = {
    "random_state": "随机种子", "n_jobs": "并行线程数", "max_iter": "最大迭代次数",
    "n_estimators": "基学习器数量", "n_neighbors": "邻居数", "n_clusters": "簇数量",
    "n_components": "输出维度", "C": "正则化强度倒数", "alpha": "正则化强度",
    "kernel": "核函数", "gamma": "核系数", "tol": "收敛容差",
    "learning_rate": "学习率", "fit_intercept": "拟合截距", "contamination": "预期异常比例",
    "novelty": "支持新样本检测", "probability": "拟合概率估计", "binarize": "二值化阈值",
    "init": "初始化方法", "max_depth": "最大树深度", "min_samples_leaf": "叶节点最小样本数",
    "solver": "求解器", "strategy": "策略", "whiten": "白化变换"
}

_EXTENDED_LABELS = {
    "n_components": "隐状态数", "n_mix": "混合分量数", "n_iter": "最大拟合迭代数",
    "tol": "收敛容差", "covariance_type": "协方差类型", "hidden_size": "隐藏层宽度",
    "num_layers": "网络层数", "batch_size": "批次大小", "learning_rate": "学习率",
    "weight_decay": "权重衰减", "max_epochs": "最大训练轮数", "patience": "早停耐心值",
    "random_state": "随机种子",
}


def _display(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)[:160]


def parameter_schema(model_id: str) -> list[dict[str, Any]]:
    """Describe supported parameters from each registered model family's schema."""
    if model_id in EXTENDED_MODEL_IDS:
        defaults = validate_extended_parameters(model_id)
        common = set(_EXTENDED_LABELS) - {"random_state", "weight_decay"}
        return [
            {
                "name": name,
                "label_zh": _EXTENDED_LABELS.get(name, name),
                "default": _display(value),
                "value_type": type(value).__name__ if value is not None else "optional",
                "group": "常用参数" if name in common else "高级结构化参数",
            }
            for name, value in defaults.items()
        ]
    estimator = build_estimator(model_id)
    defaults = estimator.get_params(deep=True)
    result = []
    for name in estimator_parameter_names(model_id, estimator):
        if "__" in name and name.rsplit("__", 1)[0] not in {"estimator", "base_estimator"}:
            continue
        value = defaults.get(name)
        result.append({
            "name": name,
            "label_zh": _COMMON_LABELS.get(name, name),
            "default": _display(value),
            "value_type": type(value).__name__ if value is not None else "optional",
            "group": "常用参数" if name in _COMMON_LABELS else "高级结构化参数",
        })
    return result


def parameter_defaults(model_id: str) -> dict[str, Any]:
    """Return defaults using the estimator family that owns this model."""
    if model_id in EXTENDED_MODEL_IDS:
        return validate_extended_parameters(model_id)
    return build_estimator(model_id).get_params(deep=True)
