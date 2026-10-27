"""Shared evaluation contracts. Undefined scores remain null with explanations."""
from __future__ import annotations

import warnings
import numpy as np
from sklearn import metrics as skm
from sklearn.utils.multiclass import unique_labels

# Insertion order also defines the default search objective.
METRICS = {
    "classification": {
        "balanced_accuracy": ("平衡准确率", "max"),
        "accuracy": ("准确率", "max"),
        "precision_macro": ("宏平均精确率", "max"),
        "recall_macro": ("宏平均召回率", "max"),
        "f1_macro": ("宏平均 F1", "max"),
        "precision_weighted": ("加权精确率", "max"),
        "recall_weighted": ("加权召回率", "max"),
        "f1_weighted": ("加权 F1", "max"),
        "mcc": ("Matthews 相关系数", "max"),
        "cohen_kappa": ("Cohen's Kappa", "max"),
        "roc_auc": ("ROC-AUC（二分类 / 多分类 OvR 宏平均）", "max"),
        "average_precision": ("AP（二分类 / 多分类宏平均）", "max"),
        "pr_auc": ("PR-AUC（梯形积分）", "max"),
        "log_loss": ("Log Loss（需要概率）", "min"),
        "brier_score": ("Brier（类别平方误差之和）", "min"),
    },
    "regression": {
        "rmse": ("RMSE", "min"), "mae": ("MAE", "min"),
        "r2": ("R²", "max"), "mse": ("MSE", "min"),
        "median_absolute_error": ("中位绝对误差", "min"),
        "mape": ("MAPE（比例）", "min"), "msle": ("MSLE", "min"),
        "rmsle": ("RMSLE", "min"), "pinball_loss": ("分位数损失", "min"),
        "smape": ("sMAPE（百分比）", "min"), "mase": ("MASE（序列训练基线）", "min"),
    },
    "clustering": {
        "silhouette": ("轮廓系数", "max"),
        "calinski_harabasz": ("Calinski-Harabasz", "max"),
        "davies_bouldin": ("Davies-Bouldin", "min"),
        "ari": ("调整兰德指数（需要参考标签）", "max"),
        "nmi": ("归一化互信息（需要参考标签）", "max"),
        "ami": ("调整互信息（需要参考标签）", "max"),
    },
    "dimensionality reduction": {"trustworthiness": ("Trustworthiness", "max")},
    "anomaly detection": {
        "balanced_accuracy": ("平衡准确率（需要标签）", "max"),
        "precision": ("异常精确率", "max"), "recall": ("异常召回率", "max"),
        "f1": ("异常 F1", "max"), "roc_auc": ("异常 ROC-AUC", "max"),
        "average_precision": ("异常 AP", "max"),
    },
    "sequence_modeling": {"log_likelihood_per_observation": ("每个观测的对数似然", "max")},
}


def metric_choices(task):
    return [(key, value[0]) for key, value in METRICS[task].items()]


def metric_direction(task, metric):
    try:
        return METRICS[task][metric][1]
    except KeyError as exc:
        raise ValueError(f"Metric {metric!r} is not supported for {task!r}") from exc


def prediction_quantile(estimator):
    value = getattr(estimator, "quantile", None)
    if value is None and getattr(estimator, "loss", None) == "quantile":
        value = getattr(estimator, "alpha", None)
    return .5 if value is None else value


def _finite(value):
    number = float(value)
    return number if np.isfinite(number) else None


def classification_metrics(actual, predicted, score_cache=None):
    actual, predicted = np.asarray(actual).reshape(-1), np.asarray(predicted).reshape(-1)
    result = {"sample_count": int(len(actual))}
    notes = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for name, function in (("accuracy", skm.accuracy_score),
                               ("balanced_accuracy", skm.balanced_accuracy_score),
                               ("mcc", skm.matthews_corrcoef), ("cohen_kappa", skm.cohen_kappa_score)):
            result[name] = _finite(function(actual, predicted))
            if result[name] is None:
                notes[name] = "该分区的类别或预测分布使指标未定义。"
    for average in ("macro", "weighted"):
        for name, function in (("precision", skm.precision_score), ("recall", skm.recall_score), ("f1", skm.f1_score)):
            result[f"{name}_{average}"] = float(function(actual, predicted, average=average, zero_division=0))
    classes = unique_labels(actual, predicted, np.asarray(score_cache["class_order"])) if score_cache and score_cache.get("available") else unique_labels(actual, predicted)
    precision, recall, f1, support = skm.precision_recall_fscore_support(actual, predicted, labels=classes, zero_division=0)
    matrix = skm.confusion_matrix(actual, predicted, labels=classes)
    per_class = []
    for i, label in enumerate(classes):
        fp = int(matrix[:, i].sum() - matrix[i, i])
        tn = int(matrix.sum() - matrix[i, :].sum() - matrix[:, i].sum() + matrix[i, i])
        per_class.append({"class_label": label.item() if isinstance(label, np.generic) else label,
                          "precision": float(precision[i]), "recall": float(recall[i]),
                          "f1": float(f1[i]), "support": int(support[i]),
                          "specificity": tn / (tn + fp) if tn + fp else None})
    result["per_class"] = per_class
    score_metrics = ("roc_auc", "average_precision", "pr_auc", "log_loss", "brier_score")
    result.update({key: None for key in score_metrics})
    cache = score_cache or {}
    if not cache.get("available"):
        reason = cache.get("reason", "模型未提供可验证的类别分数。")
        notes.update({key: reason for key in score_metrics})
    else:
        order = cache["class_order"]
        # Explicit IDs preserve the estimator's column order for arbitrary labels.
        mapping = {label: i for i, label in enumerate(order)}
        if any(label not in mapping for label in actual):
            notes.update({key: "评估标签不在训练类别映射中。" for key in score_metrics})
        else:
            ids = np.asarray([mapping[label] for label in actual], dtype=int)
            onehot = np.eye(len(order), dtype=int)[ids]
            values = np.asarray(cache["values"], dtype=float)
            columns = [1] if len(order) == 2 else list(range(len(order)))
            areas, aps, pr_areas = [], [], []
            for column in columns:
                binary = onehot[:, column]
                if len(np.unique(binary)) < 2:
                    break
                scores = values[:, 0] if values.shape[1] == 1 else values[:, column]
                areas.append(float(skm.roc_auc_score(binary, scores)))
                aps.append(float(skm.average_precision_score(binary, scores)))
                p, r, _ = skm.precision_recall_curve(binary, scores)
                pr_areas.append(float(skm.auc(r, p)))
            if len(areas) == len(columns):
                result.update(roc_auc=float(np.mean(areas)), average_precision=float(np.mean(aps)), pr_auc=float(np.mean(pr_areas)))
            else:
                notes.update({key: "该分区缺少某个训练类别的正例或负例。" for key in score_metrics[:3]})
            result["score_class_order"] = order
            result["score_averaging"] = "binary_positive_class=class_order[1]" if len(order) == 2 else "one_vs_rest_macro"
            if cache.get("score_type") == "probability" and values.shape == onehot.shape and np.allclose(values.sum(axis=1), 1, atol=1e-6):
                result["log_loss"] = float(skm.log_loss(ids, values, labels=np.arange(len(order))))
                # Unscaled multiclass Brier; for binary this equals twice the
                # single-positive-class squared error. The definition is explicit.
                result["brier_score"] = float(np.mean(np.sum((values - onehot) ** 2, axis=1)))
            else:
                notes.update({key: "需要行和为 1 的真实类别概率；决策分数不能作为概率。" for key in score_metrics[3:]})
    if notes:
        result["metric_notes"] = notes
    return result


def probability_cache(probabilities, classes):
    values = np.asarray(probabilities, dtype=float)
    return {"available": True, "class_order": list(classes), "score_type": "probability", "values": values.tolist()}


def regression_metrics(actual, predicted, *, quantile=.5, training_series=None, seasonal_period=1, horizons=None):
    quantile = .5 if quantile is None else quantile
    if isinstance(seasonal_period, bool) or not isinstance(seasonal_period, int) or seasonal_period < 1:
        raise ValueError("seasonal_period must be a positive integer")
    actual, predicted = np.asarray(actual, dtype=float).reshape(-1), np.asarray(predicted, dtype=float).reshape(-1)
    result = {"sample_count": len(actual), "r2": float(skm.r2_score(actual, predicted)) if len(actual) >= 2 else None,
              "mae": float(skm.mean_absolute_error(actual, predicted)),
              "mse": float(skm.mean_squared_error(actual, predicted)),
              "median_absolute_error": float(skm.median_absolute_error(actual, predicted)),
              "pinball_loss": float(skm.mean_pinball_loss(actual, predicted, alpha=quantile)), "pinball_quantile": float(quantile)}
    result["rmse"] = float(np.sqrt(result["mse"]))
    notes = {}
    if len(actual) < 2:
        notes["r2"] = "R² 至少需要两个评估样本。"
    result["mape"] = None
    if np.any(np.abs(actual) <= np.finfo(float).eps):
        notes["mape"] = "真实值包含零或机器精度附近的值，MAPE 不适用。"
    else:
        result["mape"] = float(skm.mean_absolute_percentage_error(actual, predicted))
    result["msle"] = result["rmsle"] = None
    if np.any(actual < 0) or np.any(predicted < 0):
        notes.update(msle="MSLE 需要非负真实值和预测值。", rmsle="RMSLE 需要非负真实值和预测值。")
    else:
        result["msle"] = float(skm.mean_squared_log_error(actual, predicted))
        result["rmsle"] = float(np.sqrt(result["msle"]))
    denominator = np.abs(actual) + np.abs(predicted)
    ratio = np.divide(2 * np.abs(actual - predicted), denominator, out=np.zeros_like(actual), where=denominator != 0)
    result["smape"] = float(100 * np.mean(ratio))
    result["mase"] = None
    if training_series is not None:
        result["mase_seasonal_period"] = seasonal_period
        series = [training_series] if isinstance(training_series, np.ndarray) or not len(training_series) or np.isscalar(training_series[0]) else training_series
        differences = [np.abs(np.asarray(part, dtype=float)[seasonal_period:] - np.asarray(part, dtype=float)[:-seasonal_period])
                       for part in series if len(part) > seasonal_period]
        scale = float(np.mean(np.concatenate(differences))) if differences else 0.
        if scale > 0 and np.isfinite(scale):
            result["mase"] = result["mae"] / scale
        else:
            notes["mase"] = "训练序列不足或训练朴素预测误差为零。"
    else:
        notes["mase"] = "需要有顺序的训练目标序列；不从评估数据估计分母。"
    if horizons is not None:
        leads = np.asarray(horizons)
        result["by_horizon"] = [{"horizon": int(h), "sample_count": int(np.sum(leads == h)),
                                  "mae": float(np.mean(np.abs(actual[leads == h] - predicted[leads == h]))),
                                  "rmse": float(np.sqrt(np.mean((actual[leads == h] - predicted[leads == h]) ** 2)))}
                                 for h in np.unique(leads)]
    result["metric_notes"] = notes
    return result


def cluster_label_metrics(actual, predicted):
    return {"ari": float(skm.adjusted_rand_score(actual, predicted)),
            "nmi": float(skm.normalized_mutual_info_score(actual, predicted)),
            "ami": float(skm.adjusted_mutual_info_score(actual, predicted)),
            "external_label_policy": "All rows included; noise (-1) is one predicted label."}


def anomaly_metrics(actual, predicted, normal_scores=None):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if not set(np.unique(actual)).issubset({-1, 1}):
        raise ValueError("Anomaly reference labels must use -1 anomaly and +1 normal")
    positive, decisions = actual == -1, predicted == -1
    result = {"balanced_accuracy": float(skm.balanced_accuracy_score(positive, decisions)),
              "precision": float(skm.precision_score(positive, decisions, zero_division=0)),
              "recall": float(skm.recall_score(positive, decisions, zero_division=0)),
              "f1": float(skm.f1_score(positive, decisions, zero_division=0)),
              "roc_auc": None, "average_precision": None}
    if normal_scores is not None and len(np.unique(positive)) == 2:
        scores = -np.asarray(normal_scores, dtype=float).reshape(-1)
        if len(scores) == len(actual) and np.isfinite(scores).all():
            result["roc_auc"] = float(skm.roc_auc_score(positive, scores))
            result["average_precision"] = float(skm.average_precision_score(positive, scores))
    if result["roc_auc"] is None:
        result["metric_notes"] = {"roc_auc": "需要两类参考标签及真实异常分数。", "average_precision": "需要两类参考标签及真实异常分数。"}
    return result
