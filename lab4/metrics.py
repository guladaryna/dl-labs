"""Якість і парне порівняння з FP32 (пункт 3)."""

import math

import numpy as np
from sklearn.metrics import f1_score


def quality_row(logits, fp32_logits, targets):
    """Рядок таблиці пункту 3 для варіанта порівняно з FP32 на тих самих зображеннях.

    b — варіант правий, FP32 помиляється; c — навпаки.
    Δacc = (b − c)/n, SE ≈ √(b + c)/n; різниця відрізняється від шуму, якщо |b − c| > 2√(b + c).
    """
    preds = logits.argmax(1)
    fp32_preds = fp32_logits.argmax(1)
    n = len(targets)
    correct = preds == targets
    fp32_correct = fp32_preds == targets
    b = int(np.sum(correct & ~fp32_correct))
    c = int(np.sum(~correct & fp32_correct))
    threshold = 2 * math.sqrt(b + c)
    return {
        "macro_f1": float(f1_score(targets, preds, average="macro")),
        "accuracy": float(correct.mean()),
        "agreement": float((preds == fp32_preds).mean()),
        "b": b,
        "c": c,
        "delta_acc": (b - c) / n,
        "se_delta_acc": math.sqrt(b + c) / n,
        "noise_threshold_2sqrt_bc": threshold,
        "differs_from_noise": abs(b - c) > threshold,
    }


def single_model_se(accuracy, n):
    """Стандартна похибка частки правильних відповідей однієї моделі: √(p(1 − p)/n)."""
    return math.sqrt(accuracy * (1 - accuracy) / n)
