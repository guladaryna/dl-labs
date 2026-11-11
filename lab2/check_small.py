"""ЛР2, пункт 3. Малі набори A/B/C і незалежний прямий перебір порогів.

Цей файл навмисно НЕ імпортує threshold.py: перевірочний алгоритм не має
використовувати ні код, ні результати швидкого. Тут звичайні цикли Python:
для кожного кандидата — окремий прохід по всій вибірці, O(n^2). Так можна лише тут.
"""

import math

# Набори з умови. У наборі C (зноска в умові): 0,8; потім одинадцять оцінок 0,5 підряд
# (перша з міткою 1, решта десять — з міткою 0); потім 0,2. Разом 13 об'єктів.
SMALL_SETS = {
    "A": {
        "s": [0.9, 0.7, 0.7, 0.4, 0.2, 0.1],
        "y": [0, 1, 0, 1, 0, 0],
    },
    "B": {
        "s": [0.9, 0.6, 0.3],
        "y": [0, 0, 0],
    },
    "C": {
        "s": [0.8] + [0.5] * 11 + [0.2],
        "y": [1] + [1] + [0] * 10 + [0],
    },
}


def brute_force_table(y: list[int], s: list[float]) -> list[dict]:
    """Для кожного кандидата прямо застосувати правило s_i >= t і порахувати все з нуля.

    Повертає список рядків {threshold, TP, FP, FN, TN, C} за спаданням порога.
    """
    # Той самий список кандидатів, що в пункті 2: +inf і всі унікальні оцінки.
    candidates = [math.inf] + sorted(set(s), reverse=True)
    rows = []
    for t in candidates:
        tp = fp = fn = tn = 0
        for label, score in zip(y, s, strict=True):
            predicted_fraud = score >= t
            if predicted_fraud and label == 1:
                tp += 1
            elif predicted_fraud and label == 0:
                fp += 1
            elif not predicted_fraud and label == 1:
                fn += 1
            else:
                tn += 1
        rows.append({"threshold": t, "TP": tp, "FP": fp, "FN": fn, "TN": tn, "C": 10 * fn + fp})
    return rows


def brute_force_best(rows: list[dict]) -> float:
    """Мінімум C; за рівної вартості — найбільший поріг."""
    best_t = None
    best_c = None
    for row in rows:
        better = best_c is None or row["C"] < best_c
        same_cost_larger_t = best_c is not None and row["C"] == best_c and row["threshold"] > best_t
        if better or same_cost_larger_t:
            best_t = row["threshold"]
            best_c = row["C"]
    return best_t
