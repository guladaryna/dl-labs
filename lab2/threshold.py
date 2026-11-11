"""ЛР2, пункт 2. Матриця плутанини, вартість помилок і швидкий пошук порога.

Правило рішення всюди одне: прогнозуємо шахрайство (клас 1), якщо s >= t.
"""

import numpy as np
import pandas as pd

COST_FN = 10  # ціна пропущеного шахрайства
COST_FP = 1   # ціна хибної тривоги


def cost(fp, fn):
    """C = 10*FN + FP. Працює і для чисел, і для масивів."""
    return COST_FN * fn + COST_FP * fp


def confusion_counts(y: np.ndarray, s: np.ndarray, t: float) -> dict[str, int]:
    """TP, FP, FN, TN і C для ОДНОГО порога t: один прохід по вибірці.

    y — мітки 0/1, форма (n,); s — оцінки, форма (n,).
    """
    pred = s >= t
    tp = int(np.count_nonzero(pred & (y == 1)))
    fp = int(np.count_nonzero(pred & (y == 0)))
    fn = int(np.count_nonzero(~pred & (y == 1)))
    tn = int(np.count_nonzero(~pred & (y == 0)))
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "C": cost(fp, fn)}


def candidate_table(y: np.ndarray, s: np.ndarray) -> pd.DataFrame:
    """Таблиця всіх кандидатів: поріг, TP, FP, FN, TN, C. Складність O(n log n).

    Кандидати — +inf і всі унікальні оцінки. Рядки йдуть за спаданням порога:
    перший рядок +inf («усі негативні»), останній — найменша оцінка («усі позитивні»).
    """
    # 1. Сортуємо оцінки за спаданням — єдина операція O(n log n).
    order = np.argsort(-s, kind="stable")
    s_sorted = s[order]
    y_sorted = y[order]

    # 2. Кумулятивні суми: скільки позитивних і негативних міток серед перших k оцінок.
    cum_pos = np.cumsum(y_sorted == 1)
    cum_neg = np.cumsum(y_sorted == 0)

    # 3. Рівні оцінки мають отримати однаковий прогноз, тому поріг t = s забирає в «позитивні»
    #    одразу всю групу рівних оцінок. Беремо суми в ОСТАННЬОМУ елементі кожної групи:
    #    це позиції, де наступна оцінка вже інша, і останній елемент масиву.
    is_group_end = np.append(s_sorted[1:] != s_sorted[:-1], True)
    ends = np.flatnonzero(is_group_end)

    # 4. Для порога t = s_sorted[ends[j]] позитивний прогноз отримують перші ends[j]+1 об'єктів.
    #    Спереду дописуємо +inf: позитивних прогнозів немає, TP = FP = 0.
    threshold = np.concatenate([[np.inf], s_sorted[ends]])
    tp = np.concatenate([[0], cum_pos[ends]])
    fp = np.concatenate([[0], cum_neg[ends]])

    n_pos = int(np.count_nonzero(y == 1))
    n_neg = int(np.count_nonzero(y == 0))
    fn = n_pos - tp
    tn = n_neg - fp
    return pd.DataFrame(
        {"threshold": threshold, "TP": tp, "FP": fp, "FN": fn, "TN": tn, "C": cost(fp, fn)}
    )


def best_threshold(table: pd.DataFrame) -> float:
    """t*: поріг із мінімальною вартістю; якщо таких кілька — найбільший із них."""
    min_cost = table["C"].min()
    tied = table[table["C"] == min_cost]
    return float(tied["threshold"].max())
