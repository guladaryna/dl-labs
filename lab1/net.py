"""Мережа ЛР1 на чистому NumPy: X -> Linear(4, 8) -> ReLU -> Linear(8, 3).

Пункт 1 умови: прямий прохід, стабільна крос-ентропія, збереження проміжних
значень і ручний зворотний прохід. Автоматичного диференціювання тут немає.

Ваги зберігаються як (in, out), як у таблиці умови: W1 (4, 8), W2 (8, 3).
Тому шар рахується як X @ W + b.
"""

import numpy as np

PARAM_NAMES = ("W1", "b1", "W2", "b2")


def init_params():
    """Початкові параметри з таблиці умови, усі у float64.

    Новий default_rng(0) (не той, що в поділі даних). Порядок викликів
    фіксований умовою: спершу W1, потім W2. Зсуви — нулі.
    """
    rng = np.random.default_rng(0)
    W1 = rng.normal(0.0, np.sqrt(2 / 4), size=(4, 8))  # He
    W2 = rng.normal(0.0, np.sqrt(2 / (8 + 3)), size=(8, 3))  # Xavier
    return {"W1": W1, "b1": np.zeros(8), "W2": W2, "b2": np.zeros(3)}


def forward(params, X):
    """Прямий прохід. X (N, 4) -> логіти Z2 (N, 3).

    У cache лежить те, що потім потрібне зворотному проходу:
    X (N, 4) — для градієнта W1, Z1 (N, 8) — для маски ReLU,
    H (N, 8) — для градієнта W2.
    """
    Z1 = X @ params["W1"] + params["b1"]  # (N, 8)
    H = np.maximum(Z1, 0.0)  # (N, 8), ReLU
    Z2 = H @ params["W2"] + params["b2"]  # (N, 3)
    cache = {"X": X, "Z1": Z1, "H": H}
    return Z2, cache


def log_softmax(Z):
    """Стабільний log-softmax по рядках. Z (N, 3) -> (N, 3).

    Від кожного рядка віднімається його максимум: результат математично той
    самий, але найбільший аргумент exp дорівнює 0, тож переповнення немає,
    і сума під логарифмом не менша за 1.
    """
    shifted = Z - Z.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def cross_entropy(Z2, y):
    """Крос-ентропія, усереднена за об'єктами.

    Z2 (N, 3) — логіти, y (N,) — номери класів.
    Повертає втрату (скаляр) і ймовірності P (N, 3), які потрібні backward().
    """
    N = Z2.shape[0]
    log_p = log_softmax(Z2)
    loss = -log_p[np.arange(N), y].mean()
    return loss, np.exp(log_p)


def compute_loss(params, X, y):
    """Лише втрата: прямий прохід + крос-ентропія. Потрібна чисельній перевірці."""
    Z2, _ = forward(params, X)
    loss, _ = cross_entropy(Z2, y)
    return loss


def backward(params, cache, P, y, divide_by_n=True):
    """Ручний зворотний прохід. Повертає градієнти тих самих форм, що й параметри.

    P (N, 3) — ймовірності з cross_entropy, y (N,) — номери класів.
    divide_by_n=False — навмисна помилка з пункту 4 умови: у градієнті за
    логітами немає ділення на N. За замовчуванням реалізація правильна.
    """
    N = y.shape[0]

    # градієнт за логітами: (P - Y) / N, де Y — one-hot міток
    dZ2 = P.copy()  # (N, 3)
    dZ2[np.arange(N), y] -= 1.0
    if divide_by_n:
        dZ2 /= N  # втрата — середнє за N об'єктами

    # другий шар: Z2 = H @ W2 + b2
    dW2 = cache["H"].T @ dZ2  # (8, 3)
    db2 = dZ2.sum(axis=0)  # (3,)

    # назад через ReLU: похідна 1 там, де Z1 > 0, інакше 0
    dH = dZ2 @ params["W2"].T  # (N, 8)
    dZ1 = dH * (cache["Z1"] > 0)  # (N, 8)

    # перший шар: Z1 = X @ W1 + b1
    dW1 = cache["X"].T @ dZ1  # (4, 8)
    db1 = dZ1.sum(axis=0)  # (8,)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
