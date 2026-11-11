"""Дані ЛР1: Iris, стратифікований поділ 70/30 і стандартизація.

Усе з розділу «Вихідні дані та фіксовані налаштування» умови.
"""

import numpy as np
from sklearn.datasets import load_iris

CLASSES = (0, 1, 2)
TRAIN_PER_CLASS = 35
TEST_PER_CLASS = 15


def load_data():
    """Повертає X форми (150, 4) у float64 і y форми (150,) з мітками 0, 1, 2."""
    iris = load_iris()
    X = iris.data.astype(np.float64)
    y = iris.target.astype(np.int64)
    return X, y


def split_indices(y):
    """Стратифікований поділ: індекси train (105,) і test (45,).

    За умовою: один default_rng(0), класи в порядку 0, 1, 2, індекси кожного
    класу перемішуються, перші 35 ідуть у train, решта 15 — у test.
    """
    rng = np.random.default_rng(0)
    train_parts = []
    test_parts = []
    for c in CLASSES:
        idx = np.flatnonzero(y == c)  # індекси об'єктів класу c за зростанням
        rng.shuffle(idx)
        train_parts.append(idx[:TRAIN_PER_CLASS])
        test_parts.append(idx[TRAIN_PER_CLASS:])
    return np.concatenate(train_parts), np.concatenate(test_parts)


def standardize(X_train, X_test):
    """Стандартизує ознаки статистиками навчальної вибірки (ddof=0).

    Середнє і стандартне відхилення рахуються лише на train; до test
    застосовуються ті самі числа. Повертає також mean і std форми (4,).
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0, ddof=0)
    return (X_train - mean) / std, (X_test - mean) / std, mean, std


def prepare_data():
    """Повний шлях: завантаження -> поділ -> стандартизація.

    Повертає словник із X_train (105, 4), y_train (105,), X_test (45, 4),
    y_test (45,), індексами поділу та статистиками стандартизації.
    """
    X, y = load_data()
    train_idx, test_idx = split_indices(y)
    X_train, X_test, mean, std = standardize(X[train_idx], X[test_idx])
    return {
        "X_train": X_train,
        "y_train": y[train_idx],
        "X_test": X_test,
        "y_test": y[test_idx],
        "train_idx": train_idx,
        "test_idx": test_idx,
        "mean": mean,
        "std": std,
    }


def check_data(data):
    """Перевіряє поділ і стандартизацію проти умови та друкує результат.

    Якщо щось не збігається, assert зупиняє програму.
    """
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    train_idx, test_idx = data["train_idx"], data["test_idx"]

    assert X_train.shape == (105, 4) and X_test.shape == (45, 4)
    assert X_train.dtype == np.float64 and X_test.dtype == np.float64

    train_counts = np.bincount(y_train, minlength=3)
    test_counts = np.bincount(y_test, minlength=3)
    assert train_counts.tolist() == [TRAIN_PER_CLASS] * 3
    assert test_counts.tolist() == [TEST_PER_CLASS] * 3

    # train і test не перетинаються і разом дають усі 150 об'єктів
    assert len(np.intersect1d(train_idx, test_idx)) == 0
    assert np.array_equal(np.sort(np.concatenate([train_idx, test_idx])), np.arange(150))

    # після стандартизації train має середнє 0 і відхилення 1 за кожною ознакою
    max_abs_mean = np.abs(X_train.mean(axis=0)).max()
    max_std_error = np.abs(X_train.std(axis=0, ddof=0) - 1.0).max()
    assert max_abs_mean < 1e-12 and max_std_error < 1e-12

    print("== Дані ==")
    print(f"train: {X_train.shape}, по класах {train_counts.tolist()}")
    print(f"test:  {X_test.shape}, по класах {test_counts.tolist()}")
    print(f"dtype: {X_train.dtype}; train і test не перетинаються")
    print(f"перші 5 індексів train: {train_idx[:5].tolist()}")
    print(f"середнє train (до стандартизації): {data['mean'].tolist()}")
    print(f"std train, ddof=0 (до стандартизації): {data['std'].tolist()}")
    print(f"після стандартизації: max|mean| = {max_abs_mean:.2e}, max|std - 1| = {max_std_error:.2e}")
    print()
