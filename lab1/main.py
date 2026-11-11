"""ЛР1: запуск усіх перевірок зворотного поширення похибки.

    uv run main.py          правильна реалізація (за замовчуванням)
    uv run main.py --bug    дослід із навмисною помилкою (пункт 4 умови)

Результати друкуються і зберігаються в results/correct.json або results/bug.json.
"""

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import sklearn
import torch

from data import check_data, prepare_data
from net import PARAM_NAMES, backward, compute_loss, cross_entropy, forward, init_params
from torch_ref import torch_loss_and_grads

EXPECTED_SHAPES = {"W1": (4, 8), "b1": (8,), "W2": (8, 3), "b2": (3,)}
TORCH_TOL = 1e-12  # критерій пункту 2
EPS = 1e-6  # крок чисельного диференціювання, пункт 3
NUM_TOL = 1e-7  # критерій пункту 3
# параметри для чисельної перевірки: W1[0,0], b1[0], W2[0,0], b2[0]
CHECKED = [("W1", (0, 0)), ("b1", (0,)), ("W2", (0, 0)), ("b2", (0,))]
RESULTS_DIR = Path(__file__).parent / "results"


def yes_no(flag):
    return "так" if flag else "НІ"


def check_params(params):
    """Перевіряє форми, тип і зсуви початкових параметрів проти таблиці умови."""
    print("== Початкові параметри ==")
    for name in PARAM_NAMES:
        p = params[name]
        assert p.shape == EXPECTED_SHAPES[name] and p.dtype == np.float64
        print(f"{name}: форма {p.shape}, dtype {p.dtype}, std = {p.std():.4f}")
    assert not params["b1"].any() and not params["b2"].any()
    print(f"очікувані std: W1 {np.sqrt(2 / 4):.4f}, W2 {np.sqrt(2 / 11):.4f} (вибірки малі: 32 і 24 числа)")
    print()


def run_numpy(params, X, y, divide_by_n):
    """Пункт 1: прямий прохід, втрата і ручні градієнти на повній train-вибірці."""
    Z2, cache = forward(params, X)
    loss, P = cross_entropy(Z2, y)
    grads = backward(params, cache, P, y, divide_by_n=divide_by_n)

    print("== 1. NumPy: прямий і зворотний прохід ==")
    print(f"X {X.shape}, Z1 {cache['Z1'].shape}, H {cache['H'].shape}, Z2 {Z2.shape}, P {P.shape}")
    assert np.allclose(P.sum(axis=1), 1.0)  # рядки softmax — розподіли
    for name in PARAM_NAMES:
        assert grads[name].shape == params[name].shape
        print(f"градієнт {name}: форма {grads[name].shape}")
    print(f"втрата NumPy: {float(loss)!r} (для порівняння ln 3 = {np.log(3):.6f})")
    print()
    return float(loss), grads


def compare_with_torch(params, X, y, loss_np, grads_np):
    """Пункт 2: втрата і всі елементи чотирьох градієнтів проти PyTorch.

    Критерій для кожної пари чисел: |a - b| <= 1e-12, і всі числа скінченні.
    Повертає втрату PyTorch, градієнти PyTorch і список рядків таблиці.
    """
    loss_pt, grads_pt = torch_loss_and_grads(params, X, y)

    print("== 2. Звірка з PyTorch ==")
    print(f"втрата NumPy:   {loss_np!r}")
    print(f"втрата PyTorch: {loss_pt!r}")

    pairs = [("Втрата", np.array(loss_np), np.array(loss_pt))]
    pairs += [(f"Градієнт {name}", grads_np[name], grads_pt[name]) for name in PARAM_NAMES]

    rows = []
    for label, a, b in pairs:
        assert a.shape == b.shape
        finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
        max_diff = float(np.abs(a - b).max())
        rows.append({"name": label, "max_abs_diff": max_diff, "finite": finite,
                     "passed": finite and max_diff <= TORCH_TOL})

    print()
    print("| Величина | Максимальна абсолютна різниця NumPy / PyTorch | Перевірку пройдено |")
    print("|---|---|---|")
    for r in rows:
        print(f"| {r['name']} | {r['max_abs_diff']:.3e} | {yes_no(r['passed'])} |")
    print()
    return loss_pt, grads_pt, rows


def gradient_ratio(grads_np, grads_pt):
    """У скільки разів ручні градієнти відрізняються від еталонних.

    Відношення рахується поелементно лише там, де еталонний градієнт ненульовий.
    Для правильної реалізації воно дорівнює 1; у досліді показує масштаб помилки.
    """
    a = np.concatenate([grads_np[name].ravel() for name in PARAM_NAMES])  # (67,)
    b = np.concatenate([grads_pt[name].ravel() for name in PARAM_NAMES])  # (67,)
    nonzero = b != 0
    ratio = a[nonzero] / b[nonzero]
    result = {
        "n_elements": int(a.size),
        "n_nonzero_reference": int(nonzero.sum()),
        "n_zero_in_both": int(((a == 0) & (b == 0)).sum()),
        "ratio_min": float(ratio.min()),
        "ratio_max": float(ratio.max()),
    }
    print("== Відношення градієнт NumPy / градієнт PyTorch ==")
    print(f"елементів градієнта: {result['n_elements']}, з них ненульових в еталоні: "
          f"{result['n_nonzero_reference']}, нульових в обох: {result['n_zero_in_both']}")
    print(f"відношення на ненульових: min = {result['ratio_min']:.12f}, max = {result['ratio_max']:.12f}")
    print()
    return result


def numerical_derivative(params, X, y, name, index):
    """Центральна різниця (L+ - L-) / (2 eps) за одним параметром params[name][index].

    Кроки рівно як у пункті 3 умови; після обчислення значення відновлюється.
    """
    original = params[name][index]
    params[name][index] = original + EPS
    loss_plus = compute_loss(params, X, y)
    params[name][index] = original - EPS
    loss_minus = compute_loss(params, X, y)
    params[name][index] = original
    return float((loss_plus - loss_minus) / (2 * EPS))


def numerical_check(params, X, y, grads_np):
    """Пункт 3: чотири елементи градієнта backward() проти чисельної похідної."""
    before = {name: params[name].copy() for name in PARAM_NAMES}

    rows = []
    for name, index in CHECKED:
        g_manual = float(grads_np[name][index])
        g_num = numerical_derivative(params, X, y, name, index)
        diff = abs(g_num - g_manual)
        finite = bool(np.isfinite(g_manual) and np.isfinite(g_num))
        label = f"{name}[{', '.join(str(i) for i in index)}]"
        rows.append({"name": label, "g_manual": g_manual, "g_num": g_num,
                     "abs_diff": diff, "passed": finite and diff <= NUM_TOL})

    # чисельна перевірка не має лишити слідів у параметрах
    for name in PARAM_NAMES:
        assert np.array_equal(params[name], before[name])

    print(f"== 3. Чисельна перевірка (eps = {EPS:g}) ==")
    print("| Параметр | Градієнт backward() | Чисельна похідна | Абсолютна різниця | Перевірку пройдено |")
    print("|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['name']} | {r['g_manual']:.12f} | {r['g_num']:.12f} "
              f"| {r['abs_diff']:.3e} | {yes_no(r['passed'])} |")
    print()
    return rows


def save_results(path, results):
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Результати збережено: {path.relative_to(Path(__file__).parent).as_posix()}")


def main():
    # щоб український текст і при перенаправленні у файл виводився в UTF-8
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="ЛР1: перевірка зворотного поширення похибки")
    parser.add_argument("--bug", action="store_true",
                        help="навмисна помилка: без ділення на N у градієнті за логітами (пункт 4)")
    args = parser.parse_args()

    mode = "bug" if args.bug else "correct"
    print(f"Режим: {'НАВМИСНА ПОМИЛКА (без ділення на N)' if args.bug else 'правильна реалізація'}")
    print()

    data = prepare_data()
    check_data(data)
    X, y = data["X_train"], data["y_train"]  # усі перевірки — на повній train-вибірці

    params = init_params()
    check_params(params)

    # ваги між перевірками не оновлюються: усе рахується на початкових параметрах
    loss_np, grads_np = run_numpy(params, X, y, divide_by_n=not args.bug)
    loss_pt, grads_pt, torch_rows = compare_with_torch(params, X, y, loss_np, grads_np)
    ratio = gradient_ratio(grads_np, grads_pt)
    numeric_rows = numerical_check(params, X, y, grads_np)

    all_passed = all(r["passed"] for r in torch_rows + numeric_rows)
    save_results(RESULTS_DIR / f"{mode}.json", {
        "mode": mode,
        "n_train": int(X.shape[0]),
        "loss_numpy": loss_np,
        "loss_pytorch": loss_pt,
        "torch_tolerance": TORCH_TOL,
        "torch_comparison": torch_rows,
        "gradient_ratio_numpy_to_pytorch": ratio,
        "eps": EPS,
        "numeric_tolerance": NUM_TOL,
        "numeric_check": numeric_rows,
        "all_passed": all_passed,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
            "torch": torch.__version__,
        },
    })

    if args.bug:
        # у досліді правильний результат — це виявлена помилка
        print("Підсумок досліду:", "помилку ВИЯВЛЕНО" if not all_passed else "помилку НЕ виявлено")
    else:
        print("Підсумок:", "усі перевірки пройдено" if all_passed else "Є НЕПРОЙДЕНІ ПЕРЕВІРКИ")
        if not all_passed:
            sys.exit(1)


if __name__ == "__main__":
    main()
