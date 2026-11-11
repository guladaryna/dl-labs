"""Графіки ЛР1. Будуються зі збережених results/correct.json і results/bug.json.

Нічого не рахують заново: беруть ті самі числа, що й таблиці в README.
Якщо один із двох json ще не створено, на графіку буде лише наявний режим.
"""

import json

import matplotlib

matplotlib.use("Agg")  # малюємо одразу у файл, вікно не потрібне
import matplotlib.pyplot as plt

# режим -> (підпис, колір, маркер, розмір); маркери різні, щоб режими розрізнялись і без кольору.
# Коло більше за квадрат: там, де значення режимів збігаються (втрата), видно обидва.
MODES = {
    "correct": ("правильна версія", "#2a78d6", "o", 13),
    "bug": ("--bug: без ділення на N", "#eb6834", "s", 7),
}
# скільки елементів у кожному градієнті, у порядку PARAM_NAMES: 4*8, 8, 8*3, 3
BLOCKS = [("W1", 32), ("b1", 8), ("W2", 24), ("b2", 3)]
FLOOR = 1e-18  # різницю рівно 0 на лог-шкалі не показати, тому малюємо її на цьому рівні
GRAY = "#898781"


def load_results(results_dir):
    """Читає наявні json. Повертає словник режим -> результати запуску."""
    results = {}
    for mode in MODES:
        path = results_dir / f"{mode}.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                results[mode] = json.load(f)
    return results


def draw_differences(ax, results, rows_key, value_key, tolerance, title):
    """Одна панель: абсолютні різниці кожного режиму і лінія критерію."""
    for mode, res in results.items():
        label, color, marker, size = MODES[mode]
        names = [row["name"].replace("Градієнт ", "∂L/∂") for row in res[rows_key]]
        values = [max(row[value_key], FLOOR) for row in res[rows_key]]
        ax.plot(names, values, marker, color=color, markersize=size, label=label)
    ax.axhline(tolerance, color=GRAY, linestyle="--", linewidth=1)
    ax.text(0.99, tolerance * 3, f"критерій {tolerance:g}", color=GRAY, ha="right",
            transform=ax.get_yaxis_transform())
    ax.set_yscale("log")
    ax.set_ylim(FLOOR, 1e3)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.margins(x=0.15)


def plot_checks(results, path):
    """Графік 1: максимальні різниці з PyTorch (пункт 2) і з чисельною похідною (пункт 3)."""
    first = next(iter(results.values()))
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    draw_differences(left, results, "torch_comparison", "max_abs_diff",
                     first["torch_tolerance"], "Звірка з PyTorch: max |NumPy − PyTorch|")
    draw_differences(right, results, "numeric_check", "abs_diff",
                     first["numeric_tolerance"], "Чисельна перевірка: |g_num − g_manual|")
    left.set_ylabel("абсолютна різниця (лог-шкала)")
    left.legend(loc="center left")
    fig.suptitle("Точка вище лінії критерію — перевірку не пройдено")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ratio(results, path):
    """Графік 2: відношення градієнт NumPy / градієнт PyTorch для кожного з 67 елементів."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for mode, res in results.items():
        label, color, marker, _ = MODES[mode]
        ratios = res["gradient_ratio_numpy_to_pytorch"]["ratio_per_element"]
        # None стоїть там, де еталонний градієнт нульовий і відношення не визначене
        points = [(i, r) for i, r in enumerate(ratios) if r is not None]
        ax.plot([i for i, _ in points], [r for _, r in points], marker, color=color,
                markersize=5, label=f"{label}: {ratios[0]:.0f}")

    # межі між чотирма градієнтами і їхні підписи
    start = 0
    for name, size in BLOCKS:
        ax.text(start + size / 2 - 0.5, 400, f"∂L/∂{name}", ha="center", color=GRAY)
        start += size
        if start < sum(size for _, size in BLOCKS):
            ax.axvline(start - 0.5, color=GRAY, linewidth=0.8, alpha=0.5)

    ax.set_yscale("log")
    ax.set_ylim(0.3, 1000)
    ax.set_yticks([1, 10, 105])
    ax.set_yticklabels(["1", "10", "105"])
    ax.set(xlabel="номер елемента градієнта (усі 67)",
           ylabel="градієнт NumPy / градієнт PyTorch",
           title="Помилка множить кожен елемент градієнта на N = 105")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="center right")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_plots(results_dir):
    """Будує обидва графіки в results_dir і повертає шляхи до них."""
    results = load_results(results_dir)
    paths = [results_dir / "checks.png", results_dir / "gradient_ratio.png"]
    plot_checks(results, paths[0])
    plot_ratio(results, paths[1])
    return paths
