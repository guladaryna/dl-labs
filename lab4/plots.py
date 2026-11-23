"""Графіки: діапазони каналів (пункт 4) і розподіл затримки (пункт 5)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def channel_ranges(convs, median_r, path):
    """r = max_c α_c / median_c α_c для кожного Conv у порядку графа."""
    names = [c["name"].replace("/Conv", "").lstrip("/") for c in convs]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(range(len(convs)), [c["r"] for c in convs], color="#4C72B0")
    ax.axhline(median_r, color="#C44E52", linestyle="--", label=f"медіана r = {median_r:.2f}")
    ax.set_xticks(range(len(convs)), names, rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("r = max α_c / median α_c")
    ax.set_title("Розкид діапазонів вихідних каналів ваг у згортках")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def latency(rounds, path):
    """Усі 300 вимірів кожного варіанта по порядку, раунди розділено вертикальними лініями."""
    colors = {"fp32": "#4C72B0", "int8": "#DD8452"}
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, by_round in rounds.items():
        values = [v for r in by_round for v in r]
        ax.plot(range(1, len(values) + 1), values, ".", markersize=3, color=colors.get(name),
                label=name.upper())
    per_round = len(next(iter(rounds.values()))[0])
    for k in range(1, len(next(iter(rounds.values())))):
        ax.axvline(k * per_round + 0.5, color="gray", linewidth=0.8)
    ax.set_xlabel("номер виміру (3 раунди × 100)")
    ax.set_ylabel("затримка sess.run, мс")
    ax.set_title("Затримка, B = 1, 1 потік")
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
