"""ЛР2, пункти 2–4. Аналіз порогів зі збережених прогнозів: без навчання і без CSV.

    uv run python analyze.py --saved results --out analysis

Порядок суворий: validation (пункт 2) -> перевірки на наборах A/B/C (пункт 3) -> test (пункт 4).
Якщо хоч одна перевірка не пройшла, програма зупиняється і оцінки test не читає.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # малюємо одразу у файл, вікно не потрібне
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from check_small import SMALL_SETS, brute_force_best, brute_force_table
from threshold import best_threshold, candidate_table, confusion_counts
from train import check_split, load_predictions, saved_parts

COUNT_COLUMNS = ["TP", "FP", "FN", "TN", "C"]


def precision_recall(c: dict[str, int]) -> tuple[float | None, float | None]:
    """precision = TP/(TP+FP), recall = TP/(TP+FN).

    Якщо знаменник нульовий, метрика не визначена — повертаємо None, а не 0.
    """
    precision = c["TP"] / (c["TP"] + c["FP"]) if c["TP"] + c["FP"] > 0 else None
    recall = c["TP"] / (c["TP"] + c["FN"]) if c["TP"] + c["FN"] > 0 else None
    return precision, recall


def threshold_for_json(t: float) -> float | str:
    """JSON не має нескінченності, тому +inf записуємо рядком."""
    return "inf" if np.isinf(t) else float(t)


# ---------- Пункт 2: validation ----------


def analyze_validation(y: np.ndarray, s: np.ndarray, out: Path) -> dict:
    """Повна таблиця кандидатів, t*, порівняння трьох правил, криві. Лише validation."""
    table = candidate_table(y, s)
    # Повна точність: float64 пишеться у CSV так, що зчитується назад біт у біт.
    table.to_csv(out / "candidates.csv", index=False)
    t_star = best_threshold(table)
    print(f"\n[2] Validation: кандидатів {len(table)} (унікальних оцінок {len(table) - 1} і +inf)")

    # Граничні пороги: +inf — усі негативні; найменша оцінка — усі позитивні.
    first, last = table.iloc[0], table.iloc[-1]
    n_pos, n_neg = int(np.count_nonzero(y == 1)), int(np.count_nonzero(y == 0))
    assert first["TP"] == 0 and first["FP"] == 0, "+inf має давати «усі негативні»"
    assert last["TP"] == n_pos and last["FP"] == n_neg, "найменша оцінка має давати «усі позитивні»"

    rules = [("Стандартний поріг", 0.5), ("Мінімум вартості", t_star), ("Усі негативні", np.inf)]
    summary = []
    for name, t in rules:
        counts = confusion_counts(y, s, t)  # прямий підрахунок для одного порога
        precision, recall = precision_recall(counts)
        summary.append(
            {"rule": name, "threshold": threshold_for_json(t), **counts,
             "precision": precision, "recall": recall}
        )
        print(f"    {name:<18} t={t:<22.17g} " + "  ".join(f"{k}={counts[k]}" for k in COUNT_COLUMNS))

    # Самоперевірка: рядок швидкої таблиці для t* збігається з прямим підрахунком.
    row = table[table["threshold"] == t_star].iloc[0]
    direct = confusion_counts(y, s, t_star)
    assert all(int(row[k]) == direct[k] for k in COUNT_COLUMNS), "таблиця і прямий підрахунок різні"

    n_tied = int(np.count_nonzero(table["C"] == table["C"].min()))
    print(f"    порогів із мінімальною вартістю: {n_tied}; вибрано найбільший t* = {t_star!r}")

    auc = float(roc_auc_score(y, s))
    print(f"    ROC-AUC = {auc:.6f}")
    plot_curves(y, s, summary[:2], out)
    return {
        "n": len(y),
        "n_fraud": n_pos,
        "n_candidates": len(table),
        "n_thresholds_with_min_cost": n_tied,
        "t_star": t_star,
        "roc_auc": auc,
        "summary": summary,
    }


def plot_curves(y: np.ndarray, s: np.ndarray, marked: list[dict], out: Path) -> None:
    """PR- і ROC-криві на validation (scikit-learn) з точками для порогів 0,5 і t*."""
    precision, recall, _ = precision_recall_curve(y, s)
    fpr, tpr, _ = roc_curve(y, s)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision)
    for row in marked:
        if row["precision"] is not None:
            ax.plot(row["recall"], row["precision"], "o",
                    label=f"{row['rule']} (t = {row['threshold']:.4f})")
    ax.set(xlabel="Повнота (recall)", ylabel="Точність (precision)",
           title="Крива точність–повнота, validation")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(out / "pr_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    for row in marked:
        ax.plot(row["FP"] / (row["FP"] + row["TN"]), row["TP"] / (row["TP"] + row["FN"]), "o",
                label=f"{row['rule']} (t = {row['threshold']:.4f})")
    ax.set(xlabel="Частка хибних тривог (FPR)", ylabel="Повнота (TPR)",
           title="ROC-крива, validation")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(out / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------- Пункт 3: перевірка пошуку на малих наборах ----------


def run_small_checks(out: Path) -> dict:
    """Порівняти швидкий пошук із незалежним прямим перебором на наборах A/B/C."""
    print("\n[3] Перевірка пошуку порога на малих наборах")
    report = {}
    csv_rows = []
    for name, data in SMALL_SETS.items():
        # Перевірочний результат: рахується в check_small.py без жодного коду з threshold.py.
        slow_rows = brute_force_table(data["y"], data["s"])
        slow_t = brute_force_best(slow_rows)

        # Швидкий алгоритм — той самий код, що працює на validation.
        fast = candidate_table(np.array(data["y"]), np.array(data["s"], dtype=np.float64))
        fast_t = best_threshold(fast)

        same_candidates = [r["threshold"] for r in slow_rows] == fast["threshold"].tolist()
        same_counts = same_candidates and all(
            int(fast.loc[i, k]) == slow_rows[i][k] for i in range(len(slow_rows)) for k in COUNT_COLUMNS
        )
        same_best = slow_t == fast_t
        passed = same_candidates and same_counts and same_best

        print(f"    Набір {name}: n={len(data['y'])}, кандидатів {len(slow_rows)}")
        for i, r in enumerate(slow_rows):
            fast_part = (
                "  ".join(f"{k}={int(fast.loc[i, k])}" for k in COUNT_COLUMNS)
                if same_candidates else "інший список кандидатів"
            )
            print(f"      t={r['threshold']:<4}  перебір: "
                  + "  ".join(f"{k}={r[k]}" for k in COUNT_COLUMNS) + f"   швидкий: {fast_part}")
            csv_rows.append(
                {"set": name, "threshold": r["threshold"],
                 **{k: r[k] for k in COUNT_COLUMNS},
                 **{f"{k}_fast": int(fast.loc[i, k]) if same_candidates else None
                    for k in COUNT_COLUMNS},
                 "chosen_brute": r["threshold"] == slow_t,
                 "chosen_fast": r["threshold"] == fast_t}
            )
        print(f"      t* перебором = {slow_t}, t* швидким = {fast_t}  ->  "
              + ("ПРОЙДЕНО" if passed else "НЕ ПРОЙДЕНО"))
        report[name] = {
            "n": len(data["y"]),
            "n_candidates": len(slow_rows),
            "t_star_brute_force": threshold_for_json(slow_t),
            "t_star_fast": threshold_for_json(fast_t),
            "same_candidates": same_candidates,
            "same_counts_and_costs": same_counts,
            "same_threshold": same_best,
            "passed": passed,
        }
    pd.DataFrame(csv_rows).to_csv(out / "small_checks.csv", index=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--saved", type=Path, required=True,
                        help="Тека зі збереженими прогнозами (вивід train.py)")
    parser.add_argument("--out", type=Path, required=True, help="Тека для результатів аналізу")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    saved = load_predictions(args.saved)
    check_split(saved_parts(saved))  # лише мітки й номери рядків, оцінки не потрібні

    # Оцінки збережено у float32. Переводимо у float64 — це точне перетворення, значення
    # не змінюються, зате поріг t* без втрат записується в CSV/JSON і зчитується назад.
    y_val = saved["validation_y"]
    s_val = saved["validation_scores"].astype(np.float64)

    results = {"validation": analyze_validation(y_val, s_val, args.out)}

    results["small_checks"] = run_small_checks(args.out)
    if not all(check["passed"] for check in results["small_checks"].values()):
        raise SystemExit("ПОМИЛКА: перевірка на малих наборах не пройдена. До test не переходимо.")
    print("    Усі три перевірки пройдено.")

    (args.out / "analysis.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"\nРезультати збережено: {args.out.resolve()}")


if __name__ == "__main__":
    main()
