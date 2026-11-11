"""ЛР2, пункти 2–4. Аналіз порогів зі збережених прогнозів: без навчання і без CSV.

    uv run python analyze.py --saved results --out analysis

Необов'язково: --compare <тека> звіряє поділ і статистики стандартизації з іншим запуском
train.py (перевірка відтворюваності повторного навчання).

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


# ---------- Пункт 4: test ----------


def analyze_test(saved: dict[str, np.ndarray], t_star: float) -> tuple[dict, pd.DataFrame]:
    """Застосувати зафіксований t* до test: матриця плутанини, C, precision, recall, помилки.

    Викликається лише після трьох перевірок. Поріг сюди приходить готовим і не змінюється.
    """
    y = saved["test_y"]
    s = saved["test_scores"].astype(np.float64)
    csv_row = saved["test_row_ids"]
    amount = saved["test_amount"]

    counts = confusion_counts(y, s, t_star)
    precision, recall = precision_recall(counts)
    print(f"\n[4] Test, t* = {t_star!r} (зафіксовано на validation)")
    print("    " + "  ".join(f"{k}={counts[k]}" for k in COUNT_COLUMNS))
    print(f"    precision = {precision}, recall = {recall}")

    # Для порівняння у висновках: ті самі лічильники за порогів 0,5 і +inf.
    # Це лише звіт; на вибір порога вони не впливають — t* уже зафіксовано.
    reference = [
        {"rule": name, "threshold": threshold_for_json(t), **confusion_counts(y, s, t)}
        for name, t in (("Стандартний поріг", 0.5), ("Усі негативні", np.inf))
    ]

    pred = s >= t_star
    masks = {"FP": pred & (y == 0), "FN": ~pred & (y == 1)}
    error_rows = []
    all_errors = {}
    for kind, mask in masks.items():
        idx = np.flatnonzero(mask)
        idx = idx[np.argsort(csv_row[idx])]  # за номером рядка CSV (від 0, без заголовка)
        # Зведення за ВСІМА помилками цього типу — щоб не судити лише за п'ятьма прикладами.
        all_errors[kind] = {
            "count": len(idx),
            "median_amount": float(np.median(amount[idx])) if len(idx) else None,
            "median_abs_distance": float(np.median(np.abs(s[idx] - t_star))) if len(idx) else None,
        }
        for i in idx[:5]:  # перші п'ять; якщо менше — усі
            error_rows.append(
                {"csv_row": int(csv_row[i]), "type": kind, "label": int(y[i]),
                 "score": float(s[i]), "amount": float(amount[i]),
                 "score_minus_threshold": float(s[i] - t_star)}
            )
    errors = pd.DataFrame(error_rows)
    print(errors.to_string(index=False))
    result = {
        "n": len(y),
        "n_fraud": int(np.count_nonzero(y == 1)),
        "threshold": t_star,
        **counts,
        "precision": precision,
        "recall": recall,
        "reference_only": reference,
        "all_errors": all_errors,
    }
    return result, errors


# ---------- Таблиці для README і звірка двох запусків ----------


def md_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def show_threshold(t) -> str:
    """Округлення лише для відображення; повні значення — в analysis.json і CSV."""
    return "+∞" if t == "inf" or np.isinf(t) else f"{t:.6f}"


def show_metric(v) -> str:
    return "не визначена" if v is None else f"{v:.4f}"


def write_tables(results: dict, errors: pd.DataFrame, out: Path) -> None:
    """Готові markdown-таблиці: README збирається з них, а не з пам'яті."""
    parts = ["## Поділ\n\n" + md_table(
        ["Частина", "Об'єктів", "Шахрайських", "Частка"],
        [[r["split"], r["n"], r["n_fraud"], f"{r['fraction']:.6f}"] for r in results["split"]],
    )]
    parts.append("## Validation\n\n" + md_table(
        ["Правило на validation", "Поріг", "TP", "FP", "FN", "TN", "C", "precision", "recall"],
        [[r["rule"], show_threshold(r["threshold"]), *(r[k] for k in COUNT_COLUMNS),
          show_metric(r["precision"]), show_metric(r["recall"])]
         for r in results["validation"]["summary"]],
    ))
    checks = pd.read_csv(out / "small_checks.csv")
    for name, group in checks.groupby("set"):
        report = results["small_checks"][name]
        parts.append(f"## Набір {name}\n\n" + md_table(
            ["Поріг", "TP", "FP", "FN", "TN", "C", "Швидкий: TP/FP/FN/TN/C", "Вибрано"],
            [["+∞" if np.isinf(r["threshold"]) else r["threshold"],
              *(r[k] for k in COUNT_COLUMNS),
              "/".join(str(r[f"{k}_fast"]) for k in COUNT_COLUMNS),
              "t*" if r["chosen_brute"] and r["chosen_fast"] else ""]
             for _, r in group.iterrows()],
        ) + f"\nt* перебором: {show_threshold(report['t_star_brute_force'])}, "
            f"t* швидким: {show_threshold(report['t_star_fast'])}, "
            f"перевірка {'пройдена' if report['passed'] else 'НЕ пройдена'}.\n")
    test = results["test"]
    parts.append("## Test\n\n" + md_table(
        ["Поріг", "TP", "FP", "FN", "TN", "C", "precision", "recall"],
        [[show_threshold(test["threshold"]), *(test[k] for k in COUNT_COLUMNS),
          show_metric(test["precision"]), show_metric(test["recall"])]],
    ))
    parts.append("## Test, лише для порівняння\n\n" + md_table(
        ["Правило", "Поріг", "TP", "FP", "FN", "TN", "C"],
        [[r["rule"], show_threshold(r["threshold"]), *(r[k] for k in COUNT_COLUMNS)]
         for r in test["reference_only"]],
    ))
    parts.append("## Помилки на test\n\n" + md_table(
        ["Рядок CSV (від 0, без заголовка)", "Тип помилки", "Мітка", "Оцінка s", "Amount", "s − t*"],
        [[r["csv_row"], r["type"], r["label"], f"{r['score']:.6f}", f"{r['amount']:.2f}",
          f"{r['score_minus_threshold']:+.6f}"] for _, r in errors.iterrows()],
    ))
    (out / "tables.md").write_text("\n".join(parts), encoding="utf-8")


def compare_runs(saved_dir: Path, other_dir: Path) -> None:
    """Чи дає повторне навчання той самий поділ і ті самі статистики стандартизації."""
    print(f"\n[порівняння] {saved_dir} <-> {other_dir}")
    a, b = load_predictions(saved_dir), load_predictions(other_dir)
    same_split = True
    for name in ("train", "validation", "test"):
        for key in (f"{name}_row_ids", f"{name}_y"):
            equal = np.array_equal(a[key], b[key])
            same_split = same_split and equal
            print(f"    {key:<20} {'однакові' if equal else 'РІЗНІ'}")
    same_scaler = True
    with np.load(saved_dir / "preprocessing.npz") as pa, np.load(other_dir / "preprocessing.npz") as pb:
        for key in ("mean", "scale"):
            equal = np.array_equal(pa[key], pb[key])
            same_scaler = same_scaler and equal
            print(f"    scaler.{key:<13} {'однакові' if equal else 'РІЗНІ'} "
                  f"(макс. різниця {np.max(np.abs(pa[key] - pb[key])):.3g})")
    # Оцінки можуть відрізнятися в молодших розрядах (версія PyTorch, платформа) — це лише довідка.
    for name in ("validation", "test"):
        diff = np.max(np.abs(a[f"{name}_scores"].astype(np.float64) - b[f"{name}_scores"]))
        print(f"    {name}_scores: макс. різниця {diff:.3g} (збіг не вимагається)")
    if not (same_split and same_scaler):
        raise SystemExit("ПОМИЛКА: поділ або статистики стандартизації не відтворилися.")
    print("    Поділ і статистики стандартизації відтворилися точно.")


def split_table(saved: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for name in ("train", "validation", "test"):
        y = saved[f"{name}_y"]
        n_fraud = int(np.count_nonzero(y == 1))
        rows.append({"split": name, "n": len(y), "n_fraud": n_fraud, "fraction": n_fraud / len(y)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--saved", type=Path, required=True,
                        help="Тека зі збереженими прогнозами (вивід train.py)")
    parser.add_argument("--out", type=Path, required=True, help="Тека для результатів аналізу")
    parser.add_argument("--compare", type=Path,
                        help="Тека іншого запуску train.py: звірити поділ і стандартизацію")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    saved = load_predictions(args.saved)
    check_split(saved_parts(saved))  # лише мітки й номери рядків, оцінки не потрібні

    # Оцінки збережено у float32. Переводимо у float64 — це точне перетворення, значення
    # не змінюються, зате поріг t* без втрат записується в CSV/JSON і зчитується назад.
    y_val = saved["validation_y"]
    s_val = saved["validation_scores"].astype(np.float64)

    results = {"split": split_table(saved)}
    results["validation"] = analyze_validation(y_val, s_val, args.out)

    results["small_checks"] = run_small_checks(args.out)
    if not all(check["passed"] for check in results["small_checks"].values()):
        raise SystemExit("ПОМИЛКА: перевірка на малих наборах не пройдена. До test не переходимо.")
    print("    Усі три перевірки пройдено.")

    # Лише тепер читаємо оцінки test. Поріг уже зафіксовано на validation.
    t_star = results["validation"]["t_star"]
    results["test"], errors = analyze_test(saved, t_star)
    errors.to_csv(args.out / "test_errors.csv", index=False)

    (args.out / "analysis.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_tables(results, errors, args.out)
    print(f"\nРезультати збережено: {args.out.resolve()}")

    if args.compare is not None:
        compare_runs(args.saved, args.compare)


if __name__ == "__main__":
    main()
