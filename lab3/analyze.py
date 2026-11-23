"""Команда аналізу ЛР3: журнали → таблиця, криві, ефекти, перевірка вибору; потім test один раз на CPU.

    uv run python analyze.py --data <тека Intel з seg_train і seg_test> [--saved .] [--out analysis]

Код, що завантажує test, викликається лише звідси.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image
from sklearn.metrics import confusion_matrix

from data import CLASSES, IMG_SIZE, SplitDataset, eval_loader, eval_transform, load_split, test_dir
from loop import macro_f1, predict_logits

RUNS = [("scratch", 1), ("frozen", 1), ("frozen", 2), ("frozen", 3),
        ("unfreeze", 1), ("unfreeze", 2), ("unfreeze", 3)]
N_ERRORS = 5


def load_logs(saved):
    return [json.loads((saved / "runs" / f"{r}_seed{s}" / "run.json").read_text(encoding="utf-8"))
            for r, s in RUNS]


def check_split_hash(saved, logs):
    """Хеш split.csv однаковий у всіх семи журналах і збігається з поточним файлом; інакше — помилка."""
    current = hashlib.sha256((saved / "split.csv").read_bytes()).hexdigest()
    hashes = {log["split_sha256"] for log in logs}
    if hashes != {current}:
        raise SystemExit(f"Хеш split.csv не збігається: у журналах {hashes}, поточний {current}")
    return current


def runs_table(logs):
    rows = []
    for i, log in enumerate(logs, start=1):
        rows.append({"n": i, "regime": log["regime"], "seed": log["seed"],
                     "train_loss_8": log["train_loss"][-1], "val_f1_8": log["final_val_macro_f1"],
                     "train_seconds": log["train_seconds"], "commit": log["git"]["commit"],
                     "uncommitted": log["git"]["uncommitted_changes"], "device": log["device"]["name"]})
    return rows


def effects(logs):
    """F̄_r і R_r для frozen і unfreeze, поріг шуму R, Δpre і Δunf."""
    f = {}
    for log in logs:
        f.setdefault(log["regime"], []).append(log["final_val_macro_f1"])
    mean = {r: float(np.mean(v)) for r, v in f.items()}
    spread = {r: float(max(v) - min(v)) for r, v in f.items() if r != "scratch"}
    R = max(spread["frozen"], spread["unfreeze"])
    d_pre = mean["frozen"] - f["scratch"][0]
    d_unf = mean["unfreeze"] - mean["frozen"]
    return {"F": f, "mean": mean, "range": spread, "R": R, "delta_pre": d_pre, "delta_unf": d_unf,
            "total_gain": d_pre + d_unf,
            "delta_pre_exceeds_R": abs(d_pre) > R, "delta_unf_exceeds_R": abs(d_unf) > R}


def selection_rule(logs, mean):
    """Режим із найбільшим F̄ (scratch — його єдине значення), у ньому — найбільший F_{r,s}."""
    regime = max(mean, key=lambda r: mean[r])
    best = sorted((l for l in logs if l["regime"] == regime),
                  key=lambda l: (-l["final_val_macro_f1"], l["seed"]))[0]
    return regime, best["seed"]


def load_model(path):
    """lab3_model.pt → ResNet-18 на CPU в eval()."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = torchvision.models.resnet18(weights=None)
    model.fc = torch.nn.Linear(512, len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def plot_curves(logs, path):
    colors = {"scratch": "#d62728", "frozen": "#1f77b4", "unfreeze": "#2ca02c"}
    styles = {1: "-", 2: "--", 3: ":"}
    epochs = range(1, len(logs[0]["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for log in logs:
        label = f"{log['regime']}, зерно {log['seed']}"
        kw = dict(color=colors[log["regime"]], linestyle=styles[log["seed"]], marker="o",
                  markersize=3, label=label)
        ax1.plot(epochs, log["train_loss"], **kw)
        ax2.plot(epochs, log["val_macro_f1"], **kw)
    ax1.set(title="Втрата на train", xlabel="Епоха", ylabel="Cross-entropy")
    ax2.set(title="macro-F1 на validation", xlabel="Епоха", ylabel="macro-F1", ylim=(0, 1))
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.set_xticks(list(epochs))
    ax2.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion(cm, path):
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(6), CLASSES, rotation=35, ha="right")
    ax.set_yticks(range(6), CLASSES)
    ax.set(xlabel="Передбачений клас", ylabel="Справжній клас", title="Матриця плутанини на test")
    for i in range(6):
        for j in range(6):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def top_error_pairs(cm):
    """Позадіагональні пари за спаданням кількості; за рівності — за зростанням (справжній, передбачений)."""
    pairs = [(int(cm[i, j]), i, j) for i in range(6) for j in range(6) if i != j and cm[i, j] > 0]
    pairs.sort(key=lambda x: (-x[0], x[1], x[2]))
    return pairs


def pick_errors(pairs, files, targets, preds):
    """По першому за іменем (як рядок) файлу для кожної з п'яти найчастіших пар; якщо пар менше
    п'яти — другий файл найчастішої пари, потім другої за частотою і т. д."""
    files_by_pair = {(i, j): sorted(f for f, t, p in zip(files, targets, preds) if t == i and p == j)
                     for _, i, j in pairs}
    picked = []
    rank = 0
    while len(picked) < N_ERRORS:
        added = False
        for count, i, j in pairs:
            if len(picked) == N_ERRORS:
                break
            if rank < len(files_by_pair[(i, j)]):
                picked.append((files_by_pair[(i, j)][rank], i, j, count))
                added = True
        if not added:
            break
        rank += 1
    return picked


def plot_errors(data_root, errors, path):
    fig, axes = plt.subplots(1, len(errors), figsize=(3.2 * len(errors), 3.9))
    for ax, e in zip(np.atleast_1d(axes), errors):
        with Image.open(test_dir(data_root) / e["true"] / e["file"]) as img:
            ax.imshow(img.convert("RGB"))
        ax.set_title(f"{e['file']}\n{e['true']} → {e['pred']}\np = {e['prob']:.3f}", fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="тека набору Intel (з seg_train і seg_test)")
    parser.add_argument("--saved", default=".", help="тека з split.csv, runs/ і lab3_model.pt")
    parser.add_argument("--out", default="analysis")
    args = parser.parse_args()
    saved, out = Path(args.saved), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    md = []

    # 1. Хеш split.csv у журналах
    logs = load_logs(saved)
    split_hash = check_split_hash(saved, logs)
    print(f"Хеш split.csv однаковий у 7 журналах і збігається з файлом: {split_hash}")

    # 2. Таблиця семи запусків і криві
    table = runs_table(logs)
    md.append("| # | Режим | Зерно | Втрата train, епоха 8 | macro-F1 validation, епоха 8 | Час навчання, с |")
    md.append("|---|---|---|---|---|---|")
    for r in table:
        md.append(f"| {r['n']} | {r['regime']} | {r['seed']} | {r['train_loss_8']:.4f} | "
                  f"{r['val_f1_8']:.4f} | {r['train_seconds']:.1f} |")
    print("\n" + "\n".join(md))
    commits = sorted({(r["commit"], r["uncommitted"]) for r in table}, key=str)
    print(f"Коміти коду в журналах (хеш, незакомічені зміни): {commits}")
    plot_curves(logs, out / "curves.png")

    # 3. Ефекти
    eff = effects(logs)
    eff_md = [
        "", "| Величина | Значення |", "|---|---|",
        f"| F̄_frozen, R_frozen | {eff['mean']['frozen']:.4f}, {eff['range']['frozen']:.4f} |",
        f"| F̄_unfreeze, R_unfreeze | {eff['mean']['unfreeze']:.4f}, {eff['range']['unfreeze']:.4f} |",
        f"| R | {eff['R']:.4f} |",
        f"| Δpre | {eff['delta_pre']:+.4f} ({'|Δ| > R' if eff['delta_pre_exceeds_R'] else '|Δ| ≤ R'}) |",
        f"| Δunf | {eff['delta_unf']:+.4f} ({'|Δ| > R' if eff['delta_unf_exceeds_R'] else '|Δ| ≤ R'}) |",
        f"| Δpre + Δunf | {eff['total_gain']:+.4f} |",
    ]
    md += eff_md
    print("\n".join(eff_md))

    # 4. Перевірка вибору: правило за журналами ↔ lab3_model.pt
    regime, seed = selection_rule(logs, eff["mean"])
    model, ckpt = load_model(saved / "lab3_model.pt")
    meta_ok = (ckpt["regime"] == regime and ckpt["seed"] == seed and ckpt["classes"] == CLASSES
               and ckpt["img_size"] == IMG_SIZE)
    # Ваги у файлі — саме цього запуску: їхні логіти validation (CPU) ≈ збережені val_logits.npy (GPU).
    rows = load_split(saved / "split.csv")
    val_set = SplitDataset(args.data, rows, "validation", eval_transform())
    val_logits_cpu = predict_logits(model, eval_loader(val_set), torch.device("cpu"))
    saved_logits = np.load(saved / "runs" / f"{regime}_seed{seed}" / "val_logits.npy")
    max_diff = float(np.abs(val_logits_cpu - saved_logits).max())
    same_argmax = float(np.mean(val_logits_cpu.argmax(1) == saved_logits.argmax(1)))
    f_selected = next(l["final_val_macro_f1"] for l in logs if (l["regime"], l["seed"]) == (regime, seed))
    f_cpu = macro_f1(val_logits_cpu, np.array(val_set.targets))
    print(f"\nПравило вибору за журналами: {regime}, зерно {seed}; lab3_model.pt: {ckpt['regime']}, "
          f"зерно {ckpt['seed']}, img_size {ckpt['img_size']} → збіг: {meta_ok}")
    print(f"Логіти validation lab3_model.pt на CPU проти збережених: max|різниця| = {max_diff:.2e}, "
          f"однаковий argmax у {same_argmax:.4%}; macro-F1 {f_cpu:.4f} (журнал {f_selected:.4f})")
    if not meta_ok or same_argmax < 0.99:
        raise SystemExit("lab3_model.pt не відповідає правилу вибору / вибраному запуску")

    # 5. Test — один раз, на CPU
    test_set = SplitDataset(args.data, rows, "test", eval_transform())
    test_logits = predict_logits(model, eval_loader(test_set), torch.device("cpu"))
    targets = np.array(test_set.targets)
    preds = test_logits.argmax(1)
    probs = torch.softmax(torch.from_numpy(test_logits), 1).numpy()
    f1_test = macro_f1(test_logits, targets)
    acc = float(np.mean(preds == targets))
    n = len(targets)
    se = float(np.sqrt(acc * (1 - acc) / n))
    cm = confusion_matrix(targets, preds, labels=list(range(6)))
    np.save(out / "test_logits.npy", test_logits.astype(np.float32))
    plot_confusion(cm, out / "confusion.png")

    test_md = ["", f"macro-F1 test = {f1_test:.4f}; accuracy = {acc:.4f} ({int((preds == targets).sum())}/{n}); "
               f"SE = sqrt(p(1-p)/n) = {se:.4f}; F_r,s вибраного запуску = {f_selected:.4f}; "
               f"різниця test − validation = {f1_test - f_selected:+.4f}",
               "", "| Справжній \\ передбачений | " + " | ".join(CLASSES) + " |",
               "|---" * 7 + "|"]
    for i, c in enumerate(CLASSES):
        test_md.append(f"| **{c}** | " + " | ".join(str(int(x)) for x in cm[i]) + " |")

    pairs = top_error_pairs(cm)
    picked = pick_errors(pairs, test_set.files, targets, preds)
    index = {f: k for k, f in enumerate(test_set.files)}
    errors = [{"file": f, "true": CLASSES[i], "pred": CLASSES[j],
               "prob": float(probs[index[f], j]), "count": count} for f, i, j, count in picked]
    test_md += ["", "| Файл | Справжній клас | Передбачений клас | Імовірність передбаченого класу | "
                "Кількість таких помилок у test |", "|---|---|---|---|---|"]
    for e in errors:
        test_md.append(f"| {e['file']} | {e['true']} | {e['pred']} | {e['prob']:.4f} | {e['count']} |")
    plot_errors(args.data, errors, out / "errors.png")
    md += test_md
    print("\n".join(test_md))

    (out / "tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    write_json({
        "split_sha256": split_hash, "runs": table, "effects": eff,
        "selection": {"regime": regime, "seed": seed, "lab3_model_matches": meta_ok,
                      "val_logits_cpu_vs_saved_max_abs_diff": max_diff,
                      "val_argmax_agreement": same_argmax, "val_macro_f1_cpu": f_cpu,
                      "val_macro_f1_logged": f_selected},
        "test": {"n": n, "macro_f1": f1_test, "accuracy": acc, "se_accuracy": se,
                 "test_minus_val": f1_test - f_selected, "confusion_matrix": cm.tolist(),
                 "top_error_pairs": [{"count": c, "true": CLASSES[i], "pred": CLASSES[j]}
                                     for c, i, j in pairs],
                 "errors": errors},
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__,
                     "torchvision": torchvision.__version__, "numpy": np.__version__},
    }, out / "analysis.json")
    print(f"\nЗбережено: {out / 'analysis.json'}, {out / 'tables.md'}, curves.png, confusion.png, errors.png")


if __name__ == "__main__":
    main()
