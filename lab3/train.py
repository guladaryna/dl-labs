"""Команда навчання ЛР3: поділ, перевірки пунктів 1–2, сім запусків, журнали, вибір і lab3_model.pt.

Test тут не завантажується і не використовується.

    uv run python train.py --data <тека Intel з seg_train і seg_test> [--out .] [--stage all]
"""

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision

from checks import (check_params, initial_loss, print_mode_check, run_mode_check,
                    save_batch_figure)
from data import (CLASSES, IMG_SIZE, MEAN, STD, SplitDataset, build_split, check_split,
                  eval_loader, eval_transform, load_split, print_split_report, save_split,
                  train_loader, train_transform)
from loop import macro_f1, predict_logits, train_epoch
from model import (BACKBONE_LR, EVAL_MODULES, HEAD_LR, TRAINABLE_MODULES, WEIGHT_DECAY,
                   build_model, describe_params)

EPOCHS = 8
RUNS = [("scratch", 1), ("frozen", 1), ("frozen", 2), ("frozen", 3),
        ("unfreeze", 1), ("unfreeze", 2), ("unfreeze", 3)]
STAGES = ["all", "checks", "bn-experiment", "runs", "select"]


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_name(device):
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "Apple MPS"
    return platform.processor() or "CPU"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_info():
    """Хеш коміту і чи є незакомічені зміни в коді лаби (*.py, pyproject.toml, uv.lock)."""
    here = Path(__file__).parent
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True,
                                text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--", "*.py", "pyproject.toml",
                                 "uv.lock"], cwd=here, capture_output=True, text=True,
                                check=True).stdout.strip()
        return {"commit": commit, "uncommitted_changes": bool(status)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "uncommitted_changes": None}


def write_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def run_settings(regime, model, optimizer):
    """Усі налаштування з розділів «Перетворення», «Модель і режими», «Цикл навчання», включно з типовими."""
    bn = model.bn1
    return {
        "transforms": {
            "train": repr(train_transform()),
            "validation_test": repr(eval_transform()),
            "img_size": IMG_SIZE, "mean": MEAN, "std": STD,
            "RandomResizedCrop": {"size": IMG_SIZE, "scale": [0.7, 1.0], "ratio": [3 / 4, 4 / 3],
                                  "interpolation": "bilinear", "antialias": True},
            "RandomHorizontalFlip": {"p": 0.5},
            "Resize": {"size": [IMG_SIZE, IMG_SIZE], "interpolation": "bilinear",
                       "antialias": True},
        },
        "model": {
            "arch": "torchvision.models.resnet18",
            "weights": None if regime == "scratch" else "ResNet18_Weights.IMAGENET1K_V1",
            "fc": "torch.nn.Linear(512, 6), стандартна ініціалізація",
            "trainable_modules": TRAINABLE_MODULES[regime] or "усі",
            "eval_modules_during_training": EVAL_MODULES[regime],
            "bn_momentum": bn.momentum, "bn_eps": bn.eps,
            "params": describe_params(model, optimizer),
        },
        "optimizer": {
            "class": "torch.optim.AdamW",
            "defaults": {k: v for k, v in optimizer.defaults.items()
                         if isinstance(v, (int, float, bool, tuple, list, type(None)))},
            "groups": [{"modules": ("усі" if regime == "scratch" else name), "lr": g["lr"],
                        "weight_decay": g["weight_decay"]}
                       for name, g in zip(["fc", "layer4"], optimizer.param_groups)],
            "head_lr": HEAD_LR, "backbone_lr": BACKBONE_LR, "weight_decay": WEIGHT_DECAY,
            "lr_schedule": None, "grad_clipping": None, "mixed_precision": False,
            "label_smoothing": 0.0, "class_weights": None, "early_stopping": None,
        },
        "loop": {
            "epochs": EPOCHS, "train_batch_size": 32, "shuffle": True, "drop_last": False,
            "num_workers": 0, "generator": None, "steps_per_epoch": 38, "total_steps": 304,
            "eval_batch_size": 64, "loss": "CrossEntropyLoss(reduction='mean')",
            "epoch_loss": "sum_b L_b*|b| / 1200, L_b до кроку оптимізатора",
            "train_mode": "set_train_mode: model.train(), потім eval() для eval_modules",
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "dtype": "float32",
        },
    }


def train_run(regime, seed, train_set, val_set, device, out, split_hash):
    """Один запуск протоколу: 8 епох, після кожної — macro-F1 на validation; журнал і логіти."""
    run_dir = out / "runs" / f"{regime}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    model, optimizer = build_model(regime, seed, device)  # кроки 1–5
    loader = train_loader(train_set)
    val = eval_loader(val_set)
    val_targets = np.array(val_set.targets)

    losses, f1s, epoch_seconds = [], [], []
    start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        t0 = time.perf_counter()
        loss = train_epoch(model, regime, optimizer, loader, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_seconds.append(time.perf_counter() - t0)
        logits = predict_logits(model, val, device)
        f1 = macro_f1(logits, val_targets)
        losses.append(loss)
        f1s.append(f1)
        print(f"  {regime} seed {seed} епоха {epoch}: втрата {loss:.4f}, macro-F1 val {f1:.4f}",
              flush=True)
    total_seconds = time.perf_counter() - start

    # Використовуються ваги й логіти після 8-ї епохи (без вибору епохи).
    np.save(run_dir / "val_logits.npy", logits.astype(np.float32))
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, run_dir / "weights.pt")

    log = {
        "regime": regime, "seed": seed,
        "settings": run_settings(regime, model, optimizer),
        "git": git_info(),
        "split_sha256": split_hash,
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "torchvision": torchvision.__version__, "numpy": np.__version__},
        "device": {"type": device.type, "name": device_name(device)},
        "os": platform.platform(),
        "train_loss": losses, "val_macro_f1": f1s,
        "final_val_macro_f1": f1s[-1],
        "train_seconds": sum(epoch_seconds),
        "train_and_val_seconds": total_seconds,
        "val_logits_shape": list(logits.shape),
    }
    write_json(log, run_dir / "run.json")
    return log


def select_run(out):
    """Правило вибору (пункт 5) за збереженими журналами: режим із найбільшим середнім F,
    у ньому — запуск із найбільшим F_{r,s} (за рівності — менше зерно)."""
    logs = [json.loads((out / "runs" / f"{r}_seed{s}" / "run.json").read_text(encoding="utf-8"))
            for r, s in RUNS]
    by_regime = {}
    for log in logs:
        by_regime.setdefault(log["regime"], []).append(log)
    means = {r: float(np.mean([l["final_val_macro_f1"] for l in ls])) for r, ls in by_regime.items()}
    best_regime = max(means, key=lambda r: means[r])
    candidates = sorted(by_regime[best_regime], key=lambda l: (-l["final_val_macro_f1"], l["seed"]))
    return best_regime, candidates[0]["seed"], means


def save_selected_model(out, regime, seed):
    state_dict = torch.load(out / "runs" / f"{regime}_seed{seed}" / "weights.pt",
                            map_location="cpu", weights_only=True)
    torch.save({"state_dict": state_dict, "regime": regime, "seed": seed, "classes": CLASSES,
                "img_size": IMG_SIZE}, out / "lab3_model.pt")


def stage_checks(data_root, rows, device, out):
    print("\n=== Пункт 1. Пакет після перетворень і початкова втрата ===")
    train_set = SplitDataset(data_root, rows, "train", train_transform())
    save_batch_figure(train_set, out / "results" / "batch.png")
    print(f"Рисунок пакета: {out / 'results' / 'batch.png'}")
    l0 = [initial_loss(r, train_set, device) for r in ["scratch", "frozen"]]
    for x in l0:
        print(f"L0 {x['regime']:<8} = {x['L0']:.4f}  (ln 6 = {x['ln6']:.4f}, різниця "
              f"{x['L0_minus_ln6']:+.4f}; std логітів {x['logit_std']:.3f}, "
              f"середня max softmax {x['mean_max_softmax']:.3f})")

    print("\n=== Пункт 2. Параметри й групи оптимізатора ===")
    params = check_params(device)
    for regime, info in params.items():
        groups = ", ".join(f"{g['params']} пар. lr={g['lr']:g}" for g in info["groups"])
        print(f"{regime:<8} усього {info['total']}, requires_grad=True {info['trainable']}; "
              f"групи: {groups}; заморожених тензорів в оптимізаторі: "
              f"{info['frozen_tensors_in_optimizer']}; збіг із таблицею: {info['match']}")
    if not all(info["match"] for info in params.values()):
        raise SystemExit("Кількості параметрів або групи не збігаються з таблицею режимів")

    print("\n=== Пункт 2. Що змінюється за 5 кроків (set_train_mode) ===")
    modes = [run_mode_check(r, train_set, device) for r in ["frozen", "unfreeze"]]
    for m in modes:
        print_mode_check(m)
    if not all(m["all_match"] for m in modes):
        raise SystemExit("Перевірка режимів не збіглася з очікуваним")

    write_json({"initial_loss": l0, "params": params, "mode_checks": modes},
               out / "results" / "checks.json")


def stage_bn_experiment(data_root, rows, device, out):
    print("\n=== Пункт 2. Дослід із навмисною помилкою: frozen з model.train() ===")
    train_set = SplitDataset(data_root, rows, "train", train_transform())
    bug = run_mode_check("frozen", train_set, device, use_set_train_mode=False)
    print_mode_check(bug)
    write_json(bug, out / "results" / "bn_experiment.json")


def stage_runs(data_root, rows, device, out, only=None):
    train_set = SplitDataset(data_root, rows, "train", train_transform())
    val_set = SplitDataset(data_root, rows, "validation", eval_transform())
    assert len(train_set) == 1200 and len(val_set) == 1498
    split_hash = sha256(out / "split.csv")
    runs = RUNS if only is None else [r for r in RUNS if f"{r[0]}_seed{r[1]}" in only]
    for regime, seed in runs:
        print(f"\n=== Запуск {regime}, зерно {seed} ({device_name(device)}) ===", flush=True)
        log = train_run(regime, seed, train_set, val_set, device, out, split_hash)
        print(f"  F = {log['final_val_macro_f1']:.4f}, час навчання {log['train_seconds']:.1f} с")


def stage_select(out):
    regime, seed, means = select_run(out)
    for r, m in means.items():
        print(f"Середнє macro-F1 validation {r:<8}: {m:.4f}")
    print(f"Вибрано: {regime}, зерно {seed} → {out / 'lab3_model.pt'}")
    save_selected_model(out, regime, seed)
    write_json({"regime": regime, "seed": seed, "mean_val_macro_f1": means},
               out / "results" / "selection.json")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="тека набору Intel (з seg_train і seg_test)")
    parser.add_argument("--out", default=".", help="куди писати split.csv, runs/, results/, lab3_model.pt")
    parser.add_argument("--stage", default="all", choices=STAGES)
    parser.add_argument("--only", nargs="*", help="для --stage runs: лише ці запуски, напр. frozen_seed1")
    parser.add_argument("--device", default="auto", help="auto, cuda, mps або cpu")
    args = parser.parse_args()

    # Детерміновані згорткові алгоритми cuDNN: менше розбіжностей між повторами на тій самій GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    print(f"Пристрій: {device} ({device_name(device)}), torch {torch.__version__}")

    # Поділ і контроль — щоразу, перед будь-яким етапом.
    rows = build_split(args.data)
    report, no_overlap = check_split(rows)
    print("\n=== Пункт 1. Контроль поділу ===")
    print_split_report(report, no_overlap)
    save_split(rows, out / "split.csv")
    assert load_split(out / "split.csv") == rows
    print(f"split.csv SHA-256: {sha256(out / 'split.csv')}")
    write_json({"table": report, "no_overlap": no_overlap, "sha256": sha256(out / "split.csv")},
               out / "results" / "split_check.json")

    if args.stage in ("all", "checks"):
        stage_checks(args.data, rows, device, out)
    if args.stage in ("all", "bn-experiment"):
        stage_bn_experiment(args.data, rows, device, out)
    if args.stage in ("all", "runs"):
        stage_runs(args.data, rows, device, out, args.only)
    if args.stage in ("all", "select"):
        stage_select(out)


if __name__ == "__main__":
    main()
