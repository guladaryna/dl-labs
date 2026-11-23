"""Перевірки до навчання: пункт 1 (пакет, L0) і пункт 2 (параметри, режими, дослід із BN)."""

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from data import CLASSES, train_loader, unnormalize
from model import (EVAL_MODULES, REGIMES, TOP_MODULES, build_model, describe_params,
                   set_train_mode)

CHECK_SEED = 1  # усі перевірки пунктів 1–2 — зі зерном 1
N_CHECK_STEPS = 5

# Контрольні числа з таблиці режимів
EXPECTED_TOTAL = 11_179_590
EXPECTED_TRAINABLE = {"scratch": 11_179_590, "frozen": 3_078, "unfreeze": 8_396_806}
EXPECTED_GROUPS = {
    "scratch": [(11_179_590, 3e-3)],
    "frozen": [(3_078, 3e-3)],
    "unfreeze": [(3_078, 3e-3), (8_393_728, 3e-4)],
}


def save_batch_figure(train_set, path):
    """Перші 16 зображень першого пакета train (зерно 1) так, як їх бачить модель."""
    torch.manual_seed(CHECK_SEED)
    images, targets = next(iter(train_loader(train_set)))
    images = unnormalize(images[:16])
    fig, axes = plt.subplots(4, 4, figsize=(8, 8.6))
    for ax, img, t in zip(axes.flat, images, targets[:16]):
        ax.imshow(img.permute(1, 2, 0).numpy())
        ax.set_title(CLASSES[int(t)], fontsize=10)
        ax.axis("off")
    fig.suptitle("Перший пакет train після аугментацій (обернене нормування), 160×160")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def initial_loss(regime, train_set, device):
    """L0 на першому пакеті до жодного кроку: після set_train_mode, під no_grad.

    Прямий прохід у train() оновлює ковзні статистики BN, тому модель після цього не використовується.
    """
    model, _ = build_model(regime, CHECK_SEED, device)
    images, targets = next(iter(train_loader(train_set)))
    set_train_mode(model, regime)
    with torch.no_grad():
        logits = model(images.to(device))
        loss = torch.nn.functional.cross_entropy(logits, targets.to(device)).item()
    return {"regime": regime, "L0": loss, "ln6": math.log(6), "L0_minus_ln6": loss - math.log(6),
            "logit_std": logits.std().item(),
            "mean_max_softmax": torch.softmax(logits, 1).max(1).values.mean().item()}


def check_params(device):
    """Кількості параметрів і групи оптимізатора для трьох режимів; звірка з таблицею режимів."""
    results = {}
    for regime in REGIMES:
        model, optimizer = build_model(regime, CHECK_SEED, device)
        info = describe_params(model, optimizer)
        got_groups = [(g["params"], g["lr"]) for g in info["groups"]]
        info["match"] = (info["total"] == EXPECTED_TOTAL
                         and info["trainable"] == EXPECTED_TRAINABLE[regime]
                         and got_groups == EXPECTED_GROUPS[regime]
                         and info["frozen_tensors_in_optimizer"] == 0)
        results[regime] = info
    return results


def module_changes(before, model):
    """Для кожного модуля верхнього рівня: чи змінився хоч один параметр / хоч один буфер BN.

    Порівняння точне (torch.equal). Буфери BN: running_mean, running_var, num_batches_tracked.
    """
    after = model.state_dict()
    param_names = {name for name, _ in model.named_parameters()}
    rows = []
    for module in TOP_MODULES:
        keys = [k for k in before if k.split(".")[0] == module]
        params = [k for k in keys if k in param_names]
        buffers = [k for k in keys if k not in param_names]
        params_changed = any(not torch.equal(before[k], after[k]) for k in params)
        if buffers:
            stats_changed = "так" if any(not torch.equal(before[k], after[k]) for k in buffers) else "ні"
        else:
            stats_changed = "немає BN"
        rows.append({"module": module, "params_changed": "так" if params_changed else "ні",
                     "bn_stats_changed": stats_changed})
    return rows


def expected_change(regime, module):
    """Очікування: модулі «У eval()» не змінюються взагалі; решта — змінюються параметри і (якщо є BN) статистики."""
    has_bn = module not in ("conv1", "fc")  # conv1 і fc не містять BatchNorm
    if module in EVAL_MODULES[regime]:
        return "ні", ("ні" if has_bn else "немає BN")
    return "так", ("так" if has_bn else "немає BN")


def run_mode_check(regime, train_set, device, use_set_train_mode=True):
    """5 кроків оптимізатора на перших 5 пакетах і порівняння state_dict до/після.

    use_set_train_mode=False — дослід із навмисною помилкою: model.train() замість set_train_mode.
    """
    model, optimizer = build_model(regime, CHECK_SEED, device)
    loader = train_loader(train_set)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if use_set_train_mode:
        set_train_mode(model, regime)
    else:
        model.train()

    loss_fn = torch.nn.CrossEntropyLoss()
    losses = []
    for step, (images, targets) in enumerate(loader):
        if step == N_CHECK_STEPS:
            break
        optimizer.zero_grad()
        loss = loss_fn(model(images.to(device)), targets.to(device))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    rows = module_changes(before, model)
    for row in rows:
        exp_params, exp_stats = expected_change(regime, row["module"])
        row["matches_expected"] = ("так" if (row["params_changed"], row["bn_stats_changed"])
                                   == (exp_params, exp_stats) else "ні")

    after = model.state_dict()
    bn = "layer1.0.bn1"
    bn_delta = {
        "running_mean_max_abs_change":
            (after[f"{bn}.running_mean"] - before[f"{bn}.running_mean"]).abs().max().item(),
        "running_var_max_abs_change":
            (after[f"{bn}.running_var"] - before[f"{bn}.running_var"]).abs().max().item(),
        "num_batches_tracked": [int(before[f"{bn}.num_batches_tracked"]),
                                int(after[f"{bn}.num_batches_tracked"])],
    }
    return {"regime": regime, "set_train_mode": use_set_train_mode, "modules": rows,
            "step_losses": losses, "layer1.0.bn1": bn_delta,
            "all_match": all(r["matches_expected"] == "так" for r in rows)}


def print_mode_check(result):
    how = "set_train_mode" if result["set_train_mode"] else "model.train() (навмисна помилка)"
    print(f"\nРежим {result['regime']}, {how}:")
    print(f"  {'Модуль':<8} {'Параметри':>10} {'Статистики BN':>14} {'Як очікувано':>13}")
    for r in result["modules"]:
        print(f"  {r['module']:<8} {r['params_changed']:>10} {r['bn_stats_changed']:>14} "
              f"{r['matches_expected']:>13}")
    d = result["layer1.0.bn1"]
    print(f"  втрати 5 кроків: {[round(x, 4) for x in result['step_losses']]}")
    print(f"  layer1.0.bn1: max|Δrunning_mean| = {d['running_mean_max_abs_change']:.6g}, "
          f"max|Δrunning_var| = {d['running_var_max_abs_change']:.6g}, "
          f"num_batches_tracked {d['num_batches_tracked'][0]} → {d['num_batches_tracked'][1]}")
