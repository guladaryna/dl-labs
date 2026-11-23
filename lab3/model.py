"""ResNet-18 у трьох режимах (розділ «Модель і режими»)."""

import numpy as np
import torch
import torchvision

N_CLASSES = 6
WEIGHT_DECAY = 1e-4
HEAD_LR = 3e-3
BACKBONE_LR = 3e-4  # layer4 у режимі unfreeze

REGIMES = ["scratch", "frozen", "unfreeze"]

# Модулі верхнього рівня, які під час навчання мають бути в eval() (стовпець «У eval()»).
EVAL_MODULES = {
    "scratch": [],
    "frozen": ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"],
    "unfreeze": ["conv1", "bn1", "layer1", "layer2", "layer3"],
}

# Що навчається (стовпець «Що навчається»); None — уся мережа.
TRAINABLE_MODULES = {
    "scratch": None,
    "frozen": ["fc"],
    "unfreeze": ["layer4", "fc"],
}

TOP_MODULES = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"]


def build_model(regime, seed, device):
    """Кроки 1–5 побудови моделі. Повертає (model, optimizer).

    Порядок важливий: зерно задається ДО створення мережі й голови, тож ініціалізація
    scratch і нової fc залежать від зерна; далі той самий глобальний генератор PyTorch
    перемішує train і робить аугментації.
    """
    # Крок 1
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Крок 2
    weights = None if regime == "scratch" else torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet18(weights=weights)

    # Крок 3: нова голова на 6 класів зі стандартною ініціалізацією nn.Linear
    model.fc = torch.nn.Linear(512, N_CLASSES)

    # Крок 4: requires_grad=False для всього, що не навчається
    trainable = TRAINABLE_MODULES[regime]
    if trainable is not None:
        for name, param in model.named_parameters():
            param.requires_grad = name.split(".")[0] in trainable

    # Крок 5
    model.to(device)
    optimizer = make_optimizer(model, regime)
    return model, optimizer


def make_optimizer(model, regime):
    """AdamW, weight_decay=1e-4 в усіх групах; параметри з requires_grad=False не передаються."""
    if regime == "scratch":
        groups = [{"params": list(model.parameters()), "lr": HEAD_LR}]
    elif regime == "frozen":
        groups = [{"params": list(model.fc.parameters()), "lr": HEAD_LR}]
    else:
        # Кістяку (layer4) — у 10 разів менша швидкість, ніж новій голові.
        groups = [{"params": list(model.fc.parameters()), "lr": HEAD_LR},
                  {"params": list(model.layer4.parameters()), "lr": BACKBONE_LR}]
    return torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)


def set_train_mode(model, regime):
    """На початку епохи: усе в train(), потім заморожені модулі — назад в eval().

    Без другого кроку BN заморожених шарів оновлювали б ковзні статистики й у train()
    нормували б за статистиками пакета, а не за збереженими ImageNet-статистиками.
    """
    model.train()
    for name in EVAL_MODULES[regime]:
        getattr(model, name).eval()


def count_params(params):
    """Кількість параметрів — сума numel(), а не кількість тензорів."""
    return sum(p.numel() for p in params)


def describe_params(model, optimizer):
    """Загальна кількість, кількість із requires_grad=True, групи оптимізатора."""
    in_optimizer = {id(p) for g in optimizer.param_groups for p in g["params"]}
    frozen_in_optimizer = sum(1 for p in model.parameters()
                              if not p.requires_grad and id(p) in in_optimizer)
    return {
        "total": count_params(model.parameters()),
        "trainable": count_params(p for p in model.parameters() if p.requires_grad),
        "groups": [{"params": count_params(g["params"]), "lr": g["lr"],
                    "weight_decay": g["weight_decay"]} for g in optimizer.param_groups],
        "frozen_tensors_in_optimizer": frozen_in_optimizer,
    }
