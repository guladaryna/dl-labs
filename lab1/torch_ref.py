"""Еталонна реалізація на PyTorch для звірки (пункт 2 умови).

PyTorch використовується тільки тут. Мережа та сама, параметри копіюються
з NumPy-мережі, градієнти рахує autograd.
"""

import torch
from torch import nn


def torch_loss_and_grads(params, X, y):
    """Втрата і чотири градієнти від PyTorch на тих самих параметрах і даних.

    params — словник NumPy-масивів (W1 (4, 8), b1 (8,), W2 (8, 3), b2 (3,)).
    Повертає втрату (float) і градієнти як NumPy-масиви в наших формах (in, out).
    """
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3)).double()  # float64

    # nn.Linear зберігає вагу як (out, in), а в нас (in, out) — тому транспонуємо
    with torch.no_grad():
        model[0].weight.copy_(torch.from_numpy(params["W1"].T))
        model[0].bias.copy_(torch.from_numpy(params["b1"]))
        model[2].weight.copy_(torch.from_numpy(params["W2"].T))
        model[2].bias.copy_(torch.from_numpy(params["b2"]))

    logits = model(torch.from_numpy(X))
    # cross_entropy за замовчуванням усереднює за об'єктами (reduction="mean")
    loss = nn.functional.cross_entropy(logits, torch.from_numpy(y))
    loss.backward()

    # градієнти ваг транспонуємо назад у форму (in, out)
    grads = {
        "W1": model[0].weight.grad.numpy().T,
        "b1": model[0].bias.grad.numpy(),
        "W2": model[2].weight.grad.numpy().T,
        "b2": model[2].bias.grad.numpy(),
    }
    return loss.item(), grads
