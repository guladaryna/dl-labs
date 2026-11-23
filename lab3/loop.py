"""Епоха навчання й оцінка на validation (розділ «Цикл навчання»)."""

import numpy as np
import torch
from sklearn.metrics import f1_score

from model import set_train_mode


def train_epoch(model, regime, optimizer, loader, device):
    """Одна епоха. Повертає втрату епохи ∑ L_b·|b| / N, де L_b — втрата пакета до кроку."""
    set_train_mode(model, regime)
    loss_fn = torch.nn.CrossEntropyLoss()  # середнє за пакетом
    total = 0.0
    n = 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(images), targets)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(targets)
        n += len(targets)
    return total / n


def predict_logits(model, loader, device):
    """Логіти в режимі eval() під no_grad, у порядку завантажувача. Форма (N, 6), float32."""
    model.eval()
    chunks = []
    with torch.no_grad():
        for images, _ in loader:
            chunks.append(model(images.to(device)).cpu())
    return torch.cat(chunks).numpy()


def macro_f1(logits, targets):
    """macro-F1: середнє F1 шести класів «клас проти решти»."""
    return float(f1_score(targets, np.argmax(logits, axis=1), average="macro",
                          labels=list(range(6))))
