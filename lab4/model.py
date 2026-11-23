"""Модель ЛР3, експорт в ONNX (пункт 1), сесії ONNX Runtime і логіти пакетами по 64."""

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torchvision

from data import CLASSES

BATCH_SIZE = 64  # таблиця фіксованих налаштувань, рядок «Якість»


def load_lab3_model(path):
    """Завантаження рівно за розділом «Модель»: resnet18(weights=None), fc -> Linear(512, 6), eval()."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if list(ckpt["classes"]) != CLASSES:
        raise SystemExit(f"Порядок класів у моделі {ckpt['classes']} не збігається з ЛР3 {CLASSES}")
    model = torchvision.models.resnet18(weights=None)
    model.fc = torch.nn.Linear(512, 6)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def export_onnx(model, img_size, path):
    """torch.onnx.export з dynamo=False, opset 17, вхід input, вихід logits, змінний розмір пакета."""
    model.eval()  # BN зливається з попередньою згорткою лише в режимі eval()
    dummy = torch.zeros(1, 3, img_size, img_size, dtype=torch.float32)
    torch.onnx.export(
        model,
        (dummy,),
        str(path),
        dynamo=False,
        opset_version=17,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )


def count_ops(path):
    """Кількість вузлів кожного типу в графі ONNX (op_type -> кількість)."""
    graph = onnx.load(str(path)).graph
    counts = {}
    for node in graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return counts


def make_session(path, disable_graph_optimizations=False):
    """Сесія рівно за таблицею: CPUExecutionProvider, 1 + 1 потік, рівень оптимізації за замовчуванням.

    disable_graph_optimizations=True — ЛИШЕ для діагностики в пункті 3: без оптимізацій графа ORT не
    зливає пари DequantizeLinear → Conv → QuantizeLinear у цілочисельну QLinearConv, а рахує згортку
    у float на деквантованих вагах і активаціях (симуляція квантизації без цілочисельного множення).
    """
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    if disable_graph_optimizations:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def onnx_logits(session, images):
    """Логіти сесії для масиву (N, 3, H, W) пакетами по 64 у тому самому порядку -> (N, 6)."""
    outputs = []
    for start in range(0, len(images), BATCH_SIZE):
        batch = images[start:start + BATCH_SIZE]
        outputs.append(session.run(["logits"], {"input": batch})[0])
    return np.concatenate(outputs)


def torch_logits(model, images):
    """Логіти моделі PyTorch під torch.no_grad() пакетами по 64 -> (N, 6)."""
    outputs = []
    with torch.no_grad():
        for start in range(0, len(images), BATCH_SIZE):
            batch = torch.from_numpy(images[start:start + BATCH_SIZE])
            outputs.append(model(batch).numpy())
    return np.concatenate(outputs)
