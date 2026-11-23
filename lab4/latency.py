"""Протокол вимірювання затримки (пункт 5)."""

import time

import numpy as np

ROUNDS = 3
WARMUP = 20
MEASURES = 100


def measure(session, feed):
    """20 прогрівальних викликів без запису, потім 100 вимірів; повертає 100 значень у мс.

    У вимірі між двома perf_counter() — лише один sess.run і нічого більше.
    """
    for _ in range(WARMUP):
        session.run(None, feed)
    times = []
    for _ in range(MEASURES):
        t0 = time.perf_counter()
        session.run(None, feed)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return [t * 1000.0 for t in times]  # переведення в мс — уже після вимірів


def run_protocol(sessions, x):
    """Три раунди, у кожному спершу FP32, потім INT8. sessions — dict у порядку {"fp32", "int8"}.

    Сесії створено заздалегідь, вхід x (1, 3, H, W) float32 підготовлено заздалегідь.
    Повертає {варіант: [[100 мс], [100 мс], [100 мс]]}.
    """
    feed = {"input": x}
    rounds = {name: [] for name in sessions}
    for round_index in range(ROUNDS):
        for name, session in sessions.items():
            rounds[name].append(measure(session, feed))
            print(f"  раунд {round_index + 1}, {name}: p50 = {np.percentile(rounds[name][-1], 50):.3f} мс")
    return rounds


def summarize(rounds_ms):
    """p50 і p95 за всіма 300 вимірами, p50 кожного раунду."""
    all_ms = np.concatenate(rounds_ms)
    return {
        "p50_ms": float(np.percentile(all_ms, 50)),
        "p95_ms": float(np.percentile(all_ms, 95)),
        "p50_by_round_ms": [float(np.percentile(r, 50)) for r in rounds_ms],
        "n_measures": int(len(all_ms)),
    }


def intervals_overlap(a, b):
    """Чи перетинаються відрізки [min, max] p50 за раундами двох варіантів."""
    return min(a) <= max(b) and min(b) <= max(a)
