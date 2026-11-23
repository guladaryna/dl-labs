"""ЛР4: FP32 проти INT8. Одна команда для пунктів 1–5; усі числа — у lab4_results.json.

uv run python run.py --model ../lab3/lab3_model.pt --split ../lab3/split.csv --data <тека набору>
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

import data
import environment
import latency
import metrics
import model as model_tools
import plots
import quant

STEPS = ["export", "quantize", "quality", "ranges", "per_tensor", "latency"]

# macro-F1 на test моделі lab3_model.pt з ЛР3 (lab3/analysis/analysis.json, "test" → "macro_f1").
LAB3_TEST_MACRO_F1 = 0.891912624642122
MAX_LOGIT_DIFF = 1e-3   # пункт 1: різниця логітів має бути меншою за 1e-3
EXPECTED_CONV = 20      # пункт 1: 20 вузлів Conv
EXPECTED_BN = 0         # пункт 1: жодного BatchNormalization


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model", required=True, help="lab3_model.pt із ЛР3")
    parser.add_argument("--split", required=True, help="split.csv із ЛР3")
    parser.add_argument("--data", required=True, help="тека з seg_train/seg_train і seg_test/seg_test")
    parser.add_argument("--out", default=".", help="куди писати .onnx, lab4_results.json і results/")
    parser.add_argument("--steps", default=",".join(STEPS),
                        help="кроки через кому (за замовчуванням усі по черзі): " + ",".join(STEPS))
    return parser.parse_args()


def file_mb(path):
    """Розмір файлу: кількість байтів / 10^6 (пункт 5)."""
    return Path(path).stat().st_size / 1e6


class Lab:
    """Спільний стан кроків: шляхи, поділ, модель, test (завантажується один раз за потреби)."""

    def __init__(self, args):
        self.data_root = args.data
        self.out = Path(args.out)
        self.results_dir = self.out / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = self.out / "lab4_results.json"
        self.fp32 = self.out / "lab4_fp32.onnx"
        self.prep = self.out / "lab4_fp32_prep.onnx"
        self.int8 = self.out / "lab4_int8.onnx"
        self.int8_pt = self.out / "lab4_int8_per_tensor.onnx"

        self.results = json.loads(self.json_path.read_text(encoding="utf-8")) if self.json_path.exists() else {}

        self.rows = data.load_split(args.split)
        self.results["split_check"] = data.check_split(self.rows)

        self.torch_model, ckpt = model_tools.load_lab3_model(args.model)
        self.img_size = int(ckpt["img_size"])  # розмір входу — з файлу моделі
        self.results["model"] = {"path": str(args.model), "regime": ckpt["regime"], "seed": ckpt["seed"],
                                 "classes": list(ckpt["classes"]), "img_size": self.img_size}
        print(f"Модель: {ckpt['regime']}, зерно {ckpt['seed']}, img_size {self.img_size}, "
              f"класи {list(ckpt['classes'])}")

        self.test_items = data.part_rows(self.rows, "test")
        self.targets = np.array([data.CLASSES.index(label) for _, label in self.test_items])
        self._test_images = None

    def test_images(self):
        """Усі 1502 зображення test у порядку split.csv, (1502, 3, H, W) float32."""
        if self._test_images is None:
            print("Завантаження 1502 зображень test…")
            self._test_images = data.load_images(self.data_root, self.test_items, "test", self.img_size)
        return self._test_images

    def save(self):
        self.json_path.write_text(json.dumps(self.results, indent=2, ensure_ascii=False), encoding="utf-8")


def step_export(lab):
    """Пункт 1: експорт і перевірка відповідності."""
    print("\n=== Пункт 1. Експорт і перевірка відповідності ===")
    t0 = time.perf_counter()
    model_tools.export_onnx(lab.torch_model, lab.img_size, lab.fp32)
    export_s = time.perf_counter() - t0

    ops = model_tools.count_ops(lab.fp32)
    n_conv, n_bn = ops.get("Conv", 0), ops.get("BatchNormalization", 0)

    images = lab.test_images()
    z_torch = model_tools.torch_logits(lab.torch_model, images)
    z_onnx = model_tools.onnx_logits(model_tools.make_session(lab.fp32), images)
    max_diff = float(np.max(np.abs(z_onnx - z_torch)))
    class_match = float(np.mean(z_onnx.argmax(1) == z_torch.argmax(1)))
    f1_torch = metrics.quality_row(z_torch, z_torch, lab.targets)["macro_f1"]
    f1_onnx = metrics.quality_row(z_onnx, z_onnx, lab.targets)["macro_f1"]
    np.save(lab.results_dir / "test_logits_pytorch.npy", z_torch)

    checks = {
        "max_logit_diff_below_1e-3": max_diff < MAX_LOGIT_DIFF,
        "class_match_is_1": class_match == 1.0,
        "f1_torch_equals_onnx": f1_torch == f1_onnx,
        "f1_equals_lab3": abs(f1_torch - LAB3_TEST_MACRO_F1) < 1e-12,
        "conv_is_20": n_conv == EXPECTED_CONV,
        "batchnorm_is_0": n_bn == EXPECTED_BN,
    }
    lab.results["export"] = {
        "file": lab.fp32.name, "export_time_s": export_s, "file_mb": file_mb(lab.fp32),
        "op_counts": ops, "n_conv": n_conv, "n_batchnorm": n_bn,
        "max_abs_logit_diff": max_diff, "class_match": class_match,
        "macro_f1_pytorch": f1_torch, "macro_f1_onnx_fp32": f1_onnx,
        "macro_f1_lab3": LAB3_TEST_MACRO_F1, "checks": checks,
    }
    lab.save()
    print(f"Експорт: {export_s:.2f} с, {file_mb(lab.fp32):.2f} МБ; Conv = {n_conv}, BatchNormalization = {n_bn}")
    print(f"max|z_ONNX − z_PyTorch| = {max_diff:.3e}; збіг класів = {class_match:.4f}")
    print(f"macro-F1: PyTorch {f1_torch:.4f}, ONNX FP32 {f1_onnx:.4f}, ЛР3 {LAB3_TEST_MACRO_F1:.4f}")
    print(f"Контроль: {checks}")
    if not all(checks.values()):
        raise SystemExit("Контроль пункту 1 не пройдено — зупинка")


def calibration_images(lab):
    calib = data.calibration_rows(lab.rows)
    lab.results["calibration"] = data.check_calibration(calib)
    lab.results["calibration"]["files"] = [file for file, _ in calib]
    return data.load_images(lab.data_root, calib, "train", lab.img_size)


def step_quantize(lab):
    """Пункт 2: калібрувальна вибірка, quant_pre_process, quantize_static (per_channel=True)."""
    print("\n=== Пункт 2. Калібрування й квантизація ===")
    calib = calibration_images(lab)

    t0 = time.perf_counter()
    quant.prepare(lab.fp32, lab.prep)
    prep_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    quant.quantize(lab.prep, lab.int8, calib, per_channel=True)
    quant_s = time.perf_counter() - t0

    lab.results["quantize"] = {
        "file": lab.int8.name, "pre_process_time_s": prep_s, "quantize_static_time_s": quant_s,
        "file_mb": file_mb(lab.int8), "op_counts": model_tools.count_ops(lab.int8),
    }
    lab.save()
    print(f"quant_pre_process: {prep_s:.2f} с; quantize_static: {quant_s:.2f} с; "
          f"{lab.int8.name}: {file_mb(lab.int8):.2f} МБ")


def fp32_test_logits(lab):
    return model_tools.onnx_logits(model_tools.make_session(lab.fp32), lab.test_images())


def step_quality(lab):
    """Пункт 3: macro-F1, accuracy, згода з FP32, b, c, Δacc ± SE."""
    print("\n=== Пункт 3. Якість ===")
    z_fp32 = fp32_test_logits(lab)
    z_int8 = model_tools.onnx_logits(model_tools.make_session(lab.int8), lab.test_images())
    np.save(lab.results_dir / "test_logits_fp32.npy", z_fp32)
    np.save(lab.results_dir / "test_logits_int8.npy", z_int8)

    fp32_row = metrics.quality_row(z_fp32, z_fp32, lab.targets)
    int8_row = metrics.quality_row(z_int8, z_fp32, lab.targets)
    n = len(lab.targets)
    se_single = metrics.single_model_se(fp32_row["accuracy"], n)

    # Діагностика (не частина таблиці): той самий lab4_int8.onnx без оптимізацій графа, тобто без
    # цілочисельних ядер. Якщо згода тут помітно вища, втрату дає цілочисельне множення на цьому CPU
    # (насичення на x86 без VNNI), а не калібрування.
    session_sim = model_tools.make_session(lab.int8, disable_graph_optimizations=True)
    z_sim = model_tools.onnx_logits(session_sim, lab.test_images())
    sim_row = metrics.quality_row(z_sim, z_fp32, lab.targets)

    lab.results["quality"] = {"n": n, "fp32": fp32_row, "int8": int8_row,
                              "se_single_model_fp32": se_single,
                              "se_ratio_single_to_paired": se_single / int8_row["se_delta_acc"]
                              if int8_row["se_delta_acc"] > 0 else math.inf,
                              "max_abs_logit_diff_int8_vs_fp32": float(np.max(np.abs(z_int8 - z_fp32))),
                              "diagnostic_int8_qdq_simulated_in_float": {
                                  **sim_row,
                                  "max_abs_logit_diff_vs_fp32": float(np.max(np.abs(z_sim - z_fp32)))}}
    lab.save()
    for name, row in (("FP32", fp32_row), ("INT8", int8_row)):
        print(f"{name}: macro-F1 {row['macro_f1']:.4f}, accuracy {row['accuracy']:.4f}, "
              f"згода {row['agreement']:.4f}, b = {row['b']}, c = {row['c']}, "
              f"Δacc = {row['delta_acc']:+.4f} ± {row['se_delta_acc']:.4f}, "
              f"|b − c| = {abs(row['b'] - row['c'])} vs 2√(b + c) = {row['noise_threshold_2sqrt_bc']:.2f} "
              f"→ відрізняється від шуму: {row['differs_from_noise']}")
    print(f"SE однієї моделі √(p(1 − p)/n) = {se_single:.4f}")
    print(f"Діагностика: INT8 без оптимізацій графа (QDQ у float): згода {sim_row['agreement']:.4f}, "
          f"accuracy {sim_row['accuracy']:.4f}, macro-F1 {sim_row['macro_f1']:.4f}, "
          f"b = {sim_row['b']}, c = {sim_row['c']}")


def step_ranges(lab):
    """Пункт 4, перша частина: діапазони каналів ваг кожного Conv графа FP32."""
    print("\n=== Пункт 4. Діапазони каналів ===")
    convs = quant.channel_ranges(lab.fp32)
    top3 = sorted(convs, key=lambda c: c["r"], reverse=True)[:3]
    median_r = float(np.median([c["r"] for c in convs]))
    lab.results["channel_ranges"] = {"convs": convs, "top3": top3, "n_conv": len(convs),
                                     "median_r": median_r, "median_log2_r": float(np.log2(median_r))}
    lab.save()
    plots.channel_ranges(convs, median_r, lab.results_dir / "channel_ranges.png")
    for c in top3:
        print(f"  r = {c['r']:.2f} ({c['log2_r']:.2f} біта)  {c['name']}  {c['shape']}")
    print(f"Медіана r за {len(convs)} згортками = {median_r:.2f}, log2 = {np.log2(median_r):.2f} біта")


def step_per_tensor(lab):
    """Пункт 4, запуск: per_channel=False, решта як у пункті 2; рядок таблиці пункту 3."""
    print("\n=== Пункт 4. INT8 з потензорними вагами ===")
    calib = calibration_images(lab)
    if not lab.prep.exists():
        quant.prepare(lab.fp32, lab.prep)
    t0 = time.perf_counter()
    quant.quantize(lab.prep, lab.int8_pt, calib, per_channel=False)
    quant_s = time.perf_counter() - t0

    z_fp32 = fp32_test_logits(lab)
    z_pt = model_tools.onnx_logits(model_tools.make_session(lab.int8_pt), lab.test_images())
    np.save(lab.results_dir / "test_logits_int8_per_tensor.npy", z_pt)
    row = metrics.quality_row(z_pt, z_fp32, lab.targets)
    lab.results["per_tensor"] = {"file": lab.int8_pt.name, "quantize_static_time_s": quant_s,
                                 "file_mb": file_mb(lab.int8_pt), "quality": row}
    lab.save()
    print(f"INT8 per-tensor: macro-F1 {row['macro_f1']:.4f}, accuracy {row['accuracy']:.4f}, "
          f"згода {row['agreement']:.4f}, b = {row['b']}, c = {row['c']}, "
          f"Δacc = {row['delta_acc']:+.4f} ± {row['se_delta_acc']:.4f}, "
          f"|b − c| = {abs(row['b'] - row['c'])} vs 2√(b + c) = {row['noise_threshold_2sqrt_bc']:.2f} "
          f"→ відрізняється від шуму: {row['differs_from_noise']}")


def step_latency(lab):
    """Пункт 5: протокол затримки для lab4_fp32.onnx і lab4_int8.onnx."""
    print("\n=== Пункт 5. Затримка ===")
    # 1. Обидві сесії — заздалегідь.
    sessions = {"fp32": model_tools.make_session(lab.fp32), "int8": model_tools.make_session(lab.int8)}
    # 2. Вхід: перше зображення test у порядку split.csv, float32 (1, 3, H, W), підготовлений заздалегідь.
    x = data.load_images(lab.data_root, lab.test_items[:1], "test", lab.img_size)
    x = np.ascontiguousarray(x, dtype=np.float32)
    # 3–4. Три раунди × (20 прогрівальних + 100 вимірів), спершу FP32, потім INT8.
    rounds = latency.run_protocol(sessions, x)

    variants = {}
    for name, path in (("fp32", lab.fp32), ("int8", lab.int8)):
        variants[name] = {"file": path.name, "file_mb": file_mb(path), **latency.summarize(rounds[name]),
                          "all_ms_by_round": rounds[name]}
    overlap = latency.intervals_overlap(variants["fp32"]["p50_by_round_ms"], variants["int8"]["p50_by_round_ms"])
    lab.results["latency"] = {
        "protocol": {"batch_size": 1, "threads": 1, "rounds": latency.ROUNDS,
                     "warmup": latency.WARMUP, "measures": latency.MEASURES,
                     "input_file": lab.test_items[0][0], "input_shape": list(x.shape)},
        "variants": variants,
        "speedup_p50": variants["fp32"]["p50_ms"] / variants["int8"]["p50_ms"],
        "ratio_p95": variants["fp32"]["p95_ms"] / variants["int8"]["p95_ms"],
        "size_ratio": variants["fp32"]["file_mb"] / variants["int8"]["file_mb"],
        "round_intervals_overlap": overlap,
        "change_is_measured": not overlap,
    }
    lab.results["environment"] = environment.describe(threads=1, batch_size=1)
    lab.save()
    plots.latency(rounds, lab.results_dir / "latency.png")
    for name, v in variants.items():
        print(f"{name}: {v['file_mb']:.2f} МБ, p50 {v['p50_ms']:.3f} мс, p95 {v['p95_ms']:.3f} мс, "
              f"p50 за раундами {[round(p, 3) for p in v['p50_by_round_ms']]}")
    print(f"Прискорення p50 = {lab.results['latency']['speedup_p50']:.2f}; "
          f"відрізки перетинаються: {overlap} → зміна виміряна: {not overlap}")
    print(f"Середовище: {lab.results['environment']}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # українські літери й символи в консолі Windows
    args = parse_args()
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        raise SystemExit(f"Невідомі кроки: {unknown}; можливі: {STEPS}")
    lab = Lab(args)
    lab.save()
    actions = {"export": step_export, "quantize": step_quantize, "quality": step_quality,
               "ranges": step_ranges, "per_tensor": step_per_tensor, "latency": step_latency}
    for name in STEPS:  # порядок кроків фіксований, незалежно від порядку в --steps
        if name in steps:
            actions[name](lab)
    print(f"\nРезультати: {lab.json_path}")


if __name__ == "__main__":
    main()
