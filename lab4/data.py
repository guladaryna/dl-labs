"""Поділ із ЛР3, контрольні числа, калібрувальна вибірка і перетворення (розділ «Дані й поділ»)."""

import csv
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import transforms

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Контроль поділу з умови: частина -> (кількість, сума числових імен файлів)
SPLIT_CONTROL = {"train": (1200, 11993084), "test": (1502, 33351444)}

# Контроль калібрувальної вибірки з умови (пункт 2)
CALIB_SIZE = 200
CALIB_NAME_SUM = 1921470
CALIB_PER_CLASS = {"buildings": 40, "forest": 35, "glacier": 36, "mountain": 33, "sea": 29, "street": 27}


def name_number(file):
    """20057.jpg -> 20057."""
    return int(Path(file).stem)


def load_split(path):
    """Рядки split.csv ЛР3 як список (file, label, part) у порядку файлу."""
    with open(path, newline="", encoding="utf-8") as f:
        return [(r["file"], r["label"], r["part"]) for r in csv.DictReader(f)]


def part_rows(rows, part):
    """Рядки однієї частини в порядку split.csv."""
    return [(file, label) for file, label, p in rows if p == part]


def check_split(rows):
    """Звіряє train і test із контролем умови (кількість і сума імен). Не збігається — зупинка."""
    report = {}
    for part, expected in SPLIT_CONTROL.items():
        items = part_rows(rows, part)
        got = (len(items), sum(name_number(file) for file, _ in items))
        report[part] = {"count": got[0], "name_sum": got[1],
                        "expected_count": expected[0], "expected_name_sum": expected[1],
                        "match": got == expected}
        print(f"Поділ {part}: {got[0]} / {got[1]}, очікується {expected[0]} / {expected[1]} "
              f"→ збіг: {got == expected}")
    if not all(r["match"] for r in report.values()):
        raise SystemExit("Поділ не збігається з контролем умови — виправити в ЛР3")
    return report


def calibration_rows(rows):
    """Калібрувальна вибірка за пунктом 2.

    Крок 1: 1200 рядків train, відсортованих за іменем файлу ЯК РЯДКИ (не як числа).
    Крок 2: idx = default_rng(0).permutation(1200)[:200], беремо рядки з номерами idx.
    """
    train = sorted(part_rows(rows, "train"), key=lambda item: item[0])
    idx = np.random.default_rng(0).permutation(len(train))[:CALIB_SIZE]
    return [train[i] for i in idx]


def check_calibration(calib):
    """Звіряє кількість, суму імен і розподіл за класами з умовою. Не збігається — зупинка."""
    per_class = {label: sum(1 for _, l in calib if l == label) for label in CLASSES}
    name_sum = sum(name_number(file) for file, _ in calib)
    ok = (len(calib) == CALIB_SIZE and name_sum == CALIB_NAME_SUM and per_class == CALIB_PER_CLASS)
    print(f"Калібрувальна вибірка: {len(calib)} зображень, сума імен {name_sum}, "
          f"за класами {per_class}")
    print(f"  очікується {CALIB_SIZE}, {CALIB_NAME_SUM}, {CALIB_PER_CLASS} → збіг: {ok}")
    if not ok:
        raise SystemExit("Калібрувальна вибірка не збігається з контролем умови")
    return {"count": len(calib), "name_sum": name_sum, "per_class": per_class, "match": ok}


def eval_transform(img_size):
    """Те саме перетворення, що для test у ЛР3: Resize -> ToTensor -> Normalize(ImageNet)."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def image_folder(data_root, part):
    """train лежить у seg_train/seg_train/<клас>/, test — у seg_test/seg_test/<клас>/."""
    if part == "train":
        return Path(data_root) / "seg_train" / "seg_train"
    return Path(data_root) / "seg_test" / "seg_test"


def load_images(data_root, items, part, img_size):
    """Масив float32 форми (N, 3, H, W) у порядку items (Pillow, .convert("RGB"))."""
    transform = eval_transform(img_size)
    folder = image_folder(data_root, part)
    arrays = []
    for file, label in items:
        with Image.open(folder / label / file) as img:
            arrays.append(transform(img.convert("RGB")).numpy())
    return np.stack(arrays).astype(np.float32)
