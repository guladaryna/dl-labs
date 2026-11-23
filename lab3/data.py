"""Поділ Intel Image Classification, перетворення і завантажувачі (розділи «Поділ», «Перетворення»)."""

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
PARTS = ["train", "validation", "test"]
IMG_SIZE = 160
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
N_TRAIN_PER_CLASS = 200

# Контрольна таблиця з умови: (клас, частина) -> (кількість, сума числових імен файлів)
CONTROL_TABLE = {
    ("buildings", "train"): (200, 1996761),
    ("buildings", "validation"): (218, 4837323),
    ("buildings", "test"): (219, 4837559),
    ("forest", "train"): (200, 2032625),
    ("forest", "validation"): (237, 5276962),
    ("forest", "test"): (237, 5291326),
    ("glacier", "train"): (200, 2183923),
    ("glacier", "validation"): (276, 6135652),
    ("glacier", "test"): (277, 6157250),
    ("mountain", "train"): (200, 1950345),
    ("mountain", "validation"): (262, 5789535),
    ("mountain", "test"): (263, 5849765),
    ("sea", "train"): (200, 1987613),
    ("sea", "validation"): (255, 5655068),
    ("sea", "test"): (255, 5668011),
    ("street", "train"): (200, 1841817),
    ("street", "validation"): (250, 5541845),
    ("street", "test"): (251, 5547533),
}
CONTROL_TOTAL = {
    "train": (1200, 11993084),
    "validation": (1498, 33236385),
    "test": (1502, 33351444),
}


def train_dir(data_root):
    return Path(data_root) / "seg_train" / "seg_train"


def test_dir(data_root):
    return Path(data_root) / "seg_test" / "seg_test"


def sorted_files(class_dir):
    """Імена файлів класу, відсортовані як рядки (як sorted() і ImageFolder): 0.jpg, 10006.jpg, 1001.jpg."""
    return sorted(p.name for p in class_dir.iterdir() if p.is_file())


def build_split(data_root):
    """Поділ за процедурою умови. Повертає список рядків (file, label, part).

    Порядок рядків: train (класи по черзі), потім validation, потім test;
    усередині класу — у порядку перемішаних індексів idx.
    """
    rows = []

    # Крок 2: ОДИН генератор на всі класи seg_train, класи у фіксованому порядку.
    rng = np.random.default_rng(0)
    for label in CLASSES:
        files = sorted_files(train_dir(data_root) / label)
        idx = np.arange(len(files))
        rng.shuffle(idx)
        for i in idx[:N_TRAIN_PER_CLASS]:
            rows.append((files[i], label, "train"))

    # Крок 3: НОВИЙ генератор default_rng(0) для seg_test, теж один на всі класи.
    rng = np.random.default_rng(0)
    validation_rows = []
    test_rows = []
    for label in CLASSES:
        files = sorted_files(test_dir(data_root) / label)
        idx = np.arange(len(files))
        rng.shuffle(idx)
        cut = len(files) // 2
        for i in idx[:cut]:
            validation_rows.append((files[i], label, "validation"))
        for i in idx[cut:]:
            test_rows.append((files[i], label, "test"))

    return rows + validation_rows + test_rows


def save_split(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "label", "part"])
        writer.writerows(rows)


def load_split(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [(r["file"], r["label"], r["part"]) for r in csv.DictReader(f)]


def split_table(rows):
    """(клас, частина) -> (кількість, сума числових імен); 20057.jpg -> 20057."""
    table = {(label, part): [0, 0] for label in CLASSES for part in PARTS}
    for file, label, part in rows:
        cell = table[(label, part)]
        cell[0] += 1
        cell[1] += int(Path(file).stem)
    return {key: tuple(value) for key, value in table.items()}


def check_split(rows):
    """Звіряє всі 18 клітинок і підсумки з умовою, перевіряє, що жоден файл не в двох частинах.

    Повертає список рядків таблиці для звіту; якщо щось не збігається — зупиняє програму.
    """
    table = split_table(rows)
    report = []
    all_ok = True
    for label in CLASSES:
        for part in PARTS:
            got = table[(label, part)]
            expected = CONTROL_TABLE[(label, part)]
            ok = got == expected
            all_ok = all_ok and ok
            report.append({"label": label, "part": part, "count": got[0], "sum": got[1],
                           "expected_count": expected[0], "expected_sum": expected[1], "match": ok})
    for part in PARTS:
        got = (sum(table[(label, part)][0] for label in CLASSES),
               sum(table[(label, part)][1] for label in CLASSES))
        expected = CONTROL_TOTAL[part]
        ok = got == expected
        all_ok = all_ok and ok
        report.append({"label": "Разом", "part": part, "count": got[0], "sum": got[1],
                       "expected_count": expected[0], "expected_sum": expected[1], "match": ok})

    # Жоден файл не входить у дві частини (імена в наборі унікальні).
    files = [file for file, _, _ in rows]
    no_overlap = len(files) == len(set(files))

    if not all_ok or not no_overlap:
        for r in report:
            if not r["match"]:
                print("НЕ ЗБІГАЄТЬСЯ:", r)
        raise SystemExit(f"Поділ не збігається з контрольною таблицею (без перетинів: {no_overlap})")
    return report, no_overlap


def print_split_report(report, no_overlap):
    print(f"{'Клас':<10} {'train':>20} {'validation':>20} {'test':>20}")
    for label in CLASSES + ["Разом"]:
        cells = [r for r in report if r["label"] == label]
        text = [f"{c['count']} / {c['sum']}" + ("" if c["match"] else " (!)") for c in cells]
        print(f"{label:<10} {text[0]:>20} {text[1]:>20} {text[2]:>20}")
    print(f"Усі 21 значення (18 клітинок + 3 підсумки) збігаються з умовою: "
          f"{all(r['match'] for r in report)}; файлів у двох частинах немає: {no_overlap}")


def train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def eval_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class SplitDataset(torch.utils.data.Dataset):
    """Зображення однієї частини split.csv у порядку рядків файлу.

    Картинки декодуються один раз і тримаються в пам'яті (PIL, RGB); перетворення —
    при кожному зверненні, тож випадкові аугментації train щоразу нові.
    """

    def __init__(self, data_root, rows, part, transform):
        folder = test_dir(data_root) if part in ("validation", "test") else train_dir(data_root)
        self.items = [(file, label) for file, label, p in rows if p == part]
        self.images = []
        for file, label in self.items:
            with Image.open(folder / label / file) as img:
                self.images.append(img.convert("RGB"))
        self.targets = [CLASSES.index(label) for _, label in self.items]
        self.files = [file for file, _ in self.items]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.transform(self.images[i]), self.targets[i]


def train_loader(dataset):
    # Без власного generator: перемішування бере випадковість із глобального генератора PyTorch,
    # засіяного torch.manual_seed(seed) на кроці 1 (розділ «Цикл навчання»).
    return torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, drop_last=False,
                                       num_workers=0)


def eval_loader(dataset):
    return torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)


def unnormalize(images):
    """Обернене нормування (для рисунка): x * std + mean, обрізане до [0, 1]."""
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0, 1)
