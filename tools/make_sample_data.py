"""Генератор демонстрационных данных для практической работы.

Скрипт создаёт набор данных для обучения перцептрона (CSV) и набор
изображений для второй и третьей частей. Пути берутся из аргументов
командной строки, значения по умолчанию - из переменной окружения
``NN_DATA_DIR``.

Примеры запуска::

    python tools/make_sample_data.py --kind csv
    python tools/make_sample_data.py --kind images --count 48
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

DEFAULT_DATA_DIR = Path(
    os.getenv("NN_DATA_DIR", Path(__file__).resolve().parents[1] / "data" / "input")
)
HEADER = ["income", "age", "debt_ratio", "credit_history", "approved"]
IMAGE_SIDE = 64


def build_csv(target: Path, rows: int, seed: int) -> Path:
    """Создаёт CSV с заявками на кредит.

    :param target: путь к будущему файлу.
    :param rows: число примеров.
    :param seed: зерно генератора для воспроизводимости.
    :return: путь к созданному файлу.
    """
    random = np.random.default_rng(seed)
    income = random.normal(85.0, 30.0, rows).clip(25.0, 220.0)
    age = random.normal(38.0, 9.0, rows).clip(21.0, 65.0)
    debt_ratio = random.beta(2.0, 3.0, rows).clip(0.02, 0.85)
    history = random.gamma(3.0, 2.0, rows).clip(0.0, 20.0)
    score = (
        0.035 * (income - 85.0)
        - 5.5 * (debt_ratio - 0.35)
        + 0.22 * (history - 6.0)
        + 0.04 * (age - 38.0)
    )
    probability = 1.0 / (1.0 + np.exp(-score))
    approved = (probability > 0.5).astype(int)
    flipped = random.random(rows) < 0.06
    approved[flipped] = 1 - approved[flipped]
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(HEADER)
        for index in range(rows):
            writer.writerow(
                [
                    f"{income[index]:.1f}",
                    f"{age[index]:.0f}",
                    f"{debt_ratio[index]:.3f}",
                    f"{history[index]:.1f}",
                    int(approved[index]),
                ]
            )
    return target


def build_examples(target: Path) -> Path:
    """Создаёт небольшой CSV с примерами для предсказания (без заголовка)."""
    rows = [
        [118.0, 41.0, 0.18, 11.0],
        [42.0, 24.0, 0.72, 1.0],
        [76.0, 33.0, 0.41, 5.5],
    ]
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)
    return target


def build_images(
    target_dir: Path, count: int, seed: int, side: int = IMAGE_SIDE
) -> int:
    """Рисует простые изображения трёх классов с шумом.

    :param target_dir: каталог, куда сохраняются PNG-файлы.
    :param count: число изображений.
    :param seed: зерно генератора.
    :param side: сторона квадратного изображения в пикселях.
    :return: фактическое число созданных файлов.
    """
    random = np.random.default_rng(seed)
    target_dir.mkdir(parents=True, exist_ok=True)
    margin = max(2, side // 6)
    shift = max(2, side // 10)
    for index in range(count):
        label = index % 3
        noise = random.integers(0, 40, (side, side, 3), dtype=np.uint8)
        image = Image.fromarray(noise + 60)
        drawer = ImageDraw.Draw(image)
        offset_x = int(random.integers(-shift, shift + 1))
        offset_y = int(random.integers(-shift, shift + 1))
        box = (
            margin + offset_x,
            margin + offset_y,
            side - margin + offset_x,
            side - margin + offset_y,
        )
        if label == 0:
            drawer.ellipse(box, fill=(220, 70, 60))
        elif label == 1:
            drawer.rectangle(box, fill=(60, 140, 220))
        else:
            drawer.polygon(
                [
                    (side // 2 + offset_x, margin + offset_y),
                    (side - margin + offset_x, side - margin + offset_y),
                    (margin + offset_x, side - margin + offset_y),
                ],
                fill=(80, 190, 110),
            )
        image.save(target_dir / f"img_{index + 1:03d}.png", format="PNG")
    return count


def build_tables(target_dir: Path, count: int, rows: int, seed: int) -> int:
    """Создаёт несколько таблиц одинаковой формы для проверки CSV-ветки.

    :param target_dir: каталог, куда сохраняются файлы.
    :param count: число таблиц.
    :param rows: число строк в каждой таблице.
    :param seed: зерно генератора.
    :return: фактическое число созданных файлов.
    """
    random = np.random.default_rng(seed + 1)
    target_dir.mkdir(parents=True, exist_ok=True)
    header = ["step", "voltage", "current", "temperature"]
    for index in range(count):
        steps = np.arange(rows)
        voltage = 12.0 + 0.4 * steps + random.normal(0.0, 0.2, rows)
        current = 2.5 + 0.05 * steps + random.normal(0.0, 0.15, rows)
        temperature = 36.0 + 0.8 * steps + random.normal(0.0, 0.5, rows)
        path = target_dir / f"table_{index + 1:02d}.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            for row in range(rows):
                writer.writerow(
                    [
                        int(steps[row]),
                        f"{voltage[row]:.3f}",
                        f"{current[row]:.3f}",
                        f"{temperature[row]:.2f}",
                    ]
                )
    return count


def main() -> int:
    """Разбирает аргументы командной строки и создаёт данные."""
    parser = argparse.ArgumentParser(description="Генерация тестовых данных")
    parser.add_argument(
        "--kind", choices=["csv", "images", "tables", "examples", "all"], default="all"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--rows", type=int, default=420)
    parser.add_argument("--count", type=int, default=48)
    parser.add_argument("--side", type=int, default=IMAGE_SIDE,
                        help="сторона изображения в пикселях")
    parser.add_argument("--images-name", default="images",
                        help="имя подкаталога с изображениями")
    parser.add_argument("--seed", type=int, default=7)
    arguments = parser.parse_args()

    if arguments.kind in {"csv", "all"}:
        path = build_csv(arguments.out / "mlp_dataset.csv", arguments.rows, arguments.seed)
        print(f"CSV: {path} ({arguments.rows} строк)")
    if arguments.kind in {"examples", "all"}:
        path = build_examples(arguments.out / "examples.csv")
        print(f"Примеры: {path}")
    if arguments.kind in {"tables", "all"}:
        created = build_tables(
            arguments.out / "tables", count=4, rows=12, seed=arguments.seed
        )
        print(f"Таблицы: {arguments.out / 'tables'} ({created} шт.)")
    if arguments.kind in {"images", "all"}:
        created = build_images(
            arguments.out / arguments.images_name,
            arguments.count,
            arguments.seed,
            arguments.side,
        )
        print(f"Изображения: {arguments.out / arguments.images_name} ({created} шт.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
