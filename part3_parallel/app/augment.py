"""Аугментация изображений: повороты, отражения и шум.

Все функции работают с массивами ``float32`` в диапазоне [0, 1]
и не меняют исходный массив.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

#: Операции, которые применяются к каждому изображению.
OPERATIONS = ("rotate", "flip", "noise")
DEFAULT_ANGLE = 15.0
DEFAULT_NOISE = 0.04


def to_image(array: np.ndarray) -> Image.Image:
    """Превращает массив [0, 1] в изображение PIL."""
    scaled = np.clip(array, 0.0, 1.0) * 255.0
    return Image.fromarray(scaled.astype(np.uint8))


def random_rotate(array: np.ndarray, angle: float) -> np.ndarray:
    """Поворачивает изображение на произвольный угол без обрезки углов."""
    rotated = to_image(array).rotate(
        angle, resample=Image.Resampling.BILINEAR, expand=False
    )
    return np.asarray(rotated, dtype=np.float32) / 255.0


def horizontal_flip(array: np.ndarray) -> np.ndarray:
    """Отражает изображение по горизонтали."""
    return np.ascontiguousarray(array[:, ::-1, :])


def add_noise(array: np.ndarray, sigma: float, generator: np.random.Generator) -> np.ndarray:
    """Добавляет гауссов шум и обрезает значения по диапазону [0, 1]."""
    noise = generator.normal(0.0, sigma, array.shape).astype(np.float32)
    return np.clip(array + noise, 0.0, 1.0)


def augment_image(
    array: np.ndarray,
    seed: int | None = None,
    operations: tuple[str, ...] = OPERATIONS,
) -> np.ndarray:
    """Применяет набор операций аугментации к одному изображению.

    :param array: изображение формы ``(height, width, channels)``.
    :param seed: зерно генератора, чтобы результат был воспроизводим.
    :param operations: какие операции применять.
    :return: новое изображение того же размера.
    """
    generator = np.random.default_rng(seed)
    result = np.array(array, dtype=np.float32, copy=True)
    if "rotate" in operations:
        angle = float(generator.uniform(-DEFAULT_ANGLE, DEFAULT_ANGLE))
        result = random_rotate(result, angle)
    if "flip" in operations and generator.random() < 0.5:
        result = horizontal_flip(result)
    if "noise" in operations:
        sigma = float(generator.uniform(0.01, DEFAULT_NOISE))
        result = add_noise(result, sigma, generator)
    return result


def augment_batch(
    arrays: np.ndarray,
    seed: int,
    operations: tuple[str, ...] = OPERATIONS,
) -> np.ndarray:
    """Аугментирует пачку изображений (используется в пуле процессов).

    :param arrays: массив формы ``(n, height, width, channels)``.
    :param seed: базовое зерно; для каждого изображения оно своё.
    :param operations: набор операций аугментации.
    :return: массив той же формы с преобразованными изображениями.
    """
    return np.stack(
        [
            augment_image(image, seed=seed + index, operations=operations)
            for index, image in enumerate(arrays)
        ]
    )


def from_uint8(array: np.ndarray) -> np.ndarray:
    """Переводит массив uint8 в float32 с диапазоном [0, 1]."""
    return array.astype(np.float32) / 255.0


def to_uint8(array: np.ndarray) -> np.ndarray:
    """Переводит массив [0, 1] обратно в uint8."""
    return (np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
