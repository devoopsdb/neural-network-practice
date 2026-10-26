"""Обход каталога, чтение файлов и сборка единого массива NumPy."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from .config import IMAGE_EXTENSIONS, TABLE_EXTENSIONS


class ScannerError(Exception):
    """Ошибка чтения или сборки данных."""


@dataclass(slots=True)
class PrepareResult:
    """Результат подготовки массива.

    :param output: путь к сохранённому файлу .npy.
    :param files: список обработанных файлов.
    :param shape: форма итогового массива.
    :param dtype: тип элементов массива.
    :param elapsed: время обработки в секундах.
    :param skipped: файлы, которые не удалось прочитать, и причина.
    """

    output: Path
    files: list[Path]
    shape: tuple[int, ...]
    dtype: str
    elapsed: float
    skipped: list[tuple[Path, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        """Превращает результат в словарь для сохранения рядом с массивом."""
        return {
            "output": str(self.output),
            "files": len(self.files),
            "shape": list(self.shape),
            "dtype": self.dtype,
            "elapsed_sec": round(self.elapsed, 4),
            "skipped": [
                {"file": str(path), "reason": reason} for path, reason in self.skipped
            ],
        }


class DataScanner:
    """Ищет файлы в каталоге и превращает их в один массив.

    :param image_size: размер, к которому приводятся изображения.
    :param logger: логгер, которому передаются сообщения о ходе работы.
    """

    def __init__(
        self,
        image_size: tuple[int, int] = (64, 64),
        logger: object | None = None,
    ) -> None:
        self.image_size = image_size
        self.logger = logger

    def _log(self, level: str, message: str) -> None:
        """Пишет сообщение в логгер, если он передан."""
        if self.logger is not None:
            getattr(self.logger, level)(message)

    # ------------------------------------------------------------------
    # обход каталога
    # ------------------------------------------------------------------
    def scan(self, directory: Path, extensions: tuple[str, ...]) -> list[Path]:
        """Рекурсивно собирает файлы с указанными расширениями.

        :param directory: корневой каталог поиска.
        :param extensions: нормализованные расширения (с точкой).
        :return: отсортированный список путей.
        """
        root = Path(directory).expanduser()
        if not root.exists():
            raise ScannerError(f"Каталог '{root}' не найден.")
        if not root.is_dir():
            raise ScannerError(f"'{root}' не является каталогом.")
        prepared = tuple(
            item.lower() if item.startswith(".") else f".{item.lower()}"
            for item in extensions
        )
        found = [
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in prepared
        ]
        found.sort(key=lambda item: str(item).lower())
        self._log("info", f"Найдено файлов: {len(found)} в каталоге {root}")
        return found

    # ------------------------------------------------------------------
    # чтение отдельного файла
    # ------------------------------------------------------------------
    def load(self, path: Path) -> np.ndarray:
        """Читает файл изображения или таблицы и возвращает массив.

        Изображения приводятся к размеру ``image_size`` и делятся на 255,
        чтобы значения попали в диапазон [0, 1].
        """
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            return self._load_image(path)
        if suffix in TABLE_EXTENSIONS:
            return self._load_table(path)
        raise ScannerError(f"Формат файла '{path.name}' не поддерживается.")

    def _load_image(self, path: Path) -> np.ndarray:
        """Читает изображение через PIL, приводит размер и масштаб."""
        try:
            with Image.open(path) as image:
                converted = image.convert("RGB").resize(self.image_size)
                pixels = np.asarray(converted, dtype=np.float32)
        except (OSError, ValueError, UnidentifiedImageError) as error:
            raise ScannerError(f"Не удалось прочитать изображение: {error}") from error
        return pixels / 255.0

    def _load_table(self, path: Path) -> np.ndarray:
        """Читает числовую таблицу, при необходимости пропуская заголовок."""
        for encoding in ("utf-8-sig", "cp1251"):
            try:
                rows = []
                with path.open("r", encoding=encoding) as stream:
                    for line in stream:
                        line = line.strip()
                        if not line:
                            continue
                        cells = [cell.strip().replace(",", ".") for cell in line.split(",")]
                        try:
                            rows.append([float(cell) for cell in cells])
                        except ValueError:
                            if rows:
                                raise ScannerError(
                                    f"Строка '{line}' в файле {path.name} не числовая."
                                ) from None
                            continue
            except UnicodeDecodeError:
                continue
            except OSError as error:
                raise ScannerError(f"Не удалось прочитать {path.name}: {error}") from error
            if not rows:
                raise ScannerError(f"В файле {path.name} нет числовых данных.")
            widths = {len(row) for row in rows}
            if len(widths) > 1:
                raise ScannerError(f"В файле {path.name} разное число столбцов.")
            return np.asarray(rows, dtype=np.float32)
        raise ScannerError(f"Не удалось определить кодировку файла {path.name}.")

    # ------------------------------------------------------------------
    # сборка массива
    # ------------------------------------------------------------------
    def build(self, files: list[Path]) -> tuple[np.ndarray, list[tuple[Path, str]]]:
        """Читает все файлы и складывает их в один массив.

        :param files: список путей, полученный из :meth:`scan`.
        :return: пара (массив, список пропущенных файлов).
        """
        if not files:
            raise ScannerError("В каталоге нет файлов с указанными расширениями.")
        arrays = []
        skipped: list[tuple[Path, str]] = []
        for index, path in enumerate(files, start=1):
            try:
                arrays.append(self.load(path))
            except ScannerError as error:
                skipped.append((path, str(error)))
                self._log("warning", f"Пропущен файл {path.name}: {error}")
                continue
            if index % 10 == 0 or index == len(files):
                self._log("info", f"Прочитано {index} из {len(files)}")
        if not arrays:
            raise ScannerError("Ни один файл не удалось прочитать.")
        shapes = {array.shape for array in arrays}
        if len(shapes) > 1:
            raise ScannerError(
                "Файлы дают массивы разной формы: "
                + ", ".join(str(shape) for shape in sorted(shapes))
            )
        stacked = np.stack(arrays)
        if all(path.suffix.lower() in TABLE_EXTENSIONS for path in files):
            stacked = self._minmax_columns(stacked)
        return stacked, skipped

    @staticmethod
    def _minmax_columns(array: np.ndarray) -> np.ndarray:
        """Приводит каждый столбец таблицы к диапазону [0, 1].

        Изображения нормализуются делением на 255 при чтении, а таблицы
        удобнее масштабировать по минимуму и максимуму столбца, иначе
        признаки вроде напряжения и температуры остаются в разных шкалах.
        """
        flat = array.reshape(-1, array.shape[-1]).astype(np.float32)
        low = flat.min(axis=0)
        high = flat.max(axis=0)
        span = np.where(high - low == 0.0, 1.0, high - low)
        return ((flat - low) / span).reshape(array.shape).astype(np.float32)

    @staticmethod
    def save(array: np.ndarray, target: Path) -> Path:
        """Сохраняет массив в формате .npy, создавая каталоги при необходимости."""
        destination = target if target.suffix == ".npy" else target.with_suffix(".npy")
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.save(destination, array)
        except OSError as error:
            raise ScannerError(f"Не удалось сохранить {destination}: {error}") from error
        return destination

    def prepare(
        self,
        directory: Path,
        extensions: tuple[str, ...],
        output: Path,
    ) -> PrepareResult:
        """Полный цикл: поиск файлов, чтение, сборка и сохранение массива."""
        started = time.perf_counter()
        files = self.scan(directory, extensions)
        array, skipped = self.build(files)
        saved = self.save(array, output)
        elapsed = time.perf_counter() - started
        return PrepareResult(
            output=saved,
            files=files,
            shape=tuple(int(size) for size in array.shape),
            dtype=str(array.dtype),
            elapsed=elapsed,
            skipped=skipped,
        )
