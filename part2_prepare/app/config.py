"""Настройки утилиты подготовки данных.

Значения берутся из переменных окружения (по желанию - из файла .env),
а аргументы командной строки имеют приоритет над ними.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Расширения, которые считаются изображениями.
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
#: Расширения, которые считаются табличными данными.
TABLE_EXTENSIONS = (".csv", ".txt")


def _load_dotenv(dotenv_path: Path | None = None) -> None:
    """Подхватывает переменные из .env, если установлен python-dotenv."""
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:
        return
    if dotenv_path is not None and dotenv_path.is_file():
        load_dotenv(dotenv_path)
        return
    found = find_dotenv(usecwd=True)
    if found:
        load_dotenv(found)


def _parse_image_size(raw: str) -> tuple[int, int]:
    """Разбирает размер изображения вида ``64x64`` или ``64,64``."""
    prepared = raw.lower().replace("x", " ").replace(",", " ")
    parts = [item for item in prepared.split() if item]
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ValueError(f"Не удалось разобрать размер изображения '{raw}'")
    width, height = int(parts[0]), int(parts[1])
    if width < 1 or height < 1:
        raise ValueError("Стороны изображения должны быть положительными")
    return width, height


def _parse_extensions(raw: str) -> tuple[str, ...]:
    """Разбирает список расширений, нормализуя точки и регистр."""
    prepared = raw.replace(";", ",").replace(" ", ",")
    parts = [item.strip().lower() for item in prepared.split(",")]
    result = []
    for part in parts:
        if not part:
            continue
        result.append(part if part.startswith(".") else f".{part}")
    return tuple(result)


@dataclass(slots=True)
class Settings:
    """Настройки одного запуска утилиты.

    :param data_path: каталог с исходными данными по умолчанию.
    :param output_dir: каталог, куда складываются результаты.
    :param log_level: уровень логирования (``DEBUG``, ``INFO``, ``WARNING``).
    :param image_size: размер, к которому приводятся изображения.
    :param extensions: расширения файлов, которые нужно обрабатывать.
    :param output_file: имя файла с результатом внутри ``output_dir``.
    """

    data_path: Path = PROJECT_ROOT / "data" / "input"
    output_dir: Path = PROJECT_ROOT / "data" / "output"
    log_level: str = "INFO"
    image_size: tuple[int, int] = (64, 64)
    extensions: tuple[str, ...] = field(default_factory=lambda: IMAGE_EXTENSIONS)
    output_file: str = "dataset.npy"

    @classmethod
    def from_environment(cls, dotenv_path: Path | None = None) -> Settings:
        """Собирает настройки из переменных окружения и значений по умолчанию."""
        _load_dotenv(dotenv_path)
        defaults = cls()
        data_path = Path(os.getenv("NN_DATA_PATH", defaults.data_path)).expanduser()
        output_dir = Path(os.getenv("NN_OUTPUT_DIR", defaults.output_dir)).expanduser()
        log_level = os.getenv("NN_LOG_LEVEL", defaults.log_level).upper()
        raw_size = os.getenv("NN_IMAGE_SIZE")
        image_size = _parse_image_size(raw_size) if raw_size else defaults.image_size
        raw_extensions = os.getenv("NN_EXTENSIONS")
        extensions = (
            _parse_extensions(raw_extensions) if raw_extensions else defaults.extensions
        )
        output_file = os.getenv("NN_OUTPUT_FILE", defaults.output_file)
        return cls(
            data_path=data_path,
            output_dir=output_dir,
            log_level=log_level,
            image_size=image_size,
            extensions=extensions,
            output_file=output_file,
        )

    def with_overrides(self, **overrides: object) -> Settings:
        """Возвращает копию настроек с заменой указанных полей."""
        prepared = {key: value for key, value in overrides.items() if value is not None}
        return replace(self, **prepared)

    def resolve_output(self, output: str | Path | None) -> Path:
        """Определяет итоговый путь файла с массивом.

        Относительный путь отсчитывается от ``output_dir``, поэтому
        результат всегда попадает в каталог, заданный в настройках.
        """
        if output is None:
            return self.output_dir / self.output_file
        candidate = Path(output).expanduser()
        if candidate.is_absolute():
            return candidate
        return self.output_dir / candidate
