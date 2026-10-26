"""Командная строка утилиты подготовки данных.

Подкоманды::

    python -m app.cli prepare --path data/input/images --ext .png \
        --output images_dataset.npy
    python -m app.cli doctor
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from .config import PROJECT_ROOT, Settings, _parse_extensions, _parse_image_size
from .logging_config import setup_logging
from .scanner import DataScanner, ScannerError

LIBRARIES = ("numpy", "PIL", "dotenv", "matplotlib", "tensorflow", "torch")
CUDA_TIMEOUT = 10.0


def build_parser() -> argparse.ArgumentParser:
    """Создаёт парсер аргументов с подкомандами prepare и doctor."""
    parser = argparse.ArgumentParser(
        prog="prepare",
        description="Подготовка данных для нейронной сети",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="собрать единый массив из файлов каталога"
    )
    prepare.add_argument("--path", type=Path, default=None, help="каталог с данными")
    prepare.add_argument(
        "--ext", nargs="+", default=None, help="расширения файлов, например .png .jpg"
    )
    prepare.add_argument("--output", default=None, help="имя или путь файла .npy")
    prepare.add_argument("--resize", default=None, help="размер изображения, например 64x64")
    prepare.add_argument("--log-level", default=None, help="уровень логирования")
    prepare.add_argument("--log-file", type=Path, default=None, help="файл журнала")

    doctor = subparsers.add_parser("doctor", help="проверить окружение")
    doctor.add_argument("--path", type=Path, default=None, help="каталог для проверки")
    doctor.add_argument("--log-level", default=None, help="уровень логирования")
    return parser


def _settings_from_args(args: argparse.Namespace) -> Settings:
    """Собирает настройки: значения по умолчанию, окружение, аргументы CLI."""
    settings = Settings.from_environment()
    overrides: dict[str, object] = {"log_level": args.log_level}
    if args.path is not None:
        overrides["data_path"] = args.path.expanduser()
    if getattr(args, "ext", None):
        overrides["extensions"] = _parse_extensions(" ".join(args.ext))
    if getattr(args, "resize", None):
        overrides["image_size"] = _parse_image_size(args.resize)
    return settings.with_overrides(**overrides)


def run_prepare(args: argparse.Namespace) -> int:
    """Выполняет подкоманду prepare и печатает статистику."""
    settings = _settings_from_args(args)
    log_file = args.log_file or (PROJECT_ROOT / "logs" / "prepare.log")
    logger = setup_logging(settings.log_level, log_file)
    output = settings.resolve_output(args.output)
    logger.debug("Настройки: путь=%s, расширения=%s, размер=%s",
                 settings.data_path, settings.extensions, settings.image_size)
    scanner = DataScanner(settings.image_size, logger)
    try:
        result = scanner.prepare(settings.data_path, settings.extensions, output)
    except ScannerError as error:
        logger.error("Ошибка подготовки данных: %s", error)
        return 2

    array = np.load(result.output, mmap_mode="r")
    print()
    print("Статистика подготовки:")
    print(f"  каталог:          {settings.data_path}")
    print(f"  расширения:       {', '.join(settings.extensions)}")
    print(f"  файлов найдено:   {len(result.files)}")
    print(f"  файлов пропущено: {len(result.skipped)}")
    print(f"  форма массива:    {result.shape}")
    print(f"  тип данных:       {result.dtype}")
    print(f"  минимум:          {float(array.min()):.4f}")
    print(f"  максимум:         {float(array.max()):.4f}")
    print(f"  время обработки:  {result.elapsed:.3f} с")
    print(f"  сохранено:        {result.output}")
    print(f"  размер файла:     {result.output.stat().st_size / 1024:.1f} КБ")

    meta_path = result.output.with_suffix(".json")
    try:
        meta_path.write_text(
            json.dumps(result.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError as error:
        logger.warning("Не удалось записать описание набора: %s", error)
    else:
        print(f"  описание:         {meta_path}")
    return 0


def _library_status(name: str) -> tuple[bool, str]:
    """Проверяет, установлена ли библиотека, и возвращает её версию."""
    if importlib.util.find_spec(name) is None:
        return False, "не установлена"
    try:
        module = __import__(name)
    except ImportError:
        return False, "не импортируется"
    return True, str(getattr(module, "__version__", "версия неизвестна"))


def _cuda_status(timeout: float = CUDA_TIMEOUT) -> tuple[bool, str]:
    """Спрашивает у nvidia-smi список видеокарт через subprocess."""
    command = ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return False, "nvidia-smi не найден в PATH"
    except subprocess.TimeoutExpired:
        return False, f"nvidia-smi не ответил за {timeout:.0f} с"
    except OSError as error:
        return False, f"не удалось запустить nvidia-smi: {error}"
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout).strip()
        return False, message.splitlines()[0] if message else "nvidia-smi вернул ошибку"
    devices = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not devices:
        return False, "устройства не найдены"
    return True, "; ".join(devices)


def run_doctor(args: argparse.Namespace) -> int:
    """Выполняет подкоманду doctor: проверяет окружение и зависимости."""
    settings = _settings_from_args(args)
    logger = setup_logging(settings.log_level)
    target_dir = Path(args.path).expanduser() if args.path else settings.output_dir
    print("Проверка окружения")
    print("-" * 60)
    print(f"Python:        {platform.python_version()} ({platform.python_implementation()})")
    print(f"Исполняемый:   {sys.executable}")
    print(f"Платформа:     {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Рабочий каталог: {Path.cwd()}")
    print()
    print("Библиотеки:")
    critical_missing = []
    for name in LIBRARIES:
        available, version = _library_status(name)
        mark = "OK  " if available else "--  "
        print(f"  [{mark}] {name:<12} {version}")
        if not available and name in {"numpy", "PIL"}:
            critical_missing.append(name)
    print()
    cuda_available, cuda_message = _cuda_status()
    print(f"CUDA:          {'доступна' if cuda_available else 'недоступна'} - {cuda_message}")
    print()
    print("Диск и каталоги:")
    try:
        usage = shutil.disk_usage(target_dir if target_dir.exists() else target_dir.parent)
        print(f"  свободно на диске: {usage.free / 1024 ** 3:.1f} ГБ из "
              f"{usage.total / 1024 ** 3:.1f} ГБ")
    except OSError as error:
        print(f"  не удалось оценить диск: {error}")
    print(f"  каталог данных:    {settings.data_path} "
          f"({'есть' if settings.data_path.exists() else 'нет'})")
    try:
        writable = os.access(
            target_dir if target_dir.exists() else target_dir.parent, os.W_OK
        )
    except OSError as error:
        writable = False
        logger.warning("Не удалось проверить права на запись: %s", error)
    print(f"  каталог вывода:    {target_dir} "
          f"({'запись разрешена' if writable else 'нет записи'})")
    print()
    if critical_missing:
        logger.error("Не хватает обязательных библиотек: %s", ", ".join(critical_missing))
        return 1
    print("Итог: окружение готово к работе.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Точка входа командной строки."""
    arguments = build_parser().parse_args(argv)
    if arguments.command == "prepare":
        return run_prepare(arguments)
    return run_doctor(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
