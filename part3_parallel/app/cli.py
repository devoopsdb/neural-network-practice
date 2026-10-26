"""Командная строка третьей части: потоки чтения и процессы аугментации.

Пример запуска::

    python -m app.cli --path ../data/input/aug_images --ext .png \\
        --output augmented.npy --report performance_report.json
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from .performance import benchmark_augmentation, compare, save_dataset, save_report
from .producer_consumer import PipelineError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-20s | %(message)s"


def _setup_logging(level: str, log_file: Path | None) -> logging.Logger:
    """Настраивает логгер приложения с выводом в консоль и файл."""
    numeric = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger("parallel")
    logger.setLevel(numeric)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter(LOG_FORMAT, "%H:%M:%S")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(numeric)
    logger.addHandler(console)
    if log_file is not None:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
        except OSError as error:
            logger.warning("Файл журнала недоступен: %s", error)
        else:
            file_handler.setFormatter(formatter)
            file_handler.setLevel(numeric)
            logger.addHandler(file_handler)
    return logger


def _env_int(name: str, default: int) -> int:
    """Читает целое число из переменной окружения."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as error:
        raise PipelineError(
            f"Переменная окружения {name} должна быть целым числом, а не '{raw}'."
        ) from error


def build_parser() -> argparse.ArgumentParser:
    """Создаёт парсер аргументов командной строки."""
    default_path = Path(
        os.getenv("NN_AUG_PATH", PROJECT_ROOT / "data" / "input" / "aug_images")
    )
    default_output = Path(os.getenv("NN_OUTPUT_DIR", PROJECT_ROOT / "data" / "output"))
    parser = argparse.ArgumentParser(
        prog="parallel",
        description="Многопоточное чтение и многопроцессорная аугментация",
    )
    parser.add_argument("--path", type=Path, default=default_path,
                        help="каталог с изображениями")
    parser.add_argument("--ext", nargs="+", default=[".png"],
                        help="расширения файлов, например .png .jpg")
    parser.add_argument("--output", default="augmented.npy",
                        help="файл для аугментированного набора")
    parser.add_argument("--report", default="performance_report.json",
                        help="файл отчёта о производительности")
    parser.add_argument("--workers", type=int, default=_env_int("NN_WORKERS", 4),
                        help="число потоков чтения")
    parser.add_argument("--processes", type=int, default=_env_int("NN_PROCESSES", 4),
                        help="число процессов аугментации")
    parser.add_argument("--log-level", default=os.getenv("NN_LOG_LEVEL", "INFO"))
    parser.add_argument("--log-file", type=Path,
                        default=PROJECT_ROOT / "logs" / "parallel.log",
                        help="файл журнала")
    parser.add_argument("--output-dir", type=Path, default=default_output,
                        help="каталог для результатов")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Точка входа: сравнивает последовательную и параллельную обработку."""
    arguments = build_parser().parse_args(argv)
    logger = _setup_logging(arguments.log_level, arguments.log_file)
    extensions = tuple(
        ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        for ext in arguments.ext
    )
    logger.debug("Аргументы: %s", arguments)
    try:
        result, dataset = compare(
            directory=arguments.path,
            extensions=extensions,
            workers=arguments.workers,
            processes=arguments.processes,
            logger=logger,
        )
    except (PipelineError, ValueError) as error:
        logger.error("Обработка не выполнена: %s", error)
        return 2

    dataset_path = save_dataset(dataset, arguments.output_dir / arguments.output)
    logger.info("Замер чистой аугментации без чтения файлов")
    augmentation = benchmark_augmentation(dataset, processes=arguments.processes)
    report = result.as_dict(arguments.path)
    report["augmentation_only"] = augmentation
    payload = save_report(report, arguments.output_dir / arguments.report)
    print()
    print("Сравнение производительности:")
    print(f"  файлов:               {result.sequential.files}")
    print(f"  форма результата:     {result.shape}")
    print(f"  последовательно:      {result.sequential.seconds:.3f} с")
    print(f"  параллельно:          {result.parallel.seconds:.3f} с "
          f"({result.workers} потоков, {result.processes} процессов)")
    print(f"  в том числе пул:      {result.parallel.pool_seconds:.3f} с")
    print(f"  только обработка:     {result.parallel.processing_seconds:.3f} с")
    print(f"  ускорение (полное):   {result.speedup_total:.2f}x")
    print(f"  ускорение обработки:  {result.speedup_processing:.2f}x")
    print(f"  чистая аугментация:   {augmentation['sequential_seconds']:.3f} с -> "
          f"{augmentation['parallel_seconds']:.3f} с "
          f"({augmentation['speedup']:.2f}x)")
    print(f"  отчёт:                {payload}")
    print(f"  данные:               {dataset_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
