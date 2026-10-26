"""Сравнение последовательной и параллельной обработки данных."""

from __future__ import annotations

import json
import logging
import os
import platform
import time
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .augment import OPERATIONS, from_uint8, to_uint8
from .producer_consumer import (
    ParallelPipeline,
    PipelineError,
    PipelineStats,
    augment_chunk,
    run_sequential,
)


@dataclass(slots=True)
class ComparisonResult:
    """Результаты двух запусков и вычисленное ускорение.

    :param sequential: статистика последовательного запуска.
    :param parallel: статистика параллельного запуска.
    :param workers: число потоков чтения.
    :param processes: число процессов аугментации.
    :param shape: форма итогового массива.
    """

    sequential: PipelineStats
    parallel: PipelineStats
    workers: int
    processes: int
    shape: tuple[int, ...]

    @property
    def speedup_total(self) -> float:
        """Отношение полного времени последовательной и параллельной версий."""
        return self.sequential.seconds / max(self.parallel.seconds, 1e-9)

    @property
    def speedup_processing(self) -> float:
        """Ускорение без времени создания пула процессов."""
        return self.sequential.processing_seconds / max(
            self.parallel.processing_seconds, 1e-9
        )

    def as_dict(self, directory: Path) -> dict[str, object]:
        """Формирует словарь для файла performance_report.json."""
        sequential_seconds = self.sequential.seconds
        parallel_seconds = self.parallel.seconds
        return {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "environment": {
                "python": platform.python_version(),
                "platform": f"{platform.system()} {platform.release()}",
                "cpu_count": os.cpu_count(),
                "processor": platform.processor() or "не определён",
            },
            "dataset": {
                "directory": str(directory),
                "files": self.sequential.files,
                "shape": list(self.shape),
            },
            "sequential": {
                "seconds": round(sequential_seconds, 4),
                "files_per_second": round(
                    self.sequential.processed / max(sequential_seconds, 1e-9), 2
                ),
            },
            "parallel": {
                "workers": self.workers,
                "processes": self.processes,
                **self.parallel.as_dict(),
                "files_per_second": round(
                    self.parallel.processed / max(parallel_seconds, 1e-9), 2
                ),
            },
            "speedup": {
                "total": round(self.speedup_total, 2),
                "processing_only": round(self.speedup_processing, 2),
            },
            "note": (
                "Ускорение processing_only считается без времени создания пула "
                "процессов, полное ускорение учитывает накладные расходы."
            ),
        }


def compare(
    directory: Path,
    extensions: tuple[str, ...],
    workers: int = 4,
    processes: int = 4,
    operations: tuple[str, ...] = OPERATIONS,
    logger: logging.Logger | None = None,
) -> tuple[ComparisonResult, np.ndarray]:
    """Запускает обе версии конвейера и собирает статистику.

    :param directory: каталог с изображениями.
    :param extensions: расширения файлов.
    :param workers: число потоков чтения.
    :param processes: число процессов аугментации.
    :param operations: операции аугментации.
    :param logger: логгер для сообщений.
    :return: результат сравнения и аугментированный набор данных.
    """
    log = logger or logging.getLogger(__name__)
    log.info("Последовательная обработка")
    sequential_data, sequential_stats = run_sequential(directory, extensions, operations)
    log.info("Последовательная версия: %.3f с", sequential_stats.seconds)

    log.info("Параллельная обработка")
    pipeline = ParallelPipeline(
        workers=workers, processes=processes, operations=operations, logger=log
    )
    parallel_data, parallel_stats = pipeline.run(directory, extensions)
    log.info("Параллельная версия: %.3f с", parallel_stats.seconds)

    if sequential_data.shape != parallel_data.shape:
        raise PipelineError(
            "Формы результатов не совпадают: "
            f"{sequential_data.shape} и {parallel_data.shape}"
        )
    return (
        ComparisonResult(
            sequential=sequential_stats,
            parallel=parallel_stats,
            workers=workers,
            processes=processes,
            shape=tuple(int(size) for size in parallel_data.shape),
        ),
        from_uint8(parallel_data),
    )


def save_report(report: dict[str, object], target: Path) -> Path:
    """Сохраняет отчёт о производительности в JSON."""
    destination = Path(target).expanduser()
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError as error:
        raise PipelineError(
            f"Не удалось сохранить отчёт {destination}: {error}"
        ) from error
    return destination


def benchmark_augmentation(
    arrays: np.ndarray,
    processes: int = 4,
    operations: tuple[str, ...] = OPERATIONS,
    chunk_size: int = 8,
) -> dict[str, float]:
    """Измеряет чистую аугментацию без чтения файлов.

    Сначала все изображения обрабатываются в одном потоке, затем те же
    данные уходят в пул процессов. Такое сравнение показывает выигрыш
    именно на CPU-bound участке конвейера.

    :param arrays: массив изображений формы ``(n, height, width, channels)``.
    :param processes: число процессов в пуле.
    :param operations: набор операций аугментации.
    :param chunk_size: сколько изображений отправлять одним заданием.
    :return: словарь с временами и ускорением.
    """
    compact = to_uint8(arrays)
    started = time.perf_counter()
    for index, image in enumerate(compact):
        augment_chunk(image[np.newaxis, ...], index + 1, operations)
    sequential = time.perf_counter() - started

    chunks = [
        (compact[start:start + chunk_size], start * 1000 + 1, operations)
        for start in range(0, len(compact), chunk_size)
    ]
    started = time.perf_counter()
    with Pool(processes=processes) as pool:
        pool.starmap(augment_chunk, chunks)
    parallel = time.perf_counter() - started
    return {
        "images": int(len(arrays)),
        "sequential_seconds": round(sequential, 4),
        "parallel_seconds": round(parallel, 4),
        "speedup": round(sequential / max(parallel, 1e-9), 2),
    }


def save_dataset(dataset: np.ndarray, target: Path) -> Path:
    """Сохраняет аугментированный набор в формат .npy."""
    destination = target if target.suffix == ".npy" else target.with_suffix(".npy")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(destination, dataset)
    except OSError as error:
        raise PipelineError(
            f"Не удалось сохранить данные {destination}: {error}"
        ) from error
    return destination
