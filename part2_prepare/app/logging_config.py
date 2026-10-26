"""Настройка логирования для утилиты подготовки данных."""

from __future__ import annotations

import logging
from pathlib import Path

LOGGER_NAME = "prepare"
DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
DATE_FORMAT = "%H:%M:%S"


def setup_logging(
    level: str = "INFO",
    log_file: Path | None = None,
    logger_name: str = LOGGER_NAME,
) -> logging.Logger:
    """Создаёт логгер с выводом в консоль и, если задан, в файл.

    :param level: текстовый уровень логирования.
    :param log_file: путь к файлу журнала; ``None`` - писать только в консоль.
    :param logger_name: имя логгера.
    :return: настроенный логгер.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger(logger_name)
    logger.setLevel(numeric_level)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter(DEFAULT_FORMAT, DATE_FORMAT)
    console = logging.StreamHandler()
    console.setLevel(numeric_level)
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
        except OSError as error:
            logger.warning("Не удалось открыть файл журнала %s: %s", log_file, error)
        else:
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    return logger
