"""Загрузка данных из CSV, нормализация и разбиение на train/test."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .exceptions import (
    DataFileNotFoundError,
    DatasetFormatError,
    EmptyDatasetError,
    InvalidParameterError,
    MismatchedDataError,
)


class DatasetManager:
    """Хранит выборку и готовит её к обучению сети.

    Первая строка CSV считается заголовком, последний столбец -
    целевой переменной, остальные столбцы - признаки. Нормализатор
    обучается только на тренировочной части, иначе в тестовую выборку
    утекает информация о её среднем и разбросе.
    """

    def __init__(self) -> None:
        self.feature_names: list[str] = []
        self.target_name: str = ""
        self.features: np.ndarray | None = None
        self.target: np.ndarray | None = None
        self.x_train: np.ndarray | None = None
        self.x_test: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.y_test: np.ndarray | None = None
        self.scaler: dict[str, np.ndarray] | None = None
        self.normalization: str = "none"
        self.source: Path | None = None

    # ------------------------------------------------------------------
    # чтение файла
    # ------------------------------------------------------------------
    def load_csv(self, path: str | Path) -> tuple[int, int]:
        """Читает CSV-файл и заполняет матрицу признаков и вектор цели.

        :param path: путь к CSV-файлу.
        :return: пара (число примеров, число признаков).
        """
        source = Path(path).expanduser()
        if not source.is_file():
            raise DataFileNotFoundError(f"Файл '{source}' не найден.")
        rows = self._read_rows(source)
        if len(rows) < 2:
            raise EmptyDatasetError(
                "В файле нет данных: нужна строка заголовка и хотя бы один пример."
            )
        header = rows[0]
        if len(header) < 2:
            raise DatasetFormatError(
                "В файле должен быть хотя бы один признак и целевой столбец."
            )
        body = rows[1:]
        widths = {len(row) for row in body}
        if widths != {len(header)}:
            raise DatasetFormatError(
                "Строки файла имеют разное число столбцов, проверьте CSV."
            )
        matrix = np.empty((len(body), len(header)), dtype=float)
        for row_index, row in enumerate(body):
            for column_index, value in enumerate(row):
                try:
                    matrix[row_index, column_index] = float(
                        str(value).replace(",", ".")
                    )
                except ValueError as error:
                    raise DatasetFormatError(
                        f"Строка {row_index + 2}, столбец '{header[column_index]}': "
                        f"значение '{value}' не является числом."
                    ) from error
        self.feature_names = [name.strip() for name in header[:-1]]
        self.target_name = header[-1].strip()
        self.features = matrix[:, :-1]
        self.target = matrix[:, -1]
        self.source = source
        self.x_train = self.x_test = self.y_train = self.y_test = None
        self.scaler = None
        self.normalization = "none"
        return int(self.features.shape[0]), int(self.features.shape[1])

    @staticmethod
    def _read_rows(source: Path) -> list[list[str]]:
        """Читает CSV в UTF-8, при неудаче пробует Windows-кодировку."""
        for encoding in ("utf-8-sig", "cp1251"):
            try:
                with source.open("r", encoding=encoding, newline="") as stream:
                    return [row for row in csv.reader(stream) if row]
            except UnicodeDecodeError:
                continue
            except OSError as error:
                raise DataFileNotFoundError(
                    f"Не удалось прочитать '{source}': {error}."
                ) from error
        raise DatasetFormatError(
            f"Не удалось определить кодировку файла '{source}'."
        )

    # ------------------------------------------------------------------
    # подготовка выборки
    # ------------------------------------------------------------------
    def split(self, test_ratio: float = 0.2, seed: int = 42) -> tuple[int, int]:
        """Делит выборку на тренировочную и тестовую части.

        :param test_ratio: доля тестовой части, должна быть в диапазоне (0, 1).
        :param seed: зерно перемешивания для воспроизводимости.
        :return: пара (размер train, размер test).
        """
        if not 0.0 < test_ratio < 1.0:
            raise InvalidParameterError(
                "Доля тестовой выборки должна быть больше 0 и меньше 1."
            )
        if self.features is None or self.target is None:
            raise EmptyDatasetError("Сначала загрузите данные из CSV.")
        total = self.features.shape[0]
        if total < 3:
            raise EmptyDatasetError("Для разбиения нужно минимум три примера.")
        test_size = max(1, int(round(total * test_ratio)))
        test_size = min(test_size, total - 1)
        indices = np.random.default_rng(seed).permutation(total)
        test_indices = indices[:test_size]
        train_indices = indices[test_size:]
        self.x_train = self.features[train_indices]
        self.x_test = self.features[test_indices]
        self.y_train = self.target[train_indices]
        self.y_test = self.target[test_indices]
        return int(self.x_train.shape[0]), int(self.x_test.shape[0])

    def normalize(self, method: str = "minmax") -> None:
        """Нормализует признаки по тренировочной части выборки.

        :param method: ``minmax`` (приведение к [0, 1]) или ``standard``
            (вычитание среднего и деление на стандартное отклонение).
        """
        if self.x_train is None:
            raise EmptyDatasetError("Сначала разбейте выборку на train/test.")
        if method not in {"minmax", "standard"}:
            raise InvalidParameterError(
                "Доступны два способа нормализации: minmax и standard."
            )
        train = self.x_train
        test = self.x_test
        if test is None:
            raise EmptyDatasetError("Тестовая часть выборки не сформирована.")
        if method == "minmax":
            low = train.min(axis=0)
            high = train.max(axis=0)
            span = np.where(high - low == 0.0, 1.0, high - low)
            self.scaler = {"low": low, "span": span}
        else:
            mean = train.mean(axis=0)
            deviation = train.std(axis=0)
            self.scaler = {
                "mean": mean,
                "std": np.where(deviation == 0.0, 1.0, deviation),
            }
        self.normalization = method
        self.x_train = self.transform(train)
        self.x_test = self.transform(test)

    def transform(self, features: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
        """Применяет сохранённый нормализатор к новым данным.

        :param features: матрица ``(n, m)`` или один вектор длины ``m``.
        :return: нормализованная копия массива.
        """
        data = np.asarray(features, dtype=float)
        if self.scaler is None:
            return data.copy()
        working = data.reshape(1, -1) if data.ndim == 1 else data
        if self.features is not None and working.shape[1] != self.features.shape[1]:
            raise MismatchedDataError(
                f"Ожидалось {self.features.shape[1]} признаков, "
                f"получено {working.shape[1]}."
            )
        if self.normalization == "minmax":
            result = (working - self.scaler["low"]) / self.scaler["span"]
        else:
            result = (working - self.scaler["mean"]) / self.scaler["std"]
        return result.reshape(data.shape)

    def is_ready(self) -> bool:
        """Проверяет, готова ли выборка к обучению сети."""
        return self.x_train is not None and self.y_train is not None

    def train_set(self) -> tuple[np.ndarray, np.ndarray]:
        """Возвращает пару (признаки, цели) тренировочной части выборки."""
        if self.x_train is None or self.y_train is None:
            raise EmptyDatasetError("Тренировочная выборка не сформирована.")
        return self.x_train, self.y_train

    def test_set(self) -> tuple[np.ndarray, np.ndarray]:
        """Возвращает пару (признаки, цели) тестовой части выборки."""
        if self.x_test is None or self.y_test is None:
            raise EmptyDatasetError("Тестовая выборка не сформирована.")
        return self.x_test, self.y_test

    def describe(self) -> list[str]:
        """Возвращает строки со статистикой по загруженным данным."""
        if self.features is None or self.target is None:
            raise EmptyDatasetError("Данные ещё не загружены.")
        classes, counts = np.unique(self.target, return_counts=True)
        lines = [
            f"Источник: {self.source}",
            f"Примеров: {self.features.shape[0]}, признаков: {self.features.shape[1]}",
            f"Признаки: {', '.join(self.feature_names)}",
            f"Целевая переменная: {self.target_name}",
            "Распределение классов: "
            + ", ".join(
                f"{int(cls)} - {int(count)}"
                for cls, count in zip(classes, counts, strict=True)
            ),
        ]
        train = self.x_train
        test = self.x_test
        if train is not None and test is not None:
            lines.append(
                f"train: {train.shape[0]}, test: {test.shape[0]}, "
                f"нормализация: {self.normalization}"
            )
        return lines
