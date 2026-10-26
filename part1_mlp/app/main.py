"""Консольное меню для работы с многослойным перцептроном.

Запуск из каталога ``part1_mlp``::

    python -m app.main
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from .exceptions import (
    EmptyDatasetError,
    InvalidParameterError,
    NeuralNetworkError,
    NotTrainedError,
)
from .manager import DatasetManager
from .models import NeuralNetwork
from .utils import (
    format_vector,
    history_summary,
    read_float,
    read_int,
    read_number_list,
    read_text,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "data" / "input" / "mlp_dataset.csv"
DEFAULT_WEIGHTS = PROJECT_ROOT / "data" / "output" / "part1_weights.json"
DEFAULT_PLOT = PROJECT_ROOT / "logs" / "part1_loss.png"

ACTIVATION_CHOICES = {
    "1": "sigmoid",
    "2": "relu",
    "3": "tanh",
    "4": "linear",
}


def _env_number(name: str, default: float) -> float:
    """Читает число из переменной окружения, иначе берёт значение по умолчанию."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.replace(",", "."))
    except ValueError as error:
        raise InvalidParameterError(
            f"Переменная окружения {name} должна содержать число, а не '{raw}'."
        ) from error


class ConsoleApp:
    """Интерактивное меню: сеть, данные, обучение, предсказания, веса."""

    MENU = [
        ("1", "Создать сеть"),
        ("2", "Загрузить данные из CSV"),
        ("3", "Обучить сеть"),
        ("4", "Предсказание"),
        ("5", "Сохранить или загрузить веса"),
        ("6", "Показать график ошибки"),
        ("7", "Выход"),
    ]

    def __init__(self) -> None:
        self.network: NeuralNetwork | None = None
        self.dataset = DatasetManager()

    # ------------------------------------------------------------------
    # основной цикл
    # ------------------------------------------------------------------
    def run(self) -> int:
        """Запускает цикл меню до выбора пункта "Выход"."""
        self._print_header()
        while True:
            self._print_menu()
            try:
                choice = read_text("Выберите пункт")
            except InvalidParameterError as error:
                print(f"  ! {error}")
                continue
            if choice == "7":
                print("Завершение работы.")
                return 0
            try:
                self._dispatch(choice)
            except NeuralNetworkError as error:
                print(f"  ! Ошибка: {error}")
            except (KeyboardInterrupt, EOFError):
                print("\nПрервано пользователем.")
                return 1

    @staticmethod
    def _print_header() -> None:
        """Печатает заголовок программы."""
        print("=" * 70)
        print("  Многослойный перцептрон: практическая работа, часть 1")
        print("=" * 70)

    def _print_menu(self) -> None:
        """Печатает список пунктов меню и состояние объектов."""
        print()
        print("-" * 70)
        state = "создана" if self.network else "не создана"
        data_state = "загружены" if self.dataset.is_ready() else "не подготовлены"
        print(f"  Сеть: {state} | Данные: {data_state}")
        for key, title in self.MENU:
            print(f"  {key}. {title}")
        print("-" * 70)

    def _dispatch(self, choice: str) -> None:
        """Вызывает обработчик выбранного пункта меню."""
        handlers = {
            "1": self._create_network,
            "2": self._load_dataset,
            "3": self._train,
            "4": self._predict_menu,
            "5": self._weights_menu,
            "6": self._plot_history,
        }
        handler = handlers.get(choice)
        if handler is None:
            raise InvalidParameterError(
                f"Пункта '{choice}' нет в меню, введите число от 1 до 7."
            )
        handler()

    # ------------------------------------------------------------------
    # пункт 1: создание сети
    # ------------------------------------------------------------------
    def _create_network(self) -> None:
        """Спрашивает размеры слоёв и функцию активации, создаёт сеть."""
        print(
            "\nРазмеры слоёв вводятся через запятую: вход, скрытые слои, выход.\n"
            "Например, для четырёх признаков и одного выхода: 4, 8, 1"
        )
        raw_sizes = read_number_list("Размеры слоёв", expected=None)
        sizes = []
        for value in raw_sizes:
            if value != int(value):
                raise InvalidParameterError(
                    "Размер слоя должен быть целым числом нейронов."
                )
            sizes.append(int(value))
        print("\nФункция активации:")
        print("  1. сигмоид (диапазон 0..1)")
        print("  2. ReLU (диапазон 0..бесконечность)")
        print("  3. гиперболический тангенс (диапазон -1..1)")
        print("  4. линейная (без преобразования)")
        choice = read_text("Вариант", "1")
        if choice not in ACTIVATION_CHOICES:
            raise InvalidParameterError("Нужно выбрать вариант от 1 до 4.")
        seed = read_int("Зерно генератора", int(os.getenv("NN_SEED", "42")))
        self.network = NeuralNetwork(
            sizes, activation=ACTIVATION_CHOICES[choice], seed=seed
        )
        print("\nСеть создана:")
        for line in self.network.describe():
            print(f"  {line}")

    # ------------------------------------------------------------------
    # пункт 2: загрузка данных
    # ------------------------------------------------------------------
    def _load_dataset(self) -> None:
        """Загружает CSV, делит выборку и нормализует признаки."""
        default_path = os.getenv("NN_DATASET_PATH") or str(DEFAULT_DATA)
        path = read_text("Путь к CSV-файлу", default_path)
        ratio = read_float(
            "Доля тестовой выборки",
            _env_number("NN_TEST_RATIO", 0.2),
            minimum=0.05,
            maximum=0.5,
        )
        print("\nСпособ нормализации:")
        print("  1. min-max (приведение к диапазону 0..1)")
        print("  2. стандартизация (среднее 0, отклонение 1)")
        method_choice = read_text("Вариант", "1")
        if method_choice not in {"1", "2"}:
            raise InvalidParameterError("Нужно выбрать 1 или 2.")
        method = "minmax" if method_choice == "1" else "standard"
        manager = DatasetManager()
        rows, columns = manager.load_csv(path)
        print(f"\nФайл прочитан: {rows} примеров, {columns} признаков.")
        train_rows, test_rows = manager.split(test_ratio=ratio, seed=42)
        manager.normalize(method)
        self.dataset = manager
        print(f"Разбиение: train {train_rows}, test {test_rows}.")
        print("Нормализация выполнена: " + method)
        print()
        for line in self.dataset.describe():
            print(f"  {line}")

    # ------------------------------------------------------------------
    # пункт 3: обучение
    # ------------------------------------------------------------------
    def _train(self) -> None:
        """Спрашивает гиперпараметры и обучает сеть на подготовленных данных."""
        network = self._require_network()
        if not self.dataset.is_ready():
            raise EmptyDatasetError("Сначала загрузите данные (пункт 2).")
        print()
        learning_rate = read_float("Скорость обучения", 0.5, minimum=0.0001)
        epochs = read_int("Число эпох", 600, minimum=1, maximum=10000)
        batch_size = read_int("Размер батча", 32, minimum=1, maximum=1024)
        print("\nОбучение:")
        train_features, train_labels = self.dataset.train_set()
        test_features, test_labels = self.dataset.test_set()
        history = network.fit(
            train_features,
            train_labels,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
        )
        print("\nИстория ошибки:")
        for line in history_summary(history):
            print(f"  {line}")
        train_loss, train_accuracy = network.evaluate(train_features, train_labels)
        test_loss, test_accuracy = network.evaluate(test_features, test_labels)
        print(f"\nИтог: train loss {train_loss:.4f}, accuracy {train_accuracy:.3f}")
        print(f"      test  loss {test_loss:.4f}, accuracy {test_accuracy:.3f}")

    # ------------------------------------------------------------------
    # пункт 4: предсказание
    # ------------------------------------------------------------------
    def _predict_menu(self) -> None:
        """Подменю предсказаний: вручную, из файла, оценка на тесте."""
        network = self._require_network()
        if not self.dataset.is_ready():
            raise EmptyDatasetError("Сначала загрузите данные (пункт 2).")
        print("\nКак подать примеры:")
        print("  1. ввести значения вручную")
        print("  2. прочитать из CSV-файла (без заголовка)")
        print("  3. оценить сеть на тестовой выборке")
        choice = read_text("Вариант", "1")
        if choice == "1":
            features = read_number_list(
                f"Признаки ({len(self.dataset.feature_names)} шт.)",
                expected=len(self.dataset.feature_names),
            )
            normalized = self.dataset.transform(np.array(features))
            self._print_prediction(network, normalized)
        elif choice == "2":
            path = read_text(
                "Путь к CSV с примерами",
                str(PROJECT_ROOT / "data" / "input" / "examples.csv"),
            )
            examples = self._read_examples(path)
            normalized = self.dataset.transform(examples)
            for index, row in enumerate(normalized, start=1):
                print(f"\nПример {index}: {format_vector(examples[index - 1])}")
                self._print_prediction(network, row)
        elif choice == "3":
            test_features, test_labels = self.dataset.test_set()
            loss, accuracy = network.evaluate(test_features, test_labels)
            print(f"\nТестовая выборка: loss {loss:.4f}, accuracy {accuracy:.3f}")
        else:
            raise InvalidParameterError("Нужно выбрать вариант от 1 до 3.")

    @staticmethod
    def _read_examples(path: str) -> np.ndarray:
        """Читает примеры из CSV без заголовка."""
        source = Path(path).expanduser()
        if not source.is_file():
            raise EmptyDatasetError(f"Файл с примерами '{source}' не найден.")
        try:
            data = np.loadtxt(source, delimiter=",", dtype=float)
        except ValueError as error:
            raise InvalidParameterError(
                f"Файл '{source}' содержит нечисловые значения."
            ) from error
        return np.atleast_2d(data)

    def _print_prediction(self, network: NeuralNetwork, features: np.ndarray) -> None:
        """Печатает выход сети и номер класса для одного примера."""
        row = features.reshape(1, -1)
        output = network.predict(row)[0]
        label = int(network.predict_classes(row)[0])
        print(f"  выход сети: {format_vector(output, precision=4)}")
        print(f"  класс: {label}")
        if output.size == 1:
            print(f"  вероятность класса 1: {output[0]:.4f}")

    # ------------------------------------------------------------------
    # пункт 5: веса
    # ------------------------------------------------------------------
    def _weights_menu(self) -> None:
        """Сохраняет сеть в JSON или загружает её из JSON."""
        print("\n  1. сохранить веса в файл")
        print("  2. загрузить веса из файла")
        choice = read_text("Вариант", "1")
        default_path = os.getenv("NN_WEIGHTS_PATH") or str(DEFAULT_WEIGHTS)
        if choice == "1":
            network = self._require_network()
            path = read_text("Путь для сохранения", default_path)
            saved = network.save_weights(path)
            print(f"Веса сохранены: {saved}")
        elif choice == "2":
            path = read_text("Путь к файлу с весами", default_path)
            network = self._require_network()
            network.load_weights(path)
            print(f"Веса загружены из {path}.")
            for line in network.describe():
                print(f"  {line}")
        else:
            raise InvalidParameterError("Нужно выбрать 1 или 2.")

    # ------------------------------------------------------------------
    # пункт 6: график
    # ------------------------------------------------------------------
    def _plot_history(self) -> None:
        """Строит график ошибки по эпохам и сохраняет его в PNG."""
        network = self._require_network()
        if not network.history:
            raise NotTrainedError("Сеть ещё не обучалась, истории ошибки нет.")
        try:
            import matplotlib
        except ImportError as error:
            raise NeuralNetworkError(
                "Для построения графика нужен matplotlib."
            ) from error
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        target = Path(os.getenv("NN_PLOT_PATH") or str(DEFAULT_PLOT))
        target.parent.mkdir(parents=True, exist_ok=True)
        figure, axes = plt.subplots(figsize=(9, 5), dpi=120)
        axes.plot(range(1, len(network.history) + 1), network.history, color="#1f77b4")
        axes.set_title("Зависимость ошибки от эпохи")
        axes.set_xlabel("Эпоха")
        axes.set_ylabel("Средняя ошибка MSE")
        axes.grid(alpha=0.3)
        figure.tight_layout()
        figure.savefig(target)
        plt.close(figure)
        print(f"График сохранён: {target}")

    # ------------------------------------------------------------------
    # служебные методы
    # ------------------------------------------------------------------
    def _require_network(self) -> NeuralNetwork:
        """Возвращает сеть или сообщает, что её надо создать."""
        if self.network is None:
            raise NotTrainedError("Сначала создайте сеть (пункт 1).")
        return self.network


def main() -> int:
    """Точка входа приложения."""
    sys.tracebacklimit = 0
    return ConsoleApp().run()


if __name__ == "__main__":
    raise SystemExit(main())
