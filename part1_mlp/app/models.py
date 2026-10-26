"""Полносвязная нейронная сеть (многослойный перцептрон).

Класс :class:`NeuralNetwork` хранит веса в списке матриц, умеет
считать прямой проход, обратный проход по алгоритму обратного
распространения ошибки и обучаться мини-батчами. Функция активации
задаётся именем, поэтому её можно менять без правки кода сети.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from .exceptions import (
    InvalidLayerSizeError,
    InvalidParameterError,
    MismatchedDataError,
    WeightsFileError,
)

Array = np.ndarray


def sigmoid(values: Array) -> Array:
    """Логистическая функция активации ``1 / (1 + exp(-x))``."""
    return 1.0 / (1.0 + np.exp(-np.clip(values, -500.0, 500.0)))


def sigmoid_gradient(activated: Array) -> Array:
    """Производная сигмоиды, выраженная через её же выход."""
    return activated * (1.0 - activated)


def relu(values: Array) -> Array:
    """Выпрямленная линейная единица."""
    return np.maximum(0.0, values)


def relu_gradient(activated: Array) -> Array:
    """Производная ReLU: единица там, где вход был положительным."""
    return (activated > 0.0).astype(float)


def tanh(values: Array) -> Array:
    """Гиперболический тангенс, диапазон значений [-1, 1]."""
    return np.tanh(values)


def tanh_gradient(activated: Array) -> Array:
    """Производная гиперболического тангенса."""
    return 1.0 - activated ** 2


def linear(values: Array) -> Array:
    """Линейная функция, используется для регрессии."""
    return values


def linear_gradient(activated: Array) -> Array:
    """Производная линейной функции равна единице."""
    return np.ones_like(activated)


#: Доступные функции активации: имя -> (функция, производная).
ACTIVATION_FUNCTIONS: dict[str, tuple[Callable, Callable]] = {
    "sigmoid": (sigmoid, sigmoid_gradient),
    "relu": (relu, relu_gradient),
    "tanh": (tanh, tanh_gradient),
    "linear": (linear, linear_gradient),
}


class NeuralNetwork:
    """Полносвязная сеть прямого распространения.

    :param layer_sizes: число нейронов по слоям, например ``[4, 8, 1]``.
        Первое значение - размер входа, последнее - размер выхода,
        промежуточные элементы описывают скрытые слои.
    :param activation: имя функции активации скрытых слоёв.
    :param output_activation: функция активации выходного слоя; если
        не задана, используется та же, что и для скрытых слоёв.
    :param seed: зерно генератора случайных чисел для воспроизводимости.
    """

    def __init__(
        self,
        layer_sizes: Sequence[int],
        activation: str = "sigmoid",
        output_activation: str | None = None,
        seed: int | None = None,
    ) -> None:
        self.layer_sizes = self._validate_layers(layer_sizes)
        self.activation = self._validate_activation(activation)
        self.output_activation = self._validate_activation(
            output_activation or activation
        )
        self.random = np.random.default_rng(seed)
        self.weights: list[list[Array]] = []
        self.history: list[float] = []
        self._init_weights()

    # ------------------------------------------------------------------
    # инициализация
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_layers(layer_sizes: Sequence[int]) -> list[int]:
        """Проверяет список размеров слоёв и приводит его к ``list[int]``."""
        try:
            sizes = [int(size) for size in layer_sizes]
        except (TypeError, ValueError) as error:
            raise InvalidLayerSizeError(
                "Размеры слоёв должны быть целыми числами."
            ) from error
        if len(sizes) < 2:
            raise InvalidLayerSizeError(
                "Нужно минимум два слоя: входной и выходной."
            )
        if any(size < 1 for size in sizes):
            raise InvalidLayerSizeError(
                "Число нейронов в слое должно быть положительным."
            )
        return sizes

    @staticmethod
    def _validate_activation(name: str) -> str:
        """Проверяет, что такая функция активации поддерживается."""
        normalized = str(name).strip().lower()
        if normalized not in ACTIVATION_FUNCTIONS:
            available = ", ".join(sorted(ACTIVATION_FUNCTIONS))
            raise InvalidParameterError(
                f"Неизвестная функция активации '{name}'. Доступны: {available}."
            )
        return normalized

    def _init_weights(self) -> None:
        """Заполняет веса случайными числами (инициализация Ксавье/Хе)."""
        self.weights = []
        for index in range(len(self.layer_sizes) - 1):
            inputs = self.layer_sizes[index]
            outputs = self.layer_sizes[index + 1]
            layer_activation = self._activation_name(index)
            if layer_activation == "relu":
                scale = np.sqrt(2.0 / inputs)
            else:
                scale = np.sqrt(1.0 / inputs)
            matrix = self.random.normal(0.0, scale, size=(inputs, outputs))
            bias = self.random.normal(0.0, 0.01, size=outputs)
            self.weights.append([matrix, bias])

    def _activation_name(self, layer_index: int) -> str:
        """Возвращает имя активации для слоя с указанным индексом."""
        if layer_index == len(self.weights) - 1:
            return self.output_activation
        return self.activation

    def _activation_functions(self, layer_index: int) -> tuple[Callable, Callable]:
        """Возвращает пару (функция, производная) для слоя."""
        return ACTIVATION_FUNCTIONS[self._activation_name(layer_index)]

    # ------------------------------------------------------------------
    # прямой и обратный проход
    # ------------------------------------------------------------------
    def forward(self, inputs: Array) -> tuple[Array, list[Array]]:
        """Считает выход сети и запоминает активации всех слоёв.

        :param inputs: матрица признаков формы ``(n_samples, n_features)``.
        :return: пара (выход сети, список активаций от входа к выходу).
        """
        data = np.asarray(inputs, dtype=float)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] != self.layer_sizes[0]:
            raise MismatchedDataError(
                f"Сеть ожидает {self.layer_sizes[0]} признаков, "
                f"а получила {data.shape[1]}."
            )
        activations = [data]
        current = data
        for index, weights in enumerate(self.weights):
            matrix, bias = weights
            activation, _ = self._activation_functions(index)
            current = activation(current @ matrix + bias)
            activations.append(current)
        return current, activations

    def _backward(
        self, activations: list[Array], target: Array, learning_rate: float
    ) -> None:
        """Считает градиенты и обновляет веса на одном батче."""
        samples = target.shape[0]
        output_error = (activations[-1] - target) * self._activation_functions(
            len(self.weights) - 1
        )[1](activations[-1])
        deltas = [output_error]
        for layer in range(len(self.weights) - 2, -1, -1):
            next_matrix = self.weights[layer + 1][0]
            propagated = deltas[0] @ next_matrix.T
            deltas.insert(
                0, propagated * self._activation_functions(layer)[1](activations[layer + 1])
            )
        for layer, delta in enumerate(deltas):
            layer_inputs = activations[layer]
            matrix_gradient = layer_inputs.T @ delta / samples
            bias_gradient = delta.mean(axis=0)
            self.weights[layer][0] -= learning_rate * matrix_gradient
            self.weights[layer][1] -= learning_rate * bias_gradient

    # ------------------------------------------------------------------
    # обучение и предсказание
    # ------------------------------------------------------------------
    def loss(self, inputs: Array, target: Array) -> float:
        """Среднеквадратичная ошибка сети на наборе данных."""
        output, _ = self.forward(inputs)
        difference = output - self._prepare_target(target, output.shape[1])
        return float(np.mean(0.5 * np.sum(difference ** 2, axis=1)))

    def _prepare_target(self, target: Array, outputs: int) -> Array:
        """Приводит целевую переменную к матрице нужной формы."""
        values = np.asarray(target, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.shape[1] != outputs:
            raise MismatchedDataError(
                f"Выход сети имеет размер {outputs}, "
                f"а целевая переменная - {values.shape[1]}."
            )
        return values

    def fit(
        self,
        inputs: Array,
        target: Array,
        epochs: int = 100,
        learning_rate: float = 0.1,
        batch_size: int | None = 32,
        shuffle: bool = True,
    ) -> list[float]:
        """Обучает сеть градиентным спуском по мини-батчам.

        :param inputs: матрица признаков обучающей выборки.
        :param target: целевые значения (метки или one-hot матрица).
        :param epochs: число полных проходов по выборке.
        :param learning_rate: шаг обучения ``alpha``.
        :param batch_size: размер батча; ``None`` - вся выборка за шаг.
        :param shuffle: перемешивать примеры перед каждой эпохой.
        :return: история средней ошибки по эпохам.
        """
        if epochs < 1:
            raise InvalidParameterError("Число эпох должно быть больше нуля.")
        if not 0.0 < learning_rate <= 10.0:
            raise InvalidParameterError(
                "Скорость обучения должна лежать в диапазоне (0, 10]."
            )
        if batch_size is not None and batch_size < 1:
            raise InvalidParameterError("Размер батча должен быть положительным.")

        features = np.asarray(inputs, dtype=float)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if features.shape[1] != self.layer_sizes[0]:
            raise MismatchedDataError(
                f"Сеть ожидает {self.layer_sizes[0]} признаков, "
                f"а выборка содержит {features.shape[1]}."
            )
        expected = self.layer_sizes[-1]
        labels = self._prepare_target(target, expected)
        if labels.shape[0] != features.shape[0]:
            raise MismatchedDataError(
                f"Примеров {features.shape[0]}, а меток {labels.shape[0]}."
            )

        self.history = []
        step = batch_size or features.shape[0]
        for epoch in range(1, epochs + 1):
            order = self.random.permutation(features.shape[0]) if shuffle else np.arange(
                features.shape[0]
            )
            for start in range(0, features.shape[0], step):
                batch = order[start:start + step]
                output, activations = self.forward(features[batch])
                self._backward(activations, labels[batch], learning_rate)
            self.history.append(self.loss(features, labels))
            if epoch == 1 or epoch % max(1, epochs // 10) == 0 or epoch == epochs:
                print(f"  эпоха {epoch:4d}  ошибка {self.history[-1]:.6f}")
        return self.history

    def predict(self, inputs: Array) -> Array:
        """Возвращает значения выходного слоя для набора примеров."""
        output, _ = self.forward(inputs)
        return output

    def predict_classes(self, inputs: Array, threshold: float = 0.5) -> Array:
        """Возвращает номера классов: порог для одного выхода, argmax иначе."""
        output = self.predict(inputs)
        if output.shape[1] == 1:
            return (output[:, 0] >= threshold).astype(int)
        return np.argmax(output, axis=1)

    def evaluate(self, inputs: Array, target: Array) -> tuple[float, float]:
        """Возвращает ошибку и долю верных ответов на наборе данных."""
        labels = self._prepare_target(target, self.layer_sizes[-1])
        predicted = self.predict_classes(inputs)
        if self.layer_sizes[-1] > 1:
            true_classes = np.argmax(labels, axis=1)
        else:
            true_classes = (labels[:, 0] >= 0.5).astype(int)
        accuracy = float(np.mean(predicted == true_classes))
        return self.loss(inputs, labels), accuracy

    # ------------------------------------------------------------------
    # сохранение и загрузка
    # ------------------------------------------------------------------
    def save_weights(self, path: str | Path) -> Path:
        """Сохраняет архитектуру и веса в JSON-файл."""
        target = Path(path)
        payload = {
            "layer_sizes": self.layer_sizes,
            "activation": self.activation,
            "output_activation": self.output_activation,
            "weights": [
                {"matrix": matrix.tolist(), "bias": bias.tolist()}
                for matrix, bias in self.weights
            ],
        }
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as error:
            raise WeightsFileError(
                f"Не удалось записать веса в файл '{target}': {error}."
            ) from error
        return target

    def load_weights(self, path: str | Path) -> None:
        """Загружает веса из JSON-файла, проверяя совместимость слоёв."""
        source = Path(path)
        if not source.exists():
            raise WeightsFileError(f"Файл с весами '{source}' не найден.")
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
            sizes = [int(size) for size in payload["layer_sizes"]]
            activation = str(payload["activation"])
            matrices = [
                [np.asarray(layer["matrix"], dtype=float),
                 np.asarray(layer["bias"], dtype=float)]
                for layer in payload["weights"]
            ]
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise WeightsFileError(
                f"Файл '{source}' повреждён или имеет неверный формат."
            ) from error
        if sizes != self.layer_sizes:
            raise MismatchedDataError(
                f"В файле архитектура {sizes}, у сети - {self.layer_sizes}."
            )
        for index, layer in enumerate(self.weights):
            loaded_matrix, loaded_bias = matrices[index]
            matrix, bias = layer
            if loaded_matrix.shape != matrix.shape or loaded_bias.shape != bias.shape:
                raise MismatchedDataError(
                    f"Форма весов слоя {index + 1} в файле не совпадает с сетью."
                )
        self.activation = self._validate_activation(activation)
        self.weights = matrices

    def describe(self) -> list[str]:
        """Возвращает строки с описанием архитектуры для консольного меню."""
        lines = [
            f"Слои: {self.layer_sizes}",
            f"Активация скрытых слоёв: {self.activation}",
            f"Активация выхода: {self.output_activation}",
            f"Параметров: {sum(matrix.size + bias.size for matrix, bias in self.weights)}",
        ]
        for index, (matrix, bias) in enumerate(self.weights, start=1):
            lines.append(
                f"  слой {index}: W {matrix.shape}, b {bias.shape}"
            )
        return lines
