"""Собственные исключения для первой части практического задания.

Все ошибки, которые может породить программа, наследуются от
``NeuralNetworkError``. Это позволяет в интерфейсе ловить одним
блоком ``except`` только "свои" ошибки и не скрывать посторонние
проблемы (например, ошибки программиста).
"""


class NeuralNetworkError(Exception):
    """Базовое исключение приложения."""


class InvalidLayerSizeError(NeuralNetworkError):
    """Некорректный размер слоя или неверное число слоёв."""


class InvalidParameterError(NeuralNetworkError):
    """Некорректное значение параметра обучения или ввода пользователя."""


class DataFileNotFoundError(NeuralNetworkError):
    """Файл с данными не найден или недоступен для чтения."""


class DatasetFormatError(NeuralNetworkError):
    """Содержимое файла не соответствует ожидаемому формату CSV."""


class EmptyDatasetError(NeuralNetworkError):
    """Датасет пуст, обучать сеть не на чем."""


class MismatchedDataError(NeuralNetworkError):
    """Размерность данных не совпадает с размерностью входов сети."""


class WeightsFileError(NeuralNetworkError):
    """Не удалось прочитать или записать файл с весами."""


class NotTrainedError(NeuralNetworkError):
    """Попытка предсказания до загрузки данных или обучения."""
