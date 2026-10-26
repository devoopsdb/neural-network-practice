"""Вспомогательные функции для разбора пользовательского ввода."""

from __future__ import annotations

from collections.abc import Iterable

from .exceptions import InvalidParameterError


def read_text(prompt: str, default: str | None = None) -> str:
    """Читает строку; при пустом вводе возвращает значение по умолчанию."""
    suffix = f" [{default}]" if default else ""
    value = input(f"{prompt}{suffix}: ").strip()
    if not value and default is not None:
        return default
    if not value:
        raise InvalidParameterError("Пустой ввод, значение обязательно.")
    return value


def read_int(
    prompt: str,
    default: int | None = None,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Читает целое число и проверяет границы."""
    raw = read_text(prompt, str(default) if default is not None else None)
    try:
        value = int(raw)
    except ValueError as error:
        raise InvalidParameterError(f"'{raw}' не является целым числом.") from error
    if minimum is not None and value < minimum:
        raise InvalidParameterError(f"Значение должно быть не меньше {minimum}.")
    if maximum is not None and value > maximum:
        raise InvalidParameterError(f"Значение должно быть не больше {maximum}.")
    return value


def read_float(
    prompt: str,
    default: float | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Читает вещественное число, принимая и запятую, и точку."""
    raw = read_text(prompt, str(default) if default is not None else None)
    try:
        value = float(raw.replace(",", "."))
    except ValueError as error:
        raise InvalidParameterError(f"'{raw}' не является числом.") from error
    if minimum is not None and value < minimum:
        raise InvalidParameterError(f"Значение должно быть не меньше {minimum}.")
    if maximum is not None and value > maximum:
        raise InvalidParameterError(f"Значение должно быть не больше {maximum}.")
    return value


def read_number_list(prompt: str, expected: int | None = None) -> list[float]:
    """Читает список чисел, разделённых пробелами, запятыми или ';'."""
    raw = read_text(prompt)
    prepared = raw.replace(";", " ").replace(",", " ")
    try:
        values = [float(item.replace(",", ".")) for item in prepared.split()]
    except ValueError as error:
        raise InvalidParameterError(
            f"Не удалось разобрать список чисел: '{raw}'."
        ) from error
    if not values:
        raise InvalidParameterError("Список чисел пуст.")
    if expected is not None and len(values) != expected:
        raise InvalidParameterError(
            f"Ожидалось {expected} чисел, введено {len(values)}."
        )
    return values


def format_vector(values: Iterable[float], precision: int = 3) -> str:
    """Форматирует вектор в строку с фиксированной точностью."""
    return "[" + ", ".join(f"{value:.{precision}f}" for value in values) + "]"


def history_summary(history: list[float], head: int = 2, tail: int = 3) -> list[str]:
    """Возвращает начало и конец истории ошибки для печати."""
    if not history:
        return []
    if len(history) <= head + tail:
        return [f"{index + 1}: {value:.6f}" for index, value in enumerate(history)]
    lines = [f"{index + 1}: {value:.6f}" for index, value in enumerate(history[:head])]
    lines.append("...")
    start = len(history) - tail
    lines.extend(
        f"{start + offset + 1}: {value:.6f}"
        for offset, value in enumerate(history[start:])
    )
    return lines
