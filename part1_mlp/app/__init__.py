"""Первая часть практического задания: полносвязная сеть на ООП."""

from .manager import DatasetManager
from .models import ACTIVATION_FUNCTIONS, NeuralNetwork

__all__ = ["ACTIVATION_FUNCTIONS", "DatasetManager", "NeuralNetwork"]
