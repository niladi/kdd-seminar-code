from abc import ABC, abstractmethod
from functools import lru_cache


class Node(ABC):
    name: str
    value: float
    threshold: float

    def __init__(self, name: str, value: str, threshold: float) -> None:
        self.name = name
        self.value = value
        self.threshold = threshold

    @lru_cache(maxsize=None)
    def calc(self, data):
        if self.is_active():
            return self.operation(data)
        return None

    @lru_cache(maxsize=None)
    @abstractmethod
    def is_active(self) -> bool:
        raise NotImplementedError()

    @lru_cache(maxsize=None)
    @abstractmethod
    def operation(self, data):
        raise NotImplementedError()
