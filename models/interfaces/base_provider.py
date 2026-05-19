from abc import ABC, abstractmethod


class BaseProvider(ABC):
    @abstractmethod
    def generate(self, messages: list[dict], model: str) -> str:
        pass