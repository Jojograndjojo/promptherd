from abc import ABC, abstractmethod


class PromptValidator(ABC):
    @abstractmethod
    def id(self) -> str:
        pass

    @abstractmethod
    def validate(self, content: str) -> dict:
        pass
