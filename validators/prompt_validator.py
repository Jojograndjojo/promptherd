from abc import ABC, abstractmethod


class PromptValidator(ABC):
    @abstractmethod
    def validate(self, prompt: str) -> dict:
        pass
