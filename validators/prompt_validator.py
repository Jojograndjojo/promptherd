from abc import ABC, abstractmethod
from models.validation import Response


class PromptValidator(ABC):
    @abstractmethod
    def id(self) -> str:
        pass

    @abstractmethod
    def validate(self, content: str) -> Response:
        pass
