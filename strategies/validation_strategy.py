from abc import ABC, abstractmethod
from models.validation import ValidationResult


class ValidationStrategy(ABC):
    @abstractmethod
    def run(self, content: str) -> ValidationResult:
        pass
