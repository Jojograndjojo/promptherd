from validators.prompt_validator import PromptValidator
from models.validation import Response


class MaxLengthValidator(PromptValidator):
    def __init__(self, max_length):
        self.max_length = max_length

    def id(self) -> str:
        """id of the validator"""
        return "max_length"

    def validate(self, content: str) -> Response:
        """validate if a string is valid json"""
        if len(content) <= self.max_length:
            return Response(valid=True, error=None)
        else:
            return Response(
                valid=False,
                error=f"Content is too long {len(content)} chars. Max length: {self.max_length}",
            )
