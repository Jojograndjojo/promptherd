import json
from json.decoder import JSONDecodeError
from models.validation import Response


class JsonShapeValidator:
    def __init__(self):
        pass

    def id(self) -> str:
        return "json_shape"

    def validate(self, content: str) -> Response:
        """validate if a string is valid json"""
        try:
            json.loads(content)
            return Response(valid=True, error=None)
        except JSONDecodeError as e:
            return Response(False, f"{e}")
