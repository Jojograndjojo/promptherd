import json
from json.decoder import JSONDecodeError
from models.validation import Response


class JsonShapeValidator:
    def __init__(self):
        pass

    def validate(self, content: str) -> Response:
        try:
            json.loads(content)
            return Response(valid=True, error=None)
        except JSONDecodeError as e:
            return Response(False, f"{e}")
