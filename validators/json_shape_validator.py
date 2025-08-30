import json
from json.decoder import JSONDecodeError
from typing import Any

from jsonschema import validate, ValidationError
from models.validation import Response


class JsonShapeValidator:
    def __init__(self, schema: dict | None = None):
        self.schema = schema

    def id(self) -> str:
        return "json_shape"

    def validate(self, content: str) -> Response:
        """validate if a string is valid json"""
        response, loaded_json = _validate_is_json(content)
        if not response.valid:
            return response

        if self.schema:
            response = self._validate_against_schema(loaded_json=loaded_json)
        return response

    def _validate_against_schema(self, loaded_json: Any) -> Response:
        try:
            validate(instance=loaded_json, schema=self.schema)
            response = Response(valid=True, error=None)
        except ValidationError as e:
            response = Response(
                valid=False, error=f"Invalid JSON Properties: {e.message}"
            )
        return response


def _validate_is_json(content: str) -> tuple[Response, Any] | tuple[Response, None]:
    try:
        loaded_json = json.loads(content)
        return Response(valid=True, error=None), loaded_json
    except JSONDecodeError as e:
        return Response(False, f"Invalid JSON: {e}"), None
