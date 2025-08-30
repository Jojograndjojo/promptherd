import json
import unittest
from unittest.mock import patch
from validators.json_shape_validator import JsonShapeValidator


class TestJsonShapeValidator(unittest.TestCase):
    @patch("validators.json_shape_validator.json")
    def test_returns_validation_error_if_json_does_not_load(self, mock_json):
        json_loading_error = json.JSONDecodeError("JSON does not load", "", 0)

        mock_json.loads.side_effect = json_loading_error

        content = "result"
        validator = JsonShapeValidator()
        result = validator.validate(content=content)

        self.assertEqual(
            result.error, "Invalid JSON: JSON does not load: line 1 column 1 (char 0)"
        )
        self.assertFalse(result.valid)

    @patch("validators.json_shape_validator.json")
    def test_returns_valid_validation_if_json_loads(self, mock_json):
        mock_json.loads.return_value = {}
        content = "result"
        validator = JsonShapeValidator()

        result = validator.validate(content=content)

        self.assertIsNone(result.error)
        self.assertTrue(result.valid)

    def test_returns_validation_error_if_json_does_not_respect_schema(self):
        schema = {
            "type": "object",
            "properties": {
                "description": {"type": "string"},
                "status": {"type": "boolean"},
                "value_a": {"type": "number"},
                "value_b": {"type": "number"},
            },
            "additionalProperties": False,
        }
        content = '{"name":"george"}'
        validator = JsonShapeValidator(schema=schema)

        response = validator.validate(content=content)
        self.assertEqual(
            response.error,
            "Invalid JSON Properties: Additional properties are not allowed ('name' was unexpected)",
        )
        self.assertFalse(response.valid)

    def test_id(self):
        validator = JsonShapeValidator()
        self.assertEqual(validator.id(), "json_shape")
