import json
import unittest
from unittest.mock import patch
from validators.json_shape_validator import JsonShapeValidator


class TestJsonShapeValidator(unittest.TestCase):
    @patch("validators.json_validator.json")
    def test_returns_validation_error_if_json_does_not_load(self, mock_json):
        json_loading_error = json.JSONDecodeError("JSON does not load", "", 0)

        mock_json.loads.side_effect = json_loading_error

        content = "result"
        validator = JsonShapeValidator()
        result = validator.validate(content=content)

        self.assertEqual(result.error, "JSON does not load: line 1 column 1 (char 0)")
        self.assertFalse(result.valid)

    @patch("validators.json_validator.json")
    def test_returns_valid_validation_if_json_loads(self, mock_json):
        mock_json.loads.return_value = {}
        content = "result"
        validator = JsonShapeValidator()

        result = validator.validate(content=content)

        self.assertIsNone(result.error)
        self.assertTrue(result.valid)
