import unittest
from validators.max_length_validator import MaxLengthValidator


class TestMaxLengthValidator(unittest.TestCase):
    def test_id(self):
        validator = MaxLengthValidator(max_length=3)

        self.assertEqual(validator.id(), "max_length")

    def test_returns_invalid_response_if_string_is_too_long(self):
        content = "abcd"
        validator = MaxLengthValidator(max_length=3)

        response = validator.validate(content=content)

        self.assertEqual(
            response.error, f"Content is too long {len(content)} chars. Max length: 3"
        )
        self.assertFalse(response.valid)

    def test_returns_valid_response_if_string_is_not_too_long(self):
        content = "abc"
        validator = MaxLengthValidator(max_length=3)

        response = validator.validate(content=content)

        self.assertIsNone(response.error)
        self.assertTrue(response.valid)
