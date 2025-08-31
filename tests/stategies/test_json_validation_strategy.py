import unittest
from unittest.mock import patch, MagicMock
from strategies.json_validation_strategy import JSONValidationStrategy
from models.validation import Response


class TestJsonValidationStrategy(unittest.TestCase):
    def setUp(self):
        self.llm = MagicMock()
        self.schema = {}
        self.max_chars = 100
        self.threshold = 0.7
        self.benchmark_content = '{"name": benchmark}'

    @patch("strategies.json_validation_strategy.JsonShapeValidator")
    @patch("strategies.json_validation_strategy.EmbeddingsSimilarityValidator")
    @patch("strategies.json_validation_strategy.MaxLengthValidator")
    def test_run(
        self,
        mock_max_length_validator,
        mock_embeddings_similarity_validator,
        mock_json_shape_validator,
    ):
        json_shape_validator_response = Response(valid=True, error=None)
        json_shape_validator_id = "json_shape_validator_id"
        json_shape_validator = MagicMock()
        json_shape_validator.validate.return_value = json_shape_validator_response
        json_shape_validator.id.return_value = json_shape_validator_id
        mock_json_shape_validator.return_value = json_shape_validator

        embeddings_similarity_validator_response = Response(
            valid=False, error="Just not valid"
        )
        embeddings_similarity_validator_id = "embeddings_similarity_validator_id"
        embeddings_similarity_validator = MagicMock()
        embeddings_similarity_validator.validate.return_value = (
            embeddings_similarity_validator_response
        )
        embeddings_similarity_validator.id.return_value = (
            embeddings_similarity_validator_id
        )
        mock_embeddings_similarity_validator.return_value = (
            embeddings_similarity_validator
        )

        max_length_validator_response = Response(valid=True, error=None)
        max_length_validator_id = "max_length_validator_id"
        max_length_validator = MagicMock()
        max_length_validator.validate.return_value = max_length_validator_response
        max_length_validator.id.return_value = max_length_validator_id
        mock_max_length_validator.return_value = max_length_validator

        content = '{\n"name": "joseph"\n}'

        strategy = JSONValidationStrategy(
            llm=self.llm,
            schema=self.schema,
            max_chars=self.max_chars,
            threshold=self.threshold,
            benchmark_content=self.benchmark_content,
        )
        validation_result = strategy.run(content=content)
        self.assertFalse(validation_result.success)
        self.assertEqual(validation_result.content, content)
        self.assertEqual(validation_result.benchmark_content, self.benchmark_content)
        self.assertEqual(validation_result.threshold, self.threshold)

        validators_results = validation_result.results
        self.assertEqual(len(validators_results), 3)
        self.assertEqual(
            validators_results[json_shape_validator_id], json_shape_validator_response
        )
        self.assertEqual(
            validators_results[embeddings_similarity_validator_id],
            embeddings_similarity_validator_response,
        )
        self.assertEqual(
            validators_results[max_length_validator_id], max_length_validator_response
        )
        mock_json_shape_validator.assert_called_with(schema=self.schema)
        mock_embeddings_similarity_validator.assert_called_with(
            threshold=self.threshold,
            llm=self.llm,
            benchmark_content=self.benchmark_content,
        )
        mock_max_length_validator.assert_called_with(max_chars=self.max_chars)
