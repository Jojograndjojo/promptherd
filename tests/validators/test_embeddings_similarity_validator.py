import unittest
from unittest.mock import patch, MagicMock
from validators.embeddings_similarity_validator import EmbeddingsSimilarityValidator


class TestEmbeddingsSimilarityValidator(unittest.TestCase):
    def test_id(self):
        llm = MagicMock()
        llm.embedding.return_value = (MagicMock(), MagicMock())
        validator = EmbeddingsSimilarityValidator(
            threshold=0.5, llm=llm, benchmark_content="benchmark"
        )
        self.assertEqual(validator.id(), "embeddings_similarity")

    @patch("validators.embeddings_similarity_validator.EmbeddingsAssessor")
    def test_returns_valid_response_if_embeddings_are_not_similar(
        self, mock_embeddings_assessor
    ):
        similarity_score = -0.5
        mock_embeddings_assessor.compare_similarity.return_value = similarity_score
        benchmark_embeddings = [1.0, 0.2, 0.3]
        benchmark_embeddings_response = {
            "object": "list",
            "data": [{"embedding": benchmark_embeddings}],
        }
        llm = MagicMock()
        llm.embedding.return_value = (
            benchmark_embeddings,
            benchmark_embeddings_response,
        )
        content = "result"
        benchmark_content = "benchmark"
        validator = EmbeddingsSimilarityValidator(
            threshold=0.5, llm=llm, benchmark_content=benchmark_content
        )

        result = validator.validate(content=content)
        self.assertEqual(
            result.error,
            f"The embeddings are not similar enough score {similarity_score}. Threshold: 0.5",
        )
        self.assertFalse(result.valid)

    @patch("validators.embeddings_similarity_validator.EmbeddingsAssessor")
    def test_returns_valid_response_if_embeddings_are_similar(
        self, mock_embeddings_assessor
    ):
        similarity_score = 0.5
        mock_embeddings_assessor.compare_similarity.return_value = similarity_score
        benchmark_embeddings = [1.0, 0.2, 0.3]
        benchmark_embeddings_response = {
            "object": "list",
            "data": [{"embedding": benchmark_embeddings}],
        }
        llm = MagicMock()
        llm.embedding.return_value = (
            benchmark_embeddings,
            benchmark_embeddings_response,
        )
        content = "result"
        benchmark_content = "benchmark"
        validator = EmbeddingsSimilarityValidator(
            threshold=0.5, llm=llm, benchmark_content=benchmark_content
        )
        result = validator.validate(content=content)
        self.assertIsNone(result.error)
        self.assertTrue(result.valid)
