import unittest
from embeddings.assessor import EmbeddingsAssessor


class TestEmbeddingsAssessor(unittest.TestCase):
    def test_compare_similarity(self):
        similarity = EmbeddingsAssessor.compare_similarity(
            embedding1=[1.0, 0.2, 0.3], embedding2=[1.0, 0.2, 0.3]
        )
        self.assertEqual(similarity, 1.0)

        similarity = EmbeddingsAssessor.compare_similarity(
            embedding1=[-0.9, -0.9, -0.9], embedding2=[0.4, 0.4, 0.4]
        )
        self.assertEqual(similarity, -1.0)
