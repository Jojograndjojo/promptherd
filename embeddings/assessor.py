import numpy as np


class EmbeddingsAssessor:
    @staticmethod
    def compare_similarity(embedding1, embedding2) -> float:
        return cosine_similarity(embedding1, embedding2)


def cosine_similarity(vec1: list[float], vec2: list[float]):
    v1 = np.array(vec1)
    v2 = np.array(vec2)

    cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return cos_sim
