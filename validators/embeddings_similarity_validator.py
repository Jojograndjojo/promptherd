from embeddings.assessor import EmbeddingsAssessor
from validators.prompt_validator import PromptValidator
from models.validation import Response
from llm.llm import Llm


class EmbeddingsSimilarityValidator(PromptValidator):
    def __init__(self, benchmark_content: str, llm: Llm, threshold: float):
        self.benchmark_content = benchmark_content
        self.llm = llm
        self.embedded_benchmark_content = self._embed_benchmark_content()
        self.threshold = threshold

    def id(self) -> str:
        """id of the validator"""
        return "embeddings_similarity"

    def validate(self, content: str) -> Response:
        """validate if content matches benchmark content"""
        similarity_score = EmbeddingsAssessor.compare_similarity(
            embedding1=self.embedded_benchmark_content,
            embedding2=self._embed_content(content=content),
        )
        if similarity_score < self.threshold:
            return Response(
                valid=False,
                error=f"The embeddings are not similar enough score {similarity_score}. Threshold: {self.threshold}",
            )
        else:
            return Response(valid=True, error=None)

    def _embed_benchmark_content(self) -> list[float]:
        embeddings, _ = self.llm.embedding(embedding_input=self.benchmark_content)
        return embeddings

    def _embed_content(self, content: str) -> list[float]:
        embeddings, _ = self.llm.embedding(embedding_input=content)
        return embeddings
