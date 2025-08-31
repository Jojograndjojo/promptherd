from strategies.validation_strategy import ValidationStrategy
from models.validation import ValidationResult
from llm.llm import Llm
from validators.json_shape_validator import JsonShapeValidator
from validators.max_length_validator import MaxLengthValidator
from validators.embeddings_similarity_validator import EmbeddingsSimilarityValidator


class JSONValidationStrategy(ValidationStrategy):
    def __init__(
        self,
        llm: Llm,
        schema: dict,
        max_chars: int | None,
        threshold: float,
        benchmark_content: str,
    ):
        self.benchmark_content = benchmark_content
        self.threshold = threshold
        json_shape_validator = JsonShapeValidator(schema=schema)
        embeddings_similarity_validator = EmbeddingsSimilarityValidator(
            benchmark_content=benchmark_content,
            llm=llm,
            threshold=threshold,
        )
        validators: list[
            JsonShapeValidator | EmbeddingsSimilarityValidator | MaxLengthValidator
        ] = [
            json_shape_validator,
            embeddings_similarity_validator,
        ]
        if max_chars is not None:
            max_length_validator = MaxLengthValidator(max_chars=max_chars)
            validators.append(max_length_validator)

        self.validators = validators

    def run(self, content: str) -> ValidationResult:
        results = {
            validator.id(): validator.validate(content=content)
            for validator in self.validators
        }

        success = True
        for result in results.values():
            if not result.valid:
                success = False
                break

        return ValidationResult(
            success=success,
            content=content,
            benchmark_content=self.benchmark_content,
            threshold=self.threshold,
            results=results,
        )
