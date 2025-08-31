from dataclasses import dataclass


@dataclass
class Response:
    valid: bool
    error: str | None


@dataclass
class ValidationResult:
    success: bool
    content: str
    benchmark_content: str
    threshold: float
    results: dict[str, Response]
