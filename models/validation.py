import yaml
from dataclasses import dataclass, asdict


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

    def to_yaml(self) -> str:
        yaml_str = yaml.dump(asdict(self), sort_keys=False)
        return yaml_str
