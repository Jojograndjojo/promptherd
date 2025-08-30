from dataclasses import dataclass


@dataclass
class Response:
    valid: bool
    error: str | None
