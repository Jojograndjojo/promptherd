from enum import Enum


class Provider(Enum):
    ANTHROPIC = "anthropic"
    DUMMY = "dummy"
    OLLAMA = "ollama"
    OPENAI = "openai"
