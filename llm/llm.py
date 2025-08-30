import os
from llm.providers import Provider


class Llm:
    def __init__(self, provider: Provider, config: dict):
        _validate_config(provider=provider, config=config)
        _set_environment_variables(provider=provider, config=config)
        self.api_key = config["api_key"]
        self.llm_model = config["llm_model"]
        self.embedding_model = config["embedding_model"]


def _validate_config(provider: Provider, config: dict):
    if provider == Provider.OPENAI:
        _validate_openai_config(config)

    if "llm_model" not in config:
        raise ValueError(f"llm_model is missing from config {config}")

    if "embedding_model" not in config:
        raise ValueError(f"embedding_model is missing from config {config}")


def _validate_openai_config(config):
    if "api_key" not in config:
        raise ValueError(f"api_key is missing from config {config}")


def _set_environment_variables(provider: Provider, config: dict):
    _set_openai_environment_variables(provider=provider, config=config)


def _set_openai_environment_variables(provider: Provider, config: dict):
    if provider == Provider.OPENAI:
        os.environ["OPENAI_API_KEY"] = config["api_key"]
