import os
from llm.providers import Provider
from litellm import embedding, completion


class Llm:
    def __init__(self, provider: Provider, config: dict):
        _validate_config(provider=provider, config=config)
        _set_environment_variables(provider=provider, config=config)
        self.api_key = config.get("api_key", "")
        self.llm_model = config.get("llm_model", "")
        self.embedding_model = config.get("embedding_model", "")
        self.total_tokens = 0

    def embedding(self, embedding_input: str) -> tuple[list, dict]:
        """returns vector embeddings representation of a string and the full embeddings response"""
        response = embedding(model=self.embedding_model, input=embedding_input)
        self.total_tokens += response["usage"]["total_tokens"]
        return response["data"][0]["embedding"], response

    def completion(self, messages: list[dict]) -> tuple[list, dict]:
        """returns a string response and the full completion response"""
        response = completion(model=self.llm_model, messages=messages)
        self.total_tokens += response["usage"]["total_tokens"]
        return response["choices"][0]["message"]["content"], response


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
