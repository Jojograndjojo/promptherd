import os
import yaml
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class Config:
    provider: str
    api_key: str
    llm_model: str
    embedding_model: str
    additional_config: dict

    @classmethod
    def from_path(cls, path: str):
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        if not data:
            raise ValueError("Config is empty")

        return cls(
            provider=__replace_by_env_var__(data.get("provider", "")),
            api_key=__replace_by_env_var__(data.get("api_key", "")),
            llm_model=__replace_by_env_var__(data.get("llm_model", "")),
            embedding_model=__replace_by_env_var__(data.get("embedding_model", "")),
            additional_config=__replace_by_env_var__(data.get("additional_config", "")),
        )


def __replace_by_env_var__(value):
    load_dotenv()
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        env_variable = value[2:-1]
        return os.environ.get(env_variable, value)
    else:
        return value
