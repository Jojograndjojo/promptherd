import os
import unittest
from config.config import Config
from unittest.mock import patch, mock_open


class TestConfig(unittest.TestCase):
    def setUp(self):
        self.plain_config = """
provider: provider1
api_key: api_key1
llm_model: llm_model1
embedding_model: embedding_model1
additional_config:
    - key1: value1
      key2: value2
"""

        self.config_with_secret_api_key = """
provider: provider1
api_key: ${API_KEY}
llm_model: llm_model1
embedding_model: embedding_model1
additional_config:
    - key1: value1
      key2: value2
"""

        self.config_path = "path/to/config.yaml"

    def test_create_plain_config(self):
        with patch("builtins.open", mock_open(read_data=self.plain_config)):
            config = Config.from_path(path=self.config_path)
            self.assertEqual(config.provider, "provider1")
            self.assertEqual(config.api_key, "api_key1")
            self.assertEqual(config.llm_model, "llm_model1")
            self.assertEqual(config.embedding_model, "embedding_model1")
            self.assertEqual(
                config.additional_config, [{"key1": "value1", "key2": "value2"}]
            )

    def test_create_config_with_secret_api_key(self):
        os.environ["API_KEY"] = "api_key1"
        with patch(
            "builtins.open", mock_open(read_data=self.config_with_secret_api_key)
        ):
            config = Config.from_path(path=self.config_path)
            self.assertEqual(config.provider, "provider1")
            self.assertEqual(config.api_key, "api_key1")
            self.assertEqual(config.llm_model, "llm_model1")
            self.assertEqual(config.embedding_model, "embedding_model1")
            self.assertEqual(
                config.additional_config, [{"key1": "value1", "key2": "value2"}]
            )
