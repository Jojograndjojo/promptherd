import unittest
from llm.llm import Llm
from llm.providers import Provider
from unittest.mock import patch


class TestLlmOpenAI(unittest.TestCase):
    def test_init_fails_if_no_api_key_in_config(self):
        config = {"llm_model": "test", "embedding_model": "test"}

        with self.assertRaises(ValueError) as context:
            Llm(provider=Provider.OPENAI, config=config)
        self.assertEqual(
            str(context.exception), f"api_key is missing from config {config}"
        )

    def test_init_fails_if_no_llm_model_in_config(self):
        config = {"api_key": "test", "embedding_model": "test"}

        with self.assertRaises(ValueError) as context:
            Llm(provider=Provider.OPENAI, config=config)
        self.assertEqual(
            str(context.exception), f"llm_model is missing from config {config}"
        )

    def test_init_fails_if_no_embedding_model_in_config(self):
        config = {"api_key": "test", "llm_model": "test"}

        with self.assertRaises(ValueError) as context:
            Llm(provider=Provider.OPENAI, config=config)
        self.assertEqual(
            str(context.exception), f"embedding_model is missing from config {config}"
        )

    @patch("llm.llm.os")
    def test_init_succeeds_if_all_required_keys_in_config(self, mock_os):
        api_key = "test_api_key"
        llm_model = "test_llm_model"
        embedding_model = "test_embedding_model"

        config = {
            "api_key": api_key,
            "llm_model": llm_model,
            "embedding_model": embedding_model,
        }

        llm = Llm(provider=Provider.OPENAI, config=config)

        self.assertEqual(llm.api_key, api_key)
        self.assertEqual(llm.llm_model, llm_model)
        self.assertEqual(llm.embedding_model, embedding_model)
        mock_os.environ.__setitem__.assert_called_with("OPENAI_API_KEY", api_key)
