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


class TestLlm(unittest.TestCase):
    def setUp(self):
        self.llm = Llm(
            provider=Provider.DUMMY,
            config={
                "llm_model": "test_llm_model",
                "embedding_model": "test_embedding_model",
            },
        )

    @patch("llm.llm.embedding")
    def test_embedding(self, mock_embedding):
        produced_embeddings = [0, 1, 1, 1]
        prompt_tokens = 10
        total_tokens = 10
        produced_response = {
            "object": "list",
            "data": [
                {"object": "embedding", "index": 0, "embedding": produced_embeddings}
            ],
            "model": "test_embedding_model",
            "usage": {"prompt_tokens": prompt_tokens, "total_tokens": total_tokens},
        }
        mock_embedding.return_value = produced_response
        embedding_input = "test input"
        embeddings, response = self.llm.embedding(embedding_input=embedding_input)
        self.assertEqual(embeddings, produced_embeddings)
        self.assertEqual(response, produced_response)
        self.assertEqual(self.llm.total_tokens, total_tokens)
        mock_embedding.assert_called_with(
            model=self.llm.embedding_model, input=embedding_input
        )

    @patch("llm.llm.completion")
    def test_generate(self, mock_completion):
        returned_content = "test response"
        total_tokens = 41
        produced_response = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {"role": "assistant", "content": returned_content},
                }
            ],
            "created": 1691429984.3852863,
            "model": "claude-instant-1",
            "usage": {
                "prompt_tokens": 18,
                "completion_tokens": 23,
                "total_tokens": total_tokens,
            },
        }
        mock_completion.return_value = produced_response
        messages = [{"role": "user", "content": "Hey!"}]
        result, response = self.llm.completion(messages=messages)
        self.assertEqual(result, returned_content)
        self.assertEqual(response, produced_response)
        self.assertEqual(self.llm.total_tokens, total_tokens)
        mock_completion.assert_called_with(model=self.llm.llm_model, messages=messages)
