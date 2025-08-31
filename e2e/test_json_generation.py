import os
import unittest
import pytest
from strategies.json_validation_strategy import JSONValidationStrategy
from llm.llm import Llm
from config.config import Config
from llm.providers import Provider
from dotenv import load_dotenv


class TestJsonGeneration(unittest.TestCase):
    def setUp(self):
        load_dotenv()
        self.benchmark_content = """
        {
            "name": "josepha",
            "job": "beautician",
        }
        """
        self.max_chars = 50
        self.threshold = 0.95
        self.config = Config(
            llm_model="gpt-3.5-turbo",
            embedding_model="text-embedding-ada-002",
            api_key=os.environ["OPENAI_API_KEY"],
            provider="openai",
            additional_config={},
        )
        self.llm = Llm(
            provider=Provider(self.config.provider),
            config=self.config,
        )
        self.schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "job": {"type": "string"},
            },
            "required": ["name", "job"],
        }
        self.strategy = JSONValidationStrategy(
            llm=self.llm,
            schema=self.schema,
            max_chars=self.max_chars,
            threshold=self.threshold,
            benchmark_content=self.benchmark_content,
        )

    @pytest.mark.skip(
        reason="This test requires openai api key to be set in .env file."
    )
    def test_validate_matching_output(self):
        content = '{"name": "joseph", "job": "beautician"}'

        validation_result = self.strategy.run(content=content)

        self.assertTrue(validation_result.success)
        self.assertEqual(validation_result.content, content)
        self.assertEqual(validation_result.benchmark_content, self.benchmark_content)
        self.assertEqual(validation_result.threshold, self.threshold)

    @pytest.mark.skip(
        reason="This test requires openai api key to be set in .env file."
    )
    def test_validate_matching_output_with_too_many_chars(self):
        content = '{"name": "joseph", "job": "beautician", "birthplace": "martinique}'

        validation_result = self.strategy.run(content=content)

        self.assertFalse(validation_result.success)
        self.assertEqual(validation_result.content, content)
        self.assertEqual(validation_result.benchmark_content, self.benchmark_content)
        self.assertEqual(validation_result.threshold, self.threshold)

    @pytest.mark.skip(
        reason="This test requires openai api key to be set in .env file."
    )
    def test_validate_matching_output_with_wrong_values(self):
        content = '{"name": "robert", "job": "professor", "age": 20}'

        validation_result = self.strategy.run(content=content)

        self.assertFalse(validation_result.success)
        self.assertEqual(validation_result.content, content)
        self.assertEqual(validation_result.benchmark_content, self.benchmark_content)
        self.assertEqual(validation_result.threshold, self.threshold)
