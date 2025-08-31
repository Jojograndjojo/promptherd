import unittest


class TestJsonGeneration(unittest.TestCase):
    def setUp(self):
        self.benchmark_content = """
        {
            "name": "josepha",
            "job": "beautician",
        }
        """
        self.max_chars = 100
        self.threshold = 0.9
