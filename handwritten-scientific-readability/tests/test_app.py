import unittest

from src.app import main_function


class TestApp(unittest.TestCase):
    def test_main_function_without_image(self):
        result = main_function()
        self.assertIsInstance(result, dict)
        self.assertIn("status", result)
        self.assertEqual(result["status"], "no_image_provided")


if __name__ == "__main__":
    unittest.main()