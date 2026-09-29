import unittest

from tools.calculator import calculate


class CalculatorRegressionTests(unittest.TestCase):
    def test_normal_numbers_are_not_calculations(self):
        self.assertIsNone(calculate("I am 25 years old"))
        self.assertIsNone(calculate("iPhone 15 price"))
        self.assertIsNone(calculate("in 2024 what happened"))

    def test_date_like_values_are_not_calculations(self):
        self.assertIsNone(calculate("2024-05-12"))
        self.assertIsNone(calculate("12/5/2024"))
        self.assertIsNone(calculate("1/5/2024"))
        self.assertIsNone(calculate("5-2024"))

    def test_real_calculations_still_work(self):
        self.assertEqual(calculate("2+3*4"), "14")
        self.assertEqual(calculate("10-2-3"), "5")
        self.assertEqual(calculate("8/2/2"), "2")


if __name__ == "__main__":
    unittest.main()
