import unittest
import numpy as np
import trader_simulator


class TestTrader(unittest.TestCase):
    def test_sequence_one(self):
        initial_capital = 10000
        current_prices = np.array([2, 2, 3, 4])
        predictions = np.array([2, 1, 10, 3])
        roi = trader_simulator.simulate_trade(current_prices, predictions, initial_capital)
        self.assertEqual(roi, (13333 - initial_capital) / initial_capital)

    def test_sequence_two(self):
        initial_capital = 10000
        current_prices = np.array([1, 2, 3, 4])
        predictions = np.array([2, 1, 10, 5])
        roi = trader_simulator.simulate_trade(current_prices, predictions, initial_capital)
        self.assertEqual(roi, (26666 - initial_capital) / initial_capital)

    def test_consecutive_buys(self):
        initial_capital = 191
        current_prices = np.array([100, 90, 3, 4, 1, 10])
        predictions = np.array([1000, 1000, 1000, 1000, 1000, 0])
        roi = trader_simulator.simulate_trade(current_prices, predictions, initial_capital)
        self.assertEqual(roi, (30 - initial_capital) / initial_capital)


if __name__ == "__main__":
    unittest.main()
