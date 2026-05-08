import sys
import os
import unittest
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from strategies.analyzer import MarketAnalyzer

class TestMarketAnalyzer(unittest.TestCase):
    def setUp(self):
        # Create a mock 15m DataFrame
        data = {
            'c': np.linspace(100, 110, 50),
            'o': np.linspace(99, 109, 50),
            'h': np.linspace(101, 111, 50),
            'l': np.linspace(98, 108, 50),
            'v': np.random.randint(100, 1000, 50)
        }
        self.df_trend = pd.DataFrame(data)
        
        # Create a mock Ranging DataFrame
        data_range = {
            'c': np.sin(np.linspace(0, 10, 50)) * 5 + 100,
            'o': np.sin(np.linspace(0, 10, 50)) * 5 + 100,
            'h': np.sin(np.linspace(0, 10, 50)) * 5 + 102,
            'l': np.sin(np.linspace(0, 10, 50)) * 5 + 98,
            'v': np.random.randint(100, 1000, 50)
        }
        self.df_range = pd.DataFrame(data_range)

    def test_regime_detection(self):
        regime_trend = MarketAnalyzer.detect_regime(self.df_trend)
        self.assertIn(regime_trend, ["TRENDING", "RANGING", "VOLATILE"])

        regime_range = MarketAnalyzer.detect_regime(self.df_range)
        self.assertIn(regime_range, ["TRENDING", "RANGING", "VOLATILE"])

    def test_ema_calculation(self):
        ema_9 = MarketAnalyzer.get_ema(self.df_trend['c'], 9)
        self.assertEqual(len(ema_9), 50)
        self.assertFalse(pd.isna(ema_9.iloc[-1]))

    def test_predict_liquidation(self):
        clusters = MarketAnalyzer.predict_liquidation_clusters(self.df_trend)
        self.assertIsNotNone(clusters)
        self.assertIn("short_liq", clusters)
        self.assertIn("long_liq", clusters)
        
        # Highest high is 111, so long liq should be below it
        # Lowest low is 98, so short liq should be above it
        self.assertTrue(all(p < 111 for p in clusters["long_liq"]))
        self.assertTrue(all(p > 98 for p in clusters["short_liq"]))

    def test_structure_detection(self):
        status, is_brk, r_h, r_l = MarketAnalyzer.detect_structure(self.df_trend)
        self.assertIn(status, ["BULLISH", "BEARISH", "CHOP"])
        self.assertIsNotNone(r_h)
        self.assertIsNotNone(r_l)

if __name__ == '__main__':
    unittest.main()
