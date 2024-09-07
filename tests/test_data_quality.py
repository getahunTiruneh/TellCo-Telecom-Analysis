import os
import unittest
import pandas as pd
import numpy as np
os.chdir('...')
from src.data_quality_check import DataQualityCheck
class TestDataQuality(unittest.TestCase):
    def setUp(self):
        """Set up sample DataFrames for testing."""
        # DataFrame with missing data
        self.df_missing = pd.DataFrame({
            'A': [1, 2, np.nan, 4],
            'B': [np.nan, 2, np.nan, 4],
            'C': [1, 2, 3, 4]
        })

        # DataFrame without missing data
        self.df_no_missing = pd.DataFrame({
            'A': [1, 2, 3, 4],
            'B': [5, 6, 7, 8],
            'C': [9, 10, 11, 12]
        })

        # DataFrame with duplicate rows
        self.df_duplicates = pd.DataFrame({
            'A': [1, 2, 2, 4],
            'B': [5, 6, 6, 8],
            'C': [9, 10, 10, 12]
        })

        # DataFrame without duplicate rows
        self.df_no_duplicates = pd.DataFrame({
            'A': [1, 2, 3, 4],
            'B': [5, 6, 7, 8],
            'C': [9, 10, 11, 12]
        })
    def test_check_missing_data(self):
        """Test the check_missing_data method."""
        dq = DataQualityCheck(self.df_missing)
        result = dq.check_missing_data(self.df_missing)

        # Check if the result is a DataFrame when missing data exists
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.iloc[0]['Feature'], 'B')
        self.assertEqual(result.iloc[0]['Missing in %'], 50.0)

        # Check for DataFrame with no missing data
        dq_no_missing = DataQualityCheck(self.df_no_missing)
        result_no_missing = dq_no_missing.check_missing_data(self.df_no_missing)
        self.assertEqual(result_no_missing, "Success: No missing data!")
    def test_check_duplicate_data(self):
        """Test the check_duplicate_data method."""
        dq = DataQualityCheck(self.df_duplicates)
        result = dq.check_duplicate_data(self.df_duplicates)

        # Check if the result is a DataFrame when duplicates exist
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)  # There should be one duplicate row

        # Check for DataFrame with no duplicate data
        dq_no_duplicates = DataQualityCheck(self.df_no_duplicates)
        result_no_duplicates = dq_no_duplicates.check_duplicate_data(self.df_no_duplicates)
        self.assertEqual(result_no_duplicates, "Success: No duplicate data!")
if __name__ == '__main__':
    unittest.main()