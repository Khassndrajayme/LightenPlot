"""
Unit Tests for SummaryGenerator Component
This file verifies the functionality of the data summarization component,
including calculation accuracy and output formatting.
"""
import unittest
import pandas as pd
import numpy as np
# Import the class specific to this component's tests
from plotease import SummaryGenerator


def load_mtcars():
    """Load mtcars dataset for testing"""
    # NOTE: This data is copied locally to ensure the test is self-contained.
    mtcars = pd.DataFrame({
        'mpg': [21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2, 
                17.8, 16.4, 17.3, 15.2, 10.4, 10.4, 14.7, 32.4, 30.4, 33.9,
                21.5, 15.5, 15.2, 13.3, 19.2, 27.3, 26.0, 30.4, 15.8, 19.7, 15.0, 21.4],
        'cyl': [6, 6, 4, 6, 8, 6, 8, 4, 4, 6, 
                6, 8, 8, 8, 8, 8, 8, 4, 4, 4,
                4, 8, 8, 8, 8, 4, 4, 4, 8, 6, 8, 4],
        'hp': [110, 110, 93, 110, 175, 105, 245, 62, 95, 123,
               123, 180, 180, 180, 205, 215, 230, 66, 52, 65,
               97, 150, 150, 245, 175, 66, 91, 113, 264, 175, 335, 109],
        'wt': [2.620, 2.875, 2.320, 3.215, 3.440, 3.460, 3.570, 3.190, 3.150, 3.440,
               3.440, 4.070, 3.730, 3.780, 5.250, 5.424, 5.345, 2.200, 1.615, 1.835,
               2.465, 3.520, 3.435, 3.840, 3.845, 1.935, 2.140, 1.513, 3.170, 2.770, 3.570, 2.780]
    })
    return mtcars


class TestSummaryGenerator(unittest.TestCase):
    """Test SummaryGenerator functionality."""
    
    def setUp(self):
        """Set up test data - using mtcars."""
        self.mtcars = load_mtcars()
        # All 4 columns are numeric in this subset
        self.numeric_cols = list(self.mtcars.columns) 
        
    def test_initialization(self):
        """Test SummaryGenerator initialization."""
        sg = SummaryGenerator(self.mtcars)
        self.assertIsNotNone(sg)
        
    def test_len_dunder(self):
        """Test __len__ method returns the number of rows."""
        sg = SummaryGenerator(self.mtcars)
        self.assertEqual(len(sg), 32)  # 32 cars in mtcars
        
    # summarize_numeric (List Output)

    def test_summarize_numeric(self):
        """Test numeric column summarization returns a list of dictionaries."""
        sg = SummaryGenerator(self.mtcars)
        numeric_summary = sg.summarize_numeric()
        
        self.assertIsInstance(numeric_summary, list)
        self.assertEqual(len(numeric_summary), len(self.numeric_cols))
        
        # Check structure of the first summary item
        first_summary = numeric_summary[0]
        self.assertIn('Column', first_summary)
        self.assertIn('Mean', first_summary)
        self.assertIn('Std', first_summary)
        self.assertIn('Min', first_summary)
        self.assertIn('Max', first_summary)

    # tabular_summary (DataFrame Output)

        
    def test_tabular_summary_full(self):
        """Test full tabular summary returns a DataFrame with all columns."""
        sg = SummaryGenerator(self.mtcars)
        summary_df = sg.tabular_summary(style='full')
        
        self.assertIsInstance(summary_df, pd.DataFrame)
        self.assertEqual(len(summary_df), len(self.numeric_cols))
        
    def test_tabular_summary_numeric_only(self):
        """Test numeric-only summary returns a DataFrame with the correct count."""
        sg = SummaryGenerator(self.mtcars)
        summary_df = sg.tabular_summary(style='numeric')
        
        self.assertIsInstance(summary_df, pd.DataFrame)
        self.assertEqual(len(summary_df), len(self.numeric_cols))
        

    # Accuracy Test
   
    def test_summary_statistics_accuracy(self):
        """Test that summary statistics for a known column are accurate."""
        sg = SummaryGenerator(self.mtcars)
        summary = sg.tabular_summary(style='numeric')
        
        # Calculate expected mean for 'mpg' using pandas built-in mean
        expected_mean = self.mtcars['mpg'].mean()
        
        # Extract 'mpg' row from the SummaryGenerator output
        mpg_row = summary[summary['Column'] == 'mpg'].iloc[0]
        
        # Verify row count
        self.assertEqual(int(mpg_row['Count']), 32)
        
        # Verify mean with a small tolerance (delta)
        actual_mean = float(mpg_row['Mean'])
        self.assertAlmostEqual(actual_mean, expected_mean, delta=0.01)


if __name__ == '__main__':
    # Run with verbose output when executed directly
    unittest.main(verbosity=2)
