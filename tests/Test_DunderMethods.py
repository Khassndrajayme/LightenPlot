"""
Unit Tests for Dunder Methods (__methods__)
This file verifies the Pythonic integration of PlotEase through operator overloading.
"""

import unittest
import pandas as pd
# Import the main class that implements/overrides the dunder methods
from plotease import PlotEase


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
        'disp': [160.0, 160.0, 108.0, 258.0, 360.0, 225.0, 360.0, 146.7, 140.8, 167.6,
                 167.6, 275.8, 275.8, 275.8, 472.0, 460.0, 440.0, 78.7, 75.7, 71.1,
                 120.1, 318.0, 304.0, 350.0, 400.0, 79.0, 120.3, 95.1, 351.0, 145.0, 301.0, 121.0],
        'hp': [110, 110, 93, 110, 175, 105, 245, 62, 95, 123,
               123, 180, 180, 180, 205, 215, 230, 66, 52, 65,
               97, 150, 150, 245, 175, 66, 91, 113, 264, 175, 335, 109],
        'wt': [2.620, 2.875, 2.320, 3.215, 3.440, 3.460, 3.570, 3.190, 3.150, 3.440,
               3.440, 4.070, 3.730, 3.780, 5.250, 5.424, 5.345, 2.200, 1.615, 1.835,
               2.465, 3.520, 3.435, 3.840, 3.845, 1.935, 2.140, 1.513, 3.170, 2.770, 3.570, 2.780]
    })
    return mtcars


class TestDunderMethods(unittest.TestCase):
    """Test dunder methods (magic methods) defined in PlotEase."""
    
    def setUp(self):
        """Set up test data - using mtcars and a subset."""
        self.mtcars = load_mtcars()
        self.mtcars_subset = self.mtcars.head(10)
    
    def test_repr(self):
        """Test __repr__ method for correct developer-friendly string representation."""
        pe = PlotEase(self.mtcars, theme='colorful')
        repr_str = repr(pe)
        # Check for class name, row count (32), column count (11), and theme
        self.assertIn('PlotEase', repr_str)
        self.assertIn('rows=32', repr_str)
        # Note: If you included all mtcars columns (11), adjust this assertion if needed
        self.assertIn('cols=5', repr_str) # Based on the subset in load_mtcars() above
        self.assertIn("theme='colorful'", repr_str)
    
    def test_len(self):
        """Test __len__ method, allowing len(pe) to return the row count."""
        pe_full = PlotEase(self.mtcars)  # 32 rows
        self.assertEqual(len(pe_full), 32)
        
        pe_subset = PlotEase(self.mtcars_subset) # 10 rows
        self.assertEqual(len(pe_subset), 10)
    
    def test_eq(self):
        """Test __eq__ method (==) for object equality based on data and theme."""
        # Case 1: Identical data and theme
        pe1 = PlotEase(self.mtcars, theme='minimal')
        pe2 = PlotEase(self.mtcars, theme='minimal')
        self.assertEqual(pe1, pe2)
        
        # Case 2: Different data (should be False)
        pe3 = PlotEase(self.mtcars_subset, theme='minimal')
        self.assertNotEqual(pe1, pe3)
        
    def test_lt(self):
        """Test __lt__ method (<) for comparison based on data size (row count)."""
        pe_full = PlotEase(self.mtcars)      # 32 cars
        pe_subset = PlotEase(self.mtcars_subset) # 10 cars
        
        self.assertFalse(pe_full < pe_subset)  # 32 < 10 = False
        self.assertTrue(pe_subset < pe_full)   # 10 < 32 = True
        self.assertFalse(pe_full < pe_full)    # 32 < 32 = False


if __name__ == '__main__':
    # Run with verbose output when executed directly
    unittest.main(verbosity=2)               