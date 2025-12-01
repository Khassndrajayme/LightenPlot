"""
Unit Tests for DiagnosticPlotter Component
This file verifies the functionality of the multi-plot, diagnostic visualization component.
"""
import unittest
import pandas as pd
# Import the classes specific to this component's tests
from plotease import DiagnosticPlotter, VisualizationBase


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


class TestDiagnosticPlotter(unittest.TestCase):
    """Test DiagnosticPlotter functionality, focusing on autoplot execution."""
    
    def setUp(self):
        """Set up test data using mtcars."""
        self.mtcars = load_mtcars()
        
    def test_initialization(self):
        """Test DiagnosticPlotter initialization."""
        dp = DiagnosticPlotter(self.mtcars)
        self.assertIsNotNone(dp)
        self.assertIsInstance(dp, DiagnosticPlotter)
    
    def test_inheritance_from_base(self):
        """Test that DiagnosticPlotter inherits from VisualizationBase."""
        dp = DiagnosticPlotter(self.mtcars)
        self.assertIsInstance(dp, VisualizationBase)
        
    def test_autoplot_runs_without_error(self):
        """Test that autoplot executes without errors for a regression target ('mpg')."""
        dp = DiagnosticPlotter(self.mtcars)
        try:
            # Test autoplot with 'mpg' as the target variable
            dp.autoplot(target='mpg', max_plots=6)
            success = True
        except Exception as e:
            print(f"DiagnosticPlotter's autoplot failed with error: {e}")
            success = False
            
        self.assertTrue(success, "Autoplot failed to execute for target 'mpg'.")
        
    def test_autoplot_with_different_targets(self):
        """Test autoplot with different target variables to ensure robustness."""
        dp = DiagnosticPlotter(self.mtcars)
        
        # Test with 'hp' (Horsepower)
        try:
            dp.autoplot(target='hp', max_plots=4)
            success_hp = True
        except Exception:
            success_hp = False
            
        # Test with 'wt' (Weight)
        try:
            dp.autoplot(target='wt', max_plots=4)
            success_wt = True
        except Exception:
            success_wt = False
            
        self.assertTrue(success_hp, "Autoplot failed to execute for target 'hp'.")
        self.assertTrue(success_wt, "Autoplot failed to execute for target 'wt'.")


if __name__ == '__main__':
    # Run with verbose output when executed directly
    unittest.main(verbosity=2)
