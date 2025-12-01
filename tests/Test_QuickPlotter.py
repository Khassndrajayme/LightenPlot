"""
Unit Tests for QuickPlotter Component
This file verifies the functionality of the QuickPlotter, focusing on
automatic plot type detection and visualization execution.
"""
import unittest
import pandas as pd
# Import the classes specific to this component's tests
from plotease import QuickPlotter, VisualizationBase


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
               2.465, 3.520, 3.435, 3.840, 3.845, 1.935, 2.140, 1.513, 3.170, 2.770, 3.570, 2.780],
        'am': [1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
               0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    })
    return mtcars


class TestQuickPlotter(unittest.TestCase):
    """Test QuickPlotter functionality, focusing on plot detection and quick plotting."""
    
    def setUp(self):
        """Set up test data using mtcars."""
        self.mtcars = load_mtcars()
        self.qp = QuickPlotter(self.mtcars)
        
    
    # Initialization and Inheritance Tests
        
    def test_initialization(self):
        """Test QuickPlotter initialization."""
        self.assertIsNotNone(self.qp)
        
    def test_inheritance_from_base(self):
        """Test that QuickPlotter inherits from VisualizationBase."""
        self.assertIsInstance(self.qp, VisualizationBase)

    # Plot Type Detection Tests

    def test_detect_plot_type_single_numeric(self):
        """Test automatic plot type detection for a single numeric variable (should be 'hist')."""
        plot_type = self.qp.detect_plot_type('mpg', None)
        self.assertEqual(plot_type, 'hist')
        
    def test_detect_plot_type_two_numeric(self):
        """Test automatic plot type detection for two numeric variables (should be 'scatter')."""
        plot_type = self.qp.detect_plot_type('mpg', 'hp')
        self.assertEqual(plot_type, 'scatter')

    def test_detect_plot_type_numeric_vs_discrete(self):
        """Test detection for numeric vs discrete variables (should be 'boxplot' or 'bar')."""
        # Cylinders ('cyl') has few unique values (4, 6, 8) and often treated as categorical/discrete.
        # Assuming the library defaults to 'boxplot' or similar for this numeric vs discrete grouping.
        # We test based on the expected output from the internal detection logic, assuming 'boxplot'.
        plot_type = self.qp.detect_plot_type('mpg', 'cyl')
        
        # Accept 'box', 'boxplot', 'bar' as valid outputs for this scenario
        valid_plots = ['box', 'boxplot', 'bar']
        self.assertTrue(plot_type in valid_plots, f"Expected one of {valid_plots}, got {plot_type}")
        
   
    # Plot Execution Tests

    def test_quick_plot_scatter(self):
        """Test quick_plot executes a scatter plot (numeric vs numeric) without error."""
        try:
            self.qp.quick_plot('hp', 'mpg', kind='scatter')
            success = True
        except Exception as e:
            print(f"QuickPlotter scatter failed: {e}")
            success = False
            
        self.assertTrue(success)

    def test_quick_plot_histogram(self):
        """Test quick_plot executes a histogram (single numeric) without error."""
        try:
            self.qp.quick_plot('mpg', kind='hist')
            success = True
        except Exception as e:
            print(f"QuickPlotter histogram failed: {e}")
            success = False
            
        self.assertTrue(success)
        
    def test_quick_plot_automatic_detection(self):
        """Test quick_plot executes successfully using automatic detection (kind=None)."""
        try:
            # Should automatically detect scatter plot
            self.qp.quick_plot('wt', 'mpg', kind=None)
            
            # Should automatically detect histogram
            self.qp.quick_plot('hp', kind=None)
            
            success = True
        except Exception as e:
            print(f"QuickPlotter automatic detection failed: {e}")
            success = False
            
        self.assertTrue(success)


if __name__ == '__main__':
    # Run with verbose output when executed directly
    unittest.main(verbosity=2)
