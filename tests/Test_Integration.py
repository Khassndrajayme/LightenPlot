"""
Integration Tests for PlotEase Library
This file verifies that all components of the library work together seamlessly
in a typical data analysis workflow using the mtcars dataset.
"""
import unittest
import pandas as pd
# Import all relevant classes to test their interactions
from plotease import PlotEase, DiagnosticPlotter, SummaryGenerator


def load_mtcars():
    """Load mtcars dataset for testing"""
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


class TestIntegration(unittest.TestCase):
    """Integration tests - test class interactions with mtcars"""
    
    def setUp(self):
        """Set up test data - using mtcars"""
        self.mtcars = load_mtcars()
    
    def test_full_workflow(self):
        """Test complete workflow using all features with mtcars"""
        # Initialize PlotEase
        pe = PlotEase(self.mtcars, theme='minimal')
        
        # Test all main methods
        try:
            # Feature 1: AutoPlot
            pe.autoplot(target='mpg', max_plots=4)
            
            # Feature 2: Summary
            summary = pe.tabular_summary(style='full')
            self.assertIsInstance(summary, pd.DataFrame)
            self.assertEqual(len(summary), 11)
            
            # Feature 3: Model Comparison
            models = {
                'Linear Reg': {'Accuracy': 0.85, 'Precision': 0.82},
                'Random Forest': {'Accuracy': 0.90, 'Precision': 0.87}
            }
            pe.compare_models(models)
            
            # Feature 4: Quick Plot
            pe.quick_plot('hp', 'mpg', kind='scatter')
            
            success = True
        except Exception as e:
            print(f"Integration test failed: {e}")
            success = False
        
        self.assertTrue(success)
    
    def test_mtcars_specific_analysis(self):
        """Test mtcars-specific analysis workflows"""
        pe = PlotEase(self.mtcars)
        
        # Test correlations between car characteristics
        summary = pe.tabular_summary(style='numeric')
        
        # Verify we can analyze key car metrics
        columns = ['mpg', 'hp', 'wt', 'cyl']
        for col in columns:
            self.assertIn(col, self.mtcars.columns)
        
        # Test plotting key relationships
        try:
            pe.quick_plot('hp', 'mpg')  # Power vs efficiency
            pe.quick_plot('wt', 'mpg')  # Weight vs efficiency
            success = True
        except Exception:
            success = False
        
        self.assertTrue(success)


if __name__ == '__main__':
    # Run with verbose output when executed directly
    unittest.main(verbosity=2)
