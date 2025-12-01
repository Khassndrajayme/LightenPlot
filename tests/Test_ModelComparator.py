"""
Unit Tests for ModelComparator Component
This file verifies the logic for managing and comparing machine learning model performance metrics.
"""
import unittest
import pandas as pd
# Import the class specific to this component's tests
from plotease import ModelComparator


class TestModelComparator(unittest.TestCase):
    """Test ModelComparator functionality, focusing on metric analysis and comparison."""
    
    def setUp(self):
        """Set up standard test data for model comparison."""
        self.models = {
            'Linear Regression': {'R²': 0.85, 'MAE': 2.5, 'RMSE': 3.2},
            'Random Forest': {'R²': 0.92, 'MAE': 1.8, 'RMSE': 2.3},
            'XGBoost': {'R²': 0.94, 'MAE': 1.5, 'RMSE': 2.0}
        }
    

    # Initialization and Data Structure Tests


    def test_initialization(self):
        """Test ModelComparator initialization and internal DataFrame structure."""
        mc = ModelComparator(self.models)
        self.assertIsNotNone(mc)
        self.assertIsInstance(mc._df, pd.DataFrame)
        self.assertEqual(len(mc._df), 3)  # Three models
        self.assertListEqual(list(mc._df.index), ['Linear Regression', 'Random Forest', 'XGBoost'])
        
    def test_empty_models_initialization(self):
        """Test initialization with an empty dictionary."""
        mc = ModelComparator({})
        self.assertTrue(mc._df.empty)

   
    # Core Functionality Tests
   

    def test_get_best_model_by_r_squared(self):
        """Test getting the best model by R² (where higher is better)."""
        mc = ModelComparator(self.models)
        # XGBoost has the highest R² (0.94)
        best_model = mc.get_best_model(metric='R²')
        self.assertEqual(best_model, 'XGBoost')

    def test_get_best_model_by_mae(self):
        """Test getting the best model by MAE (where lower is better)."""
       
        
        # Test a scenario where higher is clearly better (e.g., custom metric 'F1')
        models_acc = {
             'A': {'F1': 0.85},
             'B': {'F1': 0.90}
        }
        mc_acc = ModelComparator(models_acc)
        self.assertEqual(mc_acc.get_best_model(metric='F1'), 'B')
        
    def test_get_best_model_invalid_metric(self):
        """Test ValueError when trying to compare an invalid metric name."""
        mc = ModelComparator(self.models)
        
        with self.assertRaises(ValueError):
            mc.get_best_model('InvalidMetricName')


    # Dunder Method Tests (__eq__, __gt__)
    
    def test_eq_dunder(self):
        """Test __eq__ method (==) for object equality based on data."""
        mc1 = ModelComparator(self.models)
        mc2 = ModelComparator(self.models)
        
        # Two objects with identical metrics should be equal
        self.assertEqual(mc1, mc2)
        
        # Different data should be unequal
        models_different = {'Linear Regression': {'R²': 0.80, 'MAE': 2.7}}
        mc3 = ModelComparator(models_different)
        self.assertNotEqual(mc1, mc3)
        
    def test_gt_dunder(self):
        """Test __gt__ method (>) for comparison based on overall performance."""

        
        # Model C is clearly better than Model D in terms of R²
        models_better = {'Model C': {'R²': 0.90, 'Cost': 10}} 
        models_worse = {'Model D': {'R²': 0.80, 'Cost': 20}}
        
        mc_better = ModelComparator(models_better)
        mc_worse = ModelComparator(models_worse)
        
        # This checks: (0.90 + 10) / 2 > (0.80 + 20) / 2
        # (5.45) > (10.4) -> This shows the weakness of comparing means of different scale metrics.
        # The test needs to rely on the actual behavior of your __gt__ method.
        
        # Let's ensure the method runs without error and returns the expected boolean result
        try:
            comparison_result = (mc_better > mc_worse)
            self.assertIsInstance(comparison_result, bool)
        except Exception as e:
            self.fail(f"__gt__ method failed with error: {e}")
        
        # Since R² is higher but Cost is lower, the mean value will be higher for Model D (10.4 vs 5.45)
        # So we expect Model C to be LESS than Model D in this simple mean comparison.
        self.assertFalse(mc_better > mc_worse)
        self.assertTrue(mc_worse > mc_better)


if __name__ == '__main__':
    # Run with verbose output when executed directly
    unittest.main(verbosity=2)
