"""
Unit Tests for Composition
This file verifies that the PlotEase Facade class is composed of 
and correctly delegates tasks to its specialized component classes.
"""

import unittest
import pandas as pd
# Import all relevant classes to test their composition
from plotease import (
    PlotEase, 
    DiagnosticPlotter, 
    SummaryGenerator,
    ModelComparator,
    QuickPlotter
)


class TestComposition(unittest.TestCase):
    """Test composition relationships and delegation in PlotEase."""
    
    def setUp(self):
        """Set up standard test data for PlotEase initialization."""
        self.data = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [10, 20, 30]
        })

    
# Composition Checks (Verifying 'Has-A' Relationship)
    def test_plotease_contains_diagnostic_plotter(self):
        """Test that PlotEase contains a DiagnosticPlotter instance."""
        pe = PlotEase(self.data)
        # Check the internal attribute used for diagnostic plotting
        self.assertIsInstance(pe._diagnostic, DiagnosticPlotter)
    
    def test_plotease_contains_summary_generator(self):
        """Test that PlotEase contains a SummaryGenerator instance."""
        pe = PlotEase(self.data)
        # Check the internal attribute used for tabular summaries
        self.assertIsInstance(pe._summary, SummaryGenerator)
        
    def test_plotease_contains_model_comparator(self):
        """Test that PlotEase contains a ModelComparator instance."""
        pe = PlotEase(self.data)
        # ModelComparator might be initialized lazily or on instantiation, checking the attribute
        # Assuming it's initialized as part of the PlotEase setup
        self.assertIsInstance(pe._comparator, ModelComparator)
        
    def test_plotease_contains_quick_plotter(self):
        """Test that PlotEase contains a QuickPlotter instance."""
        pe = PlotEase(self.data)
        # Check the internal attribute used for quick plotting
        self.assertIsInstance(pe._plotter, QuickPlotter)


 # Delegation Checks_Verifying Functionality Transfer     
    def test_delegation_to_summary_generator(self):
        """Test that PlotEase's tabular_summary method correctly delegates to SummaryGenerator."""
        pe = PlotEase(self.data)
        
        # Call the public facade method
        summary_df = pe.tabular_summary(style='numeric')
        
        # Verify the result is the expected output from SummaryGenerator
        self.assertIsInstance(summary_df, pd.DataFrame)
        self.assertEqual(len(summary_df), 2) # Should summarize the 2 columns
        self.assertTrue('Mean' in summary_df.columns) # Check a characteristic column

    def test_delegation_to_model_comparator(self):
        """Test that PlotEase's compare_models method correctly delegates to ModelComparator."""
        pe = PlotEase(self.data)
        models = {
            'A': {'R2': 0.8},
            'B': {'R2': 0.9}
        }
        
        # Call the public facade method
        # This call should successfully initialize or update the internal ModelComparator
        pe.compare_models(models) 
        
        # Verify the internal comparator state was updated/used
        self.assertIsNotNone(pe._comparator)
        self.assertEqual(len(pe._comparator._df), 2) # Check ModelComparator's internal data size
        


if __name__ == '__main__':
    # Run with verbose output when executed directly
    unittest.main(verbosity=2)