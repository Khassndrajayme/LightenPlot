"""
Unit Tests for VisualizationBase (Abstract Base Class)
This file verifies the abstract nature, inheritance, and core non-abstract
methods of the VisualizationBase class.
"""
import unittest
import pandas as pd
from abc import ABC
# Import the class being tested and a concrete class for instantiation checks
from plotease import VisualizationBase, PlotEase 


def load_sample_data():
    """Load a simple DataFrame for testing"""
    return pd.DataFrame({
        'A': [1, 2, 3],
        'B': [10, 20, 30]
    })


class TestVisualizationBase(unittest.TestCase):
    """Test Abstract Base Class functionality and properties."""
    
    def setUp(self):
        """Set up test data."""
        self.data = load_sample_data()
        
    # Abstract Class Enforcement Tests
    
        
    def test_cannot_instantiate_base_class(self):
        """Test that VisualizationBase cannot be instantiated directly (is abstract)."""
        with self.assertRaises(TypeError) as context:
            VisualizationBase(self.data)
            
        self.assertIn("Can't instantiate abstract class", str(context.exception))
        
    def test_subclass_must_implement_render(self):
        """Test that a subclass not implementing 'render' raises a TypeError."""
        # Define a test subclass that deliberately misses the abstract 'render' method
        class MissingRender(VisualizationBase):
            # Missing the required abstract method: render()
            pass 
        
        with self.assertRaises(TypeError) as context:
            MissingRender(self.data)
            
        self.assertIn("Can't instantiate abstract class", str(context.exception))
        self.assertIn("render", str(context.exception))
        
   
    # Concrete Method and Property Tests (using a concrete subclass)
    

    def test_initialization_of_data_and_theme(self):
        """Test that the concrete subclass initializes _data and _theme correctly."""
        # Use PlotEase as a representative concrete subclass
        pe = PlotEase(self.data, theme='dark')
        
        self.assertTrue(pe._data.equals(self.data))
        self.assertEqual(pe._theme, 'dark')
        
    def test_get_data_method(self):
        """Test the concrete get_data() method in the base class."""
        pe = PlotEase(self.data)
        retrieved_data = pe.get_data()
        
        self.assertTrue(retrieved_data.equals(self.data))
        self.assertIsNot(retrieved_data, self.data) # Should be a copy or safe view
        
    def test_set_theme_method(self):
        """Test the concrete set_theme() method in the base class."""
        pe = PlotEase(self.data, theme='default')
        self.assertEqual(pe._theme, 'default')
        
        pe.set_theme('minimal')
        self.assertEqual(pe._theme, 'minimal')


if __name__ == '__main__':
    # Run with verbose output when executed directly
    unittest.main(verbosity=2)
