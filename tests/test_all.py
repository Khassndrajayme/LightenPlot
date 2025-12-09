# tests/test_plots.py
"""Unit tests for lightenplot."""

import unittest
import numpy as np
import pandas as pd
from lightenplot import (
    ScatterPlot, LinePlot, BarPlot, HistogramPlot,
    BoxPlot, HeatmapPlot, PlotComposer, ThemeManager
)


class TestBasePlot(unittest.TestCase):
    """Test base plot functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.data = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [2, 4, 6, 8, 10]
        })
    
    def test_scatter_creation(self):
        """Test scatter plot creation."""
        plot = ScatterPlot(self.data)
        plot.create(x='x', y='y')
        self.assertIsNotNone(plot.figure)
        self.assertIsNotNone(plot.ax)
    
    def test_line_creation(self):
        """Test line plot creation."""
        plot = LinePlot(self.data)
        plot.create(x='x', y='y')
        self.assertEqual(len(plot.ax.lines), 1)
    
    def test_bar_creation(self):
        """Test bar plot creation."""
        plot = BarPlot(self.data)
        plot.create(x='x', y='y')
        self.assertGreater(len(plot.ax.patches), 0)
    
    def test_method_chaining(self):
        """Test method chaining."""
        plot = ScatterPlot(self.data)
        result = plot.create(x='x', y='y').set_title('Test').set_labels('X', 'Y')
        self.assertIsInstance(result, ScatterPlot)
        self.assertEqual(plot._title, 'Test')


class TestHistogram(unittest.TestCase):
    """Test histogram functionality."""
    
    def test_histogram_with_array(self):
        """Test histogram with numpy array."""
        data = np.random.randn(100)
        plot = HistogramPlot(data)
        plot.create(bins=10)
        self.assertIsNotNone(plot.figure)
    
    def test_histogram_with_dataframe(self):
        """Test histogram with DataFrame column."""
        df = pd.DataFrame({'values': np.random.randn(100)})
        plot = HistogramPlot(df)
        plot.create(column='values', bins=20)
        self.assertIsNotNone(plot.ax)


class TestBoxPlot(unittest.TestCase):
    """Test box plot functionality."""
    
    def test_boxplot_multiple_columns(self):
        """Test box plot with multiple columns."""
        df = pd.DataFrame({
            'A': np.random.randn(50),
            'B': np.random.randn(50),
            'C': np.random.randn(50)
        })
        plot = BoxPlot(df)
        plot.create(columns=['A', 'B', 'C'])
        self.assertIsNotNone(plot.ax)


class TestHeatmap(unittest.TestCase):
    """Test heatmap functionality."""
    
    def test_heatmap_creation(self):
        """Test heatmap with correlation matrix."""
        data = pd.DataFrame(np.random.randn(5, 5))
        corr = data.corr()
        plot = HeatmapPlot(corr)
        plot.create()
        self.assertIsNotNone(plot.ax)


class TestPlotComposer(unittest.TestCase):
    """Test plot composition."""
    
    def setUp(self):
        """Set up test data."""
        self.data = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
    
    def test_composer_creation(self):
        """Test composer initialization."""
        composer = PlotComposer(2, 2)
        self.assertEqual(len(composer), 0)
        self.assertEqual(composer._rows, 2)
        self.assertEqual(composer._cols, 2)
    
    def test_add_plots(self):
        """Test adding plots to composer."""
        composer = PlotComposer(1, 2)
        plot1 = ScatterPlot(self.data)
        plot2 = LinePlot(self.data)
        
        composer.add_plot(plot1).add_plot(plot2)
        self.assertEqual(len(composer), 2)
    
    def test_composer_overflow(self):
        """Test that adding too many plots raises error."""
        composer = PlotComposer(1, 1)
        plot1 = ScatterPlot(self.data)
        plot2 = LinePlot(self.data)
        
        composer.add_plot(plot1)
        with self.assertRaises(ValueError):
            composer.add_plot(plot2)


class TestThemes(unittest.TestCase):
    """Test theme management."""
    
    def test_list_themes(self):
        """Test listing available themes."""
        themes = ThemeManager.list_themes()
        self.assertIn('default', themes)
        self.assertIn('dark', themes)
        self.assertIn('minimal', themes)
    
    def test_get_theme(self):
        """Test getting theme by name."""
        theme = ThemeManager.get_theme('dark')
        self.assertIsNotNone(theme)
    
    def test_invalid_theme(self):
        """Test that invalid theme raises error."""
        with self.assertRaises(ValueError):
            ThemeManager.get_theme('nonexistent')
    
    def test_apply_theme(self):
        """Test applying theme to plot."""
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        plot = ScatterPlot(df)
        plot.create(x='x', y='y')
        plot.apply_theme('minimal')
        self.assertEqual(plot._theme, 'minimal')


class TestDunderMethods(unittest.TestCase):
    """Test dunder methods implementation."""
    
    def setUp(self):
        """Set up test data."""
        self.data = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
    
    def test_repr(self):
        """Test __repr__ method."""
        plot = ScatterPlot(self.data)
        repr_str = repr(plot)
        self.assertIn('ScatterPlot', repr_str)
    
    def test_str(self):
        """Test __str__ method."""
        plot = LinePlot(self.data)
        str_rep = str(plot)
        self.assertIn('LinePlot', str_rep)
    
    def test_eq(self):
        """Test __eq__ method."""
        plot1 = ScatterPlot(self.data)
        plot2 = ScatterPlot(self.data)
        self.assertTrue(plot1 == plot2)
    
    def test_lt(self):
        """Test __lt__ method for LinePlot."""
        data1 = pd.DataFrame({'x': [1, 2], 'y': [3, 4]})
        data2 = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        
        plot1 = LinePlot(data1)
        plot2 = LinePlot(data2)
        
        self.assertTrue(plot1 < plot2)


class TestDataValidation(unittest.TestCase):
    """Test data validation."""
    
    def test_dataframe_input(self):
        """Test DataFrame input."""
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        plot = ScatterPlot(df)
        self.assertIsInstance(plot._data, pd.DataFrame)
    
    def test_dict_input(self):
        """Test dict input conversion."""
        data = {'x': [1, 2, 3], 'y': [4, 5, 6]}
        plot = ScatterPlot(data)
        self.assertIsInstance(plot._data, pd.DataFrame)
    
    def test_array_input(self):
        """Test array input."""
        data = np.array([1, 2, 3, 4, 5])
        plot = HistogramPlot(data)
        self.assertIsInstance(plot._data, np.ndarray)


if __name__ == '__main__':
    unittest.main()
