# lightenplot/__init__.py
"""
lightenplot - Simplified Data Visualization

A Python library that dramatically simplifies data visualization syntax
while maintaining the power and flexibility of matplotlib and seaborn.

Example:
    >>> import lightenplot as lp
    >>> lp.scatter(data, x='age', y='salary').show()
    
Author: RichieClan
Version: 0.4.0
"""

__version__ = '0.4.0'
__author__ = 'RichieClan'

# Import core classes
from .core import BasePlot
from .composer import PlotComposer
from .plots import (
    ScatterPlot,
    LinePlot,
    BarPlot,
    HistogramPlot,
    BoxPlot,
    HeatmapPlot
)
from .themes import ThemeManager
from .exporters import PlotExporter
from . import utils

# Quick API functions for one-liner plotting
def scatter(data=None, x=None, y=None, **kwargs):
    """
    Create a scatter plot in one line.
    
    Args:
        data: DataFrame or dict
        x: X-axis column name or data
        y: Y-axis column name or data
        **kwargs: Additional scatter plot parameters
        
    Returns:
        ScatterPlot instance
        
    Example:
        >>> lp.scatter(df, x='age', y='income', color='gender')
    """
    plot = ScatterPlot(data)
    plot.create(x, y, **kwargs)
    return plot


def line(data=None, x=None, y=None, **kwargs):
    """
    Create a line plot in one line.
    
    Args:
        data: DataFrame or dict
        x: X-axis column name or data
        y: Y-axis column name or data
        **kwargs: Additional line plot parameters
        
    Returns:
        LinePlot instance
        
    Example:
        >>> lp.line(df, x='date', y='price')
    """
    plot = LinePlot(data)
    plot.create(x, y, **kwargs)
    return plot


def bar(data=None, x=None, y=None, **kwargs):
    """
    Create a bar plot in one line.
    
    Args:
        data: DataFrame or dict
        x: X-axis categories
        y: Y-axis values
        **kwargs: Additional bar plot parameters
        
    Returns:
        BarPlot instance
        
    Example:
        >>> lp.bar(df, x='category', y='count')
    """
    plot = BarPlot(data)
    plot.create(x, y, **kwargs)
    return plot


def histogram(data=None, column=None, **kwargs):
    """
    Create a histogram in one line.
    
    Args:
        data: DataFrame or array
        column: Column name (if DataFrame)
        **kwargs: Additional histogram parameters
        
    Returns:
        HistogramPlot instance
        
    Example:
        >>> lp.histogram(df, column='age', bins=20)
    """
    plot = HistogramPlot(data)
    plot.create(column=column, **kwargs)
    return plot


def boxplot(data=None, columns=None, **kwargs):
    """
    Create a box plot in one line.
    
    Args:
        data: DataFrame or list of arrays
        columns: List of column names (if DataFrame)
        **kwargs: Additional box plot parameters
        
    Returns:
        BoxPlot instance
        
    Example:
        >>> lp.boxplot(df, columns=['A', 'B', 'C'])
    """
    plot = BoxPlot(data)
    plot.create(columns=columns, **kwargs)
    return plot


def heatmap(data=None, **kwargs):
    """
    Create a heatmap in one line.
    
    Args:
        data: 2D array or DataFrame
        **kwargs: Additional heatmap parameters
        
    Returns:
        HeatmapPlot instance
        
    Example:
        >>> lp.heatmap(correlation_matrix)
    """
    plot = HeatmapPlot(data)
    plot.create(**kwargs)
    return plot


def compose(rows=1, cols=1):
    """
    Create a plot composer for multiple subplots.
    
    Args:
        rows: Number of subplot rows
        cols: Number of subplot columns
        
    Returns:
        PlotComposer instance
        
    Example:
        >>> composer = lp.compose(2, 2)
        >>> composer.add_plot(plot1).add_plot(plot2).show()
    """
    return PlotComposer(rows, cols)


# Expose main classes and functions
__all__ = [
    # Classes
    'BasePlot',
    'ScatterPlot',
    'LinePlot', 
    'BarPlot',
    'HistogramPlot',
    'BoxPlot',
    'HeatmapPlot',
    'PlotComposer',
    'ThemeManager',
    'PlotExporter',
    
    # Quick API functions
    'scatter',
    'line',
    'bar',
    'histogram',
    'boxplot',
    'heatmap',
    'compose',
    
    # Utils
    'utils'
]