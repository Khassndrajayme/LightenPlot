"""
LightenPlot - A Python Library for Easy Data Visualization
==========================================================

LightenPlot provides an intuitive, object-oriented interface for creating
beautiful data visualizations with minimal code.

Main Classes:
    - VisualizationBase: Abstract base class for all visualization components
    - LightenPlot: Main interface for creating plots
    - DiagnosticPlotter: Automated diagnostic plots for data analysis
    - SummaryGenerator: Generate statistical summary visualizations
    - ModelComparator: Compare multiple models visually
    - QuickPlotter: Fast, one-line plotting utilities

Example:
    >>> from lightenplot import LightenPlot
    >>> import pandas as pd
    >>> 
    >>> df = pd.read_csv('data.csv')
    >>> plotter = LightenPlot(df)
    >>> plotter.scatter('x_column', 'y_column', title='My Scatter Plot')
"""

__version__ = "0.1.0"
__author__ = "Group 5: Jayme, Janog, Mahilum, Ventura, Zamoranos"
__license__ = "MIT"

from .visualization_base import VisualizationBase
from .lightenplot import LightenPlot
from .diagnostic_plotter import DiagnosticPlotter
from .summary_generator import SummaryGenerator
from .model_comparator import ModelComparator
from .quick_plotter import QuickPlotter

__all__ = [
    'VisualizationBase',
    'LightenPlot',
    'DiagnosticPlotter',
    'SummaryGenerator',
    'ModelComparator',
    'QuickPlotter'
]