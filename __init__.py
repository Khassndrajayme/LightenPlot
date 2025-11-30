"""
PlotEase: A simplified data visualization library for Python

PlotEase makes data visualization easy and intuitive, reducing the complexity
of matplotlib and ggplot while maintaining powerful features.

Example:
    >>> from plotease import PlotEase
    >>> import pandas as pd
    >>> 
    >>> data = pd.read_csv('data.csv')
    >>> pe = PlotEase(data, theme='minimal')
    >>> pe.autoplot(target='price')
"""

# Version information
__version__ = '1.0.0'
__author__ = 'PlotEase Team'
__email__ = 'your.email@example.com'
__license__ = 'MIT'

# Import main classes
from .base import VisualizationBase
from .plotease import PlotEase
from .diagnostic import DiagnosticPlotter
from .summary import SummaryGenerator
from .comparator import ModelComparator
from .plotter import QuickPlotter

# Import utilities
from . import utils

# Define what gets imported with "from plotease import *"
__all__ = [
    # Main classes
    'PlotEase',
    'VisualizationBase',
    'DiagnosticPlotter',
    'SummaryGenerator',
    'ModelComparator',
    'QuickPlotter',
    
    # Utilities module
    'utils',
    
    # Version info
    '__version__',
]

# Package metadata
PACKAGE_INFO = {
    'name': 'plotease',
    'version': __version__,
    'description': 'A simplified data visualization library for Python',
    'author': __author__,
    'email': __email__,
    'license': __license__,
    'url': 'https://github.com/yourusername/plotease',
}


def get_version():
    """Return the current version of PlotEase"""
    return __version__


def print_info():
    """Print package information"""
    print(f"PlotEase v{__version__}")
    print(f"Author: {__author__}")
    print(f"License: {__license__}")
    print(f"Documentation: https://plotease.readthedocs.io/")
  
# print(f"PlotEase v{__version__} loaded successfully!")
