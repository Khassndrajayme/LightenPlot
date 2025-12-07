"""
LightenPlot: A simplified data visualization library for Python

LightenPlot makes data visualization easy and intuitive, reducing the complexity
of matplotlib and ggplot while maintaining powerful features.

Example:
    >>> from lightenplot import LightenPlot 
    >>> import pandas as pd
    >>> 
    >>> data = pd.read_csv('data.csv')
    >>> pe = LightenPlot(data, theme='minimal') 
    >>> pe.autoplot(target='price')
"""

# Version information
__version__ = '1.0.0'
__author__ = 'RichieClan' 
__email__ = 'your.email@example.com'
__license__ = 'MIT'


# Import main classes
from .visualization import VisualizationBase
from .LightenPlot import LightenPlot 
from .diagnostic import DiagnosticPlotter
from .summary import SummaryGenerator
from .model_comp import ModelComparator
from .quick_plotter import QuickPlotter

# Import utilities
from . import utils

# Define what gets imported with "from lightenplot import *"
__all__ = [
    # Main classes
    'LightenPlot',
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
    'name': 'lightenplot', 
    'version': __version__,
    'description': 'A simplified data visualization library for Python',
    'author': __author__,
    'email': __email__,
    'license': __license__,
    'url': 'https://github.com/Khassndrajayme/lightenplot', 
}


def get_version():
    """Return the current version of LightenPlot""" 
    return __version__


def print_info():
    """Print package information"""
    print(f"LightenPlot v{__version__}") # Output updated
    print(f"Author: {__author__}")
    print(f"License: {__license__}")
    print(f"Documentation: https://lightenplot.readthedocs.io/") # Documentation link updated
 
# print(f"LightenPlot v{__version__} loaded successfully!") # Comment updated