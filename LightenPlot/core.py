"""Core base classes for LightenPlot library."""

from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class BasePlot(ABC):
    """
    Abstract base class for all plot types.
    
    Attributes:
        data: Input data (DataFrame, array, or dict)
        figure: Matplotlib figure object
        ax: Matplotlib axes object
        _title: Protected title attribute
        _theme: Protected theme attribute
    """
    
    def __init__(self, data=None, figsize=(10, 6)):
        """
        Initialize base plot.
        
        Args:
            data: Input data for plotting
            figsize: Tuple of (width, height) for figure size
        """
        self._data = self._validate_data(data) if data is not None else None
        self.figure, self.ax = plt.subplots(figsize=figsize)
        self._title = ""
        self._theme = "default"
        self._xlabel = ""
        self._ylabel = ""
    
    @abstractmethod
    def create(self, *args, **kwargs):
        """Abstract method to create the plot. Must be implemented by subclasses."""
        pass
    
    def _validate_data(self, data):
        """
        Validate and convert input data to appropriate format.
        
        Args:
            data: Input data (DataFrame, dict, array, etc.)
            
        Returns:
            Validated data in appropriate format
        """
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, dict):
            return pd.DataFrame(data)
        elif isinstance(data, (list, np.ndarray)):
            return np.array(data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def set_title(self, title):
        """Set plot title."""
        self._title = title
        self.ax.set_title(title, fontsize=14, fontweight='bold')
        return self
    
    def set_labels(self, xlabel=None, ylabel=None):
        """Set axis labels."""
        if xlabel:
            self._xlabel = xlabel
            self.ax.set_xlabel(xlabel, fontsize=12)
        if ylabel:
            self._ylabel = ylabel
            self.ax.set_ylabel(ylabel, fontsize=12)
        return self
    
    def apply_theme(self, theme_name):
        """Apply a predefined theme to the plot."""
        from .themes import ThemeManager
        theme = ThemeManager.get_theme(theme_name)
        theme.apply(self.ax, self.figure)
        self._theme = theme_name
        return self
    
    def save(self, filename, dpi=300, format=None):
        """
        Save plot to file.
        
        Args:
            filename: Output filename
            dpi: Dots per inch (resolution)
            format: File format (png, jpg, pdf, svg)
        """
        from .exporters import PlotExporter
        exporter = PlotExporter(self.figure)
        exporter.save(filename, dpi=dpi, format=format)
        return self
    
    def show(self):
        """Display the plot."""
        plt.tight_layout()
        plt.show()
        return self
    
    def __repr__(self):
        """String representation of the plot object."""
        return f"{self.__class__.__name__}(theme='{self._theme}', title='{self._title}')"
    
    def __str__(self):
        """User-friendly string representation."""
        return f"LightenPlot {self.__class__.__name__}"


class PlotComposer:
    """
    Compose multiple plots into a single figure.
    Uses composition pattern - contains multiple BasePlot instances.
    """
    
    def __init__(self, rows=1, cols=1, figsize=None):
        """
        Initialize plot composer.
        
        Args:
            rows: Number of subplot rows
            cols: Number of subplot columns
            figsize: Overall figure size
        """
        self._plots = []
        self._rows = rows
        self._cols = cols
        if figsize is None:
            figsize = (6 * cols, 4 * rows)
        self.figure, self.axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1 and cols == 1:
            self.axes = np.array([self.axes])
        self.axes = self.axes.flatten() if isinstance(self.axes, np.ndarray) else [self.axes]
        self._current_idx = 0
    
    def add_plot(self, plot_obj):
        """Add a plot to the composition."""
        if self._current_idx >= len(self.axes):
            raise ValueError(f"Cannot add more than {len(self.axes)} plots")
        self._plots.append(plot_obj)
        self._current_idx += 1
        return self
    
    def render(self):
        """Render all plots in the composition."""
        for idx, plot_obj in enumerate(self._plots):
            if idx < len(self.axes):
                plot_obj.ax = self.axes[idx]
                plot_obj.figure = self.figure
        plt.tight_layout()
        return self
    
    def show(self):
        """Display the composed plots."""
        self.render()
        plt.show()
    
    def save(self, filename, dpi=300):
        """Save the composed plots."""
        self.render()
        self.figure.savefig(filename, dpi=dpi, bbox_inches='tight')
        return self
    
    def __len__(self):
        """Return number of plots in composition."""
        return len(self._plots)
    
    def __repr__(self):
        return f"PlotComposer(plots={len(self._plots)}, grid={self._rows}x{self._cols})"