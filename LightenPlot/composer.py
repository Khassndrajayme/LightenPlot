"""Composer base classes for LightenPlot library."""

from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from .core import BasePlot

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
                plot_obj.create(*plot_obj._create_args, **plot_obj._create_kwargs)
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