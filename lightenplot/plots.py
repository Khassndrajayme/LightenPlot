"""Specific plot implementations inheriting from BasePlot."""

from .core import BasePlot
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


class ScatterPlot(BasePlot):
    """
    Scatter plot implementation with simplified syntax.
    
    Example:
        >>> plot = ScatterPlot(data)
        >>> plot.create(x='age', y='salary').show()
    """
    
    def create(self, x, y, color=None, size=None, alpha=0.7, **kwargs):
        """
        Create a scatter plot.
        
        Args:
            x: Column name or array for x-axis
            y: Column name or array for y-axis
            color: Color or column name for color coding
            size: Size or column name for size coding
            alpha: Transparency level (0-1)
            **kwargs: Additional matplotlib scatter parameters
            
        Returns:
            self for method chaining
        """

        self._create_args = (x, y)
        self._create_kwargs = dict(color=color, size=size, alpha=alpha, **kwargs)

        if isinstance(self._data, pd.DataFrame):
            x_data = self._data[x] if isinstance(x, str) else x
            y_data = self._data[y] if isinstance(y, str) else y
            
            if color and isinstance(color, str) and color in self._data.columns:
                c = self._data[color]
            else:
                c = color
            
            if size and isinstance(size, str) and size in self._data.columns:
                s = self._data[size]
            else:
                s = size if size else 50
        else:
            x_data, y_data = x, y
            c, s = color, size if size else 50
        
        self.ax.scatter(x_data, y_data, c=c, s=s, alpha=alpha, **kwargs)
        self.ax.grid(True, alpha=0.3)
        return self
    
    def __eq__(self, other):
        """Check equality based on plot type and theme."""
        if not isinstance(other, ScatterPlot):
            return False
        return self._theme == other._theme


class LinePlot(BasePlot):
    """
    Line plot implementation for time series and continuous data.
    
    Example:
        >>> plot = LinePlot(data)
        >>> plot.create(x='date', y='value').show()
    """
    
    def create(self, x, y, color='blue', linewidth=2, marker=None, **kwargs):
        """
        Create a line plot.
        
        Args:
            x: Column name or array for x-axis
            y: Column name or array for y-axis
            color: Line color
            linewidth: Width of the line
            marker: Marker style (None, 'o', 's', '^', etc.)
            **kwargs: Additional matplotlib plot parameters
            
        Returns:
            self for method chaining
        """
        self._create_args = (x, y)
        self._create_kwargs = dict(color=color, linewidth=linewidth, marker=marker, **kwargs)
        
        if isinstance(self._data, pd.DataFrame):
            x_data = self._data[x] if isinstance(x, str) else x
            y_data = self._data[y] if isinstance(y, str) else y
        else:
            x_data, y_data = x, y
        
        self.ax.plot(x_data, y_data, color=color, linewidth=linewidth, 
                     marker=marker, **kwargs)
        self.ax.grid(True, alpha=0.3)
        return self
    
    def __lt__(self, other):
        """Compare plots by number of data points."""
        if not isinstance(other, LinePlot):
            return NotImplemented
        return len(self._data) < len(other._data) if hasattr(self._data, '__len__') else False


class BarPlot(BasePlot):
    """
    Bar plot implementation for categorical data.
    
    Example:
        >>> plot = BarPlot(data)
        >>> plot.create(x='category', y='count').show()
    """
    
    def create(self, x, y, color='steelblue', horizontal=False, **kwargs):
        """
        Create a bar plot.
        
        Args:
            x: Column name or array for categories
            y: Column name or array for values
            color: Bar color(s)
            horizontal: If True, create horizontal bars
            **kwargs: Additional matplotlib bar parameters
            
        Returns:
            self for method chaining
        """
        
        self._create_args = (x, y)
        self._create_kwargs = dict(color=color, horizontal=horizontal, **kwargs)

        if isinstance(self._data, pd.DataFrame):
            x_data = self._data[x] if isinstance(x, str) else x
            y_data = self._data[y] if isinstance(y, str) else y
        else:
            x_data, y_data = x, y
        
        if horizontal:
            self.ax.barh(x_data, y_data, color=color, **kwargs)
        else:
            self.ax.bar(x_data, y_data, color=color, **kwargs)
        
        self.ax.grid(True, alpha=0.3, axis='y' if not horizontal else 'x')
        return self


class HistogramPlot(BasePlot):
    """
    Histogram plot for distribution analysis.
    
    Example:
        >>> plot = HistogramPlot(data)
        >>> plot.create(column='age', bins=20).show()
    """
    
    def create(self, column=None, data=None, bins=30, color='skyblue', 
                 edgecolor='black', **kwargs):
        """
        Create a histogram.
        
        Args:
            column: Column name (if data is DataFrame)
            data: Direct data array (alternative to column)
            bins: Number of bins
            color: Bar color
            edgecolor: Edge color of bars
            **kwargs: Additional matplotlib hist parameters
            
        Returns:
            self for method chaining
        """

        self._create_args = ()
        self._create_kwargs = dict(column=column, data=data, bins=bins, color=color, edgecolor=edgecolor, **kwargs)

        if data is not None:
            plot_data = data
        elif isinstance(self._data, pd.DataFrame) and column:
            plot_data = self._data[column]
        else:
            plot_data = self._data
        
        self.ax.hist(plot_data, bins=bins, color=color, edgecolor=edgecolor, 
                     alpha=0.7, **kwargs)
        self.ax.grid(True, alpha=0.3, axis='y')
        return self


class BoxPlot(BasePlot):
    """Box plot for distribution comparison."""
    
    def create(self, columns=None, data=None, **kwargs):
        """
        Create a box plot.
        
        Args:
            columns: List of column names (for DataFrame)
            data: Direct data (list of arrays)
            **kwargs: Additional matplotlib boxplot parameters
            
        Returns:
            self for method chaining
        """

        self._create_args = ()
        self._create_kwargs = dict(columns=columns, data=data, **kwargs)

        if data is not None:
            plot_data = data
        elif isinstance(self._data, pd.DataFrame) and columns:
            plot_data = [self._data[col].dropna() for col in columns]
        else:
            plot_data = self._data
        
        # Create boxplot first
        self.ax.boxplot(plot_data, **kwargs)
        
        # Then set tick labels if columns provided
        if isinstance(self._data, pd.DataFrame) and columns:
            self.ax.set_xticks(range(1, len(columns) + 1))  # ← FIX: Set ticks first
            self.ax.set_xticklabels(columns)                 # ← Then set labels
        
        self.ax.grid(True, alpha=0.3, axis='y')
        return self


class HeatmapPlot(BasePlot):
    """
    Heatmap for correlation or matrix data.
    
    Example:
        >>> plot = HeatmapPlot(data)
        >>> plot.create(cmap='coolwarm').show()
    """
    
    def create(self, data=None, cmap='viridis', annot=True, fmt='.2f', **kwargs):
        """
        Create a heatmap.
        
        Args:
            data: 2D array or DataFrame
            cmap: Colormap name
            annot: If True, write data values in cells
            fmt: String format for annotations
            **kwargs: Additional seaborn heatmap parameters
            
        Returns:
            self for method chaining
        """

        self._create_args = ()
        self._create_kwargs = dict(data=data, cmap=cmap, annot=annot, fmt=fmt, **kwargs)

        plot_data = data if data is not None else self._data
        
        sns.heatmap(plot_data, cmap=cmap, annot=annot, fmt=fmt, 
                    ax=self.ax, **kwargs)
        return self
    
    def __repr__(self):
        """Enhanced representation with data shape."""
        shape = self._data.shape if hasattr(self._data, 'shape') else 'unknown'
        return f"HeatmapPlot(shape={shape}, theme='{self._theme}')"