"""
Main LightenPlot class for creating various types of plots.

This module provides the primary interface for users to create
visualizations from their data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Union
from .visualization import VisualizationBase


class LightenPlot(VisualizationBase):
    """
    Main plotting interface for LightenPlot library.
    
    This class provides methods for creating various types of plots
    from pandas DataFrames with minimal code.
    
    Attributes:
        data (pd.DataFrame): The data to visualize
    """
    
    def __init__(self, data: pd.DataFrame, theme: str = 'default', **kwargs):
        """Initialize LightenPlot with data."""
        # Pass data, theme, and all extra kwargs (like figsize) to parent
        super().__init__(data=data, theme=theme, **kwargs)
    
    def render(self) -> None:
        """
        Render method - required by abstract base class.
        Displays basic information about the data.
        """
        print("=" * 60)
        print("LightenPlot Visualization Ready")
        print("=" * 60)
        print(f"Data shape: {self._data.shape}")
        print(f"Columns: {list(self._data.columns)}")
        print(f"\nUse methods like:")
        print("  - scatter(x, y) for scatter plots")
        print("  - histogram(column) for distributions")
        print("  - heatmap() for correlations")
        print("=" * 60)
    
    def scatter(self, x: str, y: str, title: Optional[str] = None,
                alpha: float = 0.6, **kwargs) -> plt.Figure:
        """
        Create a scatter plot.
        
        Args:
            x: Column name for x-axis
            y: Column name for y-axis
            title: Plot title
            alpha: Point transparency
            **kwargs: Additional scatter plot arguments
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.scatter(self._data[x], self._data[y], alpha=alpha, **kwargs)
        
        ax.set_xlabel(x, fontweight='bold')
        ax.set_ylabel(y, fontweight='bold')
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title(f'{y} vs {x}', fontsize=14, fontweight='bold')
        
        ax.grid(alpha=0.3)
        plt.tight_layout()
        return fig
    
    def line(self, x: str, y: Union[str, List[str]], 
             title: Optional[str] = None, **kwargs) -> plt.Figure:
        """
        Create a line plot.
        
        Args:
            x: Column name for x-axis
            y: Column name(s) for y-axis
            title: Plot title
            **kwargs: Additional plotting arguments
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if isinstance(y, str):
            y = [y]
        
        for col in y:
            ax.plot(self._data[x], self._data[col], label=col, linewidth=2, **kwargs)
        
        ax.set_xlabel(x, fontweight='bold')
        ax.set_ylabel('Value', fontweight='bold')
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        
        if len(y) > 1:
            ax.legend()
        
        ax.grid(alpha=0.3)
        plt.tight_layout()
        return fig
    
    def bar(self, x: str, y: str, title: Optional[str] = None, 
            horizontal: bool = False, **kwargs) -> plt.Figure:
        """
        Create a bar plot.
        
        Args:
            x: Column name for x-axis (categories)
            y: Column name for y-axis (values)
            title: Plot title
            horizontal: If True, create horizontal bar plot
            **kwargs: Additional bar plot arguments
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if horizontal:
            ax.barh(self._data[x], self._data[y], **kwargs)
            ax.set_ylabel(x, fontweight='bold')
            ax.set_xlabel(y, fontweight='bold')
        else:
            ax.bar(self._data[x], self._data[y], **kwargs)
            ax.set_xlabel(x, fontweight='bold')
            ax.set_ylabel(y, fontweight='bold')
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        
        ax.grid(alpha=0.3)
        plt.tight_layout()
        return fig
    
    def histogram(self, column: str, bins: int = 30,
                  title: Optional[str] = None, **kwargs) -> plt.Figure:
        """
        Create a histogram.
        
        Args:
            column: Column name to plot
            bins: Number of bins
            title: Plot title
            **kwargs: Additional histogram arguments
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(self._data[column].dropna(), bins=bins, edgecolor='black', 
               alpha=0.7, **kwargs)
        ax.set_xlabel(column, fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title(f'Distribution of {column}', fontsize=14, fontweight='bold')
        
        ax.grid(alpha=0.3, axis='y')
        plt.tight_layout()
        return fig
    
    def boxplot(self, columns: Union[str, List[str]], 
                title: Optional[str] = None, **kwargs) -> plt.Figure:
        """
        Create a box plot.
        
        Args:
            columns: Column name(s) to plot
            title: Plot title
            **kwargs: Additional boxplot arguments
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if isinstance(columns, str):
            columns = [columns]
        
        self._data[columns].boxplot(ax=ax, patch_artist=True, **kwargs)
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title('Box Plot', fontsize=14, fontweight='bold')
        
        ax.set_ylabel('Value', fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
        plt.tight_layout()
        return fig
    
    def heatmap(self, columns: Optional[List[str]] = None,
                title: Optional[str] = None, annot: bool = True, **kwargs) -> plt.Figure:
        """
        Create a correlation heatmap.
        
        Args:
            columns: List of columns to include (None for all numeric)
            title: Plot title
            annot: If True, write correlation values in cells
            **kwargs: Additional seaborn heatmap arguments
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if columns:
            corr_data = self._data[columns].corr()
        else:
            corr_data = self._data.select_dtypes(include=[np.number]).corr()
        
        sns.heatmap(corr_data, annot=annot, fmt='.2f', ax=ax, 
                    cmap='coolwarm', center=0, **kwargs)
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def pairplot(self, columns: Optional[List[str]] = None, 
                 hue: Optional[str] = None, **kwargs):
        """
        Create a pairplot (scatter plot matrix).
        
        Args:
            columns: Columns to include (None for all numeric)
            hue: Column name for color encoding
            **kwargs: Additional seaborn pairplot arguments
            
        Returns:
            seaborn.PairGrid: The created pairplot
        """
        if columns:
            data_to_plot = self._data[columns]
        else:
            data_to_plot = self._data.select_dtypes(include=[np.number])
        
        if hue and hue not in data_to_plot.columns:
            data_to_plot[hue] = self._data[hue]
        
        return sns.pairplot(data_to_plot, hue=hue, **kwargs)
    
    def violinplot(self, x: str, y: str, title: Optional[str] = None, 
                   **kwargs) -> plt.Figure:
        """
        Create a violin plot.
        
        Args:
            x: Column name for x-axis (categories)
            y: Column name for y-axis (values)
            title: Plot title
            **kwargs: Additional violin plot arguments
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.violinplot(data=self._data, x=x, y=y, ax=ax, **kwargs)
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        
        ax.grid(alpha=0.3, axis='y')
        plt.tight_layout()
        return fig