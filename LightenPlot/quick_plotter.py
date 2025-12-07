"""
Quick plotting utilities for fast visualizations.

This module provides simple, one-line functions for creating
common plots without extensive configuration.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Union
from .visualization_base import VisualizationBase


class QuickPlotter(VisualizationBase):
    """
    Fast, one-line plotting utilities.
    
    This class provides static methods for creating quick visualizations
    with minimal code and sensible defaults.
    """
    
    def __init__(self, data: Optional[pd.DataFrame] = None, theme: str = 'default'):
        """
        Initialize QuickPlotter.
        
        Args:
            data: Optional DataFrame
            theme: Visual theme
        """
        # QuickPlotter can work without data since it uses static methods
        if data is None:
            data = pd.DataFrame()  # Empty DataFrame
        super().__init__(data, theme)
    
    def render(self) -> None:
        """
        Render method - required by abstract base class.
        """
        print("QuickPlotter ready for quick plotting operations.")
        print("Use methods like quick_scatter(), quick_hist(), etc.")
    
    @staticmethod
    def quick_scatter(data: pd.DataFrame, x: str, y: str, 
                     title: Optional[str] = None, **kwargs) -> plt.Figure:
        """
        Create a quick scatter plot.
        
        Args:
            data: DataFrame containing the data
            x: Column name for x-axis
            y: Column name for y-axis
            title: Plot title
            **kwargs: Additional scatter arguments
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(data[x], data[y], alpha=0.6, **kwargs)
        ax.set_xlabel(x, fontweight='bold')
        ax.set_ylabel(y, fontweight='bold')
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title(f'{y} vs {x}', fontsize=14, fontweight='bold')
        
        ax.grid(alpha=0.3)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def quick_line(data: pd.DataFrame, x: str, y: Union[str, List[str]], 
                   title: Optional[str] = None, **kwargs) -> plt.Figure:
        """
        Create a quick line plot.
        
        Args:
            data: DataFrame containing the data
            x: Column name for x-axis
            y: Column name(s) for y-axis
            title: Plot title
            **kwargs: Additional line plot arguments
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if isinstance(y, str):
            y = [y]
        
        for col in y:
            ax.plot(data[x], data[col], label=col, linewidth=2, **kwargs)
        
        ax.set_xlabel(x, fontweight='bold')
        ax.set_ylabel('Value', fontweight='bold')
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title(f'Line Plot', fontsize=14, fontweight='bold')
        
        if len(y) > 1:
            ax.legend()
        
        ax.grid(alpha=0.3)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def quick_hist(data: pd.DataFrame, column: str, bins: int = 30,
                   title: Optional[str] = None, **kwargs) -> plt.Figure:
        """
        Create a quick histogram.
        
        Args:
            data: DataFrame containing the data
            column: Column name to plot
            bins: Number of bins
            title: Plot title
            **kwargs: Additional histogram arguments
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(data[column].dropna(), bins=bins, edgecolor='black', 
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
    
    @staticmethod
    def quick_box(data: pd.DataFrame, columns: Union[str, List[str]],
                  title: Optional[str] = None, **kwargs) -> plt.Figure:
        """
        Create a quick box plot.
        
        Args:
            data: DataFrame containing the data
            columns: Column name(s) to plot
            title: Plot title
            **kwargs: Additional boxplot arguments
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if isinstance(columns, str):
            columns = [columns]
        
        data[columns].boxplot(ax=ax, patch_artist=True, **kwargs)
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title('Box Plot', fontsize=14, fontweight='bold')
        
        ax.set_ylabel('Value', fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
        plt.tight_layout()
        return fig
    
    @staticmethod
    def quick_heatmap(data: pd.DataFrame, columns: Optional[List[str]] = None,
                      title: Optional[str] = None, **kwargs) -> plt.Figure:
        """
        Create a quick correlation heatmap.
        
        Args:
            data: DataFrame containing the data
            columns: Columns to include (None for all numeric)
            title: Plot title
            **kwargs: Additional heatmap arguments
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if columns:
            corr_data = data[columns].corr()
        else:
            corr_data = data.select_dtypes(include=[np.number]).corr()
        
        sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, ax=ax, **kwargs)
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_all(data: pd.DataFrame, x: str, y: str,
                 plot_types: List[str] = ['scatter', 'line', 'hist', 'box']) -> plt.Figure:
        """
        Create multiple plots at once.
        
        Args:
            data: DataFrame containing the data
            x: Column name for x-axis
            y: Column name for y-axis
            plot_types: List of plot types to create
            
        Returns:
            matplotlib.figure.Figure: The created figure with subplots
        """
        n_plots = len(plot_types)
        n_cols = min(2, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
        
        if n_plots == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, plot_type in enumerate(plot_types):
            ax = axes[idx]
            plt.sca(ax)
            
            if plot_type == 'scatter':
                ax.scatter(data[x], data[y], alpha=0.6)
                ax.set_title('Scatter Plot')
            elif plot_type == 'line':
                ax.plot(data[x], data[y], linewidth=2)
                ax.set_title('Line Plot')
            elif plot_type == 'hist':
                ax.hist(data[y].dropna(), bins=30, edgecolor='black', alpha=0.7)
                ax.set_title('Histogram')
            elif plot_type == 'box':
                ax.boxplot(data[y].dropna())
                ax.set_title('Box Plot')
            
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.grid(alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'Multiple Plot Types: {y} vs {x}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def __repr__(self) -> str:
        """Return string representation."""
        return "QuickPlotter()"