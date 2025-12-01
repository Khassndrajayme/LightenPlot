import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict
from .visualization import VisualizationBase 

class QuickPlotter(VisualizationBase):
    """
    Quick plotting with minimal syntax
    Demonstrates: Inheritance from VisualizationBase
    """
    
    def __init__(self, data: pd.DataFrame, theme: str = 'default'):
        # This super() call now works because VisualizationBase is imported
        super().__init__(data, theme)
        self._style_config = {}  # Protected attribute for custom styles
        self._apply_theme()
    
    def detect_plot_type(self, x: str, y: Optional[str]) -> str:
        """Automatically detect appropriate plot type"""
        if y is None:
            # FIX: Using pd.api.types.is_numeric_dtype is more robust than [np.number]
            if pd.api.types.is_numeric_dtype(self._data[x].dtype):
                return 'hist'
            else:
                return 'bar'
        else:
            if pd.api.types.is_numeric_dtype(self._data[x].dtype) and pd.api.types.is_numeric_dtype(self._data[y].dtype):
                return 'scatter'
            else:
                return 'bar'
    
    def quick_plot(self, x: str, y: Optional[str] = None, 
                   kind: str = 'auto', 
                   color: str = 'steelblue',
                   title: Optional[str] = None,
                   figsize: tuple = (10, 6),
                   **kwargs):
        """
        Create plots with minimal syntax
        
        Args:
            x: Column name for x-axis
            y: Column name for y-axis (optional)
            kind: Plot type ('auto', 'scatter', 'line', 'bar', 'box', 'hist')
            color: Color for the plot
            title: Custom title
            figsize: Figure size tuple
        """
        plt.figure(figsize=figsize)
        
        if kind == 'auto':
            kind = self.detect_plot_type(x, y)
        
        # Create the plot
        if kind == 'scatter' and y:
            plt.scatter(self._data[x], self._data[y], alpha=0.6, color=color, **kwargs)
            plt.xlabel(x, fontsize=12)
            plt.ylabel(y, fontsize=12)
            
        elif kind == 'line' and y:
            plt.plot(self._data[x], self._data[y], color=color, linewidth=2, **kwargs)
            plt.xlabel(x, fontsize=12)
            plt.ylabel(y, fontsize=12)
            
        elif kind == 'bar':
            if y:
                self._data.groupby(x)[y].mean().plot(kind='bar', color=color, **kwargs)
                plt.ylabel(f'Mean {y}', fontsize=12)
            else:
                self._data[x].value_counts().plot(kind='bar', color=color, **kwargs)
                plt.ylabel('Count', fontsize=12)
            plt.xlabel(x, fontsize=12)
            plt.xticks(rotation=45, ha='right')
            
        elif kind == 'hist':
            # Note: Added .dropna() for robustness against NaN values
            plt.hist(self._data[x].dropna(), bins=30, color=color, edgecolor='black', alpha=0.7, **kwargs) 
            plt.xlabel(x, fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            
        elif kind == 'box':
            if y:
                self._data.boxplot(column=y, by=x, ax=plt.gca(), patch_artist=True)
                plt.suptitle('') # Remove default pandas title
            else:
                self._data[[x]].boxplot(patch_artist=True)
            plt.xlabel(x, fontsize=12)
        
        # Styling
        if title:
            plt.title(title, fontsize=16, fontweight='bold', pad=20)
        else:
            plt.title(f'{kind.capitalize()} Plot: {x}' + (f' vs {y}' if y else ''), 
                      fontsize=16, fontweight='bold', pad=20)
        
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def set_style(self, style_dict: Dict[str, any]):
        """Apply custom styling globally"""
        if 'font_size' in style_dict:
            plt.rcParams['font.size'] = style_dict['font_size']
        if 'figure_facecolor' in style_dict:
            plt.rcParams['figure.facecolor'] = style_dict['figure_facecolor']
        if 'axes_facecolor' in style_dict:
            plt.rcParams['axes.facecolor'] = style_dict['axes_facecolor']
        if 'grid_alpha' in style_dict:
            plt.rcParams['grid.alpha'] = style_dict['grid_alpha']
            
        self._style_config.update(style_dict)
        print("✓ Custom styling applied")
    
    def render(self):
        """Implementation of abstract method"""
        print("Use quick_plot() method to render specific plots")
    
    def __repr__(self) -> str:
        """String representation"""
        return f"QuickPlotter(rows={len(self._data)}, theme='{self._theme}')"
