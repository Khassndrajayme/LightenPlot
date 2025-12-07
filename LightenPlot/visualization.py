"""
Abstract Base Class for all visualization components in LightenPlot.

This module provides the foundation for all plotting classes through
the VisualizationBase abstract class.
"""

from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any


class VisualizationBase(ABC):
    """
    Abstract base class for all visualization components.
    
    This class defines the common interface and shared functionality
    for all visualization classes in the LightenPlot library.
    
    Attributes:
        _style (str): The plotting style to use (default: 'seaborn-v0_8')
        _figsize (tuple): Default figure size (width, height)
        _color_palette (str): Default color palette name
    """
    
    def __init__(self, style: str = 'seaborn-v0_8', 
                 figsize: tuple = (10, 6),
                 color_palette: str = 'deep'):
        """
        Initialize the VisualizationBase.
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size as (width, height)
            color_palette: Seaborn color palette name
        """
        self._style = style
        self._figsize = figsize
        self._color_palette = color_palette
        self._setup_style()
    
    def _setup_style(self) -> None:
        """Apply the plotting style and color palette."""
        try:
            plt.style.use(self._style)
        except:
            plt.style.use('default')
        sns.set_palette(self._color_palette)
    
    @property
    def style(self) -> str:
        """Get the current plotting style."""
        return self._style
    
    @style.setter
    def style(self, value: str) -> None:
        """Set a new plotting style."""
        self._style = value
        self._setup_style()
    
    @property
    def figsize(self) -> tuple:
        """Get the default figure size."""
        return self._figsize
    
    @figsize.setter
    def figsize(self, value: tuple) -> None:
        """Set the default figure size."""
        if not isinstance(value, tuple) or len(value) != 2:
            raise ValueError("figsize must be a tuple of (width, height)")
        self._figsize = value
    
    @property
    def color_palette(self) -> str:
        """Get the current color palette."""
        return self._color_palette
    
    @color_palette.setter
    def color_palette(self, value: str) -> None:
        """Set a new color palette."""
        self._color_palette = value
        sns.set_palette(value)
    
    @abstractmethod
    def plot(self, *args, **kwargs) -> plt.Figure:
        """
        Abstract method to create a plot.
        
        Must be implemented by all subclasses.
        
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        pass
    
    def save_plot(self, filename: str, dpi: int = 300, 
                  bbox_inches: str = 'tight') -> None:
        """
        Save the current plot to a file.
        
        Args:
            filename: Path where the plot should be saved
            dpi: Resolution in dots per inch
            bbox_inches: Bounding box setting
        """
        plt.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
        print(f"Plot saved to {filename}")
    
    def show(self) -> None:
        """Display the current plot."""
        plt.show()
    
    def close(self) -> None:
        """Close the current plot."""
        plt.close()
    
    def __repr__(self) -> str:
        """Return a string representation of the object."""
        return f"{self.__class__.__name__}(style='{self._style}', figsize={self._figsize})"
    
    def __eq__(self, other) -> bool:
        """Check equality based on style and figsize."""
        if not isinstance(other, VisualizationBase):
            return False
        return (self._style == other._style and 
                self._figsize == other._figsize and
                self._color_palette == other._color_palette)
    
    def __lt__(self, other) -> bool:
        """Compare based on figure area."""
        if not isinstance(other, VisualizationBase):
            return NotImplemented
        return (self._figsize[0] * self._figsize[1] < 
                other._figsize[0] * other._figsize[1])
    
    def __le__(self, other) -> bool:
        """Less than or equal comparison."""
        return self < other or self == other
    
    def __gt__(self, other) -> bool:
        """Greater than comparison."""
        if not isinstance(other, VisualizationBase):
            return NotImplemented
        return (self._figsize[0] * self._figsize[1] > 
                other._figsize[0] * other._figsize[1])
    
    def __ge__(self, other) -> bool:
        """Greater than or equal comparison."""
        return self > other or self == other