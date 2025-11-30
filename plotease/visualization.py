# Necessary imports for the base class
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Union
import warnings
# Note: The original code included numpy and seaborn imports that are not strictly
# used in the base class, but we'll include the ones essential for its methods.

class VisualizationBase(ABC):
    """
    Abstract Base Class (ABC) for all visualization components.
    This class serves as a blueprint for concrete visualization classes (like BarChart, ScatterPlot, etc.),
    ensuring they all share common initialization, data handling, theme management, and core methods.

    It demonstrates the following Object-Oriented Programming (OOP) concepts:
     1. Inheritance: It is designed to be subclassed.
     2. Encapsulation: Data and theme are protected attributes (`_data`, `_theme`) managed via public methods.
     3. Polymorphism: The abstract `render()` method forces subclasses to provide their own distinct visualization implementation.
     4. Dunder Methods: Customizing built-in operations like string representation (`__repr__`) and equality (`__eq__`).

    Attributes:
        _data (pd.DataFrame): The protected attribute holding the data to be visualized.
        _theme (str): The protected attribute holding the name of the active visualization theme.
    """

    def __init__(self, data: pd.DataFrame, theme: str = 'default'):
        """
        Initializes the base visualization component with data and a theme.

        Args:
            data: The pandas DataFrame containing the data to be plotted.
            theme: The visual theme to apply. Supported built-in themes include
                   'default', 'minimal', 'dark', and 'colorful'.

        Raises:
            TypeError: If `data` is not a pandas DataFrame.
            ValueError: If `data` is an empty DataFrame.
        """
        self._data = data  # Protected attribute (encapsulation)
        self._theme = theme  # Protected attribute
        self._validate_data()
        self._apply_theme() # Apply the theme immediately upon initialization

    def _validate_data(self) -> bool:
        """
        Protected method to validate that the input data is a non-empty pandas DataFrame.
        """
        if not isinstance(self._data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame")
        if self._data.empty:
            raise ValueError("DataFrame cannot be empty")
        return True

    # Getter and Setter methods (Encapsulation)
    def get_data(self) -> pd.DataFrame:
        """
        Retrieves the underlying pandas DataFrame used for visualization.
        """
        return self._data

    def set_theme(self, theme: str):
        """
        Sets a new visualization theme and immediately applies it.
        """
        self._theme = theme
        self._apply_theme()


    def _apply_theme(self):
        """
        Protected method to map the internal theme name to a Matplotlib style
        and apply the style globally.
        """
        themes = {
            'default': 'seaborn-v0_8-darkgrid',
            'minimal': 'seaborn-v0_8-whitegrid',
            'dark': 'dark_background',
            'colorful': 'seaborn-v0_8-bright'
        }

        # Get the corresponding Matplotlib style name, defaulting if the theme is unknown
        style_name = themes.get(self._theme, 'default')
        plt.style.use(style_name)

    @abstractmethod
    def render(self):
        """
        Abstract method Polymorphism that must be implemented by all concrete subclasses.
        This method is responsible for generating and displaying the specific visualization.

        Subclasses must override this method.
        """
        pass

    #  Dunder Methods (Operator Overloading)

    def __repr__(self) -> str:
        """
        Returns a developer-friendly, informative string representation of the object.
        """
        return f"{self.__class__.__name__}(rows={len(self._data)}, cols={len(self._data.columns)}, theme='{self._theme}')"

    def __eq__(self, other) -> bool:
        """
        Implements the equality operator (==) for VisualizationBase objects.

        Two objects are considered equal if they are instances of the base class
        and their underlying dataframes and themes are identical.
        """
        if not isinstance(other, VisualizationBase):
            return False
        # pd.DataFrame.equals() compares data, index, and column names
        return self._data.equals(other._data) and self._theme == other._theme

    def __len__(self) -> int:
        """
        Implements the built-in `len()` function, allowing the object to report its size.

        Returns:
            The number of rows in the visualization's data DataFrame.
        """
        return len(self._data)
