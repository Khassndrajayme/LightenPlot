import pandas as pd
# Import necessary classes for composition
from .visualization import VisualizationBase
from .diagnostic import DiagnosticPlotter
from .summary import SummaryGenerator
from .quick_plotter import QuickPlotter
from .model_comp import ModelComparator
from typing import Optional, List, Dict

# --- CRITICAL CHANGE 1: Rename the Class ---
class LightenPlot(VisualizationBase):
    """
    Main entry point for the Lighten Plot library.
    Demonstrates Composition by aggregating specialized components.
    """
    def __init__(self, data: pd.DataFrame, theme: str = 'default'):
        # 1. Inheritance: Initialize parent class (VisualizationBase)
        super().__init__(data, theme) 
        
        # 2. Composition: Initialize component objects
        # The fixes above ensure these classes can now be instantiated correctly.
        self._diagnostic = DiagnosticPlotter(data, theme)
        self._summary = SummaryGenerator(data, theme)
        self._plotter = QuickPlotter(data, theme)
        self._comparator = None # Initialized on first use
        
        # NOTE: self._apply_theme() is inherited from the parent
    
    # --- Polymorphism: Overriding the abstract render() method ---
    def render(self):
        """
        Implementation of the abstract method. Delegates to the diagnostic plotter 
        as a sensible default for a main library class.
        """
        print("Rendering default diagnostic plots...")
        self._diagnostic.render()

    # --- Delegation Methods (Composition in Action) ---
    def autoplot(self, target: Optional[str] = None, max_plots: int = 6):
        """Delegates to DiagnosticPlotter's autoplot method."""
        self._diagnostic.autoplot(target=target, max_plots=max_plots)

    def tabular_summary(self, style: str = 'full'):
        """Delegates to SummaryGenerator's tabular_summary method."""
        return self._summary.tabular_summary(style=style)

    def quick_plot(self, x: str, y: Optional[str] = None, **kwargs):
        """Delegates to QuickPlotter's quick_plot method."""
        self._plotter.quick_plot(x, y, **kwargs)

    def set_style(self, style_dict: Dict[str, any]):
        """Delegates style setting to the quick plotter."""
        self._plotter.set_style(style_dict)

    def compare_models(self, models_results: Dict[str, Dict[str, float]]):
        """Initializes and runs the ModelComparator."""
        # Lazy initialization of the comparator
        self._comparator = ModelComparator(models_results)
        self._comparator.compare_models()

    # --- Dunder Methods ---
    def __repr__(self) -> str:
        """Returns the official string representation."""
        # --- CRITICAL CHANGE 2: Update Class Name in __repr__ ---
        return f"LightenPlot(rows={len(self._data)}, theme='{self._theme}')"

    def __len__(self) -> int:
        """Returns the number of rows in the data (used for __lt__ logic)."""
        return len(self._data)
        
    def __eq__(self, other) -> bool:
        """Compares two LightenPlot objects based on data and theme."""
        # --- CRITICAL CHANGE 3: Update Class Check in __eq__ ---
        if not isinstance(other, LightenPlot):
            return False
        return self._data.equals(other._data) and self._theme == other._theme
        
    def __lt__(self, other) -> bool:
        """Compares LightenPlot objects based on data size."""
        # --- CRITICAL CHANGE 4: Update Class Check in __lt__ ---
        if not isinstance(other, LightenPlot):
            return NotImplemented
        return len(self) < len(other)