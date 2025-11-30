# Import necessary types and libraries
import pandas as pd
from typing import Optional, List, Dict

# Import components from other files (assuming the structure recommended previously)
from .base import VisualizationBase

from .summary import SummaryGenerator
from .plotters import DiagnosticPlotter, QuickPlotter
from .comparison import ModelComparator


class PlotEase(VisualizationBase):
    """
    Main facade class that integrates all features (summary, plotting, comparison).

    Demonstrates: Composition, Inheritance, Polymorphism (via render() override).
    """

    def __init__(self, data: pd.DataFrame, theme: str = 'default'):
        # 1. Inheritance: Initialize the parent class
        super().__init__(data, theme)

        # 2. Composition - PlotEase HAS-A these components
        self._diagnostic = DiagnosticPlotter(data, theme)
        self._summary = SummaryGenerator(data)
        self._comparator = None  # Will be initialized when compare_models is called
        self._plotter = QuickPlotter(data, theme)

        # Ensure theme is applied, though super().__init__ already calls it
        self._apply_theme()

    def autoplot(self, target: Optional[str] = None, max_plots: int = 6):
        """Delegate to DiagnosticPlotter to automatically generate diagnostic plots."""
        self._diagnostic.autoplot(target, max_plots)

    def tabular_summary(self, style: str = 'full') -> pd.DataFrame:
        """Delegate to SummaryGenerator to create a detailed data summary table."""
        return self._summary.tabular_summary(style)

    def compare_models(self, models_results: Dict[str, Dict[str, float]],
                      metrics: Optional[List[str]] = None):
        """
        Delegate to ModelComparator to compare machine learning model results 
        and display comparative plots.
        """
        self._comparator = ModelComparator(models_results)
        self._comparator.compare_models(metrics)

    def quick_plot(self, x: str, y: Optional[str] = None, **kwargs):
        """Delegate to QuickPlotter for minimal-syntax, single plot generation."""
        self._plotter.quick_plot(x, y, **kwargs)

    def set_style(self, style_dict: Dict[str, any]):
        """Delegate to QuickPlotter to apply custom Matplotlib styling."""
        self._plotter.set_style(style_dict)

    # 3. Polymorphism: Implementation of the abstract method from VisualizationBase
    def render(self):
        """Override abstract method to provide the main interface overview."""
        print("PlotEase Main Interface (Facade)")
        print(f"Current Theme: '{self._theme}'")
        print("Available high-level methods:")
        print("  - autoplot(target=None): Generate diagnostic plots.")
        print("  - tabular_summary(style='full'): Generate data summary.")
        print("  - quick_plot(x, y=None): Create quick, single visualizations.")
        print("  - compare_models(results, metrics=None): Compare ML models (requires results dict).")
        print("  - set_theme(theme_name): Change the visual theme.")

    # Additional Dunder Methods (Method Overriding and Operator Overloading)
    def __repr__(self) -> str:
        """String representation - Method Overriding."""
        return f"PlotEase(rows={len(self._data)}, cols={len(self._data.columns)}, theme='{self._theme}')"

    def __lt__(self, other) -> bool:
        """Less than comparison based on data size."""
        if not isinstance(other, PlotEase):
            return NotImplemented
        return len(self._data) < len(other._data)
