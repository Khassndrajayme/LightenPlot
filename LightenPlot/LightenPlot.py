# In file: lightenplot.py (or __init__.py if all classes are here)
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any
# Assuming VisualizationBase, DiagnosticPlotter, etc., are imported correctly

# Placeholder/Minimal Classes for Composition
class DiagnosticPlotter(VisualizationBase):
    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
    
    def plot(self): # Must be implemented
        print(f"[DiagnosticPlotter] Plotting default diagnostic chart with theme '{self._theme}'.")
        
    def render(self): # Must be implemented
        return self.plot()
        
    def autoplot(self, target: Optional[str] = None, max_plots: int = 6):
        print(f"[DiagnosticPlotter] Generating {max_plots} diagnostic plots for target '{target}'.")
        # Placeholder for plt.show()
        plt.figure(figsize=self._figsize)
        plt.title(f"Autoplot Diagnostics for {target}")
        plt.text(0.5, 0.5, "Placeholder Plot", ha='center')
        plt.axis('off')
        
    def data_quality_report(self):
        print("[DiagnosticPlotter] Running data quality report.")

class SummaryGenerator(VisualizationBase):
    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
    
    def plot(self):
        print(f"[SummaryGenerator] Plotting summary heatmap with theme '{self._theme}'.")

    def render(self):
        return self.plot()
    
    def tabular_summary(self, style: str = 'full') -> pd.DataFrame:
        print(f"[SummaryGenerator] Generating {style} tabular summary.")
        # Return a simple DataFrame for the demo to print
        return pd.DataFrame({
            'Count': self._data.count(), 
            'Mean': self._data.mean(numeric_only=True)
        }).T

class QuickPlotter(VisualizationBase):
    def __init__(self, data: Optional[pd.DataFrame] = None, **kwargs):
        super().__init__(data, **kwargs)
        
    def plot(self):
        print(f"[QuickPlotter] Ready for quick plotting with theme '{self._theme}'.")

    def render(self):
        return self.plot()

    def quick_plot(self, x: str, y: Optional[str] = None, data: Optional[pd.DataFrame] = None, **kwargs):
        plot_type = kwargs.get('kind', 'scatter')
        title = kwargs.get('title', f"{x} vs {y or 'Count'}")
        
        plt.figure(figsize=self._figsize)
        plt.title(title)
        
        if plot_type == 'hist':
            plt.hist(self._data[x], color=kwargs.get('color', 'gray'))
            plt.xlabel(x)
            plt.ylabel('Frequency')
        else:
            plt.scatter(self._data[x], self._data[y], color=kwargs.get('color', 'gray'))
            plt.xlabel(x)
            plt.ylabel(y)
        
        print(f"[QuickPlotter] Created {plot_type} plot: {title}")
        
    def set_style(self, style_dict: Dict[str, Any]):
        print(f"[QuickPlotter] Applying custom style: {list(style_dict.keys())}")


class ModelComparator:
    def __init__(self, models_results: Dict[str, Dict[str, float]]):
        self._results = models_results
        
    def compare_models(self):
        print("[ModelComparator] Generating comparison charts (Grouped Bar/Radar).")

    def __repr__(self) -> str:
        return f"ModelComparator(models={len(self._results)})"
    
    def __gt__(self, other) -> bool:
        """Simple comparison based on a single metric (Accuracy of first model)."""
        my_acc = list(list(self._results.values())[0].values())[0]
        other_acc = list(list(other._results.values())[0].values())[0]
        return my_acc > other_acc

# --- Main LightenPlot Class ---
class LightenPlot(VisualizationBase):
    """Main façade class demonstrating Composition and resolving the Type Error."""
    def __init__(self, data: pd.DataFrame, theme: str = 'default', **kwargs):
        # 1. Inheritance: Initialize parent class
        super().__init__(data, theme=theme, **kwargs) 
        
        # 2. Composition: Initialize component objects
        self._diagnostic = DiagnosticPlotter(data, theme=theme, **kwargs)
        self._summary = SummaryGenerator(data, theme=theme, **kwargs)
        self._plotter = QuickPlotter(data, theme=theme, **kwargs)
        self._comparator: Optional[ModelComparator] = None
    
   
    def plot(self): 
        """Resolves the TypeError. Delegates to the primary plotting method."""
        print("Executing LightenPlot.plot() -> Delegating to render().")
        return self.render() 

    def render(self):
        """Primary plotting method (used in the demo). Delegates to autoplot."""
        print("Executing LightenPlot.render() -> Running autoplot.")
        return self.autoplot(target='mpg') 
    
    # --- Delegation Methods (Composition in Action) ---
    def autoplot(self, target: Optional[str] = None, max_plots: int = 6):
        """Delegates to DiagnosticPlotter's autoplot method."""
        self._diagnostic.autoplot(target=target, max_plots=max_plots)
        
    def tabular_summary(self, style: str = 'full'):
        """Delegates to SummaryGenerator's tabular_summary method."""
        return self._summary.tabular_summary(style=style)

    def quick_plot(self, x: str, y: Optional[str] = None, **kwargs):
        """Delegates to QuickPlotter's quick_plot method."""
        # Ensure the plotter has access to the data
        self._plotter.quick_plot(x, y, data=self._data, **kwargs)

    def set_style(self, style_dict: Dict[str, Any]):
        """Delegates style setting to the quick plotter."""
        self._plotter.set_style(style_dict)

    def compare_models(self, models_results: Dict[str, Dict[str, float]]):
        """Initializes and runs the ModelComparator."""
        self._comparator = ModelComparator(models_results)
        self._comparator.compare_models()

    # --- Dunder Methods (for Demo 6) ---
    def __repr__(self) -> str:
        return f"LightenPlot(rows={len(self._data)}, theme='{self._theme}')"

    def __len__(self) -> int:
        return len(self._data)
        
    def __eq__(self, other) -> bool:
        if not isinstance(other, LightenPlot):
            return False
        return self._data.equals(other._data) and self._theme == other._theme
        
    def __lt__(self, other) -> bool:
        if not isinstance(other, LightenPlot):
            return NotImplemented
        return len(self) < len(other)