import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict 

class ModelComparator:
    """
    Compares machine learning model performance across various metrics.
    """
    
    def __init__(self, models_results: Dict[str, Dict[str, float]]):
        """Initializes the ModelComparator with a dictionary of model results."""
        self._models_results = models_results  
        
        if not models_results:
            self._df = pd.DataFrame()
        else:
            # Convert results to a DataFrame (models as index, metrics as columns)
            self._df = pd.DataFrame(models_results).T
    
    def create_bar_chart(self, ax: plt.Axes, df: pd.DataFrame):
        """Creates and formats a grouped bar chart for model performance."""
        if df.empty:
            ax.text(0.5, 0.5, 'No Data to Plot', ha='center', va='center', fontsize=16)
            ax.set_title('Model Performance Comparison (Bar Chart)')
            return
            
        df.plot(kind='bar', ax=ax, width=0.8, edgecolor='black')
        ax.set_title('Model Performance Comparison (Bar Chart)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        ax.set_xticklabels(df.index, rotation=45, ha='right')
        ax.set_ylim(0, 1.0)
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3, fontsize=9)
    
    def create_radar_chart(self, ax: plt.Axes, df: pd.DataFrame):
        """
        Creates and formats a radar chart for model performance.
        
        """
        if df.empty or len(df.columns) < 3:
            ax.text(0.5, 0.5, 'Requires at least 3 Metrics', ha='center', va='center', fontsize=16)
            ax.set_title('Model Performance Radar')
            ax.axis('off')
            return

        n_metrics = len(df.columns)
        # Calculate angles for the radar axes (from 0 to 2*pi, closed loop)
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]
        
        for model_name, values in df.iterrows():
            values_list = values.tolist()
            values_list += values_list[:1]
            ax.plot(angles, values_list, 'o-', linewidth=2, label=model_name)
            ax.fill(angles, values_list, alpha=0.15)
        
        ax.set_xticks(angles[:-1])
        # FIX: The radar chart axes labels were incomplete in your snippet
        ax.set_xticklabels(df.columns, fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.set_title('Model Performance Radar', fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)


    def compare_models(self, metrics: Optional[List[str]] = None):
        """
        Generates and displays two visualizations comparing the performance of the models.
        """
        if self._df.empty:
            print("Error: No model results were loaded into the comparator.")
            return
        
        df = self._df.copy()
        if metrics:
            missing_metrics = [m for m in metrics if m not in df.columns]
            if missing_metrics:
                print(f"Error: The following metrics were not found: {missing_metrics}")
                return
            df = df[metrics]
        
        # Create a figure with two subplots side-by-side
        # The main figure needs to be handled carefully to allow polar projection
        fig = plt.figure(figsize=(15, 6))
        ax_bar = fig.add_subplot(1, 2, 1)
        ax_radar = fig.add_subplot(1, 2, 2, projection='polar')

        self.create_bar_chart(ax_bar, df)
        self.create_radar_chart(ax_radar, df)
        
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.show() 
        
        print("\nModel Performance Summary:")
        print("="*60)
        print(df.to_string())
        print("="*60)


    def get_best_model(self, metric: str) -> str:
        """
        Retrieves the name of the model with the highest score for a specified metric.
        """
        if metric not in self._df.columns:
            raise ValueError(f"Metric '{metric}' not found. Available metrics: {list(self._df.columns)}")
        return self._df[metric].idxmax()


    # --- Dunder Methods (Operator Overloading) ---

    @property
    def overall_mean_score(self) -> float:
        """Calculates the average score across all models and all metrics."""
        if self._df.empty:
            return 0.0
        # Calculate mean of all metrics, then take the mean of those means
        return self._df.mean().mean()

    def __repr__(self) -> str:
        """Returns a developer-friendly, official string representation of the object."""
        return f"ModelComparator(models={len(self._models_results)}, metrics={list(self._df.columns)})"

    def __eq__(self, other) -> bool:
        """Implements the equality operator (==)."""
        if not isinstance(other, ModelComparator):
            return False
        return self._df.equals(other._df)

    def __gt__(self, other) -> bool:
        """Implements the greater than operator (>)."""
        if not isinstance(other, ModelComparator):
            return NotImplemented
    
    # FIX: Explicitly cast the result to a standard Python bool
        return bool(self.overall_mean_score > other.overall_mean_score)
        
        # Comparison is based on the calculated average performance score.
        return self.overall_mean_score > other.overall_mean_score
