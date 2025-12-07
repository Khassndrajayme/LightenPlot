"""
Compare multiple machine learning models visually.

This module provides tools to compare model performance
through various visualization techniques.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union
from .visualization_base import VisualizationBase


class ModelComparator(VisualizationBase):
    """
    Compare multiple models through visualizations.
    
    This class helps visualize and compare the performance of different
    machine learning models using various metrics and plots.
    
    Attributes:
        models (dict): Dictionary of model names and their metrics
    """
    
    def __init__(self, models: Optional[Dict[str, Dict]] = None, theme: str = 'default'):
        """
        Initialize ModelComparator.
        
        Args:
            models: Dictionary with model names as keys and metric dicts as values
            theme: Visual theme to use
        """
        # ModelComparator doesn't require data, so we pass an empty DataFrame
        super().__init__(pd.DataFrame(), theme)
        self._models = models if models is not None else {}
    
    def render(self) -> None:
        """
        Render method - required by abstract base class.
        Creates a default model comparison visualization.
        """
        print("Rendering ModelComparator...")
        if self._models:
            fig = self.plot()
            plt.show()
            return fig
        else:
            print("No models added yet. Use add_model() to add models.")
    
    def add_model(self, name: str, metrics: Dict[str, float]) -> None:
        """
        Add a model with its metrics.
        
        Args:
            name: Model name
            metrics: Dictionary of metric names and values
        """
        self._models[name] = metrics
        print(f"✓ Added model '{name}' with {len(metrics)} metrics")
    
    def remove_model(self, name: str) -> None:
        """
        Remove a model from comparison.
        
        Args:
            name: Model name to remove
        """
        if name in self._models:
            del self._models[name]
            print(f"✓ Removed model '{name}'")
        else:
            print(f"Model '{name}' not found")
    
    @property
    def models(self) -> Dict[str, Dict]:
        """Get the models dictionary."""
        return self._models
    
    @property
    def model_names(self) -> List[str]:
        """Get list of model names."""
        return list(self._models.keys())
    
    def plot(self, metric: Optional[str] = None) -> plt.Figure:
        """
        Create a comparison plot for all models.
        
        Args:
            metric: Specific metric to plot (None for all metrics)
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        if not self._models:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No models to compare. Use add_model() first.', 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        if metric:
            return self._plot_single_metric(metric)
        else:
            return self._plot_all_metrics()
    
    def _plot_single_metric(self, metric: str) -> plt.Figure:
        """Plot a single metric across all models."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        model_names = []
        values = []
        
        for name, metrics in self._models.items():
            if metric in metrics:
                model_names.append(name)
                values.append(metrics[metric])
        
        if not values:
            ax.text(0.5, 0.5, f'Metric "{metric}" not found in any model',
                   ha='center', va='center', fontsize=12)
            ax.axis('off')
        else:
            # Sort by value
            sorted_pairs = sorted(zip(model_names, values), key=lambda x: x[1], reverse=True)
            model_names, values = zip(*sorted_pairs)
            
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(values)))
            bars = ax.barh(model_names, values, color=colors, edgecolor='black')
            
            ax.set_xlabel(metric.capitalize(), fontweight='bold')
            ax.set_title(f'Model Comparison: {metric.capitalize()}', 
                        fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{val:.4f}', ha='left', va='center')
        
        plt.tight_layout()
        return fig
    
    def _plot_all_metrics(self) -> plt.Figure:
        """Plot all metrics for all models."""
        # Get all unique metrics
        all_metrics = set()
        for metrics in self._models.values():
            all_metrics.update(metrics.keys())
        all_metrics = sorted(list(all_metrics))
        
        if not all_metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No metrics found',
                   ha='center', va='center', fontsize=12)
            ax.axis('off')
            return fig
        
        # Create DataFrame for heatmap
        data = []
        for model_name in self._models.keys():
            row = []
            for metric in all_metrics:
                value = self._models[model_name].get(metric, np.nan)
                row.append(value)
            data.append(row)
        
        df = pd.DataFrame(data, index=self._models.keys(), columns=all_metrics)
        
        fig, ax = plt.subplots(figsize=(12, max(6, len(self._models) * 0.8)))
        sns.heatmap(df, annot=True, fmt='.4f', cmap='RdYlGn', 
                   ax=ax, cbar_kws={'label': 'Value'}, linewidths=0.5)
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xlabel('Metrics', fontweight='bold')
        ax.set_ylabel('Models', fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def radar_plot(self, metrics: Optional[List[str]] = None, 
                   normalize: bool = True) -> plt.Figure:
        """
        Create a radar plot comparing models across metrics.
        
        Args:
            metrics: List of metrics to include (None for all)
            normalize: If True, normalize metrics to 0-1 scale
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        if not self._models:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, 'No models to compare', 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        if metrics is None:
            all_metrics = set()
            for model_metrics in self._models.values():
                all_metrics.update(model_metrics.keys())
            metrics = sorted(list(all_metrics))
        
        # Prepare data
        model_names = list(self._models.keys())
        num_metrics = len(metrics)
        
        # Create angles for radar plot
        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
        
        for idx, model_name in enumerate(model_names):
            values = []
            for metric in metrics:
                value = self._models[model_name].get(metric, 0)
                values.append(value)
            
            # Normalize if requested
            if normalize and values:
                max_val = max(max(values), 1e-6)  # Avoid division by zero
                values = [v / max_val for v in values]
            
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[idx])
            ax.fill(angles, values, alpha=0.15, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1 if normalize else None)
        ax.set_title('Model Comparison - Radar Plot', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        return fig
    
    def ranking_plot(self, metric: str, ascending: bool = False) -> plt.Figure:
        """
        Create a ranking plot for models based on a specific metric.
        
        Args:
            metric: Metric to rank by
            ascending: If True, rank in ascending order (lower is better)
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        model_scores = []
        for name, metrics in self._models.items():
            if metric in metrics:
                model_scores.append((name, metrics[metric]))
        
        if not model_scores:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f'Metric "{metric}" not found', 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        # Sort
        model_scores.sort(key=lambda x: x[1], reverse=not ascending)
        
        names, scores = zip(*model_scores)
        ranks = range(1, len(names) + 1)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(names)))
        if ascending:
            colors = colors[::-1]
        
        bars = ax.barh(names, scores, color=colors, edgecolor='black')
        
        # Add rank numbers
        for i, (bar, rank, score) in enumerate(zip(bars, ranks, scores)):
            ax.text(0.01, bar.get_y() + bar.get_height()/2,
                   f'#{rank}', ha='left', va='center',
                   fontweight='bold', fontsize=12, color='white')
            ax.text(score + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{score:.4f}', ha='left', va='center', fontweight='bold')
        
        ax.set_xlabel(metric.capitalize(), fontweight='bold')
        ax.set_title(f'Model Ranking by {metric.capitalize()}', 
                    fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def __repr__(self) -> str:
        """Return string representation."""
        return f"ModelComparator(n_models={len(self._models)})"
    
    def __len__(self) -> int:
        """Return number of models."""
        return len(self._models)
    
    def __contains__(self, model_name: str) -> bool:
        """Check if model exists."""
        return model_name in self._models
    
    def __gt__(self, other) -> bool:
        """Compare based on number of models."""
        if not isinstance(other, ModelComparator):
            return NotImplemented
        return len(self._models) > len(other._models)