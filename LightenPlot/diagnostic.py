"""
Diagnostic plotting functionality for data analysis.

This module provides automated diagnostic plots to help users
understand their data quickly.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List
from scipy import stats
from .visualization import VisualizationBase


class DiagnosticPlotter(VisualizationBase):
    """
    Create diagnostic plots for data exploration and analysis.
    
    This class provides methods for generating diagnostic visualizations
    that help identify patterns, outliers, and data quality issues.
    
    Attributes:
        data (pd.DataFrame): The data to analyze
    """
    
    def __init__(self, data: pd.DataFrame, theme: str = 'default', **kwargs):
        super().__init__(data=data, theme=theme, **kwargs)
    
    def render(self) -> None:
        """
        Render method - required by abstract base class.
        Creates a default diagnostic visualization.
        """
        print("Rendering DiagnosticPlotter visualization...")
        self.plot()
    
    def plot(self, columns: Optional[List[str]] = None) -> plt.Figure:
        """
        Create a comprehensive diagnostic dashboard.
        
        Args:
            columns: List of columns to analyze (None for all numeric)
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        if columns is None:
            columns = self._data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(columns) == 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No numeric columns found to plot', 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        n_cols = len(columns)
        fig, axes = plt.subplots(n_cols, 3, figsize=(15, 4*n_cols))
        
        if n_cols == 1:
            axes = axes.reshape(1, -1)
        
        for idx, col in enumerate(columns):
            # Histogram
            axes[idx, 0].hist(self._data[col].dropna(), bins=30, 
                             edgecolor='black', alpha=0.7)
            axes[idx, 0].set_title(f'{col} - Distribution')
            axes[idx, 0].set_xlabel(col)
            axes[idx, 0].set_ylabel('Frequency')
            axes[idx, 0].grid(alpha=0.3)
            
            # Box plot
            axes[idx, 1].boxplot(self._data[col].dropna())
            axes[idx, 1].set_title(f'{col} - Box Plot')
            axes[idx, 1].set_ylabel(col)
            axes[idx, 1].grid(alpha=0.3)
            
            # Q-Q plot
            stats.probplot(self._data[col].dropna(), dist="norm", 
                          plot=axes[idx, 2])
            axes[idx, 2].set_title(f'{col} - Q-Q Plot')
            axes[idx, 2].grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def missing_data_plot(self) -> plt.Figure:
        """
        Visualize missing data patterns.
        
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        missing = self._data.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        
        if len(missing) == 0:
            ax.text(0.5, 0.5, 'No missing data found! ✓', 
                   ha='center', va='center', fontsize=16, color='green')
            ax.axis('off')
        else:
            colors = plt.cm.Reds(np.linspace(0.4, 0.8, len(missing)))
            bars = ax.bar(range(len(missing)), missing.values, color=colors)
            ax.set_xticks(range(len(missing)))
            ax.set_xticklabels(missing.index, rotation=45, ha='right')
            ax.set_title('Missing Data Count by Column', fontsize=14, fontweight='bold')
            ax.set_xlabel('Column')
            ax.set_ylabel('Missing Count')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, missing.values)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       str(int(val)), ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def outlier_detection_plot(self, column: str, method: str = 'iqr') -> plt.Figure:
        """
        Detect and visualize outliers in a column.
        
        Args:
            column: Column name to analyze
            method: Detection method ('iqr' or 'zscore')
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        if column not in self._data.columns:
            raise ValueError(f"Column '{column}' not found in data")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        data = self._data[column].dropna()
        
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = (data < lower_bound) | (data > upper_bound)
            method_name = 'IQR Method'
        else:  # zscore
            z_scores = np.abs((data - data.mean()) / data.std())
            outliers = z_scores > 3
            method_name = 'Z-Score Method (|z| > 3)'
        
        # Scatter plot with outliers highlighted
        colors = ['red' if x else 'blue' for x in outliers]
        axes[0].scatter(range(len(data)), data, c=colors, alpha=0.6, s=30)
        axes[0].set_title(f'{column} - Outlier Detection\n{method_name}')
        axes[0].set_xlabel('Index')
        axes[0].set_ylabel(column)
        axes[0].grid(alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='blue', label='Normal'),
                          Patch(facecolor='red', label='Outlier')]
        axes[0].legend(handles=legend_elements, loc='best')
        
        # Box plot
        bp = axes[1].boxplot(data, vert=True, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        axes[1].set_title(f'{column} - Box Plot')
        axes[1].set_ylabel(column)
        axes[1].grid(alpha=0.3)
        
        # Add statistics text
        n_outliers = outliers.sum()
        pct_outliers = (n_outliers / len(data)) * 100
        stats_text = f'Outliers: {n_outliers} ({pct_outliers:.1f}%)'
        axes[1].text(0.5, 0.02, stats_text, transform=axes[1].transAxes,
                    ha='center', fontsize=10, bbox=dict(boxstyle='round', 
                    facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig
    
    def distribution_comparison(self, columns: List[str]) -> plt.Figure:
        """
        Compare distributions of multiple columns.
        
        Args:
            columns: List of column names to compare
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for col in columns:
            if col in self._data.columns:
                self._data[col].dropna().plot(kind='density', ax=ax, 
                                             label=col, linewidth=2)
        
        ax.set_title('Distribution Comparison', fontsize=14, fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend(loc='best')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def data_quality_report(self) -> plt.Figure:
        """
        Generate a comprehensive data quality report.
        
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Missing data
        ax1 = fig.add_subplot(gs[0, 0])
        missing = self._data.isnull().sum()
        missing = missing[missing > 0]
        if len(missing) > 0:
            missing.plot(kind='bar', ax=ax1, color='coral')
            ax1.set_title('Missing Values by Column')
            ax1.set_ylabel('Count')
        else:
            ax1.text(0.5, 0.5, 'No Missing Data', ha='center', va='center')
            ax1.axis('off')
        
        # Data types
        ax2 = fig.add_subplot(gs[0, 1])
        dtype_counts = self._data.dtypes.value_counts()
        ax2.pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
        ax2.set_title('Data Types Distribution')
        
        # Unique values
        ax3 = fig.add_subplot(gs[1, 0])
        unique_counts = self._data.nunique().sort_values(ascending=False)[:10]
        unique_counts.plot(kind='barh', ax=ax3, color='skyblue')
        ax3.set_title('Top 10 Columns by Unique Values')
        ax3.set_xlabel('Unique Count')
        
        # Memory usage
        ax4 = fig.add_subplot(gs[1, 1])
        memory = self._data.memory_usage(deep=True).sort_values(ascending=False)[:10] / 1024
        memory.plot(kind='barh', ax=ax4, color='lightgreen')
        ax4.set_title('Top 10 Memory Usage (KB)')
        ax4.set_xlabel('Memory (KB)')
        
        # Summary statistics
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        summary_text = f"""
        Dataset Summary:
        • Total Rows: {len(self._data):,}
        • Total Columns: {len(self._data.columns)}
        • Numeric Columns: {len(self._data.select_dtypes(include=[np.number]).columns)}
        • Categorical Columns: {len(self._data.select_dtypes(include=['object', 'category']).columns)}
        • Total Missing Values: {self._data.isnull().sum().sum():,}
        • Memory Usage: {self._data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB
        • Duplicate Rows: {self._data.duplicated().sum():,}
        """
        
        ax5.text(0.5, 0.5, summary_text, ha='center', va='center',
                fontsize=11, family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Data Quality Report', fontsize=16, fontweight='bold', y=0.98)
        return fig
    
    def __repr__(self) -> str:
        """Return string representation."""
        return f"DiagnosticPlotter(data_shape={self._data.shape})"