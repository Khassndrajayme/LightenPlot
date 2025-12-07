"""
Generate statistical summary visualizations.

This module provides functionality to create visual summaries
of statistical information from data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List
from .visualization_base import VisualizationBase


class SummaryGenerator(VisualizationBase):
    """
    Generate statistical summary visualizations.
    
    This class creates visual representations of statistical summaries
    to help users quickly understand their data characteristics.
    
    Attributes:
        data (pd.DataFrame): The data to summarize
    """
    
    def __init__(self, data: pd.DataFrame, **kwargs):
        """
        Initialize SummaryGenerator with data.
        
        Args:
            data: pandas DataFrame to summarize
            **kwargs: Additional arguments passed to VisualizationBase
        """
        super().__init__(**kwargs)
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        self._data = data
    
    @property
    def data(self) -> pd.DataFrame:
        """Get the underlying data."""
        return self._data
    
    def plot(self, columns: Optional[List[str]] = None) -> plt.Figure:
        """
        Create a summary statistics visualization.
        
        Args:
            columns: Columns to include (None for all numeric)
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        if columns is None:
            columns = self._data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(columns) == 0:
            fig, ax = plt.subplots(figsize=self._figsize)
            ax.text(0.5, 0.5, 'No numeric columns to summarize', 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        summary = self._data[columns].describe()
        
        fig, ax = plt.subplots(figsize=self._figsize)
        
        # Create heatmap of summary statistics
        sns.heatmap(summary, annot=True, fmt='.2f', cmap='YlGnBu', 
                   ax=ax, cbar_kws={'label': 'Value'})
        ax.set_title('Summary Statistics Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Columns')
        ax.set_ylabel('Statistics')
        
        plt.tight_layout()
        return fig
    
    def create_summary_table(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Generate a detailed summary statistics table.
        
        Args:
            columns: Columns to include (None for all)
            
        Returns:
            pd.DataFrame: Summary statistics
        """
        if columns is None:
            columns = self._data.columns.tolist()
        
        summary_list = []
        
        for col in columns:
            col_data = self._data[col]
            summary_dict = {
                'Column': col,
                'Type': str(col_data.dtype),
                'Count': col_data.count(),
                'Missing': col_data.isnull().sum(),
                'Missing %': f"{(col_data.isnull().sum() / len(col_data) * 100):.1f}%",
                'Unique': col_data.nunique()
            }
            
            if np.issubdtype(col_data.dtype, np.number):
                summary_dict.update({
                    'Mean': f"{col_data.mean():.2f}",
                    'Std': f"{col_data.std():.2f}",
                    'Min': f"{col_data.min():.2f}",
                    'Max': f"{col_data.max():.2f}",
                    'Median': f"{col_data.median():.2f}"
                })
            else:
                # For non-numeric columns
                most_common = col_data.mode()
                if len(most_common) > 0:
                    summary_dict['Most Common'] = str(most_common.iloc[0])
                else:
                    summary_dict['Most Common'] = 'N/A'
            
            summary_list.append(summary_dict)
        
        return pd.DataFrame(summary_list)
    
    def plot_summary_table(self, columns: Optional[List[str]] = None) -> plt.Figure:
        """
        Create a visual representation of the summary table.
        
        Args:
            columns: Columns to include (None for all)
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        summary_df = self.create_summary_table(columns)
        
        # Limit columns for display
        display_cols = ['Column', 'Type', 'Count', 'Missing', 'Unique']
        if 'Mean' in summary_df.columns:
            display_cols.extend(['Mean', 'Min', 'Max'])
        
        display_df = summary_df[display_cols]
        
        fig, ax = plt.subplots(figsize=(14, len(summary_df) * 0.5 + 1))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=display_df.values,
                        colLabels=display_df.columns,
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.15] * len(display_df.columns))
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(display_df.columns)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(display_df) + 1):
            for j in range(len(display_df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        plt.title('Data Summary Table', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        return fig
    
    def correlation_summary(self, columns: Optional[List[str]] = None,
                           threshold: float = 0.5) -> plt.Figure:
        """
        Summarize correlations above a threshold.
        
        Args:
            columns: Columns to include (None for all numeric)
            threshold: Minimum absolute correlation to display
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        if columns is None:
            columns = self._data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(columns) < 2:
            fig, ax = plt.subplots(figsize=self._figsize)
            ax.text(0.5, 0.5, 'Need at least 2 numeric columns', 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        corr = self._data[columns].corr()
        
        # Find high correlations
        high_corr = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                corr_value = corr.iloc[i, j]
                if abs(corr_value) >= threshold:
                    high_corr.append({
                        'Variable 1': corr.columns[i],
                        'Variable 2': corr.columns[j],
                        'Correlation': corr_value
                    })
        
        fig, ax = plt.subplots(figsize=self._figsize)
        
        if high_corr:
            df_corr = pd.DataFrame(high_corr)
            df_corr = df_corr.sort_values('Correlation', key=abs, ascending=False)
            
            colors = ['red' if x < 0 else 'green' for x in df_corr['Correlation']]
            bars = ax.barh(range(len(df_corr)), df_corr['Correlation'], color=colors, alpha=0.7)
            
            ax.set_yticks(range(len(df_corr)))
            ax.set_yticklabels([f"{r['Variable 1']} ↔ {r['Variable 2']}" 
                               for _, r in df_corr.iterrows()])
            ax.set_xlabel('Correlation Coefficient', fontweight='bold')
            ax.set_title(f'High Correlations (|r| ≥ {threshold})', 
                        fontsize=14, fontweight='bold')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, df_corr['Correlation'])):
                ax.text(val + 0.02 if val > 0 else val - 0.02, bar.get_y() + bar.get_height()/2,
                       f'{val:.3f}', ha='left' if val > 0 else 'right', va='center')
        else:
            ax.text(0.5, 0.5, f'No correlations found above threshold {threshold}',
                   ha='center', va='center', fontsize=12)
            ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def statistical_overview(self, column: str) -> plt.Figure:
        """
        Create a detailed statistical overview for a specific column.
        
        Args:
            column: Column name to analyze
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        if column not in self._data.columns:
            raise ValueError(f"Column '{column}' not found in data")
        
        data = self._data[column].dropna()
        
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Histogram
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(data, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        ax1.set_title('Distribution')
        ax1.set_xlabel(column)
        ax1.set_ylabel('Frequency')
        ax1.grid(alpha=0.3)
        
        # Box plot
        ax2 = fig.add_subplot(gs[0, 1])
        bp = ax2.boxplot(data, vert=True, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightgreen')
        ax2.set_title('Box Plot')
        ax2.set_ylabel(column)
        ax2.grid(alpha=0.3)
        
        # Q-Q plot
        ax3 = fig.add_subplot(gs[1, 0])
        from scipy import stats
        stats.probplot(data, dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot')
        ax3.grid(alpha=0.3)
        
        # Density plot
        ax4 = fig.add_subplot(gs[1, 1])
        data.plot(kind='density', ax=ax4, linewidth=2, color='purple')
        ax4.set_title('Kernel Density Estimation')
        ax4.set_xlabel(column)
        ax4.set_ylabel('Density')
        ax4.grid(alpha=0.3)
        
        # Statistics table
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        # Calculate statistics
        stats_dict = {
            'Statistic': ['Count', 'Mean', 'Std Dev', 'Min', 'Q1 (25%)', 
                         'Median', 'Q3 (75%)', 'Max', 'Skewness', 'Kurtosis'],
            'Value': [
                f"{len(data):,}",
                f"{data.mean():.4f}",
                f"{data.std():.4f}",
                f"{data.min():.4f}",
                f"{data.quantile(0.25):.4f}",
                f"{data.median():.4f}",
                f"{data.quantile(0.75):.4f}",
                f"{data.max():.4f}",
                f"{data.skew():.4f}",
                f"{data.kurtosis():.4f}"
            ]
        }
        
        stats_df = pd.DataFrame(stats_dict)
        
        table = ax5.table(cellText=stats_df.values,
                         colLabels=stats_df.columns,
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.3, 0.3])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(stats_df.columns)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.suptitle(f'Statistical Overview: {column}', 
                    fontsize=16, fontweight='bold', y=0.98)
        return fig
    
    def compare_groups(self, column: str, group_by: str) -> plt.Figure:
        """
        Compare summary statistics across different groups.
        
        Args:
            column: Column to analyze
            group_by: Column to group by
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        if column not in self._data.columns or group_by not in self._data.columns:
            raise ValueError("Columns not found in data")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Box plot
        self._data.boxplot(column=column, by=group_by, ax=axes[0, 0])
        axes[0, 0].set_title(f'{column} by {group_by}')
        axes[0, 0].set_xlabel(group_by)
        axes[0, 0].set_ylabel(column)
        
        # Violin plot
        groups = self._data[group_by].unique()
        positions = range(len(groups))
        data_by_group = [self._data[self._data[group_by] == g][column].dropna() 
                        for g in groups]
        
        parts = axes[0, 1].violinplot(data_by_group, positions=positions, 
                                      showmeans=True, showmedians=True)
        axes[0, 1].set_xticks(positions)
        axes[0, 1].set_xticklabels(groups)
        axes[0, 1].set_title(f'Distribution of {column} by {group_by}')
        axes[0, 1].set_xlabel(group_by)
        axes[0, 1].set_ylabel(column)
        
        # Bar plot of means
        means = self._data.groupby(group_by)[column].mean().sort_values()
        means.plot(kind='barh', ax=axes[1, 0], color='skyblue', edgecolor='black')
        axes[1, 0].set_title(f'Mean {column} by {group_by}')
        axes[1, 0].set_xlabel(f'Mean {column}')
        axes[1, 0].grid(axis='x', alpha=0.3)
        
        # Summary statistics table
        summary = self._data.groupby(group_by)[column].describe()
        axes[1, 1].axis('off')
        
        table = axes[1, 1].table(cellText=summary.round(2).values,
                                colLabels=summary.columns,
                                rowLabels=summary.index,
                                cellLoc='center',
                                loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        
        plt.suptitle(f'Group Comparison: {column} by {group_by}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def __repr__(self) -> str:
        """Return string representation."""
        return f"SummaryGenerator(data_shape={self._data.shape})"