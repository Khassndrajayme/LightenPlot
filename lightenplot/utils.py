# lightenplot/utils.py
"""Utility functions for lightenplot."""

import pandas as pd
import numpy as np


def validate_columns(data, columns):
    """
    Validate that columns exist in DataFrame.
    
    Args:
        data: pandas DataFrame
        columns: List of column names
        
    Raises:
        ValueError: If any column not found
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas DataFrame")
    
    missing = [col for col in columns if col not in data.columns]
    if missing:
        raise ValueError(f"Columns not found: {missing}")


def auto_bins(data, method='sturges'):
    """
    Calculate optimal number of bins for histogram.
    
    Args:
        data: Array-like data
        method: Method to use ('sturges', 'sqrt', 'rice')
        
    Returns:
        Optimal number of bins
    """
    n = len(data)
    if method == 'sturges':
        return int(np.ceil(np.log2(n) + 1))
    elif method == 'sqrt':
        return int(np.ceil(np.sqrt(n)))
    elif method == 'rice':
        return int(np.ceil(2 * n**(1/3)))
    else:
        return 30


def prepare_data(data, x=None, y=None):
    """
    Prepare data for plotting.
    
    Args:
        data: Input data (DataFrame, dict, or arrays)
        x: X column/data
        y: Y column/data
        
    Returns:
        Tuple of (x_data, y_data)
    """
    if isinstance(data, pd.DataFrame):
        x_data = data[x] if isinstance(x, str) else x
        y_data = data[y] if isinstance(y, str) else y
    elif isinstance(data, dict):
        df = pd.DataFrame(data)
        x_data = df[x] if isinstance(x, str) else x
        y_data = df[y] if isinstance(y, str) else y
    else:
        x_data, y_data = x, y
    
    return x_data, y_data


def normalize_data(data, method='minmax'):
    """
    Normalize numerical data.
    
    Args:
        data: Array-like numerical data
        method: Normalization method ('minmax' or 'zscore')
        
    Returns:
        Normalized data
    """
    data = np.array(data)
    if method == 'minmax':
        return (data - data.min()) / (data.max() - data.min())
    elif method == 'zscore':
        return (data - data.mean()) / data.std()
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def color_palette(n_colors, palette='default'):
    """
    Generate color palette.
    
    Args:
        n_colors: Number of colors needed
        palette: Palette name ('default', 'pastel', 'vibrant')
        
    Returns:
        List of color hex codes
    """
    palettes = {
        'default': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
        'pastel': ['#FFB3BA', '#BAFFC9', '#BAE1FF', '#FFFFBA', '#FFD8BA'],
        'vibrant': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    }
    
    colors = palettes.get(palette, palettes['default'])
    # Repeat colors if needed
    return (colors * (n_colors // len(colors) + 1))[:n_colors]
