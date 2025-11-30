"""
PlotEase Utility and Helper Functions
Week 3: Functional Implementation - Submodules
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional


# DATA VALIDATION HELPERS

def validate_dataframe(data: pd.DataFrame, min_rows: int = 1) -> bool:
    """
    Validate that data is a proper DataFrame
    
    Args:
        data: Data to validate
        min_rows: Minimum number of rows required
    
    Returns:
        True if valid
    
    Raises:
        TypeError: If data is not a DataFrame
        ValueError: If DataFrame is empty or too small
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(data).__name__}")
    
    if data.empty:
        raise ValueError("DataFrame cannot be empty")
    
    if len(data) < min_rows:
        raise ValueError(f"DataFrame must have at least {min_rows} rows. Got {len(data)}")
    
    return True


def validate_column_exists(data: pd.DataFrame, column: str) -> bool:
    """
    Validate that a column exists in DataFrame
    
    Args:
        data: DataFrame to check
        column: Column name
    
    Returns:
        True if exists
    
    Raises:
        KeyError: If column doesn't exist
    """
    if column not in data.columns:
        raise KeyError(f"Column '{column}' not found. Available columns: {list(data.columns)}")
    
    return True


def validate_numeric_column(data: pd.DataFrame, column: str) -> bool:
    """
    Validate that a column is numeric
    
    Args:
        data: DataFrame containing the column
        column: Column name
    
    Returns:
        True if numeric
    
    Raises:
        ValueError: If column is not numeric
    """
    validate_column_exists(data, column)
    
    if data[column].dtype not in [np.number, 'int64', 'float64']:
        raise ValueError(f"Column '{column}' must be numeric. Got {data[column].dtype}")
    
    return True

# DATA CLEANING HELPERS

def get_numeric_columns(data: pd.DataFrame) -> List[str]:
    """
    Get list of numeric column names
    
    Args:
        data: DataFrame to analyze
    
    Returns:
        List of numeric column names
    """
    return data.select_dtypes(include=[np.number]).columns.tolist()


def get_categorical_columns(data: pd.DataFrame) -> List[str]:
    """
    Get list of categorical column names
    
    Args:
        data: DataFrame to analyze
    
    Returns:
        List of categorical column names
    """
    return data.select_dtypes(include=['object', 'category']).columns.tolist()


def remove_missing_values(data: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Remove columns with too many missing values
    
    Args:
        data: DataFrame to clean
        threshold: Maximum proportion of missing values (0-1)
    
    Returns:
        Cleaned DataFrame
    """
    missing_ratio = data.isnull().sum() / len(data)
    columns_to_keep = missing_ratio[missing_ratio <= threshold].index
    return data[columns_to_keep]


def detect_outliers_iqr(data: pd.Series) -> pd.Series:
    """
    Detect outliers using IQR method
    
    Args:
        data: Series to analyze
    
    Returns:
        Boolean Series indicating outliers
    """
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    return (data < lower_bound) | (data > upper_bound)


# STATISTICAL HELPERS

def calculate_statistics(data: pd.Series) -> Dict[str, float]:
    """
    Calculate comprehensive statistics for a series
    
    Args:
        data: Series to analyze
    
    Returns:
        Dictionary of statistics
    """
    return {
        'count': data.count(),
        'mean': data.mean(),
        'median': data.median(),
        'std': data.std(),
        'min': data.min(),
        'max': data.max(),
        'q1': data.quantile(0.25),
        'q3': data.quantile(0.75),
        'skewness': data.skew(),
        'kurtosis': data.kurtosis()
    }


def calculate_correlation_matrix(data: pd.DataFrame, method: str = 'pearson') -> pd.DataFrame:
    """
    Calculate correlation matrix for numeric columns
    
    Args:
        data: DataFrame with numeric columns
        method: Correlation method ('pearson', 'spearman', 'kendall')
    
    Returns:
        Correlation matrix
    """
    numeric_data = data.select_dtypes(include=[np.number])
    return numeric_data.corr(method=method)


def find_highly_correlated_pairs(data: pd.DataFrame, threshold: float = 0.8) -> List[Tuple[str, str, float]]:
    """
    Find pairs of highly correlated variables
    
    Args:
        data: DataFrame to analyze
        threshold: Correlation threshold
    
    Returns:
        List of tuples (var1, var2, correlation)
    """
    corr_matrix = calculate_correlation_matrix(data)
    
    # Get upper triangle to avoid duplicates
    upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    corr_matrix = corr_matrix.where(upper_triangle)
    
    high_corr = []
    for col in corr_matrix.columns:
        for idx in corr_matrix.index:
            value = corr_matrix.loc[idx, col]
            if not pd.isna(value) and abs(value) >= threshold:
                high_corr.append((idx, col, value))
    
    return sorted(high_corr, key=lambda x: abs(x[2]), reverse=True)


# FORMATTING HELPERS

def format_number(value: float, decimals: int = 2) -> str:
    """
    Format number with specified decimals
    
    Args:
        value: Number to format
        decimals: Number of decimal places
    
    Returns:
        Formatted string
    """
    return f"{value:.{decimals}f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format value as percentage
    
    Args:
        value: Value to format (0-1 range)
        decimals: Number of decimal places
    
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


def format_large_number(value: float) -> str:
    """
    Format large numbers with K, M, B suffixes
    
    Args:
        value: Number to format
    
    Returns:
        Formatted string
    """
    if abs(value) >= 1e9:
        return f"{value/1e9:.1f}B"
    elif abs(value) >= 1e6:
        return f"{value/1e6:.1f}M"
    elif abs(value) >= 1e3:
        return f"{value/1e3:.1f}K"
    else:
        return f"{value:.0f}"

# COLOR AND STYLING HELPERS

def generate_color_palette(n_colors: int, palette: str = 'viridis') -> List[str]:
    """
    Generate color palette for visualizations
    
    Args:
        n_colors: Number of colors needed
        palette: Palette name
    
    Returns:
        List of color codes
    """
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap(palette)
    return [cmap(i/n_colors) for i in range(n_colors)]


def get_theme_colors(theme: str) -> Dict[str, str]:
    """
    Get color scheme for a theme
    
    Args:
        theme: Theme name
    
    Returns:
        Dictionary of theme colors
    """
    themes = {
        'default': {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'accent': '#2ca02c',
            'background': '#ffffff',
            'text': '#000000'
        },
        'minimal': {
            'primary': '#4c72b0',
            'secondary': '#55a868',
            'accent': '#c44e52',
            'background': '#f8f8f8',
            'text': '#333333'
        },
        'dark': {
            'primary': '#8dd3c7',
            'secondary': '#fdb462',
            'accent': '#fb8072',
            'background': '#2b2b2b',
            'text': '#ffffff'
        },
        'colorful': {
            'primary': '#e377c2',
            'secondary': '#7f7f7f',
            'accent': '#bcbd22',
            'background': '#ffffff',
            'text': '#000000'
        }
    }
    return themes.get(theme, themes['default'])


# DATA GENERATION HELPERS (FOR TESTING)

def load_mtcars() -> pd.DataFrame:
    """
    Load the mtcars dataset (Motor Trend Car Road Tests)
    
    Returns:
        mtcars DataFrame with 32 cars and 11 variables
    """
    mtcars = pd.DataFrame({
        'mpg': [21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2, 
                17.8, 16.4, 17.3, 15.2, 10.4, 10.4, 14.7, 32.4, 30.4, 33.9,
                21.5, 15.5, 15.2, 13.3, 19.2, 27.3, 26.0, 30.4, 15.8, 19.7, 15.0, 21.4],
        'cyl': [6, 6, 4, 6, 8, 6, 8, 4, 4, 6, 
                6, 8, 8, 8, 8, 8, 8, 4, 4, 4,
                4, 8, 8, 8, 8, 4, 4, 4, 8, 6, 8, 4],
        'disp': [160.0, 160.0, 108.0, 258.0, 360.0, 225.0, 360.0, 146.7, 140.8, 167.6,
                 167.6, 275.8, 275.8, 275.8, 472.0, 460.0, 440.0, 78.7, 75.7, 71.1,
                 120.1, 318.0, 304.0, 350.0, 400.0, 79.0, 120.3, 95.1, 351.0, 145.0, 301.0, 121.0],
        'hp': [110, 110, 93, 110, 175, 105, 245, 62, 95, 123,
               123, 180, 180, 180, 205, 215, 230, 66, 52, 65,
               97, 150, 150, 245, 175, 66, 91, 113, 264, 175, 335, 109],
        'drat': [3.90, 3.90, 3.85, 3.08, 3.15, 2.76, 3.21, 3.69, 3.92, 3.92,
                 3.92, 3.07, 3.07, 3.07, 2.93, 3.00, 3.23, 4.08, 4.93, 4.22,
                 3.70, 2.76, 3.15, 3.73, 3.08, 4.08, 4.43, 3.77, 4.22, 3.62, 3.54, 4.11],
        'wt': [2.620, 2.875, 2.320, 3.215, 3.440, 3.460, 3.570, 3.190, 3.150, 3.440,
               3.440, 4.070, 3.730, 3.780, 5.250, 5.424, 5.345, 2.200, 1.615, 1.835,
               2.465, 3.520, 3.435, 3.840, 3.845, 1.935, 2.140, 1.513, 3.170, 2.770, 3.570, 2.780],
        'qsec': [16.46, 17.02, 18.61, 19.44, 17.02, 20.22, 15.84, 20.00, 22.90, 18.30,
                 18.90, 17.40, 17.60, 18.00, 17.98, 17.82, 17.42, 19.47, 18.52, 19.90,
                 20.01, 16.87, 17.30, 15.41, 17.05, 18.90, 16.70, 16.90, 14.50, 15.50, 14.60, 18.60],
        'vs': [0, 0, 1, 1, 0, 1, 0, 1, 1, 1,
               1, 0, 0, 0, 0, 0, 0, 1, 1, 1,
               1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
        'am': [1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
               0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        'gear': [4, 4, 4, 3, 3, 3, 3, 4, 4, 4,
                 4, 3, 3, 3, 3, 3, 3, 4, 4, 4,
                 3, 3, 3, 3, 3, 4, 5, 5, 5, 5, 5, 4],
        'carb': [4, 4, 1, 1, 2, 1, 4, 2, 2, 4,
                 4, 3, 3, 3, 4, 4, 4, 1, 2, 1,
                 1, 2, 2, 4, 2, 1, 2, 2, 4, 6, 8, 2]
    }, index=['Mazda RX4', 'Mazda RX4 Wag', 'Datsun 710', 'Hornet 4 Drive', 'Hornet Sportabout',
              'Valiant', 'Duster 360', 'Merc 240D', 'Merc 230', 'Merc 280',
              'Merc 280C', 'Merc 450SE', 'Merc 450SL', 'Merc 450SLC', 'Cadillac Fleetwood',
              'Lincoln Continental', 'Chrysler Imperial', 'Fiat 128', 'Honda Civic', 'Toyota Corolla',
              'Toyota Corona', 'Dodge Challenger', 'AMC Javelin', 'Camaro Z28', 'Pontiac Firebird',
              'Fiat X1-9', 'Porsche 914-2', 'Lotus Europa', 'Ford Pantera L', 'Ferrari Dino',
              'Maserati Bora', 'Volvo 142E'])
    
    return mtcars


def generate_sample_data(n_rows: int = 100, seed: int = 42) -> pd.DataFrame:
    """
    Generate sample data for testing
    
    Args:
        n_rows: Number of rows
        seed: Random seed
    
    Returns:
        Sample DataFrame
    """
    np.random.seed(seed)
    
    return pd.DataFrame({
        'age': np.random.randint(20, 70, n_rows),
        'salary': np.random.randint(30000, 150000, n_rows),
        'experience': np.random.randint(0, 30, n_rows),
        'department': np.random.choice(['Sales', 'Engineering', 'Marketing', 'HR'], n_rows),
        'performance': np.random.choice(['Low', 'Medium', 'High'], n_rows),
        'satisfaction': np.random.uniform(1, 10, n_rows)
    })


def generate_model_results(n_models: int = 3, seed: int = 42) -> Dict[str, Dict[str, float]]:
    """
    Generate sample model results for testing
    
    Args:
        n_models: Number of models
        seed: Random seed
    
    Returns:
        Dictionary of model results
    """
    np.random.seed(seed)
    
    models = {}
    model_names = ['Random Forest', 'XGBoost', 'Logistic Regression', 'SVM', 'Neural Network']
    
    for i in range(min(n_models, len(model_names))):
        base_score = np.random.uniform(0.70, 0.95)
        models[model_names[i]] = {
            'Accuracy': base_score,
            'Precision': base_score + np.random.uniform(-0.05, 0.05),
            'Recall': base_score + np.random.uniform(-0.05, 0.05),
            'F1-Score': base_score + np.random.uniform(-0.03, 0.03)
        }
        
        # Ensure all metrics are between 0 and 1
        for metric in models[model_names[i]]:
            models[model_names[i]][metric] = min(1.0, max(0.0, models[model_names[i]][metric]))
    
    return models


# EXPORT HELPERS

def export_summary_to_csv(summary: pd.DataFrame, filename: str):
    """
    Export summary DataFrame to CSV
    
    Args:
        summary: Summary DataFrame
        filename: Output filename
    """
    summary.to_csv(filename, index=False)
    print(f"✓ Summary exported to {filename}")


def export_summary_to_excel(summary: pd.DataFrame, filename: str):
    """
    Export summary DataFrame to Excel
    
    Args:
        summary: Summary DataFrame
        filename: Output filename
    """
    summary.to_excel(filename, index=False)
    print(f"✓ Summary exported to {filename}")


# PRINT HELPERS

def print_section_header(title: str, width: int = 80):
    """
    Print formatted section header
    
    Args:
        title: Section title
        width: Width of header
    """
    print("\n" + "="*width)
    print(f"  {title}")
    print("="*width + "\n")


def print_data_info(data: pd.DataFrame):
    """
    Print comprehensive data information
    
    Args:
        data: DataFrame to describe
    """
    print_section_header("DATA INFORMATION")
    
    print(f"Shape: {data.shape}")
    print(f"Rows: {len(data):,}")
    print(f"Columns: {len(data.columns)}")
    print(f"Memory Usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\nColumn Types:")
    print(data.dtypes)
    
    print("\nMissing Values:")
    missing = data.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("  No missing values")
    
    print("\nNumeric Columns:", get_numeric_columns(data))
    print("Categorical Columns:", get_categorical_columns(data))


# DEMO FUNCTION

def demo_helpers():
    """Demo all helper functions using mtcars"""
    print_section_header("PLOTEASE HELPER FUNCTIONS DEMO - Using mtcars")
    
    # Load mtcars data
    print("1. Loading mtcars dataset...")
    data = load_mtcars()
    print(f"✓ Loaded mtcars data with shape {data.shape}")
    print(f"  32 cars, 11 variables")
    
    # Data validation
    print("\n2. Validating data...")
    validate_dataframe(data, min_rows=30)
    print("✓ Data validation passed")
    
    # Get column types
    print("\n3. Analyzing column types...")
    numeric_cols = get_numeric_columns(data)
    categorical_cols = get_categorical_columns(data)
    print(f"✓ Numeric columns: {numeric_cols}")
    print(f"✓ Categorical columns: {categorical_cols}")
    
    # Calculate statistics
    print("\n4. Calculating statistics for MPG...")
    stats = calculate_statistics(data['mpg'])
    print("✓ MPG (fuel efficiency) statistics:")
    for key, value in stats.items():
        print(f"  {key}: {format_number(value)}")
    
    # Find correlations
    print("\n5. Finding correlations in mtcars...")
    high_corr = find_highly_correlated_pairs(data, threshold=0.7)
    print(f"✓ Found {len(high_corr)} highly correlated pairs")
    for var1, var2, corr in high_corr[:5]:
        print(f"  {var1} <-> {var2}: {format_number(corr)}")
    
    # Detect outliers
    print("\n6. Detecting outliers in horsepower...")
    outliers = detect_outliers_iqr(data['hp'])
    print(f"✓ Found {outliers.sum()} outliers in horsepower")
    if outliers.sum() > 0:
        outlier_cars = data[outliers].index.tolist()
        print(f"  Outlier cars: {', '.join(outlier_cars[:3])}")
    
    # Generate model results
    print("\n7. Generating model results for MPG prediction...")
    models = generate_model_results(n_models=3)
    print("✓ Generated results for models:")
    for model, metrics in models.items():
        print(f"  {model}: Accuracy={format_percentage(metrics['Accuracy'])}")
    
    # Print comprehensive data info
    print("\n8. Comprehensive mtcars information:")
    print_data_info(data)
    
    print("\n" + "="*80)
    print("✓ All helper functions working correctly with mtcars!")
    print("="*80)


if __name__ == "__main__":
    demo_helpers()
