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
    """Demo all helper functions"""
    print_section_header("PLOTEASE HELPER FUNCTIONS DEMO")
    
    # Generate sample data
    print("1. Generating sample data...")
    data = generate_sample_data(n_rows=100)
    print(f"✓ Generated data with shape {data.shape}")
    
    # Data validation
    print("\n2. Validating data...")
    validate_dataframe(data, min_rows=50)
    print("✓ Data validation passed")
    
    # Get column types
    print("\n3. Analyzing column types...")
    numeric_cols = get_numeric_columns(data)
    categorical_cols = get_categorical_columns(data)
    print(f"✓ Numeric columns: {numeric_cols}")
    print(f"✓ Categorical columns: {categorical_cols}")
    
    # Calculate statistics
    print("\n4. Calculating statistics...")
    stats = calculate_statistics(data['salary'])
    print("✓ Salary statistics:")
    for key, value in stats.items():
        print(f"  {key}: {format_number(value)}")
    
    # Find correlations
    print("\n5. Finding correlations...")
    high_corr = find_highly_correlated_pairs(data, threshold=0.5)
    print(f"✓ Found {len(high_corr)} highly correlated pairs")
    for var1, var2, corr in high_corr[:3]:
        print(f"  {var1} <-> {var2}: {format_number(corr)}")
    
    # Detect outliers
    print("\n6. Detecting outliers...")
    outliers = detect_outliers_iqr(data['salary'])
    print(f"✓ Found {outliers.sum()} outliers in salary")
    
    # Generate model results
    print("\n7. Generating model results...")
    models = generate_model_results(n_models=3)
    print("✓ Generated results for models:")
    for model, metrics in models.items():
        print(f"  {model}: Accuracy={format_percentage(metrics['Accuracy'])}")
    
    print("\n" + "="*80)
    print("✓ All helper functions working correctly!")
    print("="*80)


if __name__ == "__main__":
    demo_helpers()
