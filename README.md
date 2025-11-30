# PlotEase
Our library addresses the common problem of complex coding processes required for data plotting and visualization. Many existing solutions demand extensive manual coding, creating a barrier for users who simply want to generate insightful plots quickly and efficiently. The library addresses the complex coding often needed for data plotting. We aim to solve this by providing a user-friendly library that makes plot creation more accessible, reducing the code needed for effective visualizations. 

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


![PlotEase UML Class Diagram](PlotEase_UMLDiagram/PlotEase_UMLDiagram.png)

_ _ _

## Features

### **1. AutoPlot Diagnostics**
Automatically generate comprehensive diagnostic plots based on your data types:
- Distribution plots for numeric variables
- Correlation heatmaps
- Missing value analysis
- Outlier detection with boxplots
- Target variable analysis

### **2. Tabular Summary**
Generate detailed statistical summaries with a single command:
- Comprehensive statistics for numeric columns
- Category analysis for categorical data
- Missing value reports
- Export to CSV/Excel

### **3. Model Comparison**
Visualize machine learning model performance:
- Grouped bar charts comparing metrics
- Radar charts showing model profiles
- Best model identification
- Side-by-side performance comparison

### **4. Quick Plotting**
Create beautiful plots with minimal syntax:
- Automatic plot type detection
- Scatter, line, bar, histogram, and box plots
- Custom styling and themes
- One-line plot generation


## Installation

### From PyPI (Recommended)
```bash
pip install plotease
```

### From Source
```bash
git clone https://github.com/yourusername/plotease.git
cd plotease
pip install -e .
```

### Requirements
- Python 3.8+
- pandas >= 1.3.0
- numpy >= 1.21.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- scipy >= 1.7.0

## Quick Start

### Basic Usage

```python
import pandas as pd
from plotease import PlotEase

# Load your data
data = pd.read_csv('your_data.csv')

# Initialize PlotEase
pe = PlotEase(data, theme='minimal')

# That's it! You're ready to visualize
```
### Tabular Summary

```python
# Full summary (numeric + categorical)
summary = pe.tabular_summary(style='full')

# Only numeric columns
numeric_summary = pe.tabular_summary(style='numeric')

# Only categorical columns
cat_summary = pe.tabular_summary(style='categorical')
```

**Output includes:**
- Count, Missing values, Mean, Std, Min, Max
- Unique values, Top category, Frequency
- Memory usage statistics

### Model Comparison

```python
models_results = {
    'Linear Regression': {
        'Accuracy': 0.85, 
        'Precision': 0.82, 
        'Recall': 0.84,
        'F1-Score': 0.83
    },
    'Random Forest': {
        'Accuracy': 0.92, 
        'Precision': 0.90, 
        'Recall': 0.91,
        'F1-Score': 0.90
    },
    'XGBoost': {
        'Accuracy': 0.94, 
        'Precision': 0.93, 
        'Recall': 0.94,
        'F1-Score': 0.93
    }
}

pe.compare_models(models_results)

# Compare specific metrics only
pe.compare_models(models_results, metrics=['Accuracy', 'F1-Score'])
```


### Quick Plotting

```python
# Automatic plot type detection
pe.quick_plot('age', 'salary')  # Auto-detects scatter plot

# Specific plot types
pe.quick_plot('age', 'salary', kind='scatter', color='blue')
pe.quick_plot('category', kind='bar', color='green')
pe.quick_plot('values', kind='hist', color='orange')
pe.quick_plot('var1', 'var2', kind='line', color='red')

# Custom styling
pe.quick_plot('x', 'y', kind='scatter', 
              color='coral', 
              title='My Custom Title',
              figsize=(12, 6))

# Apply custom style globally
custom_style = {
    'font_size': 14,
    'figure_facecolor': '#f9f9f9',
    'axes_facecolor': 'white',
    'grid_alpha': 0.3
}
pe.set_style(custom_style)
```

## Themes

PlotEase comes with 4 built-in themes:

```python
# Available themes
pe = PlotEase(data, theme='default')    # Classic scientific
pe = PlotEase(data, theme='minimal')    # Clean and simple
pe = PlotEase(data, theme='dark')       # Dark mode
pe = PlotEase(data, theme='colorful')   # Vibrant colors

# Change theme after initialization
pe.set_theme('dark')
```

##  Testing

Run the test suite:

```bash
# Run all tests
pytest tests/test_plotease.py -v

# Run specific test class
pytest tests/test_plotease.py::TestDunderMethods -v

# Run with coverage
pytest tests/test_plotease.py --cov=plotease --cov-report=html
```

Run the demo:

```bash
python demo.py
```

## 📖 Documentation

### Helper Functions

PlotEase includes utility functions in `plotease.utils`:

```python
from plotease.utils import (
    load_mtcars,                    # Load example dataset
    validate_dataframe,             # Validate data
    get_numeric_columns,            # Get numeric column names
    get_categorical_columns,        # Get categorical column names
    calculate_statistics,           # Comprehensive statistics
    find_highly_correlated_pairs,   # Correlation analysis
    detect_outliers_iqr,            # Outlier detection
    export_summary_to_csv,          # Export to CSV
    print_data_info                 # Print data information
)
```

## Contributors

This project was developed as part of an Object-Oriented Programming course.

### Team Members

- **[Khassandra Louise C. Jayme]** - Lead Developer - [GitHub](https://github.com/Khassndrajayme)
- **[Team Member 2]** - Developer - [GitHub](https://github.com/Sheena06J)
- **[Team Member 3]** - Developer - [GitHub](https://github.com/MysteriousHotdog)
- **[Team Member 4]** - Developer - [GitHub](https://github.com/AllenKalbo)
- **[Team Member 5]** - Developer - [GitHub](https://github.com/noryt-py)

### Acknowledgments

- Thanks to our instructor for guidance and support
- Inspired by matplotlib, seaborn, and ggplot2
- mtcars dataset from R's datasets package

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


**Made with Honeybunchsugarplumsweetypie by the PlotEase Team**
