# 🎨 LightenPlot

**Dramatically Simplify Data Visualization in Python**

LightenPlot is a Python library that reduces complex visualization code into single-line commands, while maintaining the power and flexibility of matplotlib and seaborn.

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI](https://img.shields.io/badge/pypi-lightenplot-orange.svg)](https://pypi.org/project/lightenplot/)

# **Table of Contents**

* [ Features](#-features)
* [ Installation](#-installation)
* [ Quick Start](#-quick-start)
* [ Examples](#-examples)
* [ Plot Composition](#-plot-composition)
* [ Themes](#-themes)
* [ Exporting Plots](#-exporting-plots)
* [ Architecture Overview](#-architecture-overview)
* [ API Reference](#-api-reference)
* [ Testing](#-testing)
* [ Documentation](#-documentation)
* [ Contributing](#-contributing)
* [ RichieClan Team](#-richieclan-team)
* [ License](#-license)
* [ Contact](#-contact)


## Features

1. **One-Line Plotting**: Create beautiful visualizations with single commands
2. **Method Chaining**: Fluent API for intuitive plot customization
3. **Built-in Themes**: Multiple professional themes (default, dark, minimal, colorful)
4. **Comprehensive Plot Types**: Scatter, line, bar, histogram, box, heatmap
5. **Smart Composition**: Easily combine multiple plots
6. **Easy Export**: Save in multiple formats (PNG, PDF, SVG)
7. **Full OOP Design**: Encapsulation, inheritance, polymorphism

## Installation

```bash
pip install lightenplot
```

### From Source

```bash
git clone https://github.com/khassndrajayme/lightenplot.git
cd lightenplot
pip install -e .
```

## Quick Start

### Traditional Way (10+ lines)

```python
import matplotlib.pyplot as plt
import pandas as pd

data = pd.DataFrame({'x': [1, 2, 3, 4], 'y': [2, 4, 6, 8]})
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(data['x'], data['y'], alpha=0.6)
ax.set_xlabel('X Values')
ax.set_ylabel('Y Values')
ax.set_title('My Scatter Plot')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### LightenPlot Way (1 line!)

```python
import lightenplot as lp

lp.scatter(data, x='x', y='y').set_title('My Scatter Plot').show()
```

## Examples

### Scatter Plot

```python
import lightenplot as lp
import pandas as pd

data = pd.DataFrame({
    'age': [25, 30, 35, 40, 45],
    'salary': [50000, 60000, 75000, 80000, 95000]
})

lp.scatter(data, x='age', y='salary', color='steelblue') \
  .set_title('Age vs Salary') \
  .set_labels('Age (years)', 'Salary ($)') \
  .apply_theme('minimal') \
  .save('plot.png') \
  .show()
```

### Line Plot

```python
lp.line(data, x='month', y='revenue', color='#FF6B6B', linewidth=2.5) \
  .set_title('Monthly Revenue') \
  .apply_theme('dark') \
  .show()
```

### Bar Plot

```python
lp.bar(data, x='category', y='value', color='#4ECDC4') \
  .set_title('Sales by Category') \
  .show()
```

### Histogram

```python
import numpy as np

ages = np.random.normal(35, 10, 1000)
lp.histogram(ages, bins=30, color='skyblue') \
  .set_title('Age Distribution') \
  .show()
```

### Box Plot

```python
lp.boxplot(data, columns=['Group_A', 'Group_B', 'Group_C']) \
  .set_title('Performance Comparison') \
  .show()
```

### Heatmap

```python
correlation_matrix = data.corr()
lp.heatmap(correlation_matrix, cmap='coolwarm', annot=True) \
  .set_title('Correlation Matrix') \
  .show()
```

### Plot Composition

```python
# Create individual plots
plot1 = lp.scatter(data, x='x', y='y')
plot2 = lp.line(data, x='date', y='value')
plot3 = lp.bar(data, x='cat', y='count')
plot4 = lp.histogram(values, bins=20)

# Compose into 2x2 grid
composer = lp.compose(rows=2, cols=2)
composer.add_plot(plot1).add_plot(plot2).add_plot(plot3).add_plot(plot4)
composer.show()
```

## Themes

LightenPlot includes 4 built-in themes:

- `default`: Clean, professional look
- `dark`: Dark mode with high contrast
- `minimal`: Minimalist design with no spines
- `colorful`: Vibrant, eye-catching colors

```python
# Apply theme
plot.apply_theme('dark')

# List available themes
print(lp.ThemeManager.list_themes())
```

## Exporting Plots

```python
# Single format
plot.save('output.png', dpi=300)

# Multiple formats
from lightenplot import PlotExporter
exporter = PlotExporter(plot.figure)
exporter.save_multiple('output', formats=['png', 'pdf', 'svg'])
```

## Architecture

LightenPlot is built with solid OOP principles:

### Core Classes

1. **BasePlot**: Abstract base class for all plots
2. **ScatterPlot, LinePlot, BarPlot, etc.**: Specific plot implementations
3. **PlotComposer**: Compose multiple plots (composition pattern)
4. **ThemeManager**: Manage and apply themes (singleton pattern)
5. **PlotExporter**: Handle plot exports

### OOP Features Demonstrated

1. **Encapsulation**: Private attributes (`_data`, `_theme`)  
2. **Inheritance**: All plots inherit from `BasePlot`  
3. **Polymorphism**: Theme classes with common interface  
4. **Composition**: `PlotComposer` contains `BasePlot` instances  
5. **Dunder Methods**: `__repr__`, `__str__`, `__eq__`, `__lt__`

## API Reference

### Quick Functions

| Function | Description |
|----------|-------------|
| `scatter()` | Create scatter plot |
| `line()` | Create line plot |
| `bar()` | Create bar plot |
| `histogram()` | Create histogram |
| `boxplot()` | Create box plot |
| `heatmap()` | Create heatmap |
| `compose()` | Create plot composer |

### Common Methods

| Method | Description |
|--------|-------------|
| `.set_title(title)` | Set plot title |
| `.set_labels(x, y)` | Set axis labels |
| `.apply_theme(name)` | Apply theme |
| `.save(filename)` | Save plot |
| `.show()` | Display plot |

## Testing

Run unit tests:

```bash
# Simple run
pytest tests/test_all.py

# Verbose (shows test names)
pytest tests/test_all.py -v

# Show print statements
pytest tests/test_all.py -v -s

# Stop on first failure
pytest tests/test_all.py -x

# Run specific test class
pytest tests/test_all.py::TestScatterPlot -v

# Run specific test method
pytest tests/test_all.py::TestScatterPlot::test_scatter_creation -v

# With coverage
pytest tests/test_all.py --cov=lightenplot --cov-report=term

# Run all tests in tests/ folder
pytest tests/
```

Or with unittest:

```bash
python -m unittest discover tests/
```

## Documentation

Full documentation is available in the [docs](docs/) folder or can be generated using:

```bash
# Using pdoc
pip install pdoc
pdoc --html lightenplot

# Using Sphinx
cd docs/
make html
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## RichieClan Team

- **Khassandra Louise C. Jayme** - Lead Developer - [GitHub](https://github.com/Khassndrajayme)
- **Sheena Angela T. Janog** - Developer - [GitHub](https://github.com/Sheena06J)
- **Xavier Neo Mahilum** - Developer - [GitHub](https://github.com/MysteriousHotdog)
- **Allen Floro Ventura** - Developer - [GitHub](https://github.com/AllenKalbo)
- **Genetyron Zamoranos** - Developer - [GitHub](https://github.com/noryt-py)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on top of [Matplotlib](https://matplotlib.org/)
- Inspired by [Seaborn](https://seaborn.pydata.org/)
- Thanks to our instructor for guidance and support
- Inspired by matplotlib, seaborn, and ggplot2


## Contact

For questions or feedback, please open an issue on GitHub or contact us at khassandrajayme@gmail.com

---

**Made with Love by RichieClan**
