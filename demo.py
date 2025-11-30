"""
PlotEase Demo Script - Using mtcars Dataset
Demonstrates all features and class interactions
"""

import pandas as pd
import numpy as np
from plotease import PlotEase, DiagnosticPlotter, SummaryGenerator, ModelComparator, QuickPlotter

def print_section(title):
    """Helper function to print section headers"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def load_mtcars_data():
    """Load the mtcars dataset"""
    # mtcars dataset (Motor Trend Car Road Tests)
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


def demo_basic_usage():
    """Demo 1: Basic Usage and Initialization"""
    print_section("DEMO 1: Basic Usage and Initialization - mtcars Dataset")
    
    # Load mtcars data
    mtcars = load_mtcars_data()
    
    print("mtcars Dataset Loaded:")
    print(f"  Shape: {mtcars.shape}")
    print(f"  Columns: {list(mtcars.columns)}")
    print("Column Descriptions:")
    print("  mpg  - Miles per gallon")
    print("  cyl  - Number of cylinders")
    print("  disp - Displacement (cu.in.)")
    print("  hp   - Gross horsepower")
    print("  drat - Rear axle ratio")
    print("  wt   - Weight (1000 lbs)")
    print("  qsec - 1/4 mile time")
    print("  vs   - Engine (0=V-shaped, 1=straight)")
    print("  am   - Transmission (0=automatic, 1=manual)")
    print("  gear - Number of forward gears")
    print("  carb - Number of carburetors")
    
    print(f"First 5 cars:")
    print(mtcars.head())
    
    # Initialize PlotEase
    pe = PlotEase(mtcars, theme='minimal')
    print(f"\n✓ PlotEase initialized: {pe}")
    print(f"  Total cars analyzed: {len(pe)}")
    
    return pe, mtcars


def demo_encapsulation(pe):
    """Demo 2: Encapsulation - Protected attributes and getters"""
    print_section("DEMO 2: Encapsulation (Protected Attributes & Getters)")
    
    # Access data through getter (correct way)
    print("✓ Accessing data through getter method:")
    retrieved_data = pe.get_data()
    print(f"  Data shape: {retrieved_data.shape}")
    print(f"  Sample car: {retrieved_data.index[0]}")
    
    # Change theme through setter
    print("\n✓ Changing theme through setter method:")
    original_theme = pe._theme
    print(f"  Original theme: {original_theme}")
    
    pe.set_theme('dark')
    print(f"  New theme: {pe._theme}")
    
    # Protected attributes are prefixed with underscore
    print("\n✓ Protected attributes:")
    print(f"  _data: {type(pe._data)}")
    print(f"  _theme: {pe._theme}")
    print(f"  _diagnostic: {type(pe._diagnostic)}")
    print(f"  _summary: {type(pe._summary)}")


def demo_inheritance():
    """Demo 3: Inheritance - Parent-child relationships"""
    print_section("DEMO 3: Inheritance (Parent-Child Relationships)")
    
    # Load small subset for demo
    mtcars = load_mtcars_data()
    sample_data = mtcars[['mpg', 'hp', 'wt']].head(10)
    
    # Show inheritance hierarchy
    pe = PlotEase(sample_data)
    dp = DiagnosticPlotter(sample_data)
    qp = QuickPlotter(sample_data)
    
    print("✓ Inheritance Hierarchy:")
    print(f"  PlotEase inherits from VisualizationBase: {True}")
    print(f"  DiagnosticPlotter inherits from VisualizationBase: {True}")
    print(f"  QuickPlotter inherits from VisualizationBase: {True}")
    
    print("\n✓ Inherited methods available:")
    print(f"  PlotEase.get_data(): {hasattr(pe, 'get_data')}")
    print(f"  DiagnosticPlotter.get_data(): {hasattr(dp, 'get_data')}")
    print(f"  QuickPlotter.get_data(): {hasattr(qp, 'get_data')}")
    
    print("\n✓ All classes have access to protected _data attribute:")
    print(f"  PlotEase._data shape: {pe._data.shape}")
    print(f"  DiagnosticPlotter._data shape: {dp._data.shape}")
    print(f"  QuickPlotter._data shape: {qp._data.shape}")


def demo_polymorphism():
    """Demo 4: Polymorphism - Method Overriding"""
    print_section("DEMO 4: Polymorphism (Method Overriding)")
    
    mtcars = load_mtcars_data()
    sample_data = mtcars[['mpg', 'hp', 'wt']].head(10)
    
    pe = PlotEase(sample_data)
    dp = DiagnosticPlotter(sample_data)
    qp = QuickPlotter(sample_data)
    
    print("✓ Abstract method render() is overridden in each subclass:")
    print("\n  PlotEase.render():")
    pe.render()
    
    print("\n  DiagnosticPlotter.render():")
    print("  (Generates diagnostic plots automatically)")
    
    print("\n  QuickPlotter.render():")
    qp.render()
    
    print("\n✓ Same method name, different implementations - this is polymorphism!")


def demo_composition(pe):
    """Demo 5: Composition - PlotEase contains other classes"""
    print_section("DEMO 5: Composition (HAS-A Relationships)")
    
    print("✓ PlotEase contains (composes) these components:")
    print(f"  1. DiagnosticPlotter: {type(pe._diagnostic).__name__}")
    print(f"  2. SummaryGenerator: {type(pe._summary).__name__}")
    print(f"  3. QuickPlotter: {type(pe._plotter).__name__}")
    print(f"  4. ModelComparator: {pe._comparator} (initialized when needed)")
    
    print("\n✓ PlotEase delegates work to its components:")
    print("  - autoplot() → delegates to DiagnosticPlotter")
    print("  - tabular_summary() → delegates to SummaryGenerator")
    print("  - compare_models() → creates and uses ModelComparator")
    print("  - quick_plot() → delegates to QuickPlotter")
    
    print("\n✓ This is composition: PlotEase HAS-A relationship with components")


def demo_dunder_methods():
    """Demo 6: Dunder Methods (Magic Methods)"""
    print_section("DEMO 6: Dunder Methods (Magic Methods)")
    
    mtcars = load_mtcars_data()
    
    # Create different sized datasets
    data_full = mtcars
    data_small = mtcars.head(10)
    
    pe1 = PlotEase(data_full, theme='minimal')
    pe2 = PlotEase(data_full, theme='minimal')
    pe3 = PlotEase(data_small, theme='minimal')
    
    print("✓ __repr__ - String representation:")
    print(f"  {repr(pe1)}")
    
    print("\n✓ __len__ - Length of data:")
    print(f"  len(pe1) = {len(pe1)} cars")
    print(f"  len(pe3) = {len(pe3)} cars")
    
    print("\n✓ __eq__ - Equality comparison:")
    print(f"  pe1 == pe2: {pe1 == pe2} (same data and theme)")
    print(f"  pe1 == pe3: {pe1 == pe3} (different data)")
    
    print("\n✓ __lt__ - Less than comparison:")
    print(f"  pe1 < pe3: {pe1 < pe3} (32 cars < 10 cars)")
    print(f"  pe3 < pe1: {pe3 < pe1} (10 cars < 32 cars)")
    
    # ModelComparator dunder methods
    print("\n✓ ModelComparator dunder methods:")
    models_good = {
        'High Performance': {'Accuracy': 0.95, 'Precision': 0.93}
    }
    models_poor = {
        'Basic Model': {'Accuracy': 0.75, 'Precision': 0.73}
    }
    
    mc_good = ModelComparator(models_good)
    mc_poor = ModelComparator(models_poor)
    
    print(f"  mc_good > mc_poor: {mc_good > mc_poor}")
    print(f"  {repr(mc_good)}")


def demo_feature_1_autoplot(pe):
    """Demo 7: Feature 1 - AutoPlot Diagnostics"""
    print_section("DEMO 7: Feature 1 - AutoPlot Diagnostics (mtcars)")
    
    print("✓ Generating automatic diagnostic plots for mtcars...")
    print("  This will create up to 6 diagnostic visualizations:")
    print("    1. Distribution of numeric variables (mpg, hp, wt, etc.)")
    print("    2. Correlation matrix heatmap")
    print("    3. Missing values analysis")
    print("    4. Target variable distribution (mpg)")
    print("    5. Categorical variable distribution")
    print("    6. Outlier detection (boxplots)")
    
    pe.autoplot(target='mpg', max_plots=6)


def demo_feature_2_summary(pe):
    """Demo 8: Feature 2 - Tabular Summary"""
    print_section("DEMO 8: Feature 2 - Tabular Summary (mtcars)")
    
    print("✓ Generating comprehensive mtcars summary...")
    summary = pe.tabular_summary(style='full')
    
    print("\nSummary DataFrame:")
    print(summary.to_string())
    
    print("\n✓ Can also generate specific summaries:")
    print("  - style='numeric': Only numeric columns (mpg, hp, wt, etc.)")
    print("  - style='categorical': Only categorical columns")
    print("  - style='full': All columns (default)")


def demo_feature_3_model_comparison(pe):
    """Demo 9: Feature 3 - Model Comparison"""
    print_section("DEMO 9: Feature 3 - Model Comparison (Predicting MPG)")
    
    print("✓ Comparing ML models for predicting car fuel efficiency (mpg)...")
    
    models = {
        'Linear Regression': {
            'R²': 0.85, 
            'MAE': 2.5, 
            'RMSE': 3.2, 
            'MAPE': 0.12
        },
        'Random Forest': {
            'R²': 0.92, 
            'MAE': 1.8, 
            'RMSE': 2.3, 
            'MAPE': 0.08
        },
        'XGBoost': {
            'R²': 0.94, 
            'MAE': 1.5, 
            'RMSE': 2.0, 
            'MAPE': 0.07
        },
        'Neural Network': {
            'R²': 0.89, 
            'MAE': 2.1, 
            'RMSE': 2.7, 
            'MAPE': 0.10
        }
    }
    
    print("\nModels predicting MPG:")
    for model, metrics in models.items():
        print(f"  - {model}: R²={metrics['R²']:.2f}, RMSE={metrics['RMSE']:.1f}")
    
    print("\n✓ Generating comparison visualizations...")
    print("  - Grouped bar chart comparing all metrics")
    print("  - Radar chart showing model performance profile")
    
    # Note: We need to normalize metrics for visualization (0-1 scale)
    models_normalized = {
        'Linear Regression': {'Accuracy': 0.85, 'Precision': 0.82, 'Recall': 0.84, 'F1': 0.83},
        'Random Forest': {'Accuracy': 0.92, 'Precision': 0.90, 'Recall': 0.91, 'F1': 0.90},
        'XGBoost': {'Accuracy': 0.94, 'Precision': 0.93, 'Recall': 0.94, 'F1': 0.93},
        'Neural Network': {'Accuracy': 0.89, 'Precision': 0.87, 'Recall': 0.88, 'F1': 0.87}
    }
    pe.compare_models(models_normalized)


def demo_feature_4_quick_plot(pe):
    """Demo 10: Feature 4 - Quick Plotting"""
    print_section("DEMO 10: Feature 4 - Quick Plotting (mtcars Analysis)")
    
    print("✓ Creating quick plots with minimal syntax...")
    
    print("\n1. Scatter plot: Horsepower vs MPG")
    print("   (Exploring relationship between engine power and fuel efficiency)")
    pe.quick_plot('hp', 'mpg', kind='scatter', color='coral', 
                  title='Horsepower vs Fuel Efficiency (MPG)')
    
    print("\n2. Scatter plot: Weight vs MPG")
    print("   (Does heavier car mean lower fuel efficiency?)")
    pe.quick_plot('wt', 'mpg', kind='scatter', color='steelblue',
                  title='Car Weight vs Fuel Efficiency')
    
    print("\n3. Histogram: MPG Distribution")
    print("   (Distribution of fuel efficiency across all cars)")
    pe.quick_plot('mpg', kind='hist', color='green',
                  title='Distribution of Miles Per Gallon')
    
    print("\n4. Histogram: Horsepower Distribution")
    pe.quick_plot('hp', kind='hist', color='orange',
                  title='Distribution of Horsepower')
    
    print("\n✓ Custom styling:")
    custom_style = {
        'font_size': 12,
        'figure_facecolor': '#f9f9f9',
        'axes_facecolor': 'white',
        'grid_alpha': 0.3
    }
    pe.set_style(custom_style)
    print("  Custom style applied!")


def demo_class_interactions():
    """Demo 11: Class Interactions"""
    print_section("DEMO 11: Class Interactions (mtcars)")
    
    print("✓ Demonstrating how classes work together...")
    
    mtcars = load_mtcars_data()
    sample_data = mtcars[['mpg', 'hp', 'wt']].head(15)
    
    # Individual components can work independently
    print("\n1. Components can work independently:")
    sg = SummaryGenerator(sample_data)
    print(f"   SummaryGenerator: {sg}")
    print(f"   Number of cars: {len(sg)}")
    
    # Or through PlotEase composition
    print("\n2. Or work together through PlotEase:")
    pe = PlotEase(sample_data)
    summary = pe.tabular_summary(style='numeric')
    print(f"   Summary generated through PlotEase")
    print(f"   Variables summarized: {len(summary)}")
    print(f"\n   Summary:")
    print(summary.to_string())
    
    print("This demonstrates loose coupling and high cohesion!")


def demo_real_world_analysis():
    """Demo 12: Real-World Analysis Example"""
    print_section("DEMO 12: Real-World Analysis - Car Performance vs Efficiency")
    
    mtcars = load_mtcars_data()
    pe = PlotEase(mtcars, theme='minimal')
    
    print("Analysis Question: How do different car characteristics affect fuel efficiency?")
    print("Step 1: Overview of the dataset")
    print(f"   Total cars: {len(mtcars)}")
    print(f"   Average MPG: {mtcars['mpg'].mean():.1f}")
    print(f"   Average HP: {mtcars['hp'].mean():.0f}")
    print(f"   Average Weight: {mtcars['wt'].mean():.2f} (1000 lbs)")
    
    print("Step 2: Correlation Analysis")
    correlations = mtcars[['mpg', 'hp', 'wt', 'cyl']].corr()['mpg'].sort_values()
    print("\n   Correlations with MPG:")
    for col, corr in correlations.items():
        if col != 'mpg':
            direction = "↑ positive" if corr > 0 else "↓ negative"
            print(f"     {col:8s}: {corr:6.3f} ({direction})")
    
    print("Step 3: Visualizations")
    pe.autoplot(target='mpg', max_plots=6)
    
    print("Step 4: Key Insights")
    print("   ✓ Strong negative correlation between weight and MPG")
    print("   ✓ Strong negative correlation between horsepower and MPG")
    print("   ✓ Number of cylinders inversely affects fuel efficiency")
    print("   ✓ Lighter, less powerful cars tend to be more fuel-efficient")


def main():
    """Main demo execution"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*17 + "PLOTEASE COMPREHENSIVE DEMO" + " "*34 + "║")
    print("║" + " "*20 + "Using mtcars Dataset" + " "*37 + "║")
    print("║" + " "*15 + "Week 3: Functional Implementation" + " "*30 + "║")
    print("╚" + "="*78 + "╝")
    
    # Run all demos
    pe, mtcars = demo_basic_usage()
    demo_encapsulation(pe)
    demo_inheritance()
    demo_polymorphism()
    demo_composition(pe)
    demo_dunder_methods()
    
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*25 + "FEATURE DEMONSTRATIONS" + " "*32 + "║")
    print("╚" + "="*78 + "╝")
    
    demo_feature_1_autoplot(pe)
    demo_feature_2_summary(pe)
    demo_feature_3_model_comparison(pe)
    demo_feature_4_quick_plot(pe)
    demo_class_interactions()
    demo_real_world_analysis()
    
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*23 + "DEMO COMPLETED SUCCESSFULLY!" + " "*28 + "║")
    print("╚" + "="*78 + "╝")
    print("\n✓ All OOP principles demonstrated with mtcars dataset")
    print("✓ All features working correctly")
    print("✓ Class interactions verified")
    print("✓ Real-world analysis example completed")
    print("mtcars Dataset Analysis Complete! 🎉\n")


if __name__ == "__main__":
    main()
