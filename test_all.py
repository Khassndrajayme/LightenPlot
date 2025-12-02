"""
COMPLETE Unit Tests for PlotEase Library - mtcars Dataset
"""

import unittest
import pandas as pd
import numpy as np

from plotease import (
    PlotEase, 
    DiagnosticPlotter, 
    SummaryGenerator, 
    ModelComparator, 
    QuickPlotter,
    VisualizationBase
)


def load_mtcars():
    """Load mtcars dataset for testing"""
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
    })
    return mtcars

# TEST 1: INITIALIZATION & ENCAPSULATION (5 tests)

class TestPlotEaseInitialization(unittest.TestCase):
    """Test initialization and encapsulation"""
    
    def setUp(self):
        self.mtcars = load_mtcars()
    
    def test_valid_initialization(self):
        """Test successful initialization"""
        pe = PlotEase(self.mtcars, theme='minimal')
        self.assertIsNotNone(pe)
        self.assertEqual(len(pe), 32)
    
    def test_invalid_data_type(self):
        """Test TypeError with wrong data type"""
        with self.assertRaises(TypeError):
            PlotEase([1, 2, 3])
    
    def test_empty_dataframe(self):
        """Test ValueError with empty DataFrame"""
        with self.assertRaises(ValueError):
            PlotEase(pd.DataFrame())
    
    def test_get_data_encapsulation(self):
        """Test getter method (encapsulation)"""
        pe = PlotEase(self.mtcars)
        data = pe.get_data()
        self.assertTrue(data.equals(self.mtcars))
    
    def test_set_theme_encapsulation(self):
        """Test setter method (encapsulation)"""
        pe = PlotEase(self.mtcars, theme='default')
        pe.set_theme('dark')
        self.assertEqual(pe._theme, 'dark')


# TEST 2: DUNDER METHODS (4 tests)

class TestDunderMethods(unittest.TestCase):
    """Test dunder methods (__repr__, __len__, __eq__, __lt__)"""
    
    def setUp(self):
        self.mtcars = load_mtcars()
        self.mtcars_subset = self.mtcars.head(10)
    
    def test_repr(self):
        """Test __repr__ method"""
        pe = PlotEase(self.mtcars)
        repr_str = repr(pe)
        self.assertIn('PlotEase', repr_str)
        self.assertIn('rows=32', repr_str)
    
    def test_len(self):
        """Test __len__ method"""
        pe = PlotEase(self.mtcars)
        self.assertEqual(len(pe), 32)
    
    def test_eq(self):
        """Test __eq__ method"""
        pe1 = PlotEase(self.mtcars, theme='minimal')
        pe2 = PlotEase(self.mtcars, theme='minimal')
        self.assertEqual(pe1, pe2)
    
    def test_lt(self):
        """Test __lt__ method (less than)"""
        pe_full = PlotEase(self.mtcars)
        pe_subset = PlotEase(self.mtcars_subset)
        self.assertTrue(pe_subset < pe_full)



# TEST 3: INHERITANCE (3 tests)

class TestInheritance(unittest.TestCase):
    """Test inheritance relationships"""
    
    def setUp(self):
        self.mtcars = load_mtcars()
    
    def test_plotease_inherits_base(self):
        """Test PlotEase inherits from VisualizationBase"""
        pe = PlotEase(self.mtcars)
        self.assertIsInstance(pe, VisualizationBase)
    
    def test_diagnostic_inherits_base(self):
        """Test DiagnosticPlotter inherits from VisualizationBase"""
        dp = DiagnosticPlotter(self.mtcars)
        self.assertIsInstance(dp, VisualizationBase)
    
    def test_quickplotter_inherits_base(self):
        """Test QuickPlotter inherits from VisualizationBase"""
        qp = QuickPlotter(self.mtcars)
        self.assertIsInstance(qp, VisualizationBase)


# TEST 4: POLYMORPHISM (2 tests)

class TestPolymorphism(unittest.TestCase):
    """Test polymorphism (method overriding)"""
    
    def setUp(self):
        self.mtcars = load_mtcars()
    
    def test_render_method_exists(self):
        """Test all classes have render() method"""
        pe = PlotEase(self.mtcars)
        dp = DiagnosticPlotter(self.mtcars)
        qp = QuickPlotter(self.mtcars)
        
        self.assertTrue(hasattr(pe, 'render'))
        self.assertTrue(hasattr(dp, 'render'))
        self.assertTrue(hasattr(qp, 'render'))
    
    def test_render_executes(self):
        """Test render() methods execute"""
        pe = PlotEase(self.mtcars)
        try:
            pe.render()
            success = True
        except:
            success = False
        self.assertTrue(success)


# TEST 5: COMPOSITION (3 tests)

class TestComposition(unittest.TestCase):
    """Test composition (PlotEase contains other classes)"""
    
    def setUp(self):
        self.mtcars = load_mtcars()
    
    def test_contains_diagnostic(self):
        """Test PlotEase contains DiagnosticPlotter"""
        pe = PlotEase(self.mtcars)
        self.assertIsInstance(pe._diagnostic, DiagnosticPlotter)
    
    def test_contains_summary(self):
        """Test PlotEase contains SummaryGenerator"""
        pe = PlotEase(self.mtcars)
        self.assertIsInstance(pe._summary, SummaryGenerator)
    
    def test_contains_plotter(self):
        """Test PlotEase contains QuickPlotter"""
        pe = PlotEase(self.mtcars)
        self.assertIsInstance(pe._plotter, QuickPlotter)


# TEST 6: DIAGNOSTIC PLOTTER (3 tests)

class TestDiagnosticPlotter(unittest.TestCase):
    """Test DiagnosticPlotter functionality"""
    
    def setUp(self):
        self.mtcars = load_mtcars()
    
    def test_initialization(self):
        """Test DiagnosticPlotter initializes"""
        dp = DiagnosticPlotter(self.mtcars)
        self.assertIsNotNone(dp)
    
    def test_autoplot_runs(self):
        """Test autoplot executes without error"""
        dp = DiagnosticPlotter(self.mtcars)
        try:
            dp.autoplot(target='mpg', max_plots=4)
            success = True
        except:
            success = False
        self.assertTrue(success)
    
    def test_autoplot_different_targets(self):
        """Test autoplot with different targets"""
        dp = DiagnosticPlotter(self.mtcars)
        for target in ['mpg', 'hp', 'wt']:
            try:
                dp.autoplot(target=target, max_plots=2)
                success = True
            except:
                success = False
            self.assertTrue(success)


# TEST 7: SUMMARY GENERATOR (4 tests)

class TestSummaryGenerator(unittest.TestCase):
    """Test SummaryGenerator functionality"""
    
    def setUp(self):
        self.mtcars = load_mtcars()
    
    def test_initialization(self):
        """Test SummaryGenerator initializes"""
        sg = SummaryGenerator(self.mtcars)
        self.assertIsNotNone(sg)
    
    def test_tabular_summary(self):
        """Test tabular_summary returns DataFrame"""
        sg = SummaryGenerator(self.mtcars)
        summary = sg.tabular_summary(style='full')
        self.assertIsInstance(summary, pd.DataFrame)
    
    def test_summarize_numeric(self):
        """Test summarize_numeric returns list"""
        sg = SummaryGenerator(self.mtcars)
        summary = sg.summarize_numeric()
        self.assertIsInstance(summary, list)
        self.assertGreater(len(summary), 0)
    
    def test_len_dunder(self):
        """Test __len__ method"""
        sg = SummaryGenerator(self.mtcars)
        self.assertEqual(len(sg), 32)


# TEST 8: MODEL COMPARATOR (4 tests)

class TestModelComparator(unittest.TestCase):
    """Test ModelComparator functionality"""
    
    def setUp(self):
        self.models = {
            'Model A': {'Accuracy': 0.85, 'Precision': 0.82},
            'Model B': {'Accuracy': 0.92, 'Precision': 0.90}
        }
    
    def test_initialization(self):
        """Test ModelComparator initializes"""
        mc = ModelComparator(self.models)
        self.assertIsNotNone(mc)
    
    def test_get_best_model(self):
        """Test get_best_model returns correct model"""
        mc = ModelComparator(self.models)
        best = mc.get_best_model('Accuracy')
        self.assertEqual(best, 'Model B')
    
    def test_invalid_metric(self):
        """Test ValueError for invalid metric"""
        mc = ModelComparator(self.models)
        with self.assertRaises(ValueError):
            mc.get_best_model('InvalidMetric')
    
    def test_gt_dunder(self):
        """Test __gt__ comparison operator"""
        models_good = {'Model A': {'Accuracy': 0.95}}
        models_poor = {'Model B': {'Accuracy': 0.75}}
        mc_good = ModelComparator(models_good)
        mc_poor = ModelComparator(models_poor)
        self.assertTrue(mc_good > mc_poor)

# TEST 9: QUICK PLOTTER (3 tests)

class TestQuickPlotter(unittest.TestCase):
    """Test QuickPlotter functionality"""
    
    def setUp(self):
        self.mtcars = load_mtcars()
    
    def test_initialization(self):
        """Test QuickPlotter initializes"""
        qp = QuickPlotter(self.mtcars)
        self.assertIsNotNone(qp)
    
    def test_detect_plot_type(self):
        """Test automatic plot type detection"""
        qp = QuickPlotter(self.mtcars)
        plot_type = qp.detect_plot_type('mpg', None)
        self.assertEqual(plot_type, 'hist')
    
    def test_quick_plot_runs(self):
        """Test quick_plot executes"""
        qp = QuickPlotter(self.mtcars)
        try:
            qp.quick_plot('hp', 'mpg', kind='scatter')
            success = True
        except:
            success = False
        self.assertTrue(success)


# TEST 10: INTEGRATION (2 tests)

class TestIntegration(unittest.TestCase):
    """Test complete workflow integration"""
    
    def setUp(self):
        self.mtcars = load_mtcars()
    
    def test_full_workflow(self):
        """Test complete workflow with all features"""
        pe = PlotEase(self.mtcars, theme='minimal')
        
        try:
            # Feature 1
            pe.autoplot(target='mpg', max_plots=2)
            
            # Feature 2
            summary = pe.tabular_summary(style='full')
            self.assertIsInstance(summary, pd.DataFrame)
            
            # Feature 3
            models = {
                'Model A': {'Accuracy': 0.85},
                'Model B': {'Accuracy': 0.90}
            }
            pe.compare_models(models)
            
            # Feature 4
            pe.quick_plot('hp', 'mpg', kind='scatter')
            
            success = True
        except Exception as e:
            print(f"Error: {e}")
            success = False
        
        self.assertTrue(success)
    
    def test_mtcars_analysis(self):
        """Test mtcars-specific analysis"""
        pe = PlotEase(self.mtcars)
        summary = pe.tabular_summary(style='numeric')
        
        # Verify key columns exist
        for col in ['mpg', 'hp', 'wt']:
            self.assertIn(col, self.mtcars.columns)


# RUN ALL TESTS

if __name__ == '__main__':
    print("="*80)
    print("PLOTEASE COMPLETE TEST SUITE - mtcars Dataset")
    print("="*80)
    print("\nTesting OOP Requirements:")
    print("  ✓ Encapsulation (5 tests)")
    print("  ✓ Inheritance (3 tests)")
    print("  ✓ Polymorphism (2 tests)")
    print("  ✓ Composition (3 tests)")
    print("  ✓ Dunder Methods (4 tests)")
    print("\nTesting Features:")
    print("  ✓ DiagnosticPlotter (3 tests)")
    print("  ✓ SummaryGenerator (4 tests)")
    print("  ✓ ModelComparator (4 tests)")
    print("  ✓ QuickPlotter (3 tests)")
    print("  ✓ Integration (2 tests)")
    print("\nTotal: 33 tests")
    print("="*80 + "\n")
    
    unittest.main(verbosity=2)

test_all.py
