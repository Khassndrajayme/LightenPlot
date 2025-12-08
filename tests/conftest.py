# tests/conftest.py

import pytest
import matplotlib.pyplot as plt

# The 'autouse=True' ensures this fixture runs for every test without needing to be explicitly called.
# The 'scope="function"' ensures it runs once per test function.
@pytest.fixture(autouse=True, scope="function")
def close_figures_after_test():
    """
    Fixture to automatically close all Matplotlib figures after a test has run.
    This prevents the RuntimeWarning: More than 20 figures have been opened.
    """
    # Yield control back to the test function
    yield
    
    # Teardown phase: This code runs AFTER the test completes
    plt.close('all')