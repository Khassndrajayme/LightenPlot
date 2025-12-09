# lightenplot/errors.py
"""
Error handling module for LightenPlot.

Provides custom exceptions, validation utilities, and error handling decorators
to improve user experience and debugging.
"""

import functools
import traceback
from typing import Any, Callable, List, Optional, Union
import pandas as pd
import numpy as np


# ============================================================================
# Custom Exceptions
# ============================================================================

class LightenPlotError(Exception):
    """Base exception for all LightenPlot errors."""
    
    def __init__(self, message: str, suggestion: Optional[str] = None):
        """
        Initialize LightenPlot error.
        
        Args:
            message: Error message
            suggestion: Optional suggestion for fixing the error
        """
        self.message = message
        self.suggestion = suggestion
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        """Format error message with suggestion if available."""
        if self.suggestion:
            return f"{self.message}\n Suggestion: {self.suggestion}"
        return self.message


class DataValidationError(LightenPlotError):
    """Raised when input data fails validation."""
    pass


class ColumnNotFoundError(LightenPlotError):
    """Raised when specified column doesn't exist in DataFrame."""
    
    def __init__(self, column: str, available_columns: List[str]):
        message = f"Column '{column}' not found in data"
        available = ", ".join(f"'{col}'" for col in available_columns[:5])
        if len(available_columns) > 5:
            available += f", ... ({len(available_columns)} total)"
        suggestion = f"Available columns: {available}"
        super().__init__(message, suggestion)


class InvalidThemeError(LightenPlotError):
    """Raised when an invalid theme is requested."""
    
    def __init__(self, theme_name: str, available_themes: List[str]):
        message = f"Theme '{theme_name}' not found"
        suggestion = f"Available themes: {', '.join(available_themes)}"
        super().__init__(message, suggestion)


class PlotCreationError(LightenPlotError):
    """Raised when plot creation fails."""
    pass


class ExportError(LightenPlotError):
    """Raised when plot export fails."""
    pass


class InvalidParameterError(LightenPlotError):
    """Raised when invalid parameter is provided."""
    pass


class EmptyDataError(LightenPlotError):
    """Raised when attempting to plot empty data."""
    
    def __init__(self):
        message = "Cannot create plot with empty data"
        suggestion = "Ensure your data contains at least one row/element"
        super().__init__(message, suggestion)


class IncompatibleDataError(LightenPlotError):
    """Raised when data type is incompatible with plot type."""
    
    def __init__(self, plot_type: str, data_type: str, expected_type: str):
        message = f"{plot_type} requires {expected_type} but got {data_type}"
        suggestion = f"Convert your data to {expected_type} or use a different plot type"
        super().__init__(message, suggestion)


class DimensionMismatchError(LightenPlotError):
    """Raised when data dimensions don't match expectations."""
    
    def __init__(self, expected: str, actual: str):
        message = f"Data dimension mismatch: expected {expected}, got {actual}"
        suggestion = "Check your data shape and ensure it matches the plot requirements"
        super().__init__(message, suggestion)


# ============================================================================
# Validation Functions
# ============================================================================

def validate_data_not_empty(data: Any) -> None:
    """
    Validate that data is not empty.
    
    Args:
        data: Data to validate
        
    Raises:
        EmptyDataError: If data is empty
    """
    if data is None:
        raise EmptyDataError()
    
    if isinstance(data, pd.DataFrame):
        if data.empty:
            raise EmptyDataError()
    elif isinstance(data, (list, np.ndarray)):
        if len(data) == 0:
            raise EmptyDataError()
    elif hasattr(data, '__len__'):
        if len(data) == 0:
            raise EmptyDataError()


def validate_column_exists(data: pd.DataFrame, column: Union[str, List[str]]) -> None:
    """
    Validate that column(s) exist in DataFrame.
    
    Args:
        data: DataFrame to check
        column: Column name or list of column names
        
    Raises:
        DataValidationError: If data is not a DataFrame
        ColumnNotFoundError: If column doesn't exist
    """
    if not isinstance(data, pd.DataFrame):
        raise DataValidationError(
            "Data must be a pandas DataFrame to use column names",
            "Pass data as a DataFrame or use array indices instead"
        )
    
    columns = [column] if isinstance(column, str) else column
    
    for col in columns:
        if col not in data.columns:
            raise ColumnNotFoundError(col, list(data.columns))


def validate_numeric_column(data: pd.DataFrame, column: str) -> None:
    """
    Validate that column contains numeric data.
    
    Args:
        data: DataFrame to check
        column: Column name to validate
        
    Raises:
        DataValidationError: If column is not numeric
    """
    validate_column_exists(data, column)
    
    if not pd.api.types.is_numeric_dtype(data[column]):
        raise DataValidationError(
            f"Column '{column}' must contain numeric data, got {data[column].dtype}",
            "Convert column to numeric type or choose a different column"
        )


def validate_parameter_range(param_name: str, value: float, 
                            min_val: Optional[float] = None,
                            max_val: Optional[float] = None) -> None:
    """
    Validate that parameter is within acceptable range.
    
    Args:
        param_name: Name of parameter
        value: Parameter value
        min_val: Minimum acceptable value
        max_val: Maximum acceptable value
        
    Raises:
        InvalidParameterError: If value is out of range
    """
    if min_val is not None and value < min_val:
        raise InvalidParameterError(
            f"{param_name} must be >= {min_val}, got {value}",
            f"Set {param_name} to a value between {min_val} and {max_val or '∞'}"
        )
    
    if max_val is not None and value > max_val:
        raise InvalidParameterError(
            f"{param_name} must be <= {max_val}, got {value}",
            f"Set {param_name} to a value between {min_val or '-∞'} and {max_val}"
        )


def validate_parameter_type(param_name: str, value: Any, 
                           expected_types: Union[type, tuple]) -> None:
    """
    Validate parameter type.
    
    Args:
        param_name: Name of parameter
        value: Parameter value
        expected_types: Expected type or tuple of types
        
    Raises:
        InvalidParameterError: If type doesn't match
    """
    if not isinstance(value, expected_types):
        if isinstance(expected_types, tuple):
            type_names = " or ".join(t.__name__ for t in expected_types)
        else:
            type_names = expected_types.__name__
        
        raise InvalidParameterError(
            f"{param_name} must be {type_names}, got {type(value).__name__}",
            f"Convert {param_name} to {type_names}"
        )


def validate_data_dimensions(data: Any, expected_dims: int) -> None:
    """
    Validate data dimensions.
    
    Args:
        data: Data to validate
        expected_dims: Expected number of dimensions (1 or 2)
        
    Raises:
        DimensionMismatchError: If dimensions don't match
    """
    if isinstance(data, pd.DataFrame):
        actual_dims = 2 if len(data.shape) == 2 and data.shape[1] > 1 else 1
    elif isinstance(data, np.ndarray):
        actual_dims = len(data.shape)
    else:
        actual_dims = 1
    
    if actual_dims != expected_dims:
        raise DimensionMismatchError(
            f"{expected_dims}D",
            f"{actual_dims}D"
        )


def validate_color_format(color: Any) -> None:
    """
    Validate color parameter format.
    
    Args:
        color: Color specification
        
    Raises:
        InvalidParameterError: If color format is invalid
    """
    if color is None:
        return
    
    if isinstance(color, str):
        # Check if it's a valid matplotlib color name or hex code
        if color.startswith('#') and len(color) not in [4, 7, 9]:
            raise InvalidParameterError(
                f"Invalid hex color: {color}",
                "Use format #RGB, #RRGGBB, or #RRGGBBAA"
            )
    elif not isinstance(color, (list, tuple, np.ndarray)):
        raise InvalidParameterError(
            f"Color must be string, list, or array, got {type(color).__name__}",
            "Use color name (e.g., 'red'), hex code (e.g., '#FF0000'), or RGB tuple"
        )


# ============================================================================
# Decorators for Error Handling
# ============================================================================

def handle_plot_errors(func: Callable) -> Callable:
    """
    Decorator to handle common plotting errors gracefully.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function with error handling
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except LightenPlotError:
            # Re-raise our custom errors as-is
            raise
        except KeyError as e:
            # Handle missing keys/columns
            raise ColumnNotFoundError(
                str(e).strip("'\""),
                []
            ) from e
        except ValueError as e:
            # Handle value errors
            raise DataValidationError(
                str(e),
                "Check your data format and parameter values"
            ) from e
        except TypeError as e:
            # Handle type errors
            raise InvalidParameterError(
                str(e),
                "Check parameter types match expected values"
            ) from e
        except Exception as e:
            # Handle unexpected errors
            raise PlotCreationError(
                f"Unexpected error during plot creation: {str(e)}",
                "This may be a bug. Please report it with your data structure."
            ) from e
    
    return wrapper


def validate_inputs(**validators):
    """
    Decorator to validate function inputs.
    
    Args:
        **validators: Keyword arguments mapping parameter names to validator functions
        
    Returns:
        Decorator function
        
    Example:
        @validate_inputs(
            x=lambda x: validate_column_exists(data, x),
            alpha=lambda a: validate_parameter_range('alpha', a, 0, 1)
        )
        def create(self, x, y, alpha=0.7):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature to map args to kwargs
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Run validators
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    try:
                        validator(bound_args.arguments[param_name])
                    except LightenPlotError:
                        raise
                    except Exception as e:
                        raise DataValidationError(
                            f"Validation failed for parameter '{param_name}': {str(e)}",
                            "Check the parameter value and try again"
                        ) from e
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def safe_export(func: Callable) -> Callable:
    """
    Decorator to handle export errors gracefully.
    
    Args:
        func: Export function to wrap
        
    Returns:
        Wrapped function with error handling
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (IOError, OSError, PermissionError) as e:
            raise ExportError(
                f"Failed to save file: {str(e)}",
                "Check file path, permissions, and available disk space"
            ) from e
        except Exception as e:
            raise ExportError(
                f"Unexpected error during export: {str(e)}",
                "Ensure the file format is supported and the path is valid"
            ) from e
    
    return wrapper


# ============================================================================
# Context Managers
# ============================================================================

class suppress_warnings:
    """
    Context manager to suppress matplotlib/seaborn warnings during plotting.
    
    Example:
        with suppress_warnings():
            plot.create(x='col1', y='col2')
    """
    
    def __enter__(self):
        import warnings
        self._original_filters = warnings.filters[:]
        warnings.filterwarnings('ignore')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import warnings
        warnings.filters = self._original_filters
        return False


class plot_context:
    """
    Context manager for safe plot creation with automatic cleanup.
    
    Example:
        with plot_context() as ctx:
            plot = ScatterPlot(data)
            plot.create(x='x', y='y')
    """
    
    def __init__(self, cleanup_on_error: bool = True):
        """
        Initialize plot context.
        
        Args:
            cleanup_on_error: If True, close figures on error
        """
        self.cleanup_on_error = cleanup_on_error
        self.figures = []
    
    def __enter__(self):
        import matplotlib.pyplot as plt
        self.original_figures = plt.get_fignums()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None and self.cleanup_on_error:
            import matplotlib.pyplot as plt
            # Close any new figures created during the context
            current_figures = plt.get_fignums()
            new_figures = set(current_figures) - set(self.original_figures)
            for fig_num in new_figures:
                plt.close(fig_num)
        return False


# ============================================================================
# Error Reporting Utilities
# ============================================================================

def create_error_report(error: Exception, include_traceback: bool = False) -> str:
    """
    Create a formatted error report.
    
    Args:
        error: Exception to report
        include_traceback: If True, include full traceback
        
    Returns:
        Formatted error report string
    """
    report_lines = [
        "=" * 70,
        "LightenPlot Error Report",
        "=" * 70,
        f"Error Type: {type(error).__name__}",
        f"Message: {str(error)}",
    ]
    
    if isinstance(error, LightenPlotError) and error.suggestion:
        report_lines.extend([
            "",
            f"Suggestion: {error.suggestion}",
        ])
    
    if include_traceback:
        report_lines.extend([
            "",
            "Traceback:",
            "-" * 70,
            "".join(traceback.format_tb(error.__traceback__)),
        ])
    
    report_lines.append("=" * 70)
    
    return "\n".join(report_lines)


def log_error(error: Exception, log_file: str = "lightenplot_errors.log") -> None:
    """
    Log error to file.
    
    Args:
        error: Exception to log
        log_file: Path to log file
    """
    import datetime
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"\n[{timestamp}]\n{create_error_report(error, include_traceback=True)}\n"
    
    try:
        with open(log_file, 'a') as f:
            f.write(log_entry)
    except Exception:
        # Fail silently if logging doesn't work
        pass


# ============================================================================
# Validation Helper Class
# ============================================================================

class DataValidator:
    """
    Utility class for comprehensive data validation.
    
    Example:
        validator = DataValidator(data)
        validator.check_not_empty()
        validator.check_columns(['x', 'y'])
        validator.check_numeric(['x', 'y'])
    """
    
    def __init__(self, data: Any):
        """
        Initialize validator with data.
        
        Args:
            data: Data to validate
        """
        self.data = data
        self.errors = []
    
    def check_not_empty(self) -> 'DataValidator':
        """Check data is not empty."""
        try:
            validate_data_not_empty(self.data)
        except LightenPlotError as e:
            self.errors.append(e)
        return self
    
    def check_columns(self, columns: List[str]) -> 'DataValidator':
        """Check columns exist."""
        if isinstance(self.data, pd.DataFrame):
            for col in columns:
                try:
                    validate_column_exists(self.data, col)
                except LightenPlotError as e:
                    self.errors.append(e)
        return self
    
    def check_numeric(self, columns: List[str]) -> 'DataValidator':
        """Check columns are numeric."""
        if isinstance(self.data, pd.DataFrame):
            for col in columns:
                try:
                    validate_numeric_column(self.data, col)
                except LightenPlotError as e:
                    self.errors.append(e)
        return self
    
    def check_dimensions(self, expected_dims: int) -> 'DataValidator':
        """Check data dimensions."""
        try:
            validate_data_dimensions(self.data, expected_dims)
        except LightenPlotError as e:
            self.errors.append(e)
        return self
    
    def validate(self) -> None:
        """
        Raise first error if any validations failed.
        
        Raises:
            LightenPlotError: First validation error encountered
        """
        if self.errors:
            raise self.errors[0]
    
    def get_errors(self) -> List[LightenPlotError]:
        """Get all validation errors."""
        return self.errors
    
    def is_valid(self) -> bool:
        """Check if all validations passed."""
        return len(self.errors) == 0