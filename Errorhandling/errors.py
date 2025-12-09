"""
Comprehensive error handling system for LightenPlot library.
Provides custom exceptions, decorators, and validation utilities.
"""

import functools
import pandas as pd
import numpy as np
from typing import Any, Callable, List, Optional


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class LightenPlotError(Exception):
    """Base exception class for all LightenPlot errors."""
    
    def __init__(self, message: str, suggestion: Optional[str] = None):
        """
        Initialize error with message and optional suggestion.
        
        Args:
            message: Error description
            suggestion: Helpful suggestion to fix the error
        """
        self.message = message
        self.suggestion = suggestion
        super().__init__(self.format_message())
    
    def format_message(self) -> str:
        """Format error message with suggestion if available."""
        msg = f"LightenPlot Error: {self.message}"
        if self.suggestion:
            msg += f"\nSuggestion: {self.suggestion}"
        return msg


class DataValidationError(LightenPlotError):
    """Raised when input data fails validation."""
    pass


class ColumnNotFoundError(LightenPlotError):
    """Raised when specified column doesn't exist in DataFrame."""
    pass


class InvalidThemeError(LightenPlotError):
    """Raised when an invalid theme is specified."""
    pass


class PlotCreationError(LightenPlotError):
    """Raised when plot creation fails."""
    pass


class ExportError(LightenPlotError):
    """Raised when plot export fails."""
    pass


class InvalidParameterError(LightenPlotError):
    """Raised when invalid parameters are provided."""
    pass


# ============================================================================
# DATA VALIDATOR CLASS
# ============================================================================

class DataValidator:
    """
    Static utility class for data validation.
    Provides methods to validate common data requirements.
    """
    
    @staticmethod
    def validate_dataframe(data: Any, name: str = "Data") -> None:
        """
        Validate that data is a non-empty DataFrame.
        
        Args:
            data: Data to validate
            name: Name for error messages
            
        Raises:
            DataValidationError: If validation fails
        """
        if not isinstance(data, pd.DataFrame):
            raise DataValidationError(
                f"{name} must be a pandas DataFrame, got {type(data).__name__}",
                "Convert your data to DataFrame using pd.DataFrame(data)"
            )
        
        if data.empty:
            raise DataValidationError(
                f"{name} is empty",
                "Ensure your DataFrame contains data before plotting"
            )
    
    @staticmethod
    def validate_columns_exist(data: pd.DataFrame, columns: List[str]) -> None:
        """
        Validate that all specified columns exist in DataFrame.
        
        Args:
            data: DataFrame to check
            columns: List of column names to validate
            
        Raises:
            ColumnNotFoundError: If any column is missing
        """
        if not isinstance(data, pd.DataFrame):
            return  # Skip if not a DataFrame
        
        missing_cols = [col for col in columns if col not in data.columns]
        
        if missing_cols:
            available = list(data.columns)
            raise ColumnNotFoundError(
                f"Column(s) not found: {missing_cols}",
                f"Available columns: {available}"
            )
    
    @staticmethod
    def validate_numeric(data: pd.Series, name: str = "Data") -> None:
        """
        Validate that data is numeric.
        
        Args:
            data: Series to validate
            name: Name for error messages
            
        Raises:
            DataValidationError: If data is not numeric
        """
        if not pd.api.types.is_numeric_dtype(data):
            raise DataValidationError(
                f"{name} must be numeric, got {data.dtype}",
                "Convert to numeric using pd.to_numeric() or use appropriate data"
            )
    
    @staticmethod
    def validate_range(value: float, min_val: float, max_val: float, 
                      name: str = "Value") -> None:
        """
        Validate that value is within specified range.
        
        Args:
            value: Value to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            name: Name for error messages
            
        Raises:
            InvalidParameterError: If value is out of range
        """
        if not (min_val <= value <= max_val):
            raise InvalidParameterError(
                f"{name} must be between {min_val} and {max_val}, got {value}",
                f"Use a value within the valid range [{min_val}, {max_val}]"
            )
    
    @staticmethod
    def validate_not_empty(data: Any, name: str = "Data") -> None:
        """
        Validate that data is not empty.
        
        Args:
            data: Data to validate
            name: Name for error messages
            
        Raises:
            DataValidationError: If data is empty
        """
        try:
            if len(data) == 0:
                raise DataValidationError(
                    f"{name} is empty",
                    "Provide non-empty data for plotting"
                )
        except TypeError:
            # Data doesn't support len(), assume it's valid
            pass
    
    @staticmethod
    def validate_positive(value: float, name: str = "Value") -> None:
        """
        Validate that value is positive.
        
        Args:
            value: Value to validate
            name: Name for error messages
            
        Raises:
            InvalidParameterError: If value is not positive
        """
        if value <= 0:
            raise InvalidParameterError(
                f"{name} must be positive, got {value}",
                "Use a positive value greater than 0"
            )


# ============================================================================
# DECORATOR FOR ERROR HANDLING
# ============================================================================

def handle_plot_errors(func: Callable) -> Callable:
    """
    Decorator to handle common plotting errors gracefully.
    
    Catches exceptions and provides user-friendly error messages.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function with error handling
        
    Example:
        @handle_plot_errors
        def create_plot(self, x, y):
            # plotting code
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        
        except LightenPlotError:
            # Re-raise our custom errors as-is
            raise
        
        except KeyError as e:
            raise ColumnNotFoundError(
                f"Column {e} not found in data",
                "Check your column names and ensure they exist in the DataFrame"
            ) from e
        
        except ValueError as e:
            raise DataValidationError(
                f"Invalid data: {str(e)}",
                "Verify your data format and values"
            ) from e
        
        except TypeError as e:
            raise InvalidParameterError(
                f"Invalid parameter type: {str(e)}",
                "Check parameter types match expected values"
            ) from e
        
        except Exception as e:
            raise PlotCreationError(
                f"Failed to create plot: {str(e)}",
                "Check your data and parameters, or report this as a bug"
            ) from e
    
    return wrapper


# ============================================================================
# DECORATOR FOR INPUT VALIDATION
# ============================================================================

def validate_inputs(**validators) -> Callable:
    """
    Decorator to validate function inputs before execution.
    
    Args:
        **validators: Keyword arguments mapping parameter names to 
                     validation functions
    
    Returns:
        Decorator function
        
    Example:
        @validate_inputs(
            x=lambda col: isinstance(col, str),
            alpha=lambda val: 0.0 <= val <= 1.0
        )
        def create_plot(self, x, y, alpha=0.7):
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            
            # Validate each parameter
            for param_name, validator_func in validators.items():
                if param_name in bound.arguments:
                    value = bound.arguments[param_name]
                    try:
                        if not validator_func(value):
                            raise InvalidParameterError(
                                f"Validation failed for parameter '{param_name}' with value {value}",
                                f"Check the requirements for '{param_name}'"
                            )
                    except Exception as e:
                        if isinstance(e, LightenPlotError):
                            raise
                        raise InvalidParameterError(
                            f"Validation error for '{param_name}': {str(e)}",
                            f"Ensure '{param_name}' meets the required constraints"
                        ) from e
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


# ============================================================================
# WARNING SYSTEM
# ============================================================================

class PlotWarning:
    """Class for handling non-critical warnings."""
    
    _warnings_enabled = True
    
    @classmethod
    def warn(cls, message: str, category: str = "General") -> None:
        """
        Issue a warning message.
        
        Args:
            message: Warning message
            category: Warning category
        """
        if cls._warnings_enabled:
            print(f"LightenPlot {category} Warning: {message}")
    
    @classmethod
    def enable_warnings(cls) -> None:
        """Enable warning messages."""
        cls._warnings_enabled = True
    
    @classmethod
    def disable_warnings(cls) -> None:
        """Disable warning messages."""
        cls._warnings_enabled = False


# ============================================================================
# CONTEXT MANAGER FOR ERROR HANDLING
# ============================================================================

class safe_plot:
    """
    Context manager for safe plotting operations.
    
    Example:
        with safe_plot("Creating scatter plot"):
            plot.create(x='age', y='salary')
    """
    
    def __init__(self, operation: str = "Plot operation"):
        """
        Initialize context manager.
        
        Args:
            operation: Description of the operation
        """
        self.operation = operation
    
    def __enter__(self):
        """Enter context."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and handle exceptions."""
        if exc_type is not None:
            if issubclass(exc_type, LightenPlotError):
                # Our custom errors are already well-formatted
                return False
            
            # Wrap other exceptions
            print(f"✘ Error during {self.operation}")
            print(f"   {exc_type.__name__}: {exc_val}")
            print("Check your data and parameters")
            return True  # Suppress the exception
        
        return False# lightenplot/errors2.py
"""
Comprehensive error handling system for LightenPlot library.
Provides custom exceptions, decorators, and validation utilities.
"""

import functools
import pandas as pd
import numpy as np
from typing import Any, Callable, List, Optional


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class LightenPlotError(Exception):
    """Base exception class for all LightenPlot errors."""
    
    def __init__(self, message: str, suggestion: Optional[str] = None):
        """
        Initialize error with message and optional suggestion.
        
        Args:
            message: Error description
            suggestion: Helpful suggestion to fix the error
        """
        self.message = message
        self.suggestion = suggestion
        super().__init__(self.format_message())
    
    def format_message(self) -> str:
        """Format error message with suggestion if available."""
        msg = f"LightenPlot Error: {self.message}"
        if self.suggestion:
            msg += f"\nSuggestion: {self.suggestion}"
        return msg


class DataValidationError(LightenPlotError):
    """Raised when input data fails validation."""
    pass


class ColumnNotFoundError(LightenPlotError):
    """Raised when specified column doesn't exist in DataFrame."""
    pass


class InvalidThemeError(LightenPlotError):
    """Raised when an invalid theme is specified."""
    pass


class PlotCreationError(LightenPlotError):
    """Raised when plot creation fails."""
    pass


class ExportError(LightenPlotError):
    """Raised when plot export fails."""
    pass


class InvalidParameterError(LightenPlotError):
    """Raised when invalid parameters are provided."""
    pass


# ============================================================================
# DATA VALIDATOR CLASS
# ============================================================================

class DataValidator:
    """
    Static utility class for data validation.
    Provides methods to validate common data requirements.
    """
    
    @staticmethod
    def validate_dataframe(data: Any, name: str = "Data") -> None:
        """
        Validate that data is a non-empty DataFrame.
        
        Args:
            data: Data to validate
            name: Name for error messages
            
        Raises:
            DataValidationError: If validation fails
        """
        if not isinstance(data, pd.DataFrame):
            raise DataValidationError(
                f"{name} must be a pandas DataFrame, got {type(data).__name__}",
                "Convert your data to DataFrame using pd.DataFrame(data)"
            )
        
        if data.empty:
            raise DataValidationError(
                f"{name} is empty",
                "Ensure your DataFrame contains data before plotting"
            )
    
    @staticmethod
    def validate_columns_exist(data: pd.DataFrame, columns: List[str]) -> None:
        """
        Validate that all specified columns exist in DataFrame.
        
        Args:
            data: DataFrame to check
            columns: List of column names to validate
            
        Raises:
            ColumnNotFoundError: If any column is missing
        """
        if not isinstance(data, pd.DataFrame):
            return  # Skip if not a DataFrame
        
        missing_cols = [col for col in columns if col not in data.columns]
        
        if missing_cols:
            available = list(data.columns)
            raise ColumnNotFoundError(
                f"Column(s) not found: {missing_cols}",
                f"Available columns: {available}"
            )
    
    @staticmethod
    def validate_numeric(data: pd.Series, name: str = "Data") -> None:
        """
        Validate that data is numeric.
        
        Args:
            data: Series to validate
            name: Name for error messages
            
        Raises:
            DataValidationError: If data is not numeric
        """
        if not pd.api.types.is_numeric_dtype(data):
            raise DataValidationError(
                f"{name} must be numeric, got {data.dtype}",
                "Convert to numeric using pd.to_numeric() or use appropriate data"
            )
    
    @staticmethod
    def validate_range(value: float, min_val: float, max_val: float, 
                      name: str = "Value") -> None:
        """
        Validate that value is within specified range.
        
        Args:
            value: Value to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            name: Name for error messages
            
        Raises:
            InvalidParameterError: If value is out of range
        """
        if not (min_val <= value <= max_val):
            raise InvalidParameterError(
                f"{name} must be between {min_val} and {max_val}, got {value}",
                f"Use a value within the valid range [{min_val}, {max_val}]"
            )
    
    @staticmethod
    def validate_not_empty(data: Any, name: str = "Data") -> None:
        """
        Validate that data is not empty.
        
        Args:
            data: Data to validate
            name: Name for error messages
            
        Raises:
            DataValidationError: If data is empty
        """
        try:
            if len(data) == 0:
                raise DataValidationError(
                    f"{name} is empty",
                    "Provide non-empty data for plotting"
                )
        except TypeError:
            # Data doesn't support len(), assume it's valid
            pass
    
    @staticmethod
    def validate_positive(value: float, name: str = "Value") -> None:
        """
        Validate that value is positive.
        
        Args:
            value: Value to validate
            name: Name for error messages
            
        Raises:
            InvalidParameterError: If value is not positive
        """
        if value <= 0:
            raise InvalidParameterError(
                f"{name} must be positive, got {value}",
                "Use a positive value greater than 0"
            )


# ============================================================================
# DECORATOR FOR ERROR HANDLING
# ============================================================================

def handle_plot_errors(func: Callable) -> Callable:
    """
    Decorator to handle common plotting errors gracefully.
    
    Catches exceptions and provides user-friendly error messages.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function with error handling
        
    Example:
        @handle_plot_errors
        def create_plot(self, x, y):
            # plotting code
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        
        except LightenPlotError:
            # Re-raise our custom errors as-is
            raise
        
        except KeyError as e:
            raise ColumnNotFoundError(
                f"Column {e} not found in data",
                "Check your column names and ensure they exist in the DataFrame"
            ) from e
        
        except ValueError as e:
            raise DataValidationError(
                f"Invalid data: {str(e)}",
                "Verify your data format and values"
            ) from e
        
        except TypeError as e:
            raise InvalidParameterError(
                f"Invalid parameter type: {str(e)}",
                "Check parameter types match expected values"
            ) from e
        
        except Exception as e:
            raise PlotCreationError(
                f"Failed to create plot: {str(e)}",
                "Check your data and parameters, or report this as a bug"
            ) from e
    
    return wrapper


# ============================================================================
# DECORATOR FOR INPUT VALIDATION
# ============================================================================

def validate_inputs(**validators) -> Callable:
    """
    Decorator to validate function inputs before execution.
    
    Args:
        **validators: Keyword arguments mapping parameter names to 
                     validation functions
    
    Returns:
        Decorator function
        
    Example:
        @validate_inputs(
            x=lambda col: isinstance(col, str),
            alpha=lambda val: 0.0 <= val <= 1.0
        )
        def create_plot(self, x, y, alpha=0.7):
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            
            # Validate each parameter
            for param_name, validator_func in validators.items():
                if param_name in bound.arguments:
                    value = bound.arguments[param_name]
                    try:
                        if not validator_func(value):
                            raise InvalidParameterError(
                                f"Validation failed for parameter '{param_name}' with value {value}",
                                f"Check the requirements for '{param_name}'"
                            )
                    except Exception as e:
                        if isinstance(e, LightenPlotError):
                            raise
                        raise InvalidParameterError(
                            f"Validation error for '{param_name}': {str(e)}",
                            f"Ensure '{param_name}' meets the required constraints"
                        ) from e
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


# ============================================================================
# WARNING SYSTEM
# ============================================================================

class PlotWarning:
    """Class for handling non-critical warnings."""
    
    _warnings_enabled = True
    
    @classmethod
    def warn(cls, message: str, category: str = "General") -> None:
        """
        Issue a warning message.
        
        Args:
            message: Warning message
            category: Warning category
        """
        if cls._warnings_enabled:
            print(f"LightenPlot {category} Warning: {message}")
    
    @classmethod
    def enable_warnings(cls) -> None:
        """Enable warning messages."""
        cls._warnings_enabled = True
    
    @classmethod
    def disable_warnings(cls) -> None:
        """Disable warning messages."""
        cls._warnings_enabled = False


# ============================================================================
# CONTEXT MANAGER FOR ERROR HANDLING
# ============================================================================

class safe_plot:
    """
    Context manager for safe plotting operations.
    
    Example:
        with safe_plot("Creating scatter plot"):
            plot.create(x='age', y='salary')
    """
    
    def __init__(self, operation: str = "Plot operation"):
        """
        Initialize context manager.
        
        Args:
            operation: Description of the operation
        """
        self.operation = operation
    
    def __enter__(self):
        """Enter context."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and handle exceptions."""
        if exc_type is not None:
            if issubclass(exc_type, LightenPlotError):
                # Our custom errors are already well-formatted
                return False
            
            # Wrap other exceptions
            print(f"✘ Error during {self.operation}")
            print(f"   {exc_type.__name__}: {exc_val}")
            print("Check your data and parameters")
            return True  # Suppress the exception
        
        return False