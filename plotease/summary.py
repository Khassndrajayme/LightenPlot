import pandas as pd
import numpy as np
from .visualization import VisualizationBase 
from typing import Optional, List, Dict 

class SummaryGenerator(VisualizationBase):
    """
    Generates tabular data summaries (mean, median, missing, etc.).
    Inherits from VisualizationBase.
    """
    
    def __init__(self, data: pd.DataFrame, theme: str = 'default'): 
        """
        Initializes the SummaryGenerator.
        
        Args:
            data: The pandas DataFrame.
            theme: The visual theme string (passed to the base class).
        """
        super().__init__(data, theme) 

    def render(self, style: str = 'full') -> pd.DataFrame:
        """
        Implements the abstract 'render' method from VisualizationBase.
        For SummaryGenerator, this method returns the comprehensive
        tabular summary.
        """
        # Call the primary method of this class
        return self.tabular_summary(style=style)
    
    def tabular_summary(self, style: str = 'full') -> pd.DataFrame:
        """
        Generates a comprehensive statistical summary of the data.
        """
        df = self._data
        
        # Helper function to get summary stats for numeric columns
        # FIX: Renamed to get_numeric_summary for consistency and removed 'self'
        def get_numeric_summary(data: pd.DataFrame) -> pd.DataFrame:
            # FIX: Used numeric_only=True on describe and skew/kurtosis for safety
            stats = data.describe(include=[np.number], numeric_only=True).T 
            stats['missing'] = data.isnull().sum()
            stats['% missing'] = (stats['missing'] / len(data)) * 100
            stats['skew'] = data.skew(numeric_only=True)
            stats['kurtosis'] = data.kurtosis(numeric_only=True)
            return stats[['count', 'missing', '% missing', 'mean', 'std', 'min', 'max', 'skew', 'kurtosis']]

        # Helper function to get summary stats for categorical columns
        def get_categorical_summary(data: pd.DataFrame) -> pd.DataFrame:
            # FIX: Included common non-numeric types
            data_non_numeric = data.select_dtypes(exclude=[np.number, 'datetime'])
            stats = pd.DataFrame(data_non_numeric.dtypes, columns=['DType'])
            stats['count'] = data_non_numeric.count()
            stats['missing'] = data_non_numeric.isnull().sum()
            stats['% missing'] = (stats['missing'] / len(data)) * 100
            stats['unique'] = data_non_numeric.nunique()
            # Handle empty DataFrame case for mode/top_value
            if not data_non_numeric.empty:
                 stats['top_value'] = data_non_numeric.mode().iloc[0]
                 stats['top_freq'] = data_non_numeric.apply(lambda x: x.value_counts().max() if not x.empty else 0)
            else:
                 stats['top_value'] = np.nan
                 stats['top_freq'] = 0

       
