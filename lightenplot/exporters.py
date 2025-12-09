"""Export utilities for saving plots."""

import os
from pathlib import Path


class PlotExporter:
    """
    Handle plot exports in various formats.
    Demonstrates composition - uses figure object.
    """
    
    SUPPORTED_FORMATS = ['png', 'jpg', 'jpeg', 'pdf', 'svg', 'eps']
    
    def __init__(self, figure):
        """
        Initialize exporter with figure.
        
        Args:
            figure: Matplotlib figure object
        """
        self._figure = figure
        self._export_count = 0
    
    def save(self, filename, dpi=300, format=None, **kwargs):
        """
        Save figure to file.
        
        Args:
            filename: Output filename
            dpi: Resolution in dots per inch
            format: File format (auto-detected from filename if None)
            **kwargs: Additional savefig parameters
        """
        # Auto-detect format from filename
        if format is None:
            format = Path(filename).suffix.lstrip('.')
        
        if format.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {format}. Use: {self.SUPPORTED_FORMATS}")
        
        # Create directory if needed
        output_dir = Path(filename).parent
        if output_dir and not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save figure
        self._figure.savefig(filename, dpi=dpi, format=format, 
                            bbox_inches='tight', **kwargs)
        self._export_count += 1
        
        print(f"✓ Plot saved: {filename} ({format.upper()}, {dpi} DPI)")
    
    def save_multiple(self, base_filename, formats=['png', 'pdf'], dpi=300):
        """
        Save plot in multiple formats.
        
        Args:
            base_filename: Base filename without extension
            formats: List of formats to save
            dpi: Resolution
        """
        base = Path(base_filename).stem
        directory = Path(base_filename).parent
        
        for fmt in formats:
            filename = directory / f"{base}.{fmt}"
            self.save(str(filename), dpi=dpi, format=fmt)
    
    @property
    def export_count(self):
        """Get number of exports performed."""
        return self._export_count
    
    def __repr__(self):
        return f"PlotExporter(exports={self._export_count})"
    
    def __eq__(self, other):
        """Compare exporters by export count."""
        if not isinstance(other, PlotExporter):
            return False
        return self._export_count == other._export_count