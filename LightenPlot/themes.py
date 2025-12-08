"""Theme management for LightenPlot."""

import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class Theme(ABC):
    """Abstract base class for themes."""
    
    @abstractmethod
    def apply(self, ax, fig):
        """Apply theme to axes and figure."""
        pass


class DefaultTheme(Theme):
    """Default clean theme."""
    
    def apply(self, ax, fig):
        """Apply default theme settings."""
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.tick_params(labelsize=10)
        fig.patch.set_facecolor('white')


class DarkTheme(Theme):
    """Dark mode theme."""
    
    def apply(self, ax, fig):
        """Apply dark theme settings."""
        fig.patch.set_facecolor('#1e1e1e')
        ax.set_facecolor('#2d2d2d')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.tick_params(colors='white', labelsize=10)
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')


class MinimalTheme(Theme):
    """Minimal theme with no spines."""
    
    def apply(self, ax, fig):
        """Apply minimal theme settings."""
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(left=False, bottom=False)
        ax.grid(True, alpha=0.2, linestyle='--')
        fig.patch.set_facecolor('white')


class ColorfulTheme(Theme):
    """Vibrant, colorful theme."""
    
    def apply(self, ax, fig):
        """Apply colorful theme settings."""
        fig.patch.set_facecolor('#f0f8ff')
        ax.set_facecolor('#ffffff')
        ax.spines['top'].set_color('#ff6b6b')
        ax.spines['right'].set_color('#4ecdc4')
        ax.spines['left'].set_color('#45b7d1')
        ax.spines['bottom'].set_color('#f9ca24')
        ax.spines['top'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.tick_params(labelsize=10)


class ThemeManager:
    """
    Singleton-like manager for themes.
    Demonstrates polymorphism through different theme implementations.
    """
    
    _themes = {
        'default': DefaultTheme(),
        'dark': DarkTheme(),
        'minimal': MinimalTheme(),
        'colorful': ColorfulTheme()
    }
    
    @classmethod
    def get_theme(cls, name):
        """
        Get theme by name.
        
        Args:
            name: Theme name ('default', 'dark', 'minimal', 'colorful')
            
        Returns:
            Theme instance
            
        Raises:
            ValueError: If theme name not found
        """
        if name not in cls._themes:
            raise ValueError(f"Theme '{name}' not found. Available: {list(cls._themes.keys())}")
        return cls._themes[name]
    
    @classmethod
    def register_theme(cls, name, theme):
        """
        Register a custom theme.
        
        Args:
            name: Theme name
            theme: Theme instance (must inherit from Theme)
        """
        if not isinstance(theme, Theme):
            raise TypeError("Theme must inherit from Theme base class")
        cls._themes[name] = theme
    
    @classmethod
    def list_themes(cls):
        """Return list of available theme names."""
        return list(cls._themes.keys())
    
    def __repr__(self):
        return f"ThemeManager(themes={self.list_themes()})"