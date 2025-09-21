"""
ThemeClouds Package

A package for extracting themes from documents using LLMs and creating 
visualizations of theme frequency.
"""

from .core import ThemeExtractor
from .theme_analyzer import ThemeAnalyzer
from .visualizer import WordCloudGenerator

__version__ = "0.1.0"
__all__ = ["ThemeClouds"]
