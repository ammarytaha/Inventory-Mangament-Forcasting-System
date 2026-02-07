"""
FreshFlow AI - Intelligent Inventory Decision Engine
=====================================================

AI-powered inventory management system for Fresh Flow Markets.
Provides location-specific recommendations, demand forecasting,
and actionable insights for reducing waste and preventing stockouts.

Modules:
- config: Configuration management
- data_processor: Data loading and transformation
- forecaster: Multi-model demand forecasting
- recommendation_engine: AI-driven recommendations
- context_engine: External factors (holidays, events, weather)
- explanation_generator: Human-readable explanations

Usage:
    from freshflow_ai import FreshFlowEngine
    
    engine = FreshFlowEngine(data_path='path/to/data')
    recommendations = engine.get_recommendations(place_id=94025)
"""

__version__ = "1.0.0"
__author__ = "FreshFlow AI Team"

from .config import Config
from .data_processor import DataProcessor
from .forecaster import ForecastEngine
from .recommendation_engine import RecommendationEngine
from .context_engine import ContextEngine
from .explanation_generator import ExplanationGenerator

__all__ = [
    'Config',
    'DataProcessor', 
    'ForecastEngine',
    'RecommendationEngine',
    'ContextEngine',
    'ExplanationGenerator'
]
