"""
Services Package
=================
Core business logic services for the Fresh Flow Markets Inventory Management System.

Modules:
- data_loader: Robust data loading with validation
- forecaster: Demand forecasting (Holt-Winters + fallbacks)
- context_adjustments: Event and weather-based demand adjustments
- inventory_health: Health scoring and risk assessment
- recommendation_engine: Actionable recommendations with explanations
- output_generator: Unified output generation
"""

from services.data_loader import DataLoader
from services.forecaster import DemandForecaster
from services.context_adjustments import ContextAdjuster
from services.inventory_health import InventoryHealthScorer
from services.recommendation_engine import RecommendationEngine
from services.output_generator import OutputGenerator

__all__ = [
    'DataLoader',
    'DemandForecaster',
    'ContextAdjuster',
    'InventoryHealthScorer',
    'RecommendationEngine',
    'OutputGenerator'
]
