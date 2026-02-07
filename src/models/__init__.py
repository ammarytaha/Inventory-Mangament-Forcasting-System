"""
Models Package
===============
Data models and structures for the Fresh Flow Markets Inventory Management System.

Modules:
- explanation: Business explanation layer for generating plain English explanations
"""

from models.explanation import (
    ExplanationGenerator,
    BusinessExplanation,
    generate_demand_explanation,
    generate_risk_explanation,
    generate_action_explanation
)

__all__ = [
    'ExplanationGenerator',
    'BusinessExplanation',
    'generate_demand_explanation',
    'generate_risk_explanation',
    'generate_action_explanation'
]
