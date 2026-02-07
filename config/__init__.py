# FreshFlow Configuration Package

"""
Configuration module for FreshFlow Inventory Management System.
Import settings from this module to access configuration values.
"""

from .settings import (
    BASE_DIR,
    DATA_DIR,
    OUTPUT_DIR,
    SRC_DIR,
    DATA_CONFIG,
    FORECAST_CONFIG,
    DECISION_CONFIG,
    DASHBOARD_CONFIG,
    LOGGING_CONFIG,
)

__all__ = [
    "BASE_DIR",
    "DATA_DIR",
    "OUTPUT_DIR",
    "SRC_DIR",
    "DATA_CONFIG",
    "FORECAST_CONFIG",
    "DECISION_CONFIG",
    "DASHBOARD_CONFIG",
    "LOGGING_CONFIG",
]
