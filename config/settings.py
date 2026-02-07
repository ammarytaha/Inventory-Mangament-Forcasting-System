# FreshFlow Configuration
# Centralized configuration settings for the FreshFlow solution

"""
Configuration settings for the FreshFlow Inventory Management System.
Adjust these settings based on your environment and data characteristics.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
SRC_DIR = BASE_DIR / "src"

# Data Configuration
DATA_CONFIG = {
    "fact_tables_prefix": "fct_",
    "dimension_tables_prefix": "dim_",
    "date_columns": ["created_at", "updated_at", "order_date", "expiry_date"],
    "id_columns": ["id", "place_id", "user_id", "order_id", "item_id", "sku_id"],
}

# Forecasting Configuration
FORECAST_CONFIG = {
    "default_horizon_days": 30,
    "confidence_interval": 0.95,
    "seasonality_mode": "multiplicative",
    "changepoint_prior_scale": 0.05,
}

# Decision Engine Configuration
DECISION_CONFIG = {
    "safety_stock_days": 3,
    "reorder_lead_days": 2,
    "expiry_alert_days": 7,
    "overstock_threshold": 1.5,  # 150% of average demand
    "understock_threshold": 0.5,  # 50% of safety stock
}

# Dashboard Configuration
DASHBOARD_CONFIG = {
    "page_title": "FreshFlow Inventory Dashboard",
    "theme": "light",
    "refresh_interval_seconds": 300,
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": OUTPUT_DIR / "freshflow.log",
}
