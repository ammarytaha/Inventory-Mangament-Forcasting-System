"""
System-Wide Constants and Configurations
==========================================
Centralized location for all configuration values, thresholds, and schemas.

Design Principles:
- All magic numbers should be defined here
- Schemas define expected columns and types
- Thresholds are configurable for different business contexts
- Comments explain WHY values were chosen
"""

from typing import Dict, List, Any

# =============================================================================
# DATA SCHEMAS
# =============================================================================
# These schemas define the expected structure of input files.
# Required columns will trigger validation errors if missing.
# Optional columns are validated if present but won't fail if absent.

SALES_SCHEMA = {
    "name": "sales_data",
    "description": "Order and order item data for demand analysis",
    "files": ["fct_orders.csv", "fct_order_items.csv"],
    "required_columns": {
        "fct_orders.csv": ["id", "created"],
        "fct_order_items.csv": ["id", "order_id", "item_id", "quantity"]
    },
    "optional_columns": {
        "fct_orders.csv": ["place_id", "status", "total", "payment_method"],
        "fct_order_items.csv": ["price", "cost", "discount"]
    },
    "timestamp_columns": ["created", "updated", "completed_at"],
    "numeric_columns": ["quantity", "price", "cost", "total", "discount"]
}

INVENTORY_SCHEMA = {
    "name": "inventory_data",
    "description": "Stock levels and SKU information",
    "files": ["dim_skus.csv", "fct_inventory_reports.csv"],
    "required_columns": {
        "dim_skus.csv": ["id"],
        "fct_inventory_reports.csv": []  # May be empty in some datasets
    },
    "optional_columns": {
        "dim_skus.csv": ["title", "quantity", "unit", "low_stock_threshold", "place_id"],
        "fct_inventory_reports.csv": ["sku_id", "quantity", "created"]
    },
    "timestamp_columns": ["created", "updated", "expires_at"],
    "numeric_columns": ["quantity", "low_stock_threshold"]
}

EXPIRY_SCHEMA = {
    "name": "expiry_data",
    "description": "Expiration and shelf-life related data",
    "files": ["dim_skus.csv"],
    "required_columns": {
        "dim_skus.csv": ["id"]
    },
    "optional_columns": {
        "dim_skus.csv": ["expires_at", "shelf_life_days", "is_perishable"]
    },
    "timestamp_columns": ["expires_at", "production_date"],
    "numeric_columns": ["shelf_life_days"]
}

MENU_SCHEMA = {
    "name": "menu_data",
    "description": "Menu items and bill of materials",
    "files": ["dim_menu_items.csv", "dim_bill_of_materials.csv"],
    "required_columns": {
        "dim_menu_items.csv": ["id", "title"],
        "dim_bill_of_materials.csv": ["parent_sku_id", "child_sku_id"]
    },
    "optional_columns": {
        "dim_menu_items.csv": ["price", "status", "category_id"],
        "dim_bill_of_materials.csv": ["quantity", "unit"]
    },
    "timestamp_columns": ["created", "updated"],
    "numeric_columns": ["price", "quantity"]
}

# =============================================================================
# DATA QUALITY THRESHOLDS
# =============================================================================
# These thresholds determine when to flag data quality issues.
# Adjust based on business tolerance for data issues.

DATA_QUALITY_THRESHOLDS = {
    # Missing values: percentage threshold for warnings
    "missing_warning_pct": 0.05,     # 5% missing = warning
    "missing_critical_pct": 0.20,    # 20% missing = critical
    
    # Duplicates: count threshold for warnings
    "duplicate_warning_count": 10,
    "duplicate_critical_count": 100,
    
    # Date ranges: reasonable bounds for timestamp validation
    "min_valid_date": "2020-01-01",  # Data before this is suspicious
    "max_valid_date": "2027-12-31",  # Data after this is suspicious
    
    # Numeric bounds: for outlier detection
    "quantity_min": 0,
    "quantity_max": 100000,
    "price_min": 0,
    "price_max": 100000
}

# =============================================================================
# FORECASTING CONFIGURATION
# =============================================================================
# Parameters for the forecasting module.

FORECAST_CONFIG = {
    # Minimum data points for Holt-Winters
    # Assumption: At least 14 days of data needed for reliable patterns
    "min_data_points_holt_winters": 14,
    
    # Fallback to moving average if less data
    "min_data_points_moving_avg": 3,
    
    # Default forecast horizon (days)
    "default_forecast_horizon": 7,
    
    # Maximum forecast horizon (beyond this, uncertainty is too high)
    "max_forecast_horizon": 30,
    
    # Seasonal period (7 = weekly seasonality for restaurants)
    "seasonal_period": 7,
    
    # Confidence level for prediction intervals
    "confidence_level": 0.95,
    
    # Smoothing parameters (can be tuned)
    "holt_winters_defaults": {
        "trend": "add",           # Additive trend
        "seasonal": "add",        # Additive seasonality
        "damped_trend": True,     # Dampen trend for stability
        "seasonal_periods": 7     # Weekly pattern
    }
}

# =============================================================================
# INVENTORY HEALTH SCORING
# =============================================================================
# Weights and thresholds for inventory health calculations.

HEALTH_SCORE_CONFIG = {
    # Component weights (must sum to 1.0)
    "weights": {
        "stock_level": 0.30,       # Current stock vs demand
        "days_to_expiry": 0.35,    # Expiration risk
        "demand_trend": 0.20,      # Is demand increasing/decreasing?
        "turnover_rate": 0.15      # How fast is inventory moving?
    },
    
    # Risk level thresholds (health score ranges)
    "risk_thresholds": {
        "low": 70,      # Score >= 70 = LOW risk
        "medium": 40,   # Score >= 40 = MEDIUM risk
        # Score < 40 = HIGH risk
    },
    
    # Days to expiry thresholds
    "expiry_thresholds": {
        "critical": 3,    # <= 3 days = critical
        "warning": 7,     # <= 7 days = warning
        "acceptable": 14  # <= 14 days = monitor
    }
}

# =============================================================================
# CONTEXT ADJUSTMENT FACTORS
# =============================================================================
# Multipliers for event-based and weather-based demand adjustments.

EVENT_IMPACT_FACTORS = {
    # Event types and their demand multipliers
    "holiday": 1.50,           # 50% increase on holidays
    "promotion": 1.30,         # 30% increase during promotions
    "local_event": 1.20,       # 20% increase for local events
    "slow_day": 0.80,          # 20% decrease on slow days
    "closure": 0.00,           # No demand if closed
    
    # Day of week adjustments (1 = Monday, 7 = Sunday)
    "day_of_week": {
        1: 0.85,   # Monday - typically slower
        2: 0.90,   # Tuesday
        3: 0.95,   # Wednesday
        4: 1.00,   # Thursday - baseline
        5: 1.15,   # Friday - busier
        6: 1.25,   # Saturday - peak
        7: 1.10    # Sunday
    }
}

WEATHER_IMPACT_FACTORS = {
    # Weather types and impact on demand
    "sunny": 1.10,             # Slightly higher demand
    "cloudy": 1.00,            # Baseline
    "rainy": 0.85,             # Lower demand
    "stormy": 0.70,            # Significantly lower
    "snow": 0.60,              # Much lower
    "extreme_heat": 0.90,      # Slight decrease
    "extreme_cold": 0.85,      # Slight decrease
    
    # Temperature ranges (Celsius) and multipliers
    "temperature_adjustments": {
        "hot_threshold": 30,    # Above this = hot beverages drop
        "cold_threshold": 10,   # Below this = cold beverages drop
        "hot_beverage_cold_weather_boost": 1.30,
        "cold_beverage_hot_weather_boost": 1.40,
        "salad_hot_weather_boost": 1.20,
        "soup_cold_weather_boost": 1.35
    },
    
    # Product categories affected by weather
    "weather_sensitive_categories": [
        "beverages_cold",
        "beverages_hot", 
        "ice_cream",
        "soup",
        "salad"
    ]
}

# =============================================================================
# RECOMMENDATION THRESHOLDS
# =============================================================================
# Thresholds for generating different types of recommendations.

RECOMMENDATION_CONFIG = {
    # Safety stock multiplier (buffer above expected demand)
    "safety_stock_multiplier": 1.20,  # 20% buffer
    
    # Reorder point calculation
    "lead_time_days": 2,              # Days to receive order
    
    # Overstock thresholds
    "overstock_days_of_supply": 14,   # >14 days supply = overstock
    
    # Understock thresholds  
    "understock_days_of_supply": 2,   # <2 days supply = understock
    
    # Waste reduction thresholds
    "waste_risk_expiry_days": 5,      # <5 days to expiry with high stock
    "discount_trigger_pct": 0.60,     # 60% of shelf life remaining
    "bundle_trigger_pct": 0.40,       # 40% of shelf life remaining
    
    # Minimum order quantities
    "min_order_quantity": 1,
    
    # Promotion thresholds
    "slow_mover_days": 14,            # No sales in 14 days = slow mover
    "promotion_discount_pct": 0.20    # Suggest 20% discount for slow movers
}

# =============================================================================
# OUTPUT CONFIGURATION
# =============================================================================
# Settings for output file generation.

OUTPUT_CONFIG = {
    # Output directories
    "output_base_dir": "outputs",
    "analytics_subdir": "analytics",
    "decisions_subdir": "decisions",
    "cleaned_subdir": "cleaned_data",
    "forecasts_subdir": "forecasts",
    
    # File naming
    "file_prefix": {
        "analytics": "analytics_",
        "decisions": "decision_",
        "cleaned": "cleaned_",
        "forecast": "forecast_"
    },
    
    # CSV export settings
    "csv_encoding": "utf-8",
    "csv_index": False,
    "float_format": "%.4f"
}
