"""
FreshFlow AI - Configuration Module
====================================

Centralized configuration for the inventory decision engine.
Supports environment-based settings and data path configuration.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime


@dataclass
class ModelConfig:
    """Configuration for forecasting models"""
    smooth_models: List[str] = field(default_factory=lambda: ['prophet', 'ets'])
    erratic_models: List[str] = field(default_factory=lambda: ['lightgbm', 'xgboost'])
    intermittent_models: List[str] = field(default_factory=lambda: ['croston', 'sba'])
    lumpy_models: List[str] = field(default_factory=lambda: ['isbts', 'lightgbm'])
    fallback_model: str = 'moving_average'
    
    # Forecasting horizons
    forecast_horizon_weeks: int = 4
    backtest_periods: int = 4
    
    # Model parameters
    prophet_seasonality_mode: str = 'multiplicative'
    croston_alpha: float = 0.1


@dataclass
class InventoryConfig:
    """Configuration for inventory rules"""
    # Safety stock parameters by demand type
    safety_stock_multipliers: Dict[str, float] = field(default_factory=lambda: {
        'Smooth': 1.5,
        'Erratic': 2.0,
        'Intermittent': 2.5,
        'Lumpy': 3.0,
        'Insufficient Data': 2.0
    })
    
    # Reorder point parameters (days of coverage)
    reorder_coverage_days: int = 7
    lead_time_days: int = 2
    
    # Expiry thresholds
    critical_expiry_days: int = 2
    warning_expiry_days: int = 5
    
    # Stock level thresholds
    overstocked_threshold: float = 1.5  # 150% of optimal
    understocked_threshold: float = 0.5  # 50% of optimal


@dataclass
class BusinessConfig:
    """Business rules and thresholds"""
    # Risk levels
    risk_thresholds: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'critical': {'threshold': 0.9, 'color': '#d32f2f'},
        'high': {'threshold': 0.7, 'color': '#f57c00'},
        'medium': {'threshold': 0.4, 'color': '#fbc02d'},
        'low': {'threshold': 0.0, 'color': '#388e3c'}
    })
    
    # Promotion recommendations
    discount_tiers: Dict[str, float] = field(default_factory=lambda: {
        'light': 0.10,    # 10% off
        'moderate': 0.20,  # 20% off
        'aggressive': 0.35, # 35% off
        'clearance': 0.50   # 50% off
    })
    
    # Confidence thresholds
    high_confidence: float = 0.8
    medium_confidence: float = 0.6


@dataclass
class Config:
    """
    Master configuration for FreshFlow AI
    
    Usage:
        config = Config(data_path='path/to/data')
        config.model.forecast_horizon_weeks = 8
    """
    
    # Paths
    data_path: Path = field(default_factory=lambda: Path.cwd() / 'data')
    output_path: Path = field(default_factory=lambda: Path.cwd() / 'outputs')
    analysis_path: Path = field(default_factory=lambda: Path.cwd() / 'docs' / 'data_analysis' / 'data')
    
    # Component configs
    model: ModelConfig = field(default_factory=ModelConfig)
    inventory: InventoryConfig = field(default_factory=InventoryConfig)
    business: BusinessConfig = field(default_factory=BusinessConfig)
    
    # Data settings
    date_column: str = 'created'
    timestamp_unit: str = 's'  # Unix epoch in seconds
    
    # Calendar settings
    week_start: str = 'Monday'
    timezone: str = 'UTC'
    country_code: str = 'DK'  # For holiday calculations
    
    # Danish holidays for 2024/2025/2026
    holidays: Dict[str, List[dict]] = field(default_factory=lambda: {
        '2024': [
            {'date': '2024-01-01', 'name': 'New Year', 'impact': 0.2},
            {'date': '2024-03-28', 'name': 'Maundy Thursday', 'impact': 0.7},
            {'date': '2024-03-29', 'name': 'Good Friday', 'impact': 0.5},
            {'date': '2024-03-31', 'name': 'Easter Sunday', 'impact': 0.3},
            {'date': '2024-04-01', 'name': 'Easter Monday', 'impact': 0.5},
            {'date': '2024-05-09', 'name': 'Ascension Day', 'impact': 0.7},
            {'date': '2024-05-19', 'name': 'Whit Sunday', 'impact': 0.7},
            {'date': '2024-05-20', 'name': 'Whit Monday', 'impact': 0.6},
            {'date': '2024-06-05', 'name': 'Constitution Day', 'impact': 0.8},
            {'date': '2024-12-24', 'name': 'Christmas Eve', 'impact': 0.2},
            {'date': '2024-12-25', 'name': 'Christmas Day', 'impact': 0.1},
            {'date': '2024-12-26', 'name': 'Boxing Day', 'impact': 0.4},
            {'date': '2024-12-31', 'name': 'New Year Eve', 'impact': 0.3},
        ],
        '2025': [
            {'date': '2025-01-01', 'name': 'New Year', 'impact': 0.2},
            {'date': '2025-04-17', 'name': 'Maundy Thursday', 'impact': 0.7},
            {'date': '2025-04-18', 'name': 'Good Friday', 'impact': 0.5},
            {'date': '2025-04-20', 'name': 'Easter Sunday', 'impact': 0.3},
            {'date': '2025-04-21', 'name': 'Easter Monday', 'impact': 0.5},
            {'date': '2025-05-29', 'name': 'Ascension Day', 'impact': 0.7},
            {'date': '2025-06-08', 'name': 'Whit Sunday', 'impact': 0.7},
            {'date': '2025-06-09', 'name': 'Whit Monday', 'impact': 0.6},
            {'date': '2025-06-05', 'name': 'Constitution Day', 'impact': 0.8},
            {'date': '2025-12-24', 'name': 'Christmas Eve', 'impact': 0.2},
            {'date': '2025-12-25', 'name': 'Christmas Day', 'impact': 0.1},
            {'date': '2025-12-26', 'name': 'Boxing Day', 'impact': 0.4},
            {'date': '2025-12-31', 'name': 'New Year Eve', 'impact': 0.3},
        ],
        '2026': [
            {'date': '2026-01-01', 'name': 'New Year', 'impact': 0.2},
            {'date': '2026-04-02', 'name': 'Maundy Thursday', 'impact': 0.7},
            {'date': '2026-04-03', 'name': 'Good Friday', 'impact': 0.5},
            {'date': '2026-04-05', 'name': 'Easter Sunday', 'impact': 0.3},
            {'date': '2026-04-06', 'name': 'Easter Monday', 'impact': 0.5},
            {'date': '2026-05-14', 'name': 'Ascension Day', 'impact': 0.7},
            {'date': '2026-05-24', 'name': 'Whit Sunday', 'impact': 0.7},
            {'date': '2026-05-25', 'name': 'Whit Monday', 'impact': 0.6},
            {'date': '2026-06-05', 'name': 'Constitution Day', 'impact': 0.8},
            {'date': '2026-12-24', 'name': 'Christmas Eve', 'impact': 0.2},
            {'date': '2026-12-25', 'name': 'Christmas Day', 'impact': 0.1},
            {'date': '2026-12-26', 'name': 'Boxing Day', 'impact': 0.4},
            {'date': '2026-12-31', 'name': 'New Year Eve', 'impact': 0.3},
        ]
    })
    
    # Weekly patterns (multiplicative factors)
    weekly_seasonality: Dict[int, float] = field(default_factory=lambda: {
        0: 0.89,   # Monday (trough)
        1: 1.02,   # Tuesday
        2: 1.14,   # Wednesday
        3: 1.22,   # Thursday
        4: 1.39,   # Friday (peak)
        5: 1.32,   # Saturday
        6: 0.82    # Sunday (trough)
    })
    
    def __post_init__(self):
        """Convert string paths to Path objects and create directories"""
        if isinstance(self.data_path, str):
            self.data_path = Path(self.data_path)
        if isinstance(self.output_path, str):
            self.output_path = Path(self.output_path)
        if isinstance(self.analysis_path, str):
            self.analysis_path = Path(self.analysis_path)
            
        # Create output directories if they don't exist
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_workspace(cls, workspace_path: str) -> 'Config':
        """
        Create config from workspace root path.
        
        Args:
            workspace_path: Root path of the Inventory Management workspace
            
        Returns:
            Configured Config instance
        """
        workspace = Path(workspace_path)
        return cls(
            data_path=workspace / 'data',
            output_path=workspace / 'outputs',
            analysis_path=workspace / 'docs' / 'data_analysis' / 'data'
        )
    
    def get_holiday_impact(self, date: datetime) -> tuple:
        """
        Get holiday impact for a specific date.
        
        Returns:
            Tuple of (is_holiday, holiday_name, impact_factor)
        """
        year = str(date.year)
        date_str = date.strftime('%Y-%m-%d')
        
        if year in self.holidays:
            for holiday in self.holidays[year]:
                if holiday['date'] == date_str:
                    return True, holiday['name'], holiday['impact']
        
        return False, None, 1.0
    
    def get_weekly_factor(self, day_of_week: int) -> float:
        """Get the weekly seasonality factor for a day of week (0=Monday)"""
        return self.weekly_seasonality.get(day_of_week, 1.0)


# Default configuration instance
DEFAULT_CONFIG = Config()
