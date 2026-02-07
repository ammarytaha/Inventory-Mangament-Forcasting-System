"""
Context Adjustments Service
============================
Apply event-based and weather-aware adjustments to demand forecasts.

Design Principles:
- All adjustments are multiplicative (easy to explain and trace)
- Every adjustment is logged and traceable
- Rules are configurable and business-driven
- No external APIs - uses mock/input data

Why Context Adjustments Matter:
- Raw forecasts don't account for known future events
- A holiday can spike demand by 50%+
- Weather dramatically affects food/beverage choices
- Manual adjustments lose traceability

How It Works:
1. Take a base forecast
2. Apply event multipliers (holidays, promotions)
3. Apply weather multipliers (temperature, conditions)
4. Return adjusted forecast with full audit trail
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, date

from utils.logger import get_logger
from utils.constants import EVENT_IMPACT_FACTORS, WEATHER_IMPACT_FACTORS

logger = get_logger(__name__)


@dataclass
class Event:
    """
    Represents a business event that affects demand.
    
    Attributes
    ----------
    date : date
        Date of the event
    event_type : str
        Type of event (holiday, promotion, local_event, etc.)
    name : str
        Name of the event for display
    impact_factor : float, optional
        Override impact factor. If None, uses default from config.
    affected_categories : List[str], optional
        Product categories affected. If None, affects all.
    """
    date: date
    event_type: str
    name: str
    impact_factor: Optional[float] = None
    affected_categories: Optional[List[str]] = None
    
    def get_impact_factor(self) -> float:
        """Get the impact factor, using default if not specified."""
        if self.impact_factor is not None:
            return self.impact_factor
        return EVENT_IMPACT_FACTORS.get(self.event_type, 1.0)


@dataclass
class WeatherCondition:
    """
    Represents weather conditions for a date.
    
    Attributes
    ----------
    date : date
        Date of the weather condition
    weather_type : str
        Type of weather (sunny, rainy, cloudy, etc.)
    temperature : float
        Temperature in Celsius
    rainfall_mm : float, optional
        Rainfall in millimeters
    """
    date: date
    weather_type: str
    temperature: float
    rainfall_mm: float = 0.0


@dataclass
class AdjustmentRecord:
    """
    Record of an adjustment applied to a forecast.
    
    This provides full traceability for every adjustment.
    """
    item_id: Any
    date: date
    original_forecast: float
    adjusted_forecast: float
    adjustment_type: str  # 'event', 'weather', 'day_of_week'
    adjustment_factor: float
    reason: str


class ContextAdjuster:
    """
    Applies context-based adjustments to demand forecasts.
    
    This service takes base forecasts and adjusts them based on:
    1. Known events (holidays, promotions)
    2. Weather conditions
    3. Day of week patterns
    
    All adjustments are:
    - Multiplicative (factor * base_forecast)
    - Logged for traceability
    - Configurable via constants
    
    Usage
    -----
    >>> adjuster = ContextAdjuster()
    >>> adjuster.add_event(Event(date(2026, 2, 14), "holiday", "Valentine's Day"))
    >>> adjusted = adjuster.apply_adjustments(forecast_df)
    
    Why Multiplicative Adjustments?
    - Easy to explain: "Holiday increases demand by 50%"
    - Composable: Multiple factors can be combined
    - Stable: Works with any scale of demand
    """
    
    def __init__(
        self,
        apply_day_of_week: bool = True,
        apply_weather: bool = True,
        apply_events: bool = True
    ):
        """
        Initialize the context adjuster.
        
        Parameters
        ----------
        apply_day_of_week : bool
            Whether to apply day-of-week adjustments
        apply_weather : bool
            Whether to apply weather adjustments
        apply_events : bool
            Whether to apply event adjustments
        """
        self.apply_day_of_week = apply_day_of_week
        self.apply_weather = apply_weather
        self.apply_events = apply_events
        
        # Storage for events and weather
        self.events: List[Event] = []
        self.weather_conditions: List[WeatherCondition] = []
        
        # Audit trail
        self.adjustment_records: List[AdjustmentRecord] = []
        
        # Category mappings for weather-sensitive products
        self.weather_sensitive_items: Dict[Any, str] = {}
        
        logger.info(
            f"ContextAdjuster initialized: day_of_week={apply_day_of_week}, "
            f"weather={apply_weather}, events={apply_events}"
        )
    
    def add_event(self, event: Event) -> None:
        """
        Add an event that affects demand.
        
        Parameters
        ----------
        event : Event
            The event to add
        """
        self.events.append(event)
        logger.info(f"Added event: {event.name} on {event.date}")
    
    def add_events_from_dataframe(self, events_df: pd.DataFrame) -> None:
        """
        Add events from a DataFrame.
        
        Expected columns:
        - date: Date of the event
        - event_type: Type (holiday, promotion, etc.)
        - name: Event name
        - impact_factor: Optional override factor
        """
        for _, row in events_df.iterrows():
            event = Event(
                date=pd.to_datetime(row['date']).date(),
                event_type=row['event_type'],
                name=row.get('name', row['event_type']),
                impact_factor=row.get('impact_factor', None)
            )
            self.add_event(event)
    
    def add_weather(self, weather: WeatherCondition) -> None:
        """
        Add weather conditions for a date.
        
        Parameters
        ----------
        weather : WeatherCondition
            The weather condition to add
        """
        self.weather_conditions.append(weather)
    
    def add_weather_from_dataframe(self, weather_df: pd.DataFrame) -> None:
        """
        Add weather from a DataFrame.
        
        Expected columns:
        - date: Date
        - weather_type: Type (sunny, rainy, etc.)
        - temperature: Temperature in Celsius
        - rainfall_mm: Optional rainfall
        """
        for _, row in weather_df.iterrows():
            weather = WeatherCondition(
                date=pd.to_datetime(row['date']).date(),
                weather_type=row['weather_type'],
                temperature=row['temperature'],
                rainfall_mm=row.get('rainfall_mm', 0.0)
            )
            self.add_weather(weather)
    
    def set_weather_sensitive_items(
        self,
        item_categories: Dict[Any, str]
    ) -> None:
        """
        Map items to weather-sensitive categories.
        
        Parameters
        ----------
        item_categories : Dict[Any, str]
            Mapping of item_id to category
            (e.g., {123: 'beverages_cold', 456: 'soup'})
        """
        self.weather_sensitive_items = item_categories
    
    def apply_adjustments(
        self,
        forecast_df: pd.DataFrame,
        item_column: str = 'item_id',
        date_column: str = 'date',
        forecast_column: str = 'forecast'
    ) -> pd.DataFrame:
        """
        Apply all enabled adjustments to a forecast DataFrame.
        
        Parameters
        ----------
        forecast_df : pd.DataFrame
            Base forecast with item, date, and forecast columns
        item_column : str
            Name of item identifier column
        date_column : str
            Name of date column
        forecast_column : str
            Name of forecast value column
        
        Returns
        -------
        pd.DataFrame
            Adjusted forecast with new columns:
            - adjusted_forecast: The adjusted value
            - adjustment_factor: Combined adjustment factor
            - adjustment_reasons: List of reasons for adjustments
        """
        df = forecast_df.copy()
        
        # Initialize adjustment columns
        df['adjustment_factor'] = 1.0
        df['adjustment_reasons'] = ''
        
        # Clear previous records
        self.adjustment_records = []
        
        # Apply adjustments in order
        if self.apply_day_of_week:
            df = self._apply_day_of_week_adjustments(df, date_column)
        
        if self.apply_events and len(self.events) > 0:
            df = self._apply_event_adjustments(df, item_column, date_column)
        
        if self.apply_weather and len(self.weather_conditions) > 0:
            df = self._apply_weather_adjustments(df, item_column, date_column)
        
        # Calculate adjusted forecast
        df['adjusted_forecast'] = df[forecast_column] * df['adjustment_factor']
        
        # Ensure non-negative
        df['adjusted_forecast'] = df['adjusted_forecast'].clip(lower=0)
        
        # Round to reasonable precision
        df['adjusted_forecast'] = df['adjusted_forecast'].round(2)
        
        # Log summary
        total_adjustments = (df['adjustment_factor'] != 1.0).sum()
        logger.info(
            f"Applied adjustments to {total_adjustments} of {len(df)} forecast rows"
        )
        
        return df
    
    def _apply_day_of_week_adjustments(
        self,
        df: pd.DataFrame,
        date_column: str
    ) -> pd.DataFrame:
        """
        Apply day-of-week patterns to forecast.
        
        Business Logic:
        - Restaurants typically see patterns: slower Mon-Tue, busier Fri-Sat
        - These factors are configurable in constants.py
        """
        df = df.copy()
        
        # Get day of week (1=Monday, 7=Sunday)
        df['_dow'] = pd.to_datetime(df[date_column]).dt.dayofweek + 1
        
        # Apply factors
        dow_factors = EVENT_IMPACT_FACTORS.get('day_of_week', {})
        
        for dow, factor in dow_factors.items():
            mask = df['_dow'] == dow
            if mask.any():
                df.loc[mask, 'adjustment_factor'] *= factor
                
                day_name = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][dow-1]
                reason = f"Day-of-week ({day_name}): {factor:.2f}x"
                df.loc[mask, 'adjustment_reasons'] += reason + '; '
        
        df = df.drop(columns=['_dow'])
        
        return df
    
    def _apply_event_adjustments(
        self,
        df: pd.DataFrame,
        item_column: str,
        date_column: str
    ) -> pd.DataFrame:
        """
        Apply event-based adjustments.
        
        Events can:
        - Affect all products (e.g., holiday)
        - Affect specific categories (e.g., beverage promotion)
        """
        df = df.copy()
        
        for event in self.events:
            # Find matching dates
            mask = pd.to_datetime(df[date_column]).dt.date == event.date
            
            if not mask.any():
                continue
            
            # Apply category filter if specified
            if event.affected_categories:
                # Only adjust items in specified categories
                # (Requires weather_sensitive_items mapping)
                category_mask = df[item_column].apply(
                    lambda x: self.weather_sensitive_items.get(x) in event.affected_categories
                )
                mask = mask & category_mask
            
            if mask.any():
                factor = event.get_impact_factor()
                df.loc[mask, 'adjustment_factor'] *= factor
                
                reason = f"Event ({event.name}): {factor:.2f}x"
                df.loc[mask, 'adjustment_reasons'] += reason + '; '
                
                logger.info(
                    f"Applied event '{event.name}' to {mask.sum()} rows "
                    f"(factor: {factor})"
                )
        
        return df
    
    def _apply_weather_adjustments(
        self,
        df: pd.DataFrame,
        item_column: str,
        date_column: str
    ) -> pd.DataFrame:
        """
        Apply weather-based adjustments.
        
        Weather Logic:
        1. Overall weather impact (rainy = lower demand)
        2. Temperature-based category adjustments
           - Hot day: cold beverages up, hot beverages down
           - Cold day: soup up, salad down
        """
        df = df.copy()
        
        temp_config = WEATHER_IMPACT_FACTORS.get('temperature_adjustments', {})
        
        for weather in self.weather_conditions:
            # Find matching dates
            mask = pd.to_datetime(df[date_column]).dt.date == weather.date
            
            if not mask.any():
                continue
            
            # 1. Apply general weather factor
            weather_factor = WEATHER_IMPACT_FACTORS.get(weather.weather_type, 1.0)
            
            if weather_factor != 1.0:
                df.loc[mask, 'adjustment_factor'] *= weather_factor
                reason = f"Weather ({weather.weather_type}): {weather_factor:.2f}x"
                df.loc[mask, 'adjustment_reasons'] += reason + '; '
            
            # 2. Apply temperature-based category adjustments
            self._apply_temperature_adjustments(
                df, mask, weather, item_column, temp_config
            )
        
        return df
    
    def _apply_temperature_adjustments(
        self,
        df: pd.DataFrame,
        date_mask: pd.Series,
        weather: WeatherCondition,
        item_column: str,
        temp_config: Dict[str, Any]
    ) -> None:
        """
        Apply temperature-specific adjustments to categories.
        """
        hot_threshold = temp_config.get('hot_threshold', 30)
        cold_threshold = temp_config.get('cold_threshold', 10)
        
        is_hot = weather.temperature >= hot_threshold
        is_cold = weather.temperature <= cold_threshold
        
        if not (is_hot or is_cold):
            return
        
        # Define category adjustments
        adjustments = []
        
        if is_hot:
            adjustments.extend([
                ('beverages_cold', temp_config.get('cold_beverage_hot_weather_boost', 1.4)),
                ('ice_cream', temp_config.get('cold_beverage_hot_weather_boost', 1.4)),
                ('salad', temp_config.get('salad_hot_weather_boost', 1.2)),
                ('beverages_hot', 0.7),  # Hot beverages decrease in hot weather
                ('soup', 0.6)
            ])
        
        if is_cold:
            adjustments.extend([
                ('beverages_hot', temp_config.get('hot_beverage_cold_weather_boost', 1.3)),
                ('soup', temp_config.get('soup_cold_weather_boost', 1.35)),
                ('beverages_cold', 0.7),
                ('ice_cream', 0.5),
                ('salad', 0.8)
            ])
        
        for category, factor in adjustments:
            # Find items in this category
            category_mask = df[item_column].apply(
                lambda x: self.weather_sensitive_items.get(x) == category
            )
            combined_mask = date_mask & category_mask
            
            if combined_mask.any():
                df.loc[combined_mask, 'adjustment_factor'] *= factor
                
                temp_desc = "hot" if is_hot else "cold"
                reason = f"Temperature ({temp_desc}, {category}): {factor:.2f}x"
                df.loc[combined_mask, 'adjustment_reasons'] += reason + '; '
    
    def get_adjustment_summary(self) -> pd.DataFrame:
        """
        Get a summary of all adjustments applied.
        
        Returns
        -------
        pd.DataFrame
            Summary of adjustments by type
        """
        if len(self.adjustment_records) == 0:
            return pd.DataFrame()
        
        records_df = pd.DataFrame([
            {
                'item_id': r.item_id,
                'date': r.date,
                'original': r.original_forecast,
                'adjusted': r.adjusted_forecast,
                'type': r.adjustment_type,
                'factor': r.adjustment_factor,
                'reason': r.reason
            }
            for r in self.adjustment_records
        ])
        
        return records_df


def create_sample_events(start_date: date, num_days: int = 30) -> List[Event]:
    """
    Create sample events for demonstration.
    
    This function generates realistic events for hackathon demos.
    In production, events would come from a calendar system.
    """
    events = []
    
    # Weekly promotion (every Friday)
    current = start_date
    end_date = start_date + pd.Timedelta(days=num_days)
    
    while current < end_date:
        # Friday = weekday 4
        if current.weekday() == 4:
            events.append(Event(
                date=current,
                event_type='promotion',
                name='Friday Happy Hour',
                impact_factor=1.25
            ))
        
        # Weekend effect
        if current.weekday() in [5, 6]:
            events.append(Event(
                date=current,
                event_type='local_event',
                name='Weekend',
                impact_factor=1.15
            ))
        
        current += pd.Timedelta(days=1)
    
    return events


def create_sample_weather(start_date: date, num_days: int = 30) -> List[WeatherCondition]:
    """
    Create sample weather data for demonstration.
    
    Generates realistic weather patterns.
    In production, this would come from a weather API.
    """
    np.random.seed(42)  # Reproducible for demo
    
    conditions = []
    weather_types = ['sunny', 'cloudy', 'rainy', 'cloudy', 'sunny']  # Weighted towards good weather
    
    current = start_date
    
    for _ in range(num_days):
        # Seasonal temperature (assuming winter in northern hemisphere)
        base_temp = 5 + 10 * np.sin(current.timetuple().tm_yday / 365 * 2 * np.pi)
        temp = base_temp + np.random.normal(0, 5)
        
        weather_type = np.random.choice(weather_types)
        rainfall = np.random.uniform(0, 10) if weather_type == 'rainy' else 0
        
        conditions.append(WeatherCondition(
            date=current,
            weather_type=weather_type,
            temperature=round(temp, 1),
            rainfall_mm=round(rainfall, 1)
        ))
        
        current += pd.Timedelta(days=1)
    
    return conditions
