"""
FreshFlow AI - Context Engine
==============================

Handles external factors that impact demand:
- Holidays and special dates
- Local events (sports, festivals, etc.)
- Weather conditions
- Day-of-week patterns
- Promotional periods

Provides context-aware adjustments for forecasts and recommendations.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging

from .config import Config, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class ContextEvent:
    """Represents an external event that impacts demand"""
    name: str
    start_date: date
    end_date: Optional[date] = None
    event_type: str = "event"  # holiday, event, weather, promotion
    impact_factor: float = 1.0  # Multiplicative impact on demand
    affected_categories: List[str] = field(default_factory=list)
    notes: str = ""
    location_specific: bool = False
    place_ids: List[int] = field(default_factory=list)


class ContextEngine:
    """
    Context engine for external factor analysis.
    
    Provides location-aware context adjustments based on:
    - Calendar (holidays, weekends)
    - Events (local and national)
    - Weather (if integrated)
    - Historical patterns
    
    Usage:
        context = ContextEngine(config)
        factors = context.get_context_factors(place_id, date)
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the ContextEngine.
        
        Args:
            config: Configuration object
        """
        self.config = config or DEFAULT_CONFIG
        self._events: List[ContextEvent] = []
        self._load_default_events()
        
    def _load_default_events(self):
        """Load default events from config"""
        # Load holidays from config
        for year, holidays in self.config.holidays.items():
            for holiday in holidays:
                self._events.append(ContextEvent(
                    name=holiday['name'],
                    start_date=datetime.strptime(holiday['date'], '%Y-%m-%d').date(),
                    event_type='holiday',
                    impact_factor=holiday['impact'],
                    notes=f"Danish national holiday"
                ))
                
        # Add common recurring events
        self._add_recurring_events()
        
    def _add_recurring_events(self):
        """Add common recurring events"""
        current_year = datetime.now().year
        
        # Black Friday (last Friday of November)
        for year in range(current_year-1, current_year+2):
            # Find last Friday of November
            nov_30 = date(year, 11, 30)
            days_until_friday = (4 - nov_30.weekday()) % 7
            if days_until_friday > 0:
                days_until_friday -= 7
            black_friday = nov_30 + timedelta(days=days_until_friday)
            
            self._events.append(ContextEvent(
                name="Black Friday",
                start_date=black_friday,
                end_date=black_friday + timedelta(days=1),
                event_type='promotion',
                impact_factor=1.5,
                notes="Major shopping day"
            ))
            
        # School holidays (approximations for Denmark)
        for year in range(current_year-1, current_year+2):
            # Summer holiday (late June to mid-August)
            self._events.append(ContextEvent(
                name="Summer Holiday Period",
                start_date=date(year, 6, 25),
                end_date=date(year, 8, 15),
                event_type='holiday',
                impact_factor=0.85,  # Lower demand during vacation
                notes="School summer holiday"
            ))
            
            # Winter break (around Christmas/New Year)
            self._events.append(ContextEvent(
                name="Winter Break",
                start_date=date(year, 12, 20),
                end_date=date(year+1, 1, 5) if year < current_year+1 else date(year, 12, 31),
                event_type='holiday',
                impact_factor=0.7,
                notes="Christmas/New Year period"
            ))
    
    def add_event(self, event: ContextEvent):
        """
        Add a custom event to the context engine.
        
        Args:
            event: ContextEvent object
        """
        self._events.append(event)
        logger.info(f"Added event: {event.name} on {event.start_date}")
        
    def add_local_event(
        self,
        name: str,
        start_date: date,
        place_ids: List[int],
        impact_factor: float = 1.2,
        end_date: Optional[date] = None,
        event_type: str = "event"
    ):
        """
        Add a location-specific event.
        
        Args:
            name: Event name
            start_date: Event start date
            place_ids: List of affected location IDs
            impact_factor: Demand impact multiplier
            end_date: Event end date (optional)
            event_type: Type of event
        """
        event = ContextEvent(
            name=name,
            start_date=start_date,
            end_date=end_date,
            event_type=event_type,
            impact_factor=impact_factor,
            location_specific=True,
            place_ids=place_ids
        )
        self._events.append(event)
        
    def get_context_factors(
        self,
        place_id: int,
        target_date: Optional[date] = None,
        forecast_horizon_days: int = 7
    ) -> Dict:
        """
        Get all context factors for a location and date range.
        
        Args:
            place_id: Location identifier
            target_date: Target date (defaults to today)
            forecast_horizon_days: Days to look ahead
            
        Returns:
            Dictionary with context factors and adjustments
        """
        if target_date is None:
            target_date = datetime.now().date()
            
        end_date = target_date + timedelta(days=forecast_horizon_days)
        
        context = {
            'date': target_date,
            'place_id': place_id,
            'base_factors': self._get_base_factors(target_date),
            'events': self._get_active_events(place_id, target_date, end_date),
            'weekly_pattern': self._get_weekly_pattern_info(target_date),
            'upcoming_impacts': self._get_upcoming_impacts(place_id, target_date, end_date),
            'combined_factor': 1.0,
            'recommendations': []
        }
        
        # Calculate combined impact factor
        combined = context['base_factors']['day_factor']
        for event in context['events']:
            combined *= event['impact_factor']
            
        context['combined_factor'] = round(combined, 2)
        
        # Add contextual recommendations
        context['recommendations'] = self._generate_context_recommendations(context)
        
        return context
    
    def _get_base_factors(self, target_date: date) -> Dict:
        """Get base calendar factors for a date"""
        day_of_week = target_date.weekday()
        week_of_year = target_date.isocalendar()[1]
        month = target_date.month
        
        return {
            'day_of_week': day_of_week,
            'day_name': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                        'Friday', 'Saturday', 'Sunday'][day_of_week],
            'day_factor': self.config.get_weekly_factor(day_of_week),
            'is_weekend': day_of_week >= 5,
            'week_of_year': week_of_year,
            'month': month,
            'quarter': (month - 1) // 3 + 1
        }
    
    def _get_active_events(
        self,
        place_id: int,
        start_date: date,
        end_date: date
    ) -> List[Dict]:
        """Get events active within the date range"""
        active_events = []
        
        for event in self._events:
            event_end = event.end_date or event.start_date
            
            # Check if event overlaps with date range
            if event.start_date <= end_date and event_end >= start_date:
                # Check if event applies to this location
                if event.location_specific and place_id not in event.place_ids:
                    continue
                    
                active_events.append({
                    'name': event.name,
                    'type': event.event_type,
                    'start_date': event.start_date,
                    'end_date': event_end,
                    'impact_factor': event.impact_factor,
                    'notes': event.notes,
                    'is_local': event.location_specific
                })
                
        return active_events
    
    def _get_weekly_pattern_info(self, target_date: date) -> Dict:
        """Get weekly pattern information"""
        patterns = {}
        day = target_date.weekday()
        
        # Add relative day position info
        if day == 4:  # Friday
            patterns['is_peak_day'] = True
            patterns['note'] = "Friday is typically the highest demand day (+39%)"
        elif day == 6:  # Sunday
            patterns['is_trough_day'] = True
            patterns['note'] = "Sunday is typically the lowest demand day (-18%)"
        elif day in [3, 5]:  # Thursday, Saturday
            patterns['is_high_day'] = True
            patterns['note'] = f"{'Thursday' if day == 3 else 'Saturday'} is a high-demand day"
        else:
            patterns['is_high_day'] = False
            patterns['note'] = "Standard weekday demand pattern"
            
        # Weekly pattern summary
        patterns['weekly_factors'] = {
            'Monday': 0.89,
            'Tuesday': 1.02,
            'Wednesday': 1.14,
            'Thursday': 1.22,
            'Friday': 1.39,
            'Saturday': 1.32,
            'Sunday': 0.82
        }
        
        return patterns
    
    def _get_upcoming_impacts(
        self,
        place_id: int,
        start_date: date,
        end_date: date
    ) -> List[Dict]:
        """Get significant upcoming events/impacts"""
        impacts = []
        
        # Get events in the next period
        events = self._get_active_events(place_id, start_date, end_date)
        
        for event in events:
            if event['impact_factor'] != 1.0:
                impact_type = "increase" if event['impact_factor'] > 1 else "decrease"
                impact_pct = abs(event['impact_factor'] - 1) * 100
                
                impacts.append({
                    'event': event['name'],
                    'date': event['start_date'],
                    'impact_type': impact_type,
                    'impact_percent': round(impact_pct, 0),
                    'description': f"{event['name']} expected to {impact_type} demand by {impact_pct:.0f}%"
                })
                
        # Sort by date
        impacts.sort(key=lambda x: x['date'])
        
        return impacts
    
    def _generate_context_recommendations(self, context: Dict) -> List[str]:
        """Generate recommendations based on context"""
        recommendations = []
        
        # Weekly pattern recommendations
        day_factor = context['base_factors']['day_factor']
        if day_factor > 1.2:
            recommendations.append(
                f"High-demand day expected ({context['base_factors']['day_name']}). "
                f"Increase prep quantities by {(day_factor-1)*100:.0f}%."
            )
        elif day_factor < 0.9:
            recommendations.append(
                f"Lower-demand day expected ({context['base_factors']['day_name']}). "
                f"Consider reducing prep by {(1-day_factor)*100:.0f}%."
            )
            
        # Event recommendations
        for event in context['events']:
            if event['impact_factor'] < 0.8:
                recommendations.append(
                    f"⚠️ {event['name']}: Expect {(1-event['impact_factor'])*100:.0f}% lower demand. "
                    f"Reduce orders to prevent waste."
                )
            elif event['impact_factor'] > 1.2:
                recommendations.append(
                    f"📈 {event['name']}: Expect {(event['impact_factor']-1)*100:.0f}% higher demand. "
                    f"Increase stock levels."
                )
                
        # Combined factor warning
        if context['combined_factor'] < 0.7:
            recommendations.append(
                "⚠️ Multiple factors suggest significantly lower demand. "
                "Review all orders and consider promotional bundles."
            )
        elif context['combined_factor'] > 1.5:
            recommendations.append(
                "📈 Multiple factors suggest significantly higher demand. "
                "Ensure adequate stock and staffing."
            )
            
        return recommendations
    
    def get_forecast_adjustments(
        self,
        place_id: int,
        forecast_dates: List[date]
    ) -> Dict[date, float]:
        """
        Get adjustment factors for a list of forecast dates.
        
        Args:
            place_id: Location identifier
            forecast_dates: List of dates to get adjustments for
            
        Returns:
            Dictionary mapping dates to adjustment factors
        """
        adjustments = {}
        
        for forecast_date in forecast_dates:
            context = self.get_context_factors(
                place_id, 
                target_date=forecast_date,
                forecast_horizon_days=0
            )
            adjustments[forecast_date] = context['combined_factor']
            
        return adjustments
    
    def get_holiday_calendar(
        self,
        year: int,
        place_id: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get a calendar view of holidays and events for a year.
        
        Args:
            year: Calendar year
            place_id: Optional location filter
            
        Returns:
            DataFrame with holiday/event calendar
        """
        start_date = date(year, 1, 1)
        end_date = date(year, 12, 31)
        
        events = self._get_active_events(
            place_id or 0,
            start_date,
            end_date
        )
        
        # Filter if place_id provided
        if place_id:
            events = [e for e in events if not e['is_local'] or place_id in e.get('place_ids', [])]
            
        df = pd.DataFrame(events)
        if len(df) > 0:
            df = df.sort_values('start_date')
            
        return df
    
    def analyze_historical_context(
        self,
        place_id: int,
        historical_data: pd.DataFrame,
        date_column: str = 'week_start',
        demand_column: str = 'demand'
    ) -> Dict:
        """
        Analyze historical data to detect context patterns.
        
        Args:
            place_id: Location identifier
            historical_data: DataFrame with historical demand
            date_column: Name of date column
            demand_column: Name of demand column
            
        Returns:
            Analysis of context patterns in historical data
        """
        df = historical_data.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        
        analysis = {
            'place_id': place_id,
            'period_analyzed': {
                'start': df[date_column].min(),
                'end': df[date_column].max()
            },
            'patterns': {}
        }
        
        # Day of week patterns
        df['day_of_week'] = df[date_column].dt.dayofweek
        dow_pattern = df.groupby('day_of_week')[demand_column].mean()
        overall_mean = df[demand_column].mean()
        
        analysis['patterns']['weekly'] = {
            day: round(demand / overall_mean, 2) if overall_mean > 0 else 1.0
            for day, demand in dow_pattern.items()
        }
        
        # Monthly patterns
        df['month'] = df[date_column].dt.month
        month_pattern = df.groupby('month')[demand_column].mean()
        
        analysis['patterns']['monthly'] = {
            month: round(demand / overall_mean, 2) if overall_mean > 0 else 1.0
            for month, demand in month_pattern.items()
        }
        
        # Holiday impact analysis
        holiday_impacts = []
        for event in self._events:
            if event.event_type == 'holiday':
                # Find data around holiday
                mask = (df[date_column].dt.date >= event.start_date - timedelta(days=3)) & \
                       (df[date_column].dt.date <= (event.end_date or event.start_date) + timedelta(days=3))
                       
                holiday_data = df[mask]
                if len(holiday_data) > 0:
                    holiday_mean = holiday_data[demand_column].mean()
                    impact = holiday_mean / overall_mean if overall_mean > 0 else 1.0
                    holiday_impacts.append({
                        'holiday': event.name,
                        'observed_impact': round(impact, 2),
                        'configured_impact': event.impact_factor
                    })
                    
        analysis['patterns']['holidays'] = holiday_impacts
        
        return analysis
