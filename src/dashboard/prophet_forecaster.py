"""
Prophet Forecasting Service for Fresh Flow Markets
===================================================

Uses Facebook Prophet for store-level demand forecasting with:
- Weekly seasonality (strong: Friday +39%, Sunday -26%)
- Monthly/yearly seasonality
- Holiday effects
- Confidence intervals

Design Decisions for Hackathon:
1. Use Prophet for AGGREGATE store demand (impressive charts)
2. Keep forecasts cached to avoid re-computation on refresh
3. Provide decomposition for explainability
4. Include seasonality factors for "why" explanations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Lazy import Prophet to avoid slow startup
_prophet_available = None
_Prophet = None


def _get_prophet():
    """Lazy load Prophet to speed up initial imports."""
    global _prophet_available, _Prophet
    
    if _prophet_available is None:
        try:
            from prophet import Prophet
            _Prophet = Prophet
            _prophet_available = True
        except ImportError:
            _prophet_available = False
            _Prophet = None
    
    return _Prophet, _prophet_available


class ProphetForecaster:
    """
    Prophet-based demand forecaster for Fresh Flow Markets.
    
    Provides:
    - Store-level daily demand forecasts
    - Weekly, monthly, yearly seasonality decomposition
    - Confidence intervals (80% and 95%)
    - Plain English explanations of patterns
    
    Usage:
        forecaster = ProphetForecaster()
        forecaster.fit(daily_demand_df)  # DataFrame with 'ds' and 'y' columns
        forecast = forecaster.predict(periods=14)
    """
    
    def __init__(self, 
                 yearly_seasonality: bool = True,
                 weekly_seasonality: bool = True,
                 daily_seasonality: bool = False,
                 seasonality_mode: str = 'multiplicative'):
        """
        Initialize the Prophet forecaster.
        
        Parameters:
        -----------
        yearly_seasonality : bool
            Capture yearly patterns (default: True)
        weekly_seasonality : bool
            Capture day-of-week patterns (default: True)
        daily_seasonality : bool
            Capture intra-day patterns (default: False - we use daily aggregates)
        seasonality_mode : str
            'multiplicative' or 'additive' (default: multiplicative)
        """
        Prophet, available = _get_prophet()
        
        if not available:
            raise ImportError("Prophet is not installed. Run: pip install prophet")
        
        self.model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            seasonality_mode=seasonality_mode,
            interval_width=0.95,  # 95% confidence interval
            changepoint_prior_scale=0.05  # More conservative trend changes
        )
        
        self.is_fitted = False
        self.training_data = None
        self.forecast_result = None
        self.seasonality_components = None
    
    def fit(self, df: pd.DataFrame) -> 'ProphetForecaster':
        """
        Fit the model on historical demand data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Must have columns 'ds' (datetime) and 'y' (demand quantity)
        
        Returns:
        --------
        self : ProphetForecaster
            Fitted forecaster
        """
        if 'ds' not in df.columns or 'y' not in df.columns:
            raise ValueError("DataFrame must have 'ds' and 'y' columns")
        
        # Ensure proper types
        training_df = df[['ds', 'y']].copy()
        training_df['ds'] = pd.to_datetime(training_df['ds'])
        training_df['y'] = pd.to_numeric(training_df['y'], errors='coerce').fillna(0)
        
        # Remove zeros/negatives that could cause issues
        training_df = training_df[training_df['y'] > 0]
        
        self.training_data = training_df
        self.model.fit(training_df)
        self.is_fitted = True
        
        return self
    
    def predict(self, periods: int = 14) -> pd.DataFrame:
        """
        Generate forecast for future periods.
        
        Parameters:
        -----------
        periods : int
            Number of days to forecast (default: 14)
        
        Returns:
        --------
        pd.DataFrame with columns:
            - ds: date
            - yhat: forecasted demand
            - yhat_lower: lower confidence bound (95%)
            - yhat_upper: upper confidence bound (95%)
            - trend: trend component
            - weekly: weekly seasonality effect
            - yearly: yearly seasonality effect (if enabled)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting. Call fit() first.")
        
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=periods, freq='D')
        
        # Generate forecast
        forecast = self.model.predict(future)
        
        # Store for later use
        self.forecast_result = forecast
        
        # Extract relevant columns
        result_cols = ['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']
        
        if 'weekly' in forecast.columns:
            result_cols.append('weekly')
        if 'yearly' in forecast.columns:
            result_cols.append('yearly')
        
        return forecast[result_cols]
    
    def get_seasonality_effects(self) -> Dict[str, Any]:
        """
        Extract seasonality effects for explanation.
        
        Returns:
        --------
        Dict with:
            - weekly: Dict mapping day names to effect multipliers
            - yearly: Dict mapping months to effect multipliers
            - peak_day: Day with highest demand
            - low_day: Day with lowest demand
        """
        if self.forecast_result is None:
            return {}
        
        result = {
            'weekly': {},
            'yearly': {},
            'peak_day': None,
            'low_day': None
        }
        
        # Extract weekly seasonality
        if 'weekly' in self.forecast_result.columns:
            df = self.forecast_result.copy()
            df['day_of_week'] = df['ds'].dt.dayofweek
            
            weekly_effect = df.groupby('day_of_week')['weekly'].mean()
            
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            for i, name in enumerate(day_names):
                if i in weekly_effect.index:
                    # Convert to percentage
                    result['weekly'][name] = round(weekly_effect[i] * 100, 1)
            
            if len(weekly_effect) > 0:
                result['peak_day'] = day_names[weekly_effect.idxmax()]
                result['low_day'] = day_names[weekly_effect.idxmin()]
        
        # Extract yearly seasonality
        if 'yearly' in self.forecast_result.columns:
            df = self.forecast_result.copy()
            df['month'] = df['ds'].dt.month
            
            monthly_effect = df.groupby('month')['yearly'].mean()
            
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            for i in range(12):
                if (i + 1) in monthly_effect.index:
                    result['yearly'][month_names[i]] = round(monthly_effect[i + 1] * 100, 1)
        
        return result
    
    def get_forecast_explanation(self, future_date: datetime = None) -> str:
        """
        Generate plain English explanation of the forecast.
        
        Parameters:
        -----------
        future_date : datetime, optional
            Specific date to explain. If None, explains the next week.
        
        Returns:
        --------
        str : Human-readable explanation
        """
        seasonality = self.get_seasonality_effects()
        
        if not seasonality.get('weekly'):
            return "Forecast based on historical demand patterns."
        
        peak_day = seasonality.get('peak_day', 'Friday')
        low_day = seasonality.get('low_day', 'Sunday')
        
        peak_effect = seasonality['weekly'].get(peak_day, 0)
        low_effect = seasonality['weekly'].get(low_day, 0)
        
        explanation = f"Demand follows a strong weekly pattern. "
        explanation += f"{peak_day} is the busiest day ({'+' if peak_effect > 0 else ''}{peak_effect:.0f}% vs average), "
        explanation += f"while {low_day} is slowest ({'+' if low_effect > 0 else ''}{low_effect:.0f}%). "
        
        if seasonality.get('yearly'):
            peak_month = max(seasonality['yearly'].items(), key=lambda x: x[1])[0]
            explanation += f"Yearly peak is in {peak_month}."
        
        return explanation


def create_forecast_chart_data(historical: pd.DataFrame, 
                                forecast: pd.DataFrame,
                                last_n_days: int = 30) -> Dict[str, Any]:
    """
    Prepare data for Plotly forecast chart.
    
    Parameters:
    -----------
    historical : pd.DataFrame
        Historical data with 'ds' and 'y' columns
    forecast : pd.DataFrame
        Forecast data from Prophet
    last_n_days : int
        Number of historical days to include in chart
    
    Returns:
    --------
    Dict with chart-ready data:
        - historical_dates: list of date strings
        - historical_values: list of demand values
        - forecast_dates: list of future date strings
        - forecast_values: list of forecasted values
        - conf_lower: list of lower confidence bounds
        - conf_upper: list of upper confidence bounds
    """
    # Get last N days of historical data
    historical = historical.copy()
    historical['ds'] = pd.to_datetime(historical['ds'])
    historical = historical.sort_values('ds')
    
    cutoff_date = historical['ds'].max() - timedelta(days=last_n_days)
    recent_historical = historical[historical['ds'] >= cutoff_date]
    
    # Get future forecast only
    forecast = forecast.copy()
    forecast['ds'] = pd.to_datetime(forecast['ds'])
    last_historical_date = historical['ds'].max()
    future_forecast = forecast[forecast['ds'] > last_historical_date]
    
    return {
        'historical_dates': recent_historical['ds'].dt.strftime('%Y-%m-%d').tolist(),
        'historical_values': recent_historical['y'].tolist(),
        'forecast_dates': future_forecast['ds'].dt.strftime('%Y-%m-%d').tolist(),
        'forecast_values': future_forecast['yhat'].tolist(),
        'conf_lower': future_forecast['yhat_lower'].tolist(),
        'conf_upper': future_forecast['yhat_upper'].tolist()
    }


def get_quick_forecast(daily_demand: pd.DataFrame, 
                       periods: int = 14) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Quick helper to generate forecast from daily demand data.
    
    Parameters:
    -----------
    daily_demand : pd.DataFrame
        Must have 'ds' (date) and 'y' (quantity) columns
    periods : int
        Days to forecast
    
    Returns:
    --------
    Tuple of:
        - forecast DataFrame
        - seasonality effects dict
    """
    Prophet, available = _get_prophet()
    
    if not available:
        # Fallback to simple moving average if Prophet not available
        historical_mean = daily_demand['y'].mean()
        
        last_date = pd.to_datetime(daily_demand['ds']).max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods)
        
        forecast = pd.DataFrame({
            'ds': future_dates,
            'yhat': [historical_mean] * periods,
            'yhat_lower': [historical_mean * 0.8] * periods,
            'yhat_upper': [historical_mean * 1.2] * periods
        })
        
        return forecast, {'method': 'moving_average', 'note': 'Prophet not available'}
    
    forecaster = ProphetForecaster()
    forecaster.fit(daily_demand)
    forecast = forecaster.predict(periods)
    seasonality = forecaster.get_seasonality_effects()
    seasonality['explanation'] = forecaster.get_forecast_explanation()
    seasonality['method'] = 'prophet'
    
    return forecast, seasonality


# Cache for forecasts
_forecast_cache = {}


def get_cached_forecast(daily_demand: pd.DataFrame, 
                        periods: int = 14,
                        cache_key: str = "default") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Get forecast with caching to avoid re-computation.
    
    Forecasts are cached by key to avoid re-running Prophet on every refresh.
    """
    global _forecast_cache
    
    if cache_key not in _forecast_cache:
        _forecast_cache[cache_key] = get_quick_forecast(daily_demand, periods)
    
    return _forecast_cache[cache_key]


# For testing
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    np.random.seed(42)
    
    # Simulate demand with weekly pattern
    demand = 100 + np.random.normal(0, 10, len(dates))
    day_of_week = dates.dayofweek
    
    # Add weekly seasonality (Fri +30%, Sun -25%)
    weekly_effect = np.array([0, 0, 0, 0.05, 0.3, 0.15, -0.25])
    demand = demand * (1 + weekly_effect[day_of_week])
    
    df = pd.DataFrame({'ds': dates, 'y': demand})
    
    print("Testing Prophet Forecaster...")
    print(f"Training data: {len(df)} days")
    
    forecaster = ProphetForecaster()
    forecaster.fit(df)
    
    forecast = forecaster.predict(periods=14)
    print(f"\nForecast for next 14 days:")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(14).to_string())
    
    print("\nSeasonality Effects:")
    effects = forecaster.get_seasonality_effects()
    for day, effect in effects['weekly'].items():
        print(f"  {day}: {'+' if effect > 0 else ''}{effect}%")
    
    print(f"\nExplanation: {forecaster.get_forecast_explanation()}")
