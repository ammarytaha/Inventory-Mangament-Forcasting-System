"""
Demand Forecasting Service
===========================
Implements demand forecasting using Holt-Winters Exponential Smoothing
with intelligent fallbacks and confidence bands.

Design Principles:
- Explainable: All forecasts include reasoning
- Robust: Fallback to simpler methods when data insufficient
- No black-box ML: Uses statistical methods only
- Configurable: All parameters exposed and documented

Key Algorithms:
1. Holt-Winters Triple Exponential Smoothing (primary)
   - Captures level, trend, and seasonality
   - Requires at least 2 seasonal periods of data
   
2. Moving Average (fallback)
   - Used when insufficient data for Holt-Winters
   - Simple, explainable, stable

Assumptions (documented for hackathon):
- Weekly seasonality (7-day cycle) for restaurants
- Additive seasonality (constant seasonal effect)
- Damped trend to prevent runaway forecasts
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from utils.logger import get_logger, LogContext
from utils.constants import FORECAST_CONFIG

logger = get_logger(__name__)


@dataclass
class ForecastResult:
    """
    Result of a forecast operation for a single item.
    
    Attributes
    ----------
    item_id : Any
        Identifier of the forecasted item
    method : str
        Forecasting method used ("holt_winters" or "moving_average")
    forecast : pd.DataFrame
        Forecasted values with columns: date, forecast, lower_bound, upper_bound
    model_params : Dict[str, Any]
        Parameters used in the model
    quality_metrics : Dict[str, float]
        Model quality indicators
    explanation : str
        Human-readable explanation of the forecast
    """
    item_id: Any
    method: str
    forecast: pd.DataFrame
    model_params: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""


class DemandForecaster:
    """
    Demand forecasting engine with Holt-Winters and fallback methods.
    
    This class provides:
    - Product-level demand forecasting
    - Automatic method selection based on data availability
    - Confidence intervals for uncertainty
    - Clear explanations for each forecast
    
    Usage
    -----
    >>> forecaster = DemandForecaster(horizon=7)
    >>> results = forecaster.forecast_all(daily_demand_df)
    >>> for result in results:
    ...     print(f"{result.item_id}: {result.explanation}")
    
    Technical Notes
    ---------------
    Why Holt-Winters instead of ARIMA/Prophet?
    - Holt-Winters is simpler and more explainable
    - Works well with weekly seasonality in retail/restaurant
    - No complex hyperparameter tuning needed
    - Built into Python's statsmodels (no extra dependencies)
    
    Why not deep learning?
    - Overkill for this use case
    - Requires more data than typically available
    - Harder to explain to business stakeholders
    """
    
    def __init__(
        self,
        horizon: int = None,
        seasonal_periods: int = None,
        confidence_level: float = None
    ):
        """
        Initialize the forecaster.
        
        Parameters
        ----------
        horizon : int, optional
            Number of days to forecast ahead. Default from config.
        seasonal_periods : int, optional
            Length of seasonal cycle (7 = weekly). Default from config.
        confidence_level : float, optional
            Confidence level for prediction intervals (0-1). Default: 0.95
        """
        self.horizon = horizon or FORECAST_CONFIG["default_forecast_horizon"]
        self.seasonal_periods = seasonal_periods or FORECAST_CONFIG["seasonal_period"]
        self.confidence_level = confidence_level or FORECAST_CONFIG["confidence_level"]
        
        # Minimum data requirements
        self.min_points_hw = FORECAST_CONFIG["min_data_points_holt_winters"]
        self.min_points_ma = FORECAST_CONFIG["min_data_points_moving_avg"]
        
        logger.info(
            f"Forecaster initialized: horizon={self.horizon} days, "
            f"seasonal_periods={self.seasonal_periods}"
        )
    
    def forecast_all(
        self,
        daily_demand: pd.DataFrame,
        item_column: str = 'item_id',
        date_column: str = 'date',
        demand_column: str = 'demand'
    ) -> List[ForecastResult]:
        """
        Forecast demand for all items in the dataset.
        
        Parameters
        ----------
        daily_demand : pd.DataFrame
            Daily demand data with item, date, and demand columns
        item_column : str
            Name of the item identifier column
        date_column : str
            Name of the date column
        demand_column : str
            Name of the demand quantity column
        
        Returns
        -------
        List[ForecastResult]
            Forecast results for each item
        """
        results = []
        
        if len(daily_demand) == 0:
            logger.warning("Empty demand data provided")
            return results
        
        items = daily_demand[item_column].unique()
        logger.info(f"Forecasting demand for {len(items)} items")
        
        for item_id in items:
            item_data = daily_demand[daily_demand[item_column] == item_id].copy()
            item_data = item_data.sort_values(date_column)
            
            result = self.forecast_single(
                time_series=item_data[demand_column].values,
                dates=item_data[date_column].values,
                item_id=item_id
            )
            results.append(result)
        
        # Summary logging
        hw_count = sum(1 for r in results if r.method == 'holt_winters')
        ma_count = sum(1 for r in results if r.method == 'moving_average')
        insufficient_count = sum(1 for r in results if r.method == 'insufficient_data')
        
        logger.info(
            f"Forecasting complete: {hw_count} Holt-Winters, "
            f"{ma_count} moving average, {insufficient_count} insufficient data"
        )
        
        return results
    
    def forecast_single(
        self,
        time_series: np.ndarray,
        dates: np.ndarray,
        item_id: Any
    ) -> ForecastResult:
        """
        Forecast demand for a single item.
        
        Parameters
        ----------
        time_series : np.ndarray
            Historical demand values
        dates : np.ndarray
            Corresponding dates
        item_id : Any
            Item identifier
        
        Returns
        -------
        ForecastResult
            Forecast with method used and explanation
        """
        n_points = len(time_series)
        
        # Determine which method to use based on data availability
        if n_points >= self.min_points_hw:
            return self._forecast_holt_winters(time_series, dates, item_id)
        elif n_points >= self.min_points_ma:
            return self._forecast_moving_average(time_series, dates, item_id)
        else:
            return self._insufficient_data_forecast(time_series, dates, item_id)
    
    def _forecast_holt_winters(
        self,
        time_series: np.ndarray,
        dates: np.ndarray,
        item_id: Any
    ) -> ForecastResult:
        """
        Apply Holt-Winters Triple Exponential Smoothing.
        
        Holt-Winters decomposes the time series into:
        - Level: The baseline value
        - Trend: The direction and rate of change
        - Seasonality: Repeating patterns (weekly)
        
        We use:
        - Additive seasonality: seasonal effect is constant
        - Damped trend: prevents unrealistic long-term projections
        """
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
        except ImportError:
            logger.warning("statsmodels not available, falling back to moving average")
            return self._forecast_moving_average(time_series, dates, item_id)
        
        try:
            # Ensure positive values for multiplicative models
            ts_adjusted = np.maximum(time_series, 0.1)  # Avoid zeros
            
            # Determine seasonality type
            hw_params = FORECAST_CONFIG["holt_winters_defaults"]
            
            # Fit the model
            model = ExponentialSmoothing(
                ts_adjusted,
                trend=hw_params["trend"],
                damped_trend=hw_params["damped_trend"],
                seasonal=hw_params["seasonal"],
                seasonal_periods=self.seasonal_periods,
                initialization_method='estimated'
            )
            
            fitted_model = model.fit(optimized=True)
            
            # Generate forecast
            forecast_values = fitted_model.forecast(self.horizon)
            
            # Ensure non-negative forecasts
            forecast_values = np.maximum(forecast_values, 0)
            
            # Calculate confidence bands using historical variance
            residuals = fitted_model.resid
            std_residual = np.std(residuals)
            
            # Z-score for confidence level
            from scipy import stats
            z_score = stats.norm.ppf((1 + self.confidence_level) / 2)
            
            lower_bound = np.maximum(forecast_values - z_score * std_residual, 0)
            upper_bound = forecast_values + z_score * std_residual
            
            # Generate forecast dates
            last_date = pd.to_datetime(dates[-1])
            forecast_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=self.horizon,
                freq='D'
            )
            
            forecast_df = pd.DataFrame({
                'date': forecast_dates,
                'forecast': forecast_values,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            })
            
            # Model quality metrics
            mape = self._calculate_mape(time_series, fitted_model.fittedvalues)
            
            explanation = (
                f"Forecast generated using Holt-Winters with weekly seasonality. "
                f"Based on {len(time_series)} days of historical data. "
                f"Model captures trend and 7-day seasonal patterns. "
                f"Forecast accuracy (MAPE): {mape:.1f}%."
            )
            
            return ForecastResult(
                item_id=item_id,
                method='holt_winters',
                forecast=forecast_df,
                model_params={
                    'seasonal_periods': self.seasonal_periods,
                    'trend': hw_params['trend'],
                    'seasonal': hw_params['seasonal'],
                    'damped': hw_params['damped_trend']
                },
                quality_metrics={
                    'mape': mape,
                    'std_residual': std_residual,
                    'data_points': len(time_series)
                },
                explanation=explanation
            )
            
        except Exception as e:
            logger.warning(f"Holt-Winters failed for item {item_id}: {e}")
            return self._forecast_moving_average(time_series, dates, item_id)
    
    def _forecast_moving_average(
        self,
        time_series: np.ndarray,
        dates: np.ndarray,
        item_id: Any
    ) -> ForecastResult:
        """
        Fallback: Simple Moving Average forecast.
        
        Why use this?
        - More stable with limited data
        - No seasonality assumptions needed
        - Easy to explain: "Based on average of recent days"
        """
        # Use 7-day moving average (or all available data if less)
        window = min(7, len(time_series))
        recent_values = time_series[-window:]
        
        # Forecast is the mean of recent values
        forecast_value = np.mean(recent_values)
        
        # Standard deviation for confidence bands
        std_dev = np.std(recent_values) if len(recent_values) > 1 else forecast_value * 0.2
        
        # Z-score for confidence level
        try:
            from scipy import stats
            z_score = stats.norm.ppf((1 + self.confidence_level) / 2)
        except ImportError:
            z_score = 1.96  # Default for 95% confidence
        
        lower_bound = max(0, forecast_value - z_score * std_dev)
        upper_bound = forecast_value + z_score * std_dev
        
        # Generate forecast dates
        last_date = pd.to_datetime(dates[-1])
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=self.horizon,
            freq='D'
        )
        
        # Constant forecast (typical for simple MA)
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'forecast': [forecast_value] * self.horizon,
            'lower_bound': [lower_bound] * self.horizon,
            'upper_bound': [upper_bound] * self.horizon
        })
        
        explanation = (
            f"Forecast generated using {window}-day moving average. "
            f"Limited historical data ({len(time_series)} days) prevented use of "
            f"more sophisticated methods. Average daily demand: {forecast_value:.1f} units."
        )
        
        return ForecastResult(
            item_id=item_id,
            method='moving_average',
            forecast=forecast_df,
            model_params={
                'window': window,
                'method': 'simple_moving_average'
            },
            quality_metrics={
                'std_dev': std_dev,
                'data_points': len(time_series),
                'mean_demand': forecast_value
            },
            explanation=explanation
        )
    
    def _insufficient_data_forecast(
        self,
        time_series: np.ndarray,
        dates: np.ndarray,
        item_id: Any
    ) -> ForecastResult:
        """
        Handle cases with too little data.
        
        Returns a conservative forecast with high uncertainty.
        """
        if len(time_series) > 0:
            mean_value = np.mean(time_series)
        else:
            mean_value = 0
        
        # High uncertainty due to limited data
        uncertainty = max(mean_value * 0.5, 1)
        
        last_date = datetime.now() if len(dates) == 0 else pd.to_datetime(dates[-1])
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=self.horizon,
            freq='D'
        )
        
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'forecast': [mean_value] * self.horizon,
            'lower_bound': [max(0, mean_value - uncertainty)] * self.horizon,
            'upper_bound': [mean_value + uncertainty] * self.horizon
        })
        
        explanation = (
            f"Insufficient historical data ({len(time_series)} data points) for reliable forecast. "
            f"Using mean value with high uncertainty band. "
            f"Recommend monitoring actual demand closely and adjusting."
        )
        
        return ForecastResult(
            item_id=item_id,
            method='insufficient_data',
            forecast=forecast_df,
            model_params={
                'data_points': len(time_series),
                'required_minimum': self.min_points_ma
            },
            quality_metrics={
                'confidence': 'low',
                'data_points': len(time_series)
            },
            explanation=explanation
        )
    
    def _calculate_mape(
        self,
        actual: np.ndarray,
        predicted: np.ndarray
    ) -> float:
        """
        Calculate Mean Absolute Percentage Error.
        
        MAPE interpretation:
        - < 10%: Excellent accuracy
        - 10-20%: Good accuracy
        - 20-50%: Reasonable accuracy
        - > 50%: Low accuracy
        """
        # Avoid division by zero
        mask = actual != 0
        if not np.any(mask):
            return 100.0
        
        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
        return min(mape, 100.0)  # Cap at 100%


def combine_forecast_results(
    results: List[ForecastResult]
) -> pd.DataFrame:
    """
    Combine multiple forecast results into a single DataFrame.
    
    Parameters
    ----------
    results : List[ForecastResult]
        List of individual forecast results
    
    Returns
    -------
    pd.DataFrame
        Combined forecasts with columns:
        - item_id, date, forecast, lower_bound, upper_bound, method
    """
    all_forecasts = []
    
    for result in results:
        df = result.forecast.copy()
        df['item_id'] = result.item_id
        df['method'] = result.method
        all_forecasts.append(df)
    
    if len(all_forecasts) == 0:
        return pd.DataFrame()
    
    combined = pd.concat(all_forecasts, ignore_index=True)
    
    # Reorder columns
    cols = ['item_id', 'date', 'forecast', 'lower_bound', 'upper_bound', 'method']
    combined = combined[[c for c in cols if c in combined.columns]]
    
    return combined
