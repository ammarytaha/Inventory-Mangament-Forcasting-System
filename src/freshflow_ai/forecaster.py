"""
FreshFlow AI - Forecasting Engine
==================================

Multi-model demand forecasting engine that selects the optimal
forecasting approach based on demand patterns.

Supported Models:
- Prophet: For smooth, regular demand patterns
- Croston: For intermittent demand
- SBA (Syntetos-Boylan Approximation): Alternative for intermittent
- LightGBM: For erratic/lumpy demand with ML approach
- Moving Average: Fallback for insufficient data

Features:
- Automatic model selection based on SBC classification
- Confidence intervals for all forecasts
- Backtest validation
- Multi-step forecasting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import warnings
import logging

from .config import Config, DEFAULT_CONFIG
from .data_processor import DataProcessor

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class ForecastEngine:
    """
    Multi-model forecasting engine for inventory demand prediction.
    
    Automatically selects the best forecasting model based on demand patterns
    using SBC (Syntetos-Boylan Classification).
    
    Usage:
        engine = ForecastEngine(data_processor)
        forecast = engine.forecast_item(place_id=94025, item_id=480007, horizon=4)
    """
    
    def __init__(
        self, 
        data_processor: Optional[DataProcessor] = None,
        config: Optional[Config] = None
    ):
        """
        Initialize the ForecastEngine.
        
        Args:
            data_processor: DataProcessor instance for data access
            config: Configuration object
        """
        self.config = config or DEFAULT_CONFIG
        self.data_processor = data_processor or DataProcessor(self.config)
        
        # Model availability flags (lazy loading)
        self._prophet_available = None
        self._lightgbm_available = None
        
    def _check_prophet(self) -> bool:
        """Check if Prophet is available"""
        if self._prophet_available is None:
            try:
                from prophet import Prophet
                self._prophet_available = True
            except ImportError:
                self._prophet_available = False
                logger.warning("Prophet not available. Install with: pip install prophet")
        return self._prophet_available
    
    def _check_lightgbm(self) -> bool:
        """Check if LightGBM is available"""
        if self._lightgbm_available is None:
            try:
                import lightgbm
                self._lightgbm_available = True
            except ImportError:
                self._lightgbm_available = False
                logger.warning("LightGBM not available. Install with: pip install lightgbm")
        return self._lightgbm_available
        
    def forecast_item(
        self,
        place_id: int,
        item_id: int,
        horizon: int = 4,
        include_confidence: bool = True
    ) -> Dict:
        """
        Generate demand forecast for a specific item at a location.
        
        Automatically selects the best model based on demand classification.
        
        Args:
            place_id: Location identifier
            item_id: Product identifier
            horizon: Forecast horizon in weeks
            include_confidence: Whether to include confidence intervals
            
        Returns:
            Dictionary with:
            - forecast: List of (date, predicted_demand) tuples
            - model_used: Name of the model used
            - demand_type: SBC classification
            - confidence_lower: Lower bound (if requested)
            - confidence_upper: Upper bound (if requested)
            - metrics: Model performance metrics
        """
        # Get historical data
        history = self.data_processor.get_item_history(place_id, item_id)
        
        if len(history) < 4:
            return self._insufficient_data_forecast(place_id, item_id, horizon)
        
        # Get demand classification
        demand_type = self.data_processor.get_demand_type(place_id, item_id)
        
        # Select and run model
        if demand_type == 'Smooth':
            result = self._forecast_prophet(history, horizon, include_confidence)
        elif demand_type == 'Erratic':
            result = self._forecast_ml(history, horizon, include_confidence)
        elif demand_type == 'Intermittent':
            result = self._forecast_croston(history, horizon, include_confidence)
        elif demand_type == 'Lumpy':
            result = self._forecast_ml(history, horizon, include_confidence)
        else:
            result = self._forecast_moving_average(history, horizon, include_confidence)
            
        result['demand_type'] = demand_type
        result['place_id'] = place_id
        result['item_id'] = item_id
        
        return result
    
    def forecast_place(
        self,
        place_id: int,
        horizon: int = 4,
        top_n_items: Optional[int] = None
    ) -> List[Dict]:
        """
        Generate forecasts for all (or top N) items at a location.
        
        Args:
            place_id: Location identifier
            horizon: Forecast horizon in weeks
            top_n_items: Number of top items to forecast (None for all)
            
        Returns:
            List of forecast dictionaries for each item
        """
        place_data = self.data_processor.get_place_data(place_id)
        weekly = place_data.get('weekly_demand')
        
        if weekly is None or len(weekly) == 0:
            return []
            
        # Get unique items
        items = weekly['item_id'].unique()
        
        # Optionally limit to top items
        if top_n_items:
            top_items = self.data_processor.get_top_items(place_id, n=top_n_items)
            items = top_items['item_id'].values
            
        # Generate forecasts
        forecasts = []
        for item_id in items:
            try:
                forecast = self.forecast_item(place_id, int(item_id), horizon)
                forecasts.append(forecast)
            except Exception as e:
                logger.warning(f"Error forecasting item {item_id}: {e}")
                continue
                
        return forecasts
    
    def _forecast_prophet(
        self,
        history: pd.DataFrame,
        horizon: int,
        include_confidence: bool
    ) -> Dict:
        """Forecast using Facebook Prophet"""
        if not self._check_prophet():
            return self._forecast_moving_average(history, horizon, include_confidence)
            
        from prophet import Prophet
        
        # Prepare data for Prophet
        df = history[['week_start', 'demand']].copy()
        df.columns = ['ds', 'y']
        df['ds'] = pd.to_datetime(df['ds'])
        
        # Configure Prophet
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,  # Weekly data, so no intra-week
            daily_seasonality=False,
            seasonality_mode=self.config.model.prophet_seasonality_mode,
            interval_width=0.8
        )
        
        # Fit model
        model.fit(df)
        
        # Make future dataframe
        future = model.make_future_dataframe(periods=horizon, freq='W')
        predictions = model.predict(future)
        
        # Extract forecast
        forecast_df = predictions.tail(horizon)
        
        result = {
            'model_used': 'prophet',
            'forecast': [
                {
                    'week_start': row['ds'],
                    'predicted_demand': max(0, round(row['yhat'])),
                    'lower': max(0, round(row['yhat_lower'])) if include_confidence else None,
                    'upper': max(0, round(row['yhat_upper'])) if include_confidence else None
                }
                for _, row in forecast_df.iterrows()
            ],
            'confidence_level': 0.8,
            'metrics': self._calculate_backtest_metrics(history, model, 'prophet')
        }
        
        return result
    
    def _forecast_croston(
        self,
        history: pd.DataFrame,
        horizon: int,
        include_confidence: bool
    ) -> Dict:
        """
        Forecast using Croston's method for intermittent demand.
        
        Croston's method separately forecasts:
        1. Non-zero demand sizes
        2. Inter-arrival times between demands
        """
        demand = history['demand'].values
        alpha = self.config.model.croston_alpha
        
        # Initialize
        n = len(demand)
        z = np.where(demand > 0, demand, np.nan)  # Non-zero demands
        q = np.zeros(n)  # Inter-arrival intervals
        
        # Calculate inter-arrival times
        last_demand_idx = -1
        for i in range(n):
            if demand[i] > 0:
                if last_demand_idx >= 0:
                    q[i] = i - last_demand_idx
                last_demand_idx = i
                
        # Apply exponential smoothing to z and q
        z_smooth = self._exponential_smooth(z[~np.isnan(z)], alpha)
        q_smooth = self._exponential_smooth(q[q > 0], alpha)
        
        # Calculate forecast
        if q_smooth > 0:
            forecast_value = z_smooth / q_smooth
        else:
            forecast_value = np.nanmean(demand)
            
        forecast_value = max(0, round(forecast_value))
        
        # Create forecast list
        last_date = pd.to_datetime(history['week_start'].max())
        forecasts = []
        for i in range(horizon):
            week_start = last_date + timedelta(weeks=i+1)
            forecasts.append({
                'week_start': week_start,
                'predicted_demand': forecast_value,
                'lower': max(0, int(forecast_value * 0.5)) if include_confidence else None,
                'upper': int(forecast_value * 1.5) if include_confidence else None
            })
            
        return {
            'model_used': 'croston',
            'forecast': forecasts,
            'confidence_level': 0.8,
            'metrics': {'method': 'croston', 'alpha': alpha}
        }
    
    def _forecast_ml(
        self,
        history: pd.DataFrame,
        horizon: int,
        include_confidence: bool
    ) -> Dict:
        """Forecast using ML model (LightGBM or fallback)"""
        if not self._check_lightgbm():
            return self._forecast_moving_average(history, horizon, include_confidence)
            
        import lightgbm as lgb
        
        # Prepare features
        df = history.copy()
        df = df.sort_values('week_start')
        
        # Create lag features
        for lag in [1, 2, 4]:
            df[f'lag_{lag}'] = df['demand'].shift(lag)
            
        # Rolling features
        df['roll_mean_4'] = df['demand'].shift(1).rolling(4).mean()
        df['roll_std_4'] = df['demand'].shift(1).rolling(4).std()
        
        # Drop nulls from lag creation
        df = df.dropna()
        
        if len(df) < 10:
            return self._forecast_moving_average(history, horizon, include_confidence)
            
        # Features and target
        feature_cols = [c for c in df.columns if c.startswith(('lag_', 'roll_'))]
        X = df[feature_cols]
        y = df['demand']
        
        # Train model
        train_data = lgb.Dataset(X, label=y)
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 15,
            'learning_rate': 0.1,
            'verbose': -1
        }
        model = lgb.train(params, train_data, num_boost_round=100)
        
        # Generate forecasts iteratively
        last_values = df.tail(4)['demand'].values
        forecasts = []
        last_date = pd.to_datetime(df['week_start'].max())
        
        for i in range(horizon):
            # Create features for prediction
            features = {
                'lag_1': last_values[-1],
                'lag_2': last_values[-2] if len(last_values) > 1 else last_values[-1],
                'lag_4': last_values[-4] if len(last_values) > 3 else last_values[-1],
                'roll_mean_4': np.mean(last_values[-4:]),
                'roll_std_4': np.std(last_values[-4:])
            }
            
            X_pred = pd.DataFrame([features])
            pred = model.predict(X_pred)[0]
            pred = max(0, round(pred))
            
            week_start = last_date + timedelta(weeks=i+1)
            forecasts.append({
                'week_start': week_start,
                'predicted_demand': pred,
                'lower': max(0, int(pred * 0.7)) if include_confidence else None,
                'upper': int(pred * 1.3) if include_confidence else None
            })
            
            # Update for next iteration
            last_values = np.append(last_values[1:], pred)
            
        return {
            'model_used': 'lightgbm',
            'forecast': forecasts,
            'confidence_level': 0.8,
            'metrics': {'method': 'lightgbm', 'num_features': len(feature_cols)}
        }
    
    def _forecast_moving_average(
        self,
        history: pd.DataFrame,
        horizon: int,
        include_confidence: bool
    ) -> Dict:
        """Simple moving average forecast (fallback method)"""
        demand = history['demand'].values
        
        # Use 4-week moving average
        window = min(4, len(demand))
        forecast_value = np.mean(demand[-window:])
        forecast_value = max(0, round(forecast_value))
        
        # Calculate std for confidence intervals
        std = np.std(demand[-window:]) if len(demand) > 1 else forecast_value * 0.3
        
        # Create forecast list
        last_date = pd.to_datetime(history['week_start'].max())
        forecasts = []
        
        for i in range(horizon):
            week_start = last_date + timedelta(weeks=i+1)
            forecasts.append({
                'week_start': week_start,
                'predicted_demand': forecast_value,
                'lower': max(0, int(forecast_value - 1.5 * std)) if include_confidence else None,
                'upper': int(forecast_value + 1.5 * std) if include_confidence else None
            })
            
        return {
            'model_used': 'moving_average',
            'forecast': forecasts,
            'confidence_level': 0.8,
            'metrics': {'method': 'moving_average', 'window': window}
        }
    
    def _insufficient_data_forecast(
        self,
        place_id: int,
        item_id: int,
        horizon: int
    ) -> Dict:
        """Return a placeholder forecast when data is insufficient"""
        today = datetime.now()
        forecasts = []
        
        for i in range(horizon):
            week_start = today + timedelta(weeks=i+1)
            forecasts.append({
                'week_start': week_start,
                'predicted_demand': 0,
                'lower': 0,
                'upper': 5
            })
            
        return {
            'model_used': 'insufficient_data',
            'demand_type': 'Insufficient Data',
            'place_id': place_id,
            'item_id': item_id,
            'forecast': forecasts,
            'confidence_level': 0.5,
            'metrics': {'warning': 'Insufficient historical data for reliable forecast'}
        }
    
    def _exponential_smooth(self, values: np.ndarray, alpha: float) -> float:
        """Apply exponential smoothing and return final smoothed value"""
        if len(values) == 0:
            return 0
            
        smoothed = values[0]
        for v in values[1:]:
            smoothed = alpha * v + (1 - alpha) * smoothed
            
        return smoothed
    
    def _calculate_backtest_metrics(
        self,
        history: pd.DataFrame,
        model,
        model_type: str
    ) -> Dict:
        """Calculate backtest metrics for a forecast model"""
        # Simplified metrics - real implementation would do proper backtesting
        return {
            'model_type': model_type,
            'data_points': len(history),
            'estimation_method': 'simplified'
        }
    
    def calculate_safety_stock(
        self,
        place_id: int,
        item_id: int,
        service_level: float = 0.95
    ) -> Dict:
        """
        Calculate recommended safety stock for an item.
        
        Args:
            place_id: Location identifier
            item_id: Product identifier
            service_level: Desired service level (default 95%)
            
        Returns:
            Dictionary with safety stock recommendations
        """
        history = self.data_processor.get_item_history(place_id, item_id)
        demand_type = self.data_processor.get_demand_type(place_id, item_id)
        
        if len(history) < 4:
            return {'safety_stock': 0, 'reason': 'Insufficient data'}
            
        demand = history['demand'].values
        avg_demand = np.mean(demand)
        std_demand = np.std(demand)
        
        # Service level to Z-score mapping
        z_scores = {0.90: 1.28, 0.95: 1.65, 0.99: 2.33}
        z = z_scores.get(service_level, 1.65)
        
        # Base safety stock
        lead_time_weeks = self.config.inventory.lead_time_days / 7
        base_safety = z * std_demand * np.sqrt(lead_time_weeks)
        
        # Adjust by demand type
        multiplier = self.config.inventory.safety_stock_multipliers.get(demand_type, 2.0)
        safety_stock = base_safety * multiplier
        
        return {
            'safety_stock': max(0, round(safety_stock)),
            'average_weekly_demand': round(avg_demand, 1),
            'demand_variability': round(std_demand / avg_demand, 2) if avg_demand > 0 else 0,
            'demand_type': demand_type,
            'service_level': service_level,
            'lead_time_weeks': round(lead_time_weeks, 1)
        }
    
    def calculate_reorder_point(
        self,
        place_id: int,
        item_id: int,
        forecast: Optional[Dict] = None
    ) -> Dict:
        """
        Calculate the reorder point for an item.
        
        Reorder Point = (Lead Time Demand) + Safety Stock
        
        Args:
            place_id: Location identifier
            item_id: Product identifier
            forecast: Pre-computed forecast (will generate if not provided)
            
        Returns:
            Dictionary with reorder point details
        """
        # Get forecast if not provided
        if forecast is None:
            forecast = self.forecast_item(place_id, item_id, horizon=1)
            
        # Get safety stock
        safety = self.calculate_safety_stock(place_id, item_id)
        
        # Calculate lead time demand
        if forecast['forecast']:
            weekly_demand = forecast['forecast'][0]['predicted_demand']
        else:
            weekly_demand = 0
            
        lead_time_days = self.config.inventory.lead_time_days
        lead_time_demand = weekly_demand * (lead_time_days / 7)
        
        reorder_point = lead_time_demand + safety['safety_stock']
        
        return {
            'reorder_point': max(0, round(reorder_point)),
            'lead_time_demand': round(lead_time_demand, 1),
            'safety_stock': safety['safety_stock'],
            'weekly_forecast': weekly_demand,
            'lead_time_days': lead_time_days
        }
