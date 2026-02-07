"""
Real Data Loader for FreshFlow AI Dashboard
============================================

This module replaces mock_data.py with real data from CSV files.
It loads, transforms, and caches data for the Streamlit dashboard.

Data Sources:
- data/dim_menu_items.csv: Product catalog
- data/fct_order_items.csv: Order line items (~2M rows)
- outputs/decision_engine_recommendations.csv: Pre-computed recommendations
- outputs/analytics/analytics_demand_per_sku.csv: Demand analytics
- outputs/analytics/analytics_stock_levels.csv: Stock levels

Forecasting:
- Uses Facebook Prophet for store-level demand forecasting
- Captures weekly seasonality (Friday +39%, Sunday -26%)
- Includes confidence intervals
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import streamlit as st
import json
import warnings

warnings.filterwarnings('ignore')

# Prophet lazy loading for faster startup
_prophet_model = None
_prophet_forecast = None
_prophet_seasonality = None

# Base paths
SRC_DIR = Path(__file__).parent.parent
PROJECT_ROOT = SRC_DIR.parent  # Go up to root level
DATA_DIR = PROJECT_ROOT / "data"  # Raw CSV data
DATA_ANALYSIS_DIR = PROJECT_ROOT / "docs" / "data_analysis" / "data"  # Analyzed parquet data
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
ANALYTICS_DIR = OUTPUTS_DIR / "analytics"


# ============================================================
# DATA LOADING FUNCTIONS
# ============================================================

@st.cache_data(ttl=600)  # Cache for 10 minutes
def load_menu_items() -> pd.DataFrame:
    """
    Load menu items (products) from dim_menu_items.csv
    
    Returns DataFrame with columns:
    - id, title, price, section_id, type, status, rating, purchases
    """
    file_path = DATA_DIR / "dim_menu_items.csv"
    
    df = pd.read_csv(
        file_path,
        usecols=['id', 'section_id', 'title', 'type', 'status', 'rating', 'purchases', 'price'],
        dtype={
            'id': 'int64',
            'section_id': 'int64',
            'title': 'str',
            'type': 'str',
            'status': 'str',
            'rating': 'float64',
            'purchases': 'float64',
            'price': 'float64'
        }
    )
    
    # Clean up: fill missing values
    df['title'] = df['title'].fillna('Unknown Item')
    df['price'] = df['price'].fillna(0)
    df['rating'] = df['rating'].fillna(0)
    df['purchases'] = df['purchases'].fillna(0)
    df['type'] = df['type'].fillna('Normal')
    df['status'] = df['status'].fillna('Active')
    
    return df


@st.cache_data(ttl=600)
def load_order_items_aggregated() -> pd.DataFrame:
    """
    Load and aggregate order items from fct_order_items.csv
    
    Optimized for memory efficiency:
    - Only loads necessary columns
    - Reads in chunks for large files
    - Pre-aggregates by item_id and date
    
    Returns DataFrame with columns:
    - item_id, date, daily_quantity, daily_revenue
    """
    file_path = DATA_DIR / "fct_order_items.csv"
    
    # Define columns we need
    usecols = ['item_id', 'created', 'quantity', 'price']
    
    # Read in chunks for memory efficiency
    chunk_size = 100_000
    chunks = []
    
    try:
        for chunk in pd.read_csv(
            file_path,
            usecols=usecols,
            dtype={
                'item_id': 'float64',  # May have nulls
                'quantity': 'float64',
                'price': 'float64',
                'created': 'int64'
            },
            chunksize=chunk_size
        ):
            # Drop rows with null item_id
            chunk = chunk.dropna(subset=['item_id'])
            chunk['item_id'] = chunk['item_id'].astype('int64')
            
            # Convert UNIX timestamp (seconds) to date
            chunk['date'] = pd.to_datetime(chunk['created'], unit='s').dt.date
            
            # Fill missing quantities/prices
            chunk['quantity'] = chunk['quantity'].fillna(1)
            chunk['price'] = chunk['price'].fillna(0)
            
            # Aggregate by item_id and date within chunk
            agg = chunk.groupby(['item_id', 'date']).agg({
                'quantity': 'sum',
                'price': 'sum'
            }).reset_index()
            
            chunks.append(agg)
        
        # Combine all chunks and re-aggregate
        if chunks:
            combined = pd.concat(chunks, ignore_index=True)
            result = combined.groupby(['item_id', 'date']).agg({
                'quantity': 'sum',
                'price': 'sum'
            }).reset_index()
            result.columns = ['item_id', 'date', 'daily_quantity', 'daily_revenue']
            return result
        else:
            return pd.DataFrame(columns=['item_id', 'date', 'daily_quantity', 'daily_revenue'])
            
    except Exception as e:
        st.warning(f"Error loading order items: {e}")
        return pd.DataFrame(columns=['item_id', 'date', 'daily_quantity', 'daily_revenue'])


@st.cache_data(ttl=600)
def load_recommendations() -> pd.DataFrame:
    """
    Load pre-computed recommendations from decision_engine_recommendations.csv
    
    Returns DataFrame with recommendation data including:
    - item_id, recommendation_type, risk_level, avg_daily_demand, etc.
    """
    file_path = OUTPUTS_DIR / "decision_engine_recommendations.csv"
    
    try:
        df = pd.read_csv(file_path)
        
        # Convert item_id to int where possible
        df['item_id'] = pd.to_numeric(df['item_id'], errors='coerce')
        df = df.dropna(subset=['item_id'])
        df['item_id'] = df['item_id'].astype('int64')
        
        # Fill missing values
        df['risk_level'] = df['risk_level'].fillna('low')
        df['risk_level'] = df['risk_level'].str.upper()
        df['avg_daily_demand'] = pd.to_numeric(df['avg_daily_demand'], errors='coerce').fillna(0)
        df['days_since_last_sale'] = pd.to_numeric(df['days_since_last_sale'], errors='coerce').fillna(0)
        df['current_stock'] = pd.to_numeric(df['current_stock'], errors='coerce').fillna(0)
        df['confidence'] = df['confidence'].fillna('medium')
        df['expiry_risk_score'] = pd.to_numeric(df['expiry_risk_score'], errors='coerce').fillna(0)
        
        return df
        
    except Exception as e:
        st.warning(f"Error loading recommendations: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=600)
def load_demand_analytics() -> pd.DataFrame:
    """
    Load demand analytics from analytics_demand_per_sku.csv
    """
    file_path = ANALYTICS_DIR / "analytics_demand_per_sku.csv"
    
    try:
        df = pd.read_csv(file_path)
        df['item_id'] = pd.to_numeric(df['item_id'], errors='coerce')
        df = df.dropna(subset=['item_id'])
        df['item_id'] = df['item_id'].astype('int64')
        df['period'] = pd.to_datetime(df['period'])
        return df
    except Exception as e:
        st.warning(f"Error loading demand analytics: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=600)
def load_stock_levels() -> pd.DataFrame:
    """
    Load stock levels from analytics_stock_levels.csv
    """
    file_path = ANALYTICS_DIR / "analytics_stock_levels.csv"
    
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.warning(f"Error loading stock levels: {e}")
        return pd.DataFrame()


# ============================================================
# DATA TRANSFORMATION FUNCTIONS
# ============================================================

def calculate_item_metrics(menu_items: pd.DataFrame, order_items: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate demand metrics for each menu item based on order history.
    
    Metrics calculated:
    - total_quantity_sold: All-time quantity
    - avg_daily_demand: Average daily sales
    - days_since_last_sale: Days since most recent order
    - sale_frequency: Average days between sales
    """
    today = datetime.now().date()
    
    if order_items.empty:
        # Return menu items with default metrics
        metrics = menu_items.copy()
        metrics['total_quantity_sold'] = 0
        metrics['avg_daily_demand'] = 0
        metrics['days_since_last_sale'] = 999
        metrics['last_sale_date'] = None
        return metrics
    
    # Aggregate order data by item
    item_agg = order_items.groupby('item_id').agg({
        'daily_quantity': 'sum',
        'date': ['min', 'max', 'count']
    })
    item_agg.columns = ['total_quantity', 'first_sale', 'last_sale', 'sale_days']
    item_agg = item_agg.reset_index()
    
    # Calculate average daily demand
    item_agg['days_active'] = item_agg.apply(
        lambda row: max(1, (row['last_sale'] - row['first_sale']).days + 1)
        if pd.notna(row['last_sale']) else 1,
        axis=1
    )
    item_agg['avg_daily_demand'] = item_agg['total_quantity'] / item_agg['days_active']
    
    # Days since last sale
    item_agg['days_since_last_sale'] = item_agg['last_sale'].apply(
        lambda x: (today - x).days if pd.notna(x) else 999
    )
    
    # Merge with menu items
    metrics = menu_items.merge(
        item_agg[['item_id', 'total_quantity', 'avg_daily_demand', 'days_since_last_sale', 'last_sale']],
        left_on='id',
        right_on='item_id',
        how='left'
    )
    
    # Fill missing values for items with no orders
    metrics['total_quantity'] = metrics['total_quantity'].fillna(0)
    metrics['avg_daily_demand'] = metrics['avg_daily_demand'].fillna(0)
    metrics['days_since_last_sale'] = metrics['days_since_last_sale'].fillna(999)
    
    # Rename for consistency
    metrics = metrics.rename(columns={
        'total_quantity': 'total_quantity_sold',
        'last_sale': 'last_sale_date'
    })
    
    return metrics


def calculate_health_scores(item_metrics: pd.DataFrame, recommendations: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate health scores for each item based on various metrics.
    
    Health score components:
    - Recency: How recently the item sold (40% weight)
    - Demand: Average daily demand level (30% weight)
    - Risk: From decision engine recommendations (30% weight)
    
    Returns DataFrame with health_score column (0-100)
    """
    df = item_metrics.copy()
    
    # Recency score (0-100): Lower days_since_last_sale = higher score
    # 0 days = 100, 30+ days = 0
    df['recency_score'] = 100 - np.minimum(df['days_since_last_sale'] / 30 * 100, 100)
    
    # Demand score (0-100): Based on avg_daily_demand
    # Normalize to 0-100 range
    max_demand = df['avg_daily_demand'].max() if df['avg_daily_demand'].max() > 0 else 1
    df['demand_score'] = (df['avg_daily_demand'] / max_demand * 100).clip(0, 100)
    
    # Risk score from recommendations
    if not recommendations.empty:
        # Get unique item risk levels
        risk_mapping = {'CRITICAL': 0, 'HIGH': 25, 'MEDIUM': 50, 'LOW': 100}
        
        # Get the most severe risk level per item
        item_risks = recommendations.groupby('item_id')['risk_level'].first().reset_index()
        item_risks['risk_score'] = item_risks['risk_level'].map(risk_mapping).fillna(75)
        
        df = df.merge(item_risks[['item_id', 'risk_score']], left_on='id', right_on='item_id', how='left')
        df['risk_score'] = df['risk_score'].fillna(75)  # Default to medium-low risk
    else:
        df['risk_score'] = 75
    
    # Calculate weighted health score
    df['health_score'] = (
        df['recency_score'] * 0.40 +
        df['demand_score'] * 0.30 +
        df['risk_score'] * 0.30
    ).round(0).astype(int).clip(0, 100)
    
    return df


def assign_risk_levels(df: pd.DataFrame, recommendations: pd.DataFrame) -> pd.DataFrame:
    """
    Assign risk levels to items based on recommendations and metrics.
    """
    result = df.copy()
    
    if not recommendations.empty:
        # Get risk levels from recommendations
        item_risks = recommendations.groupby('item_id')['risk_level'].first().reset_index()
        result = result.merge(item_risks, left_on='id', right_on='item_id', how='left', suffixes=('', '_rec'))
        result['risk_level'] = result['risk_level'].fillna('LOW')
    else:
        # Derive risk from health score
        conditions = [
            result['health_score'] <= 25,
            result['health_score'] <= 50,
            result['health_score'] <= 75
        ]
        choices = ['CRITICAL', 'HIGH', 'MEDIUM']
        result['risk_level'] = np.select(conditions, choices, default='LOW')
    
    return result


def generate_recommendations_from_data(
    item_metrics: pd.DataFrame, 
    recommendations_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate dashboard-ready recommendations by combining item metrics
    with pre-computed decision engine recommendations.
    """
    today = datetime.now()
    recs = []
    
    for _, item in item_metrics.iterrows():
        item_id = item['id']
        product_name = item['title']
        price = item['price']
        health_score = item['health_score']
        risk_level = item.get('risk_level', 'LOW')
        days_since_sale = item['days_since_last_sale']
        avg_demand = item['avg_daily_demand']
        stock = item.get('current_stock', 0)
        
        # Get recommendation details from decision engine
        item_recs = recommendations_df[recommendations_df['item_id'] == item_id]
        
        # Determine action based on risk and recommendation
        if risk_level == 'CRITICAL':
            action = 'DISCOUNT'
            urgency = 'URGENT'
            discount_pct = 30
            
            if days_since_sale > 30:
                explanation = f"No sales in {int(days_since_sale)} days. Consider 30% markdown to clear inventory."
                why = f"Product hasn't sold in {int(days_since_sale)} days. Deep discount may recover some value."
                impact = f"Potential to recover 70% of remaining inventory value"
            else:
                explanation = f"Critical risk level detected. Immediate action needed."
                why = f"Multiple risk factors identified by decision engine."
                impact = "Reduce waste risk significantly"
                
        elif risk_level == 'HIGH':
            if avg_demand < 1:
                action = 'DISCOUNT'
                urgency = 'HIGH'
                discount_pct = 20
                explanation = f"Low demand ({avg_demand:.1f} units/day). Consider 20% discount."
                why = f"Demand is below threshold. Promotional pricing may boost sales."
                impact = "Expected 50% increase in sales velocity"
            else:
                action = 'MONITOR'
                urgency = 'HIGH'
                discount_pct = None
                explanation = f"Elevated risk detected. Monitor closely."
                why = f"Risk score indicates attention needed."
                impact = "Prevent potential stockout or waste"
                
        elif risk_level == 'MEDIUM':
            if stock > avg_demand * 14:  # More than 2 weeks of stock
                action = 'REDUCE ORDER'
                urgency = 'MEDIUM'
                discount_pct = None
                explanation = f"Stock levels high relative to demand."
                why = f"Current stock exceeds 2 weeks of forecasted demand."
                impact = "Optimize inventory carrying costs"
            else:
                action = 'MAINTAIN'
                urgency = 'LOW'
                discount_pct = None
                explanation = f"Inventory levels are balanced."
                why = "Stock aligns with expected demand patterns."
                impact = "Continue current operations"
        else:
            action = 'MAINTAIN'
            urgency = 'LOW'
            discount_pct = None
            explanation = f"Healthy inventory status. Average {avg_demand:.1f} units/day."
            why = "All metrics within normal ranges."
            impact = "No action required"
        
        # Get expiry info if available
        expiry_score = item_recs['expiry_risk_score'].max() if not item_recs.empty and 'expiry_risk_score' in item_recs.columns else 0
        days_to_expiry = max(1, int(30 - expiry_score / 3)) if expiry_score > 0 else 30
        
        # Calculate waste risk
        waste_risk = min(1.0, expiry_score / 100) if expiry_score > 0 else max(0, (100 - health_score) / 200)
        
        # Calculate potential waste value
        unit_cost = price * 0.6 if price else 0  # Assume 40% margin
        potential_waste = round(stock * unit_cost * waste_risk, 2)
        
        # Calculate ROI / Savings for each action
        if action == 'DISCOUNT':
            # Discount recovers (100 - discount_pct)% of inventory value that would otherwise waste
            recovery_rate = (100 - (discount_pct or 0)) / 100
            estimated_savings = round(potential_waste * recovery_rate * 0.8, 2)  # 80% sell-through
            roi_explanation = f"Save ~${estimated_savings:.0f} by recovering {recovery_rate*100:.0f}% of at-risk value"
        elif action == 'REDUCE ORDER':
            # Reducing order saves carrying costs (est 2% of inventory value per week)
            excess_weeks = max(0, (stock - avg_demand * 7) / (avg_demand * 7)) if avg_demand > 0 else 0
            carrying_cost = stock * unit_cost * 0.02 * excess_weeks
            estimated_savings = round(carrying_cost, 2)
            roi_explanation = f"Save ~${estimated_savings:.0f} in carrying costs"
        elif action == 'REORDER':
            # Preventing stockout saves lost margins (est 40% of missed sales)
            stockout_days = max(0, 7 - (stock / avg_demand if avg_demand > 0 else 7))
            lost_margin = stockout_days * avg_demand * price * 0.4 if price else 0
            estimated_savings = round(lost_margin, 2)
            roi_explanation = f"Prevent ~${estimated_savings:.0f} in lost sales"
        else:
            estimated_savings = 0
            roi_explanation = "No immediate ROI - maintaining healthy status"
        
        recs.append({
            'product_id': item_id,
            'product_name': product_name,
            'category': item.get('type', 'Normal'),
            'action': action,
            'urgency': urgency,
            'discount_percent': discount_pct,
            'order_quantity': None,
            'explanation': explanation,
            'reasoning': why,
            'expected_impact': impact,
            'risk_level': risk_level,
            'health_score': health_score,
            'days_to_expiry': days_to_expiry,
            'stock_on_hand': int(stock),
            'waste_risk': round(waste_risk, 2),
            'potential_waste_value': potential_waste,
            'avg_daily_demand': round(avg_demand, 1),
            'days_since_last_sale': int(days_since_sale),
            'estimated_savings': estimated_savings,
            'roi_explanation': roi_explanation
        })
    
    return pd.DataFrame(recs)


def generate_forecasts_from_history(
    item_metrics: pd.DataFrame,
    order_items: pd.DataFrame
) -> Dict[int, Dict[str, Any]]:
    """
    Generate demand forecasts based on actual historical order data.
    """
    today = datetime.now()
    forecasts = {}
    
    for _, item in item_metrics.iterrows():
        item_id = item['id']
        avg_demand = item.get('avg_daily_demand', 0)
        
        # Get historical data for this item
        item_orders = order_items[order_items['item_id'] == item_id].copy()
        
        if not item_orders.empty:
            # Get last 14 days of actual data
            item_orders['date'] = pd.to_datetime(item_orders['date'])
            item_orders = item_orders.sort_values('date')
            
            # Take last 30 data points for historical
            recent = item_orders.tail(30)
            
            historical_dates = recent['date'].dt.strftime('%Y-%m-%d').tolist()[-14:]
            historical_demand = recent['daily_quantity'].tolist()[-14:]
            
            # Pad if less than 14 points
            while len(historical_dates) < 14:
                historical_dates.insert(0, (today - timedelta(days=len(historical_dates)+1)).strftime('%Y-%m-%d'))
                historical_demand.insert(0, int(avg_demand))
        else:
            # Generate synthetic historical based on avg_demand
            historical_dates = [(today - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(14, 0, -1)]
            historical_demand = [max(0, int(avg_demand + np.random.normal(0, max(1, avg_demand * 0.2)))) for _ in range(14)]
        
        # Generate 7-day forecast based on average demand
        forecast_dates = [(today + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 8)]
        base_forecast = [round(avg_demand, 1) for _ in range(7)]
        
        # Simple adjustments for weekends
        adjustment_factors = []
        for i in range(1, 8):
            day = (today + timedelta(days=i)).weekday()
            if day >= 5:  # Weekend
                adjustment_factors.append(1.15)
            else:
                adjustment_factors.append(1.0)
        
        adjusted_forecast = [round(base * adj, 1) for base, adj in zip(base_forecast, adjustment_factors)]
        
        # Confidence intervals
        conf_lower = [max(0, round(f * 0.7, 1)) for f in adjusted_forecast]
        conf_upper = [round(f * 1.3, 1) for f in adjusted_forecast]
        
        forecasts[item_id] = {
            'historical_dates': historical_dates,
            'historical_demand': historical_demand,
            'forecast_dates': forecast_dates,
            'base_forecast': base_forecast,
            'adjusted_forecast': adjusted_forecast,
            'conf_lower': conf_lower,
            'conf_upper': conf_upper,
            'adjustment_events': [
                {'date': d, 'event': 'Weekend', 'impact': '+15%'}
                for d, adj in zip(forecast_dates, adjustment_factors) if adj > 1.0
            ],
            'model_confidence': round(min(0.95, 0.6 + (len(item_orders) / 100) * 0.35), 2) if not item_orders.empty else 0.5
        }
    
    return forecasts


def generate_context_factors() -> Dict[str, Any]:
    """
    Generate context factors for demand adjustments.
    Based on current date and typical business patterns.
    """
    today = datetime.now()
    
    # Determine season
    month = today.month
    if month in [12, 1, 2]:
        season = "Winter"
    elif month in [3, 4, 5]:
        season = "Spring"
    elif month in [6, 7, 8]:
        season = "Summer"
    else:
        season = "Autumn"
    
    return {
        'events': [
            {
                'name': 'Weekend Rush',
                'type': 'REGULAR',
                'date': (today + timedelta(days=(5 - today.weekday()) % 7)).strftime('%Y-%m-%d'),
                'impact': '+15% demand uplift',
                'affected_categories': ['All Categories']
            }
        ],
        'weather': {
            'today': {'condition': 'Normal', 'temp_high': 50, 'impact': 'Normal demand'},
            'tomorrow': {'condition': 'Normal', 'temp_high': 52, 'impact': 'Normal demand'},
            'day_3': {'condition': 'Normal', 'temp_high': 48, 'impact': 'Normal demand'},
            'day_4': {'condition': 'Normal', 'temp_high': 45, 'impact': 'Normal demand'},
            'day_5': {'condition': 'Normal', 'temp_high': 50, 'impact': 'Normal demand'},
            'day_6': {'condition': 'Normal', 'temp_high': 55, 'impact': 'Normal demand'},
            'day_7': {'condition': 'Normal', 'temp_high': 52, 'impact': 'Normal demand'},
        },
        'seasonality': {
            'current_season': season,
            'trend': f'{season} seasonal patterns in effect',
            'yoy_growth': '+2.5%'
        }
    }


# ============================================================
# PROPHET FORECASTING
# ============================================================

@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_prophet_forecast(order_items: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generate store-level demand forecast using Prophet.
    
    This captures:
    - Weekly seasonality (Friday peak, Sunday low)
    - Trend patterns
    - Confidence intervals
    
    Returns:
    - forecast DataFrame with predictions
    - seasonality dict with pattern explanations
    """
    try:
        from prophet import Prophet
    except ImportError:
        # Fallback if Prophet not installed
        return _fallback_forecast(order_items)
    
    # Aggregate to daily store-level demand
    daily = order_items.groupby('date').agg({
        'daily_quantity': 'sum'
    }).reset_index()
    
    daily['date'] = pd.to_datetime(daily['date'])
    daily = daily.sort_values('date')
    daily = daily.rename(columns={'date': 'ds', 'daily_quantity': 'y'})
    
    # Need at least 14 days of data
    if len(daily) < 14:
        return _fallback_forecast(order_items)
    
    # Suppress Prophet logging
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Initialize and fit Prophet
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            interval_width=0.95,
            changepoint_prior_scale=0.05
        )
        
        model.fit(daily)
        
        # Generate forecast for next 14 days
        future = model.make_future_dataframe(periods=14, freq='D')
        forecast = model.predict(future)
    
    # Extract seasonality effects
    forecast['day_of_week'] = forecast['ds'].dt.dayofweek
    weekly_effects = forecast.groupby('day_of_week')['weekly'].mean()
    
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    seasonality = {
        'weekly': {},
        'peak_day': None,
        'low_day': None,
        'method': 'prophet',
        'explanation': ''
    }
    
    for i, name in enumerate(day_names):
        if i in weekly_effects.index:
            pct = float(weekly_effects[i] * 100)  # Convert to native float
            seasonality['weekly'][name] = round(pct, 1)
    
    if len(weekly_effects) > 0:
        seasonality['peak_day'] = day_names[int(weekly_effects.idxmax())]
        seasonality['low_day'] = day_names[int(weekly_effects.idxmin())]
        
        peak_pct = seasonality['weekly'].get(seasonality['peak_day'], 0)
        low_pct = seasonality['weekly'].get(seasonality['low_day'], 0)
        
        seasonality['explanation'] = (
            f"Demand follows a strong weekly pattern. "
            f"{seasonality['peak_day']} is the busiest day ({'+' if peak_pct > 0 else ''}{peak_pct:.0f}% vs average), "
            f"while {seasonality['low_day']} is slowest ({'+' if low_pct > 0 else ''}{low_pct:.0f}%)."
        )
    
    # Create a pickle-friendly forecast DataFrame with only essential columns
    forecast_clean = pd.DataFrame({
        'ds': forecast['ds'].tolist(),
        'yhat': forecast['yhat'].astype(float).tolist(),
        'yhat_lower': forecast['yhat_lower'].astype(float).tolist(),
        'yhat_upper': forecast['yhat_upper'].astype(float).tolist()
    })
    forecast_clean['ds'] = pd.to_datetime(forecast_clean['ds'])
    
    return forecast_clean, seasonality


def _fallback_forecast(order_items: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fallback forecast when Prophet is not available.
    Uses simple moving average with day-of-week adjustments.
    """
    today = datetime.now()
    
    # Calculate daily totals
    daily = order_items.groupby('date').agg({
        'daily_quantity': 'sum'
    }).reset_index()
    
    daily['date'] = pd.to_datetime(daily['date'])
    daily = daily.sort_values('date')
    
    # Calculate day-of-week patterns from historical data
    daily['day_of_week'] = daily['date'].dt.dayofweek
    dow_avg = daily.groupby('day_of_week')['daily_quantity'].mean()
    overall_avg = daily['daily_quantity'].mean()
    
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly = {}
    for i, name in enumerate(day_names):
        if i in dow_avg.index and overall_avg > 0:
            pct = ((dow_avg[i] / overall_avg) - 1) * 100
            weekly[name] = round(pct, 1)
        else:
            weekly[name] = 0
    
    # Find peak and low days
    if weekly:
        peak_day = max(weekly.items(), key=lambda x: x[1])[0]
        low_day = min(weekly.items(), key=lambda x: x[1])[0]
    else:
        peak_day, low_day = 'Friday', 'Sunday'
    
    # Generate simple forecast
    future_dates = pd.date_range(start=today, periods=14, freq='D')
    forecast_values = []
    
    for date in future_dates:
        dow = date.dayofweek
        adjustment = (weekly.get(day_names[dow], 0) / 100) + 1
        forecast_values.append(overall_avg * adjustment)
    
    forecast = pd.DataFrame({
        'ds': future_dates,
        'yhat': forecast_values,
        'yhat_lower': [v * 0.75 for v in forecast_values],
        'yhat_upper': [v * 1.25 for v in forecast_values]
    })
    
    seasonality = {
        'weekly': weekly,
        'peak_day': peak_day,
        'low_day': low_day,
        'method': 'moving_average',
        'explanation': f"Based on historical patterns. {peak_day} is busiest, {low_day} is slowest."
    }
    
    return forecast, seasonality


# ============================================================
# MAIN DATA LOADING FUNCTION
# ============================================================

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_dashboard_data() -> Dict[str, Any]:
    """
    Main function to load all dashboard data from real CSV files.
    
    Returns a dictionary containing:
    - products: Product catalog
    - inventory: Current inventory status with health scores
    - forecasts: Demand forecasts based on historical data
    - recommendations: Actionable recommendations
    - context: Event and weather factors
    - summary: Executive summary statistics
    """
    # Load raw data
    menu_items = load_menu_items()
    order_items = load_order_items_aggregated()
    recommendations_raw = load_recommendations()
    
    # Generate Prophet forecast (store-level)
    prophet_forecast, prophet_seasonality = get_prophet_forecast(order_items)
    
    # Calculate metrics
    item_metrics = calculate_item_metrics(menu_items, order_items)
    item_metrics = calculate_health_scores(item_metrics, recommendations_raw)
    item_metrics = assign_risk_levels(item_metrics, recommendations_raw)
    
    # Add stock information from recommendations
    if not recommendations_raw.empty:
        stock_info = recommendations_raw.groupby('item_id')['current_stock'].first().reset_index()
        item_metrics = item_metrics.merge(stock_info, left_on='id', right_on='item_id', how='left', suffixes=('', '_stock'))
        item_metrics['current_stock'] = item_metrics['current_stock'].fillna(0)
    else:
        item_metrics['current_stock'] = 0
    
    # Generate derived data
    recommendations = generate_recommendations_from_data(item_metrics, recommendations_raw)
    forecasts = generate_forecasts_from_history(item_metrics, order_items)
    context = generate_context_factors()
    
    # Build inventory DataFrame (for compatibility with existing app.py)
    inventory = item_metrics.rename(columns={
        'id': 'product_id',
        'title': 'product_name',
        'type': 'category'
    })
    
    # Add required columns
    inventory['stock_on_hand'] = inventory['current_stock'].fillna(0).astype(int)
    inventory['forecasted_demand_7d'] = (inventory['avg_daily_demand'] * 7).round(0).astype(int)
    inventory['days_of_stock'] = np.where(
        inventory['avg_daily_demand'] > 0,
        (inventory['stock_on_hand'] / inventory['avg_daily_demand']).round(1),
        0
    )
    
    # Estimate days to expiry based on risk
    risk_to_expiry = {'CRITICAL': 2, 'HIGH': 5, 'MEDIUM': 10, 'LOW': 30}
    inventory['days_to_expiry'] = inventory['risk_level'].map(risk_to_expiry).fillna(30)
    inventory['expiry_date'] = inventory['days_to_expiry'].apply(
        lambda x: (datetime.now() + timedelta(days=int(x))).strftime('%Y-%m-%d')
    )
    
    # Unit cost from price (estimate margin)
    inventory['unit_cost'] = (inventory['price'] * 0.6).fillna(0)  # Assume 40% margin
    
    # Waste risk
    inventory['waste_risk'] = np.where(
        inventory['days_to_expiry'] <= 2, 0.6,
        np.where(inventory['days_to_expiry'] <= 5, 0.3,
        np.where(inventory['days_to_expiry'] <= 10, 0.1, 0.02))
    )
    
    # Ensure we have scenario column for compatibility
    inventory['scenario'] = np.where(
        inventory['risk_level'] == 'CRITICAL', 'critical',
        np.where(inventory['risk_level'] == 'HIGH', 'expiring_soon',
        np.where(inventory['risk_level'] == 'MEDIUM', 'overstock', 'healthy'))
    )
    
    # Calculate summary statistics (convert to native Python types for pickle compatibility)
    summary = {
        'total_products': int(len(inventory)),
        'critical_items': int(len(inventory[inventory['risk_level'] == 'CRITICAL'])),
        'high_risk_items': int(len(inventory[inventory['risk_level'] == 'HIGH'])),
        'medium_risk_items': int(len(inventory[inventory['risk_level'] == 'MEDIUM'])),
        'low_risk_items': int(len(inventory[inventory['risk_level'] == 'LOW'])),
        'total_waste_risk_value': float(round((inventory['waste_risk'] * inventory['unit_cost'] * inventory['stock_on_hand']).sum(), 2)),
        'items_expiring_today': int(len(inventory[inventory['days_to_expiry'] <= 1])),
        'items_expiring_this_week': int(len(inventory[inventory['days_to_expiry'] <= 7])),
        'avg_health_score': float(round(inventory['health_score'].mean(), 1)),
        'items_needing_reorder': int(len(recommendations[recommendations['action'] == 'REORDER'])),
        'items_needing_discount': int(len(recommendations[recommendations['action'] == 'DISCOUNT'])),
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M')
    }
    
    # Build products DataFrame
    products = menu_items.rename(columns={
        'id': 'id',
        'title': 'name',
        'type': 'category',
        'price': 'unit_cost'
    })
    products['shelf_life_days'] = 14  # Default
    
    return {
        'products': products,
        'inventory': inventory,
        'forecasts': forecasts,
        'recommendations': recommendations,
        'context': context,
        'summary': summary,
        'prophet_forecast': prophet_forecast,
        'prophet_seasonality': prophet_seasonality
    }


# Alias for backward compatibility
def get_mock_dashboard_data() -> Dict[str, Any]:
    """Alias for get_dashboard_data() for backward compatibility."""
    return get_dashboard_data()


# For testing
if __name__ == "__main__":
    print("Testing data loader...")
    
    # Test individual loaders
    print("\n1. Loading menu items...")
    menu = load_menu_items()
    print(f"   Loaded {len(menu)} menu items")
    print(f"   Sample: {menu['title'].head(3).tolist()}")
    
    print("\n2. Loading order items (aggregated)...")
    orders = load_order_items_aggregated()
    print(f"   Loaded {len(orders)} aggregated order records")
    
    print("\n3. Loading recommendations...")
    recs = load_recommendations()
    print(f"   Loaded {len(recs)} recommendation records")
    
    print("\n4. Getting full dashboard data...")
    data = get_dashboard_data()
    print(f"   Products: {len(data['products'])}")
    print(f"   Inventory items: {len(data['inventory'])}")
    print(f"   Recommendations: {len(data['recommendations'])}")
    print(f"\n   Risk Distribution:")
    print(data['inventory']['risk_level'].value_counts())
    print(f"\n   Action Distribution:")
    print(data['recommendations']['action'].value_counts())
    print(f"\n   Summary: {data['summary']}")
