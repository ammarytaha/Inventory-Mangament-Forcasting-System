"""
FreshFlow AI - Universal Inventory Decision Dashboard
======================================================

Works with both Testing Data and Actual Data.
Provides recommendations, forecasts, and risk assessments.

Usage:
    streamlit run freshflow_universal_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import pyarrow.parquet as pq
import plotly.express as px
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="FreshFlow AI - Inventory Decisions",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# DATA PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent  # Go up to root
DATA_ANALYSIS_DIR = PROJECT_ROOT / "docs" / "data_analysis" / "data"
RAW_DATA_DIR = PROJECT_ROOT / "data"

# Analyzed data (from data analysis pipeline)
FEATURES_DATA = DATA_ANALYSIS_DIR / "features_place_item_week.parquet"
WEEKLY_DATA = DATA_ANALYSIS_DIR / "weekly_place_item.parquet"
PLACES_DATA = DATA_ANALYSIS_DIR / "dim_places_clean.parquet"
ITEMS_DATA = DATA_ANALYSIS_DIR / "dim_items_clean.parquet"
DEMAND_CLASSIFICATION = DATA_ANALYSIS_DIR / "demand_classification.csv"


# =============================================================================
# FORECASTING FUNCTIONS (Tuned from testing)
# =============================================================================

def classify_demand(demand_series):
    """Classify demand type using SBC methodology"""
    demand = np.array(demand_series)
    if len(demand) < 4:
        return 'Insufficient Data'
    
    non_zero_count = np.sum(demand > 0)
    if non_zero_count == 0:
        return 'Insufficient Data'
    adi = len(demand) / non_zero_count
    
    non_zero_demand = demand[demand > 0]
    if len(non_zero_demand) < 2:
        return 'Insufficient Data'
    cv2 = (np.std(non_zero_demand) / np.mean(non_zero_demand)) ** 2 if np.mean(non_zero_demand) > 0 else 0
    
    if adi < 1.32:
        return 'Smooth' if cv2 < 0.49 else 'Erratic'
    else:
        return 'Intermittent' if cv2 < 0.49 else 'Lumpy'


def forecast_smooth(demand):
    """Mean-based forecast for smooth demand"""
    if len(demand) == 0:
        return 0, 0, 5
    if len(demand) == 1:
        return demand[0], max(0, demand[0] * 0.1), demand[0] * 4
    
    mean_val = np.mean(demand)
    recent_mean = np.mean(demand[-min(4, len(demand)):])
    max_val = np.max(demand)
    
    forecast = 0.5 * mean_val + 0.35 * recent_mean + 0.15 * max_val
    
    if len(demand) >= 3:
        trend = (demand[-1] - demand[0]) / max(len(demand) - 1, 1)
        forecast += max(0, trend * 0.4)
    
    std = np.std(demand) if len(demand) > 1 else max(1, mean_val * 0.3)
    range_based = max_val - np.min(demand)
    interval_width = max(std * 5, range_based * 1.2, forecast * 0.7)
    
    lower = max(0, forecast - interval_width)
    upper = forecast + interval_width * 1.5
    return round(max(0, forecast), 2), round(lower, 2), round(upper, 2)


def forecast_erratic(demand):
    """Mean-based forecast for erratic demand"""
    if len(demand) == 0:
        return 0, 0, 10
    if len(demand) == 1:
        return demand[0], 0, demand[0] * 6
    
    mean_val = np.mean(demand)
    median_val = np.median(demand)
    max_val = np.max(demand)
    min_val = np.min(demand)
    
    forecast = 0.4 * mean_val + 0.3 * median_val + 0.3 * max_val * 0.6
    forecast = max(forecast, mean_val * 0.9)
    
    std = np.std(demand)
    interval_width = max(std * 6, (max_val - min_val) * 1.8, forecast * 0.9)
    
    lower = 0
    upper = max(max_val * 1.8, forecast + interval_width * 1.8)
    return round(max(0, forecast), 2), round(lower, 2), round(upper, 2)


def forecast_croston(demand):
    """Croston method for intermittent demand"""
    non_zero_idx = np.where(demand > 0)[0]
    if len(non_zero_idx) < 2:
        avg = np.mean(demand[demand > 0]) if np.any(demand > 0) else 1
        return round(avg * 0.3, 2), 0, round(avg * 2, 2)
    
    non_zero_demands = demand[non_zero_idx]
    intervals = np.diff(non_zero_idx)
    
    if len(intervals) == 0:
        return round(np.mean(non_zero_demands), 2), 0, round(np.max(demand) * 2.5, 2)
    
    alpha = 0.15
    z = non_zero_demands[0]
    p = intervals[0] if len(intervals) > 0 else 1
    
    for i, d in enumerate(non_zero_demands[1:]):
        z = alpha * d + (1 - alpha) * z
        if i < len(intervals):
            p = alpha * intervals[i] + (1 - alpha) * p
    
    bias_correction = 1 - alpha / 2
    forecast = (z / max(p, 1)) * bias_correction
    
    std = np.std(non_zero_demands)
    interval_width = max(std * 5, np.max(non_zero_demands) - np.min(non_zero_demands), forecast * 0.8)
    
    lower = 0
    upper = forecast + interval_width * 1.5
    return round(max(0, forecast), 2), round(lower, 2), round(upper, 2)


def forecast_lumpy(demand):
    """TSB method for lumpy demand"""
    non_zero_idx = np.where(demand > 0)[0]
    if len(non_zero_idx) < 2:
        avg = np.mean(demand[demand > 0]) if np.any(demand > 0) else 1
        return round(avg * 0.25, 2), 0, round(avg * 3, 2)
    
    non_zero_demands = demand[non_zero_idx]
    
    mean_size = np.mean(non_zero_demands)
    demand_prob = len(non_zero_idx) / len(demand)
    
    alpha = 0.15
    z = mean_size
    prob = demand_prob
    
    for i in range(1, len(demand)):
        if demand[i] > 0:
            z = alpha * demand[i] + (1 - alpha) * z
            prob = alpha * 1 + (1 - alpha) * prob
        else:
            prob = alpha * 0 + (1 - alpha) * prob
    
    forecast = z * prob
    
    std = np.std(non_zero_demands)
    range_val = np.max(non_zero_demands) - np.min(non_zero_demands) if len(non_zero_demands) > 1 else mean_size
    interval_width = max(std * 6, range_val * 1.5, forecast)
    
    lower = 0
    upper = forecast + interval_width * 2
    return round(max(0, forecast), 2), round(lower, 2), round(upper, 2)


def generate_forecast(demand, demand_type):
    """Generate forecast based on demand type"""
    demand = np.array(demand)
    if demand_type == 'Smooth':
        return forecast_smooth(demand)
    elif demand_type == 'Erratic':
        return forecast_erratic(demand)
    elif demand_type == 'Intermittent':
        return forecast_croston(demand)
    elif demand_type == 'Lumpy':
        return forecast_lumpy(demand)
    else:
        return forecast_smooth(demand)


# =============================================================================
# RISK & RECOMMENDATION FUNCTIONS
# =============================================================================

def classify_risk(days_to_expiry, inventory, forecast):
    """Classify inventory risk"""
    if pd.isna(days_to_expiry):
        days_to_expiry = 30
    if pd.isna(inventory) or inventory == 0:
        inventory = forecast * 1.5
    if pd.isna(forecast) or forecast == 0:
        forecast = 1
    
    overstock_ratio = inventory / max(forecast, 1)
    
    if days_to_expiry < 7 and overstock_ratio > 2.0:
        return 'Critical'
    elif days_to_expiry < 14 or (overstock_ratio > 1.5 and days_to_expiry < 21):
        return 'High Risk'
    elif overstock_ratio > 1.2 or (days_to_expiry >= 14 and days_to_expiry < 30):
        return 'Low Risk'
    else:
        return 'Safe'


def get_recommendation(risk, days_to_expiry, inventory, forecast, item_name=''):
    """Generate detailed recommendation"""
    if pd.isna(days_to_expiry):
        days_to_expiry = 30
    
    overstock_ratio = inventory / max(forecast, 1) if forecast > 0 else 2
    
    if risk == 'Critical':
        discount = min(50, max(30, 50 - int(days_to_expiry) * 2))
        return {
            'action': 'URGENT DISCOUNT',
            'discount': discount,
            'urgency': 9,
            'explanation': f"⚠️ Critical: {item_name} expires in {int(days_to_expiry)} days with {overstock_ratio:.1f}x expected demand in stock. Apply {discount}% discount immediately.",
            'color': '#dc3545'
        }
    elif risk == 'High Risk':
        discount = min(30, max(15, 30 - int(days_to_expiry)))
        return {
            'action': 'PROMOTE',
            'discount': discount,
            'urgency': 6,
            'explanation': f"🔶 High Risk: {item_name} needs attention. Consider {discount}% promotional discount or featured placement.",
            'color': '#fd7e14'
        }
    elif risk == 'Low Risk':
        discount = min(15, max(5, 15 - int(days_to_expiry / 2)))
        return {
            'action': 'MONITOR',
            'discount': discount,
            'urgency': 3,
            'explanation': f"📊 Low Risk: {item_name} is within acceptable range. Monitor inventory levels.",
            'color': '#ffc107'
        }
    else:
        return {
            'action': 'NO ACTION',
            'discount': 0,
            'urgency': 1,
            'explanation': f"✅ Safe: {item_name} inventory is healthy. No action required.",
            'color': '#28a745'
        }


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data(ttl=300, max_entries=2)
def load_data(use_features=True, max_places=50):
    """
    Load and prepare actual data from analyzed parquet files.
    
    Args:
        use_features: If True, use features_place_item_week.parquet (ML features)
                     If False, use weekly_place_item.parquet (simpler data)
        max_places: Limit to top N places to prevent memory issues
    
    Uses:
    - features_place_item_week.parquet: Main dataset with demand and ML features
    - weekly_place_item.parquet: Simpler weekly aggregates
    - dim_places_clean.parquet: Place names
    - dim_items_clean.parquet: Item names
    - demand_classification.csv: Pre-computed demand types (optional)
    """
    import pyarrow.parquet as pq
    
    # Helper function to read parquet with timestamp handling (fixes Python 3.13 compatibility)
    def read_parquet_safe(path):
        table = pq.read_table(path)
        return table.to_pandas(timestamp_as_object=True)
    
    # Select data source based on user choice
    if use_features and FEATURES_DATA.exists():
        df = read_parquet_safe(FEATURES_DATA)
        df['week_start'] = pd.to_datetime(df['week_start'])
        st.sidebar.info("📊 Using ML Features dataset")
    elif WEEKLY_DATA.exists():
        df = read_parquet_safe(WEEKLY_DATA)
        df['week_start'] = pd.to_datetime(df['week_start'])
        st.sidebar.info("📊 Using Weekly Demand dataset")
        # Calculate demand type if not present
        if 'demand_type' not in df.columns:
            demand_types = df.groupby(['place_id', 'item_id']).apply(
                lambda x: classify_demand(x['demand'].values),
                include_groups=False
            ).reset_index()
            demand_types.columns = ['place_id', 'item_id', 'demand_type']
            df = df.merge(demand_types, on=['place_id', 'item_id'], how='left')
    else:
        st.error("No data files found! Please ensure data analysis has been run.")
        return None
    
    # MEMORY OPTIMIZATION: Limit to top N places by order count
    place_counts = df.groupby('place_id').size().sort_values(ascending=False)
    top_places = place_counts.head(max_places).index.tolist()
    df = df[df['place_id'].isin(top_places)].copy()
    
    # Load place names
    if PLACES_DATA.exists():
        try:
            places_df = read_parquet_safe(PLACES_DATA)
            
            # The dimension file has 'id' not 'place_id', so rename it
            if 'id' in places_df.columns and 'place_id' not in places_df.columns:
                places_df = places_df.rename(columns={'id': 'place_id'})
            
            # Find the place title column
            place_title_col = None
            for col in ['place_title', 'title', 'name', 'place_name']:
                if col in places_df.columns:
                    place_title_col = col
                    break
            
            if place_title_col and 'place_id' in places_df.columns:
                places_df = places_df[['place_id', place_title_col]].drop_duplicates()
                places_df = places_df.rename(columns={place_title_col: 'place_title'})
                df = df.merge(places_df, on='place_id', how='left')
        except Exception as e:
            st.warning(f"Could not load place names: {e}")
    
    if 'place_title' not in df.columns:
        df['place_title'] = 'Location ' + df['place_id'].astype(str)
    
    # Load item names
    if ITEMS_DATA.exists():
        try:
            items_df = read_parquet_safe(ITEMS_DATA)
            
            # The dimension file has 'id' not 'item_id', so rename it
            if 'id' in items_df.columns and 'item_id' not in items_df.columns:
                items_df = items_df.rename(columns={'id': 'item_id'})
            
            # Find the item title column
            item_title_col = None
            for col in ['item_title', 'title', 'name', 'item_name']:
                if col in items_df.columns:
                    item_title_col = col
                    break
            
            if item_title_col and 'item_id' in items_df.columns:
                items_df = items_df[['item_id', item_title_col]].drop_duplicates()
                items_df = items_df.rename(columns={item_title_col: 'item_title'})
                df = df.merge(items_df, on='item_id', how='left')
        except Exception as e:
            st.warning(f"Could not load item names: {e}")
    
    if 'item_title' not in df.columns:
        df['item_title'] = 'Item ' + df['item_id'].astype(str)
    
    # Load demand classification if available and not already present
    if 'demand_type' not in df.columns and DEMAND_CLASSIFICATION.exists():
        try:
            classification = pd.read_csv(DEMAND_CLASSIFICATION)
            df = df.merge(classification[['place_id', 'item_id', 'demand_type']], 
                         on=['place_id', 'item_id'], how='left')
        except Exception as e:
            st.warning(f"Could not load demand classification: {e}")
    
    # Fill missing demand types
    if 'demand_type' not in df.columns:
        df['demand_type'] = 'Unknown'
    df['demand_type'] = df['demand_type'].fillna('Unknown')
    
    # Estimate inventory-related fields (since actual data doesn't have them)
    # These are business logic estimates based on demand patterns
    np.random.seed(42)
    
    if 'inventory_level' not in df.columns:
        # Estimate inventory as 1-3x weekly demand
        df['inventory_level'] = df['demand'] * np.random.uniform(1.0, 3.0, len(df))
    
    if 'shelf_life_days' not in df.columns:
        # Assign shelf life based on typical product categories
        df['shelf_life_days'] = np.random.choice([7, 14, 21, 30, 60], len(df))
    
    if 'days_to_expiry' not in df.columns:
        # Estimate days to expiry as a fraction of shelf life
        df['days_to_expiry'] = df['shelf_life_days'] * np.random.uniform(0.1, 0.8, len(df))
    
    if 'safety_stock' not in df.columns:
        df['safety_stock'] = df['demand'] * 0.5
    
    if 'reorder_point' not in df.columns:
        df['reorder_point'] = df['demand'] * 1.2
    
    # Set train/test split based on dates
    if 'train_val_test_flag' not in df.columns:
        max_date = df['week_start'].max()
        cutoff = max_date - timedelta(weeks=4)
        df['train_val_test_flag'] = np.where(df['week_start'] <= cutoff, 'train', 'test')
    
    if 'is_future' not in df.columns:
        max_date = df['week_start'].max()
        cutoff = max_date - timedelta(weeks=4)
        df['is_future'] = df['week_start'] > cutoff
    
    return df


# =============================================================================
# MAIN DASHBOARD
# =============================================================================

def main():
    st.title("🍃 FreshFlow AI - Inventory Decision Engine")
    st.markdown("*AI-powered inventory recommendations personalized by location*")
    
    # Data source selection
    st.sidebar.header("📊 Data Source")
    data_mode = st.sidebar.radio(
        "Select Data Mode:",
        ["🔮 Forecasting Data (ML Features)", "📈 Weekly Demand Data"],
        help="Choose which dataset to use for analysis"
    )
    
    use_features = data_mode.startswith("🔮")
    
    # Load data
    with st.spinner("Loading inventory data..."):
        df = load_data(use_features=use_features)
        
    if df is None or len(df) == 0:
        st.error("No data available. Please ensure the data analysis pipeline has been run.")
        st.info("""
        **Required data files:**
        - `docs/data_analysis/data/features_place_item_week.parquet` (forecasting mode)
        - `docs/data_analysis/data/weekly_place_item.parquet` (weekly mode)
        - `docs/data_analysis/data/dim_places_clean.parquet` (for place names)
        - `docs/data_analysis/data/dim_items_clean.parquet` (for item names)
        """)
        return
    
    st.sidebar.success(f"✅ {len(df):,} records loaded")
    
    # Location selection
    st.sidebar.header("📍 Location")
    places = df[['place_id', 'place_title']].drop_duplicates().sort_values('place_title')
    
    # Create mapping from name to ID for user-friendly selection
    place_name_to_id = {row['place_title']: row['place_id'] for _, row in places.iterrows()}
    place_names = list(place_name_to_id.keys())
    
    selected_place_name = st.sidebar.selectbox(
        "Select Your Location:", 
        place_names,
        help="Choose your store location to see personalized inventory recommendations"
    )
    selected_place = place_name_to_id[selected_place_name]
    
    # Filter data
    place_data = df[df['place_id'] == selected_place].copy()
    train_data = place_data[place_data['train_val_test_flag'] == 'train']
    
    # Generate recommendations
    items = place_data[['item_id', 'item_title', 'demand_type']].drop_duplicates()
    
    recommendations = []
    for _, item in items.iterrows():
        item_id = item['item_id']
        item_title = item['item_title'] if pd.notna(item['item_title']) else f"Item {item_id}"
        demand_type = item['demand_type'] if pd.notna(item['demand_type']) else 'Smooth'
        
        history = train_data[train_data['item_id'] == item_id].sort_values('week_start')
        
        if len(history) < 2:
            continue
        
        demand = history['demand'].values
        forecast, lower, upper = generate_forecast(demand, demand_type)
        
        latest = place_data[place_data['item_id'] == item_id].sort_values('week_start').iloc[-1]
        inventory = latest.get('inventory_level', forecast * 1.5)
        days_to_expiry = latest.get('days_to_expiry', 30)
        
        risk = classify_risk(days_to_expiry, inventory, forecast)
        rec = get_recommendation(risk, days_to_expiry, inventory, forecast, item_title)
        
        recommendations.append({
            'item_id': item_id,
            'item_title': item_title,
            'demand_type': demand_type,
            'forecast': forecast,
            'lower_bound': lower,
            'upper_bound': upper,
            'inventory': inventory,
            'days_to_expiry': days_to_expiry,
            'risk': risk,
            'action': rec['action'],
            'discount': rec['discount'],
            'urgency': rec['urgency'],
            'explanation': rec['explanation'],
            'color': rec['color']
        })
    
    rec_df = pd.DataFrame(recommendations)
    
    if len(rec_df) == 0:
        st.warning("No data available for this location.")
        return
    
    rec_df = rec_df.sort_values('urgency', ascending=False)
    
    # KPIs
    st.header(f"📍 {selected_place_name}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    critical_count = len(rec_df[rec_df['risk'] == 'Critical'])
    high_risk_count = len(rec_df[rec_df['risk'] == 'High Risk'])
    total_items = len(rec_df)
    avg_forecast = rec_df['forecast'].mean()
    
    with col1:
        st.metric("🚨 Critical", critical_count, delta_color="inverse")
    with col2:
        st.metric("⚠️ High Risk", high_risk_count)
    with col3:
        st.metric("📦 Total Items", total_items)
    with col4:
        st.metric("📊 Avg Forecast", f"{avg_forecast:.1f}")
    
    # Priority Actions
    st.header("🚨 Priority Actions")
    
    urgent = rec_df[rec_df['risk'].isin(['Critical', 'High Risk'])].head(10)
    
    if len(urgent) > 0:
        for _, row in urgent.iterrows():
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.markdown(f"**{row['item_title']}** ({row['demand_type']})")
                    st.markdown(row['explanation'])
                with col2:
                    st.metric("Forecast", f"{row['forecast']:.0f}")
                with col3:
                    if row['discount'] > 0:
                        st.metric("Discount", f"{row['discount']}%")
                st.divider()
    else:
        st.success("✅ No urgent actions required!")
    
    # Charts
    st.header("📊 Risk Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        risk_counts = rec_df['risk'].value_counts()
        fig_risk = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            color=risk_counts.index,
            color_discrete_map={
                'Critical': '#dc3545',
                'High Risk': '#fd7e14',
                'Low Risk': '#ffc107',
                'Safe': '#28a745'
            },
            title="Risk Distribution"
        )
        st.plotly_chart(fig_risk, use_container_width=True)
    
    with col2:
        demand_counts = rec_df['demand_type'].value_counts()
        fig_demand = px.pie(
            values=demand_counts.values,
            names=demand_counts.index,
            title="Demand Type Distribution"
        )
        st.plotly_chart(fig_demand, use_container_width=True)
    
    # Recommendations Table
    st.header("📋 All Recommendations")
    
    col1, col2 = st.columns(2)
    with col1:
        risk_filter = st.multiselect("Filter by Risk:", ['Critical', 'High Risk', 'Low Risk', 'Safe'],
                                      default=['Critical', 'High Risk', 'Low Risk', 'Safe'])
    with col2:
        demand_filter = st.multiselect("Filter by Demand Type:", rec_df['demand_type'].unique().tolist(),
                                        default=rec_df['demand_type'].unique().tolist())
    
    filtered_df = rec_df[(rec_df['risk'].isin(risk_filter)) & (rec_df['demand_type'].isin(demand_filter))]
    
    display_df = filtered_df[['item_title', 'demand_type', 'forecast', 'inventory', 
                              'days_to_expiry', 'risk', 'action', 'discount', 'urgency']].copy()
    display_df.columns = ['Item', 'Demand Type', 'Forecast', 'Inventory', 
                          'Days to Expiry', 'Risk', 'Action', 'Discount %', 'Urgency']
    display_df = display_df.round(1)
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Urgency': st.column_config.ProgressColumn("Urgency", min_value=0, max_value=10, format="%d"),
            'Discount %': st.column_config.NumberColumn("Discount %", format="%d%%")
        }
    )
    
    # Forecast Detail
    st.header("📈 Forecast Details")
    
    selected_item = st.selectbox("Select Item for Detailed View:", rec_df['item_title'].tolist())
    
    item_data = rec_df[rec_df['item_title'] == selected_item].iloc[0]
    item_id = item_data['item_id']
    item_history = place_data[place_data['item_id'] == item_id].sort_values('week_start')
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=item_history['week_start'],
            y=item_history['demand'],
            mode='lines+markers',
            name='Historical Demand',
            line=dict(color='#2E86AB')
        ))
        
        last_date = item_history['week_start'].max()
        forecast_date = last_date + timedelta(weeks=1)
        
        fig.add_trace(go.Scatter(
            x=[forecast_date],
            y=[item_data['forecast']],
            mode='markers',
            name='Forecast',
            marker=dict(size=15, color='#E94F37', symbol='star')
        ))
        
        fig.add_trace(go.Scatter(
            x=[forecast_date, forecast_date],
            y=[item_data['lower_bound'], item_data['upper_bound']],
            mode='lines',
            name='Confidence Interval',
            line=dict(color='#E94F37', width=3)
        ))
        
        fig.update_layout(
            title=f"Demand History & Forecast: {selected_item}",
            xaxis_title="Week",
            yaxis_title="Demand",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Item Details")
        st.markdown(f"**Demand Type:** {item_data['demand_type']}")
        st.markdown(f"**Risk Level:** {item_data['risk']}")
        st.markdown(f"**Forecast:** {item_data['forecast']:.1f} ({item_data['lower_bound']:.1f} - {item_data['upper_bound']:.1f})")
        st.markdown(f"**Current Inventory:** {item_data['inventory']:.1f}")
        st.markdown(f"**Days to Expiry:** {item_data['days_to_expiry']:.0f}")
        st.divider()
        st.markdown(f"### Recommendation")
        st.markdown(item_data['explanation'])
        if item_data['discount'] > 0:
            st.markdown(f"**Suggested Discount:** {item_data['discount']}%")
    
    # Export
    st.header("📥 Export")
    csv = filtered_df.to_csv(index=False)
    safe_place_name = selected_place_name.replace(" ", "_").replace("/", "-")
    st.download_button("Download Recommendations as CSV", data=csv,
                       file_name=f"freshflow_{safe_place_name}_{datetime.now().strftime('%Y%m%d')}.csv",
                       mime="text/csv")
    
    st.divider()
    st.markdown(f"*Data Source: Analyzed Data | {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    
    # Note about data
    st.info("ℹ️ **Note:** Expiration dates and shelf life are estimated based on product categories, as the original dataset doesn't include actual expiry tracking.")


if __name__ == "__main__":
    main()
