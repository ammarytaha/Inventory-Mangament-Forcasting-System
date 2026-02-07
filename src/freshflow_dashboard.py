"""
FreshFlow AI - Main Dashboard Application
==========================================

AI-powered inventory decision engine with location-based personalization.
Designed for operations managers, store owners, and regional supervisors.

Key Features:
- Location selection and personalization
- AI-driven recommendations with explanations
- Demand forecasting with confidence intervals
- Context-aware adjustments (holidays, events)
- Actionable insights for reducing waste

Run with: streamlit run freshflow_dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import sys
import warnings

warnings.filterwarnings('ignore')

# Add the project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import FreshFlow AI modules
from freshflow_ai.config import Config
from freshflow_ai.data_processor import DataProcessor
from freshflow_ai.forecaster import ForecastEngine
from freshflow_ai.recommendation_engine import RecommendationEngine, RecommendationType, RiskLevel
from freshflow_ai.context_engine import ContextEngine
from freshflow_ai.explanation_generator import ExplanationGenerator


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FreshFlow AI - Inventory Decision Engine",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
        padding: 0rem 1rem;
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1a472a 0%, #2d5a3d 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1rem;
    }
    
    /* Location Badge */
    .location-badge {
        background: rgba(255,255,255,0.15);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        margin-top: 0.5rem;
    }
    
    /* KPI Cards */
    .kpi-container {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .kpi-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem;
        flex: 1;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border-left: 4px solid #2d5a3d;
    }
    
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1a472a;
        margin: 0.3rem 0;
    }
    
    .kpi-label {
        font-size: 0.85rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .kpi-change {
        font-size: 0.8rem;
        color: #388e3c;
    }
    
    .kpi-change.negative {
        color: #d32f2f;
    }
    
    /* Recommendation Cards */
    .rec-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border-left: 4px solid #2d5a3d;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .rec-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .rec-card.critical {
        border-left-color: #d32f2f;
        background: #fff5f5;
    }
    
    .rec-card.high {
        border-left-color: #f57c00;
        background: #fff8e1;
    }
    
    .rec-card.medium {
        border-left-color: #fbc02d;
    }
    
    .rec-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.8rem;
    }
    
    .rec-card-title {
        font-weight: 600;
        font-size: 1.05rem;
        color: #333;
    }
    
    .rec-card-action {
        background: #e8f5e9;
        color: #2e7d32;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    .rec-card-reason {
        color: #666;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    
    .rec-card-impact {
        color: #1976d2;
        font-size: 0.85rem;
        font-style: italic;
    }
    
    /* Risk Badges */
    .risk-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .risk-critical {
        background: #ffebee;
        color: #c62828;
    }
    
    .risk-high {
        background: #fff3e0;
        color: #e65100;
    }
    
    .risk-medium {
        background: #fffde7;
        color: #f9a825;
    }
    
    .risk-low {
        background: #e8f5e9;
        color: #2e7d32;
    }
    
    /* Context Panel */
    .context-panel {
        background: #f5f5f5;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .context-chip {
        display: inline-block;
        background: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        margin: 0.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Section Headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e0e0e0;
    }
    
    .section-header h2 {
        margin: 0;
        font-size: 1.2rem;
        color: #333;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Sidebar styling */
    .css-1d391kg {
        padding-top: 1rem;
    }
    
    /* Custom metric styling */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
    }
    
</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

def init_session_state():
    """Initialize session state variables"""
    if 'config' not in st.session_state:
        st.session_state.config = Config.from_workspace(str(PROJECT_ROOT))
        
    if 'data_processor' not in st.session_state:
        st.session_state.data_processor = DataProcessor(st.session_state.config)
        
    if 'forecast_engine' not in st.session_state:
        st.session_state.forecast_engine = ForecastEngine(
            st.session_state.data_processor,
            st.session_state.config
        )
        
    if 'recommendation_engine' not in st.session_state:
        st.session_state.recommendation_engine = RecommendationEngine(
            st.session_state.data_processor,
            st.session_state.forecast_engine,
            st.session_state.config
        )
        
    if 'context_engine' not in st.session_state:
        st.session_state.context_engine = ContextEngine(st.session_state.config)
        
    if 'explanation_generator' not in st.session_state:
        st.session_state.explanation_generator = ExplanationGenerator(st.session_state.config)
        
    if 'selected_place_id' not in st.session_state:
        st.session_state.selected_place_id = None
        
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = []


init_session_state()


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(ttl=600)
def load_places():
    """Load available places for selection"""
    processor = st.session_state.data_processor
    try:
        processor.load_data()
        places = processor.get_active_places()
        return places
    except Exception as e:
        st.error(f"Error loading places: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def get_place_recommendations(_place_id):
    """Get recommendations for a place (cached)"""
    engine = st.session_state.recommendation_engine
    context_engine = st.session_state.context_engine
    
    # Get context for today
    context = context_engine.get_context_factors(_place_id)
    
    recommendations = engine.generate_recommendations(
        place_id=_place_id,
        forecast_horizon=4,
        top_n_items=30,
        context=context
    )
    
    return recommendations, context


# ============================================================
# SIDEBAR - LOCATION SELECTION
# ============================================================

def render_sidebar():
    """Render the sidebar with location selection"""
    with st.sidebar:
        st.markdown("## 🏪 Select Your Location")
        st.markdown("---")
        
        places = load_places()
        
        if len(places) == 0:
            st.warning("No locations available. Check data connection.")
            return None
            
        # Create location options - showing only names for user-friendliness
        place_name_to_id = {}
        place_names = []
        
        for _, row in places.head(50).iterrows():  # Limit to top 50
            place_id = row['id']
            title = row.get('title', f'Location {place_id}')
            # Only add if we have a real name
            if title and title != f'Location {place_id}':
                place_name_to_id[title] = place_id
                place_names.append(title)
            else:
                # Fallback to ID-based name
                fallback_name = f"Location {place_id}"
                place_name_to_id[fallback_name] = place_id
                place_names.append(fallback_name)
        
        # Sort alphabetically for easier finding
        place_names.sort()
        
        # Add "Demo Location" option for testing at the top
        demo_label = "📊 Demo Location (Sample Data)"
        place_name_to_id[demo_label] = -1
        place_names.insert(0, demo_label)
        
        selected_label = st.selectbox(
            "Choose your location",
            options=place_names,
            index=0,
            help="Select your store to see personalized inventory recommendations"
        )
        
        selected_place_id = place_name_to_id[selected_label]
        
        st.markdown("---")
        
        # Date context
        st.markdown("### 📅 Current Context")
        st.markdown(f"**Date:** {datetime.now().strftime('%B %d, %Y')}")
        st.markdown(f"**Day:** {datetime.now().strftime('%A')}")
        
        # Weekly pattern info
        day_of_week = datetime.now().weekday()
        day_factor = st.session_state.config.get_weekly_factor(day_of_week)
        
        if day_factor > 1.1:
            st.success(f"📈 High-demand day (+{(day_factor-1)*100:.0f}%)")
        elif day_factor < 0.9:
            st.info(f"📉 Lower-demand day ({(day_factor-1)*100:.0f}%)")
        else:
            st.info("Normal demand expected")
            
        st.markdown("---")
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
            
        if st.button("📊 Export Report", use_container_width=True):
            st.info("Report generation coming soon!")
            
        st.markdown("---")
        
        # Help
        with st.expander("ℹ️ How to use"):
            st.markdown("""
            1. **Select your location** from the dropdown above
            2. Review the **AI recommendations** on the main panel
            3. Click on any recommendation for **detailed explanation**
            4. Use forecasts to plan **prep quantities**
            5. Check the **context panel** for external factors
            """)
        
        # Data note
        with st.expander("📋 About the Data"):
            st.markdown("""
            **Expiration Dates:** Estimated based on product categories 
            (the original dataset doesn't include actual expiry tracking).
            
            **Recommendations:** Generated using AI forecasting models 
            calibrated on historical demand patterns.
            """)
            
        return selected_place_id


# ============================================================
# MAIN CONTENT AREA
# ============================================================

def render_header(place_id, context):
    """Render the main header"""
    place_info = st.session_state.data_processor.get_place_data(place_id).get('place_info', {})
    place_name = place_info.get('title', f'Location {place_id}')
    
    st.markdown(f"""
    <div class="main-header">
        <h1>🍃 FreshFlow AI - Inventory Decision Engine</h1>
        <p>Reduce waste. Prevent stockouts. Make confident inventory decisions.</p>
        <div class="location-badge">
            📍 {place_name}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_kpis(recommendations, context):
    """Render KPI cards"""
    # Calculate KPIs from recommendations
    total_recs = len(recommendations)
    critical_count = sum(1 for r in recommendations if r.risk_level == RiskLevel.CRITICAL)
    high_count = sum(1 for r in recommendations if r.risk_level == RiskLevel.HIGH)
    
    reorder_count = sum(1 for r in recommendations if r.recommendation_type == RecommendationType.REORDER)
    prep_count = sum(1 for r in recommendations if r.recommendation_type == RecommendationType.PREP_ADJUST)
    
    demand_factor = context.get('combined_factor', 1.0)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Recommendations",
            value=total_recs,
            delta=f"{critical_count} critical" if critical_count > 0 else "All under control"
        )
        
    with col2:
        st.metric(
            label="Items to Reorder",
            value=reorder_count,
            delta=f"{high_count} high priority" if high_count > 0 else None
        )
        
    with col3:
        st.metric(
            label="Prep Adjustments",
            value=prep_count,
            delta="Updated for today"
        )
        
    with col4:
        st.metric(
            label="Demand Factor",
            value=f"{demand_factor:.0%}",
            delta="vs average" if demand_factor != 1.0 else "Normal"
        )


def render_context_panel(context):
    """Render the context factors panel"""
    st.markdown("### 🌍 Today's Context")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Day information
        base = context.get('base_factors', {})
        st.markdown(f"""
        <div class="context-panel">
            <strong>📅 Day Pattern</strong><br>
            <span class="context-chip">{base.get('day_name', 'Unknown')}</span>
            <span class="context-chip">Factor: {base.get('day_factor', 1.0):.0%}</span>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        # Events
        events = context.get('events', [])
        if events:
            event_chips = ''.join([f'<span class="context-chip">🎯 {e["name"]}</span>' for e in events])
            st.markdown(f"""
            <div class="context-panel">
                <strong>🗓️ Active Events</strong><br>
                {event_chips}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="context-panel">
                <strong>🗓️ Active Events</strong><br>
                <span class="context-chip">No special events</span>
            </div>
            """, unsafe_allow_html=True)
            
    # Context recommendations
    context_recs = context.get('recommendations', [])
    if context_recs:
        with st.expander("💡 Context-Based Suggestions", expanded=True):
            for rec in context_recs:
                st.info(rec)


def render_recommendations(recommendations):
    """Render the recommendations list"""
    st.markdown("### 🎯 AI Recommendations")
    st.markdown("*Sorted by priority - most urgent first*")
    
    if not recommendations:
        st.success("✅ No urgent actions needed. Your inventory is well-balanced!")
        return
        
    # Group by type
    by_type = {}
    for rec in recommendations:
        type_name = rec.recommendation_type.value
        if type_name not in by_type:
            by_type[type_name] = []
        by_type[type_name].append(rec)
        
    # Create tabs for different types
    tab_names = list(by_type.keys())
    if len(tab_names) > 0:
        tabs = st.tabs([f"{name.title()} ({len(by_type[name])})" for name in tab_names])
        
        for tab, type_name in zip(tabs, tab_names):
            with tab:
                for rec in by_type[type_name][:10]:  # Show top 10 per type
                    render_recommendation_card(rec)


def render_recommendation_card(rec):
    """Render a single recommendation card"""
    explanation = st.session_state.explanation_generator.explain_recommendation(rec)
    risk_class = rec.risk_level.value
    
    # Determine card class
    card_class = f"rec-card {risk_class}"
    
    # Risk badge
    risk_badges = {
        'critical': '🔴 Critical',
        'high': '🟠 High',
        'medium': '🟡 Medium',
        'low': '🟢 Low'
    }
    
    # Type icons
    type_icons = {
        'reorder': '📦',
        'discount': '🏷️',
        'bundle': '🎁',
        'prep_adjust': '👨‍🍳',
        'alert': '⚠️'
    }
    
    icon = type_icons.get(rec.recommendation_type.value, '📋')
    risk_badge = risk_badges.get(risk_class, '⚪ Unknown')
    
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"**{icon} {rec.item_name}**")
            st.markdown(f"*{rec.rationale[:150]}...*" if len(rec.rationale) > 150 else f"*{rec.rationale}*")
            
        with col2:
            st.markdown(f"<span class='risk-badge risk-{risk_class}'>{risk_badge}</span>", unsafe_allow_html=True)
            
        # Action button
        col_action, col_info = st.columns([2, 1])
        with col_action:
            st.success(f"**Action:** {rec.action}")
        with col_info:
            st.caption(f"Confidence: {explanation['confidence']['visual']}")
            
        # Expandable details
        with st.expander("📖 See full explanation"):
            st.markdown(f"**Why this recommendation:**")
            st.write(rec.rationale)
            
            st.markdown(f"**Expected Impact:**")
            st.info(rec.expected_impact)
            
            st.markdown(f"**Confidence Level:** {explanation['confidence']['level']} ({explanation['confidence']['percentage']}%)")
            st.caption(explanation['confidence']['explanation'])
            
            if rec.additional_data:
                st.markdown("**Additional Details:**")
                for key, value in rec.additional_data.items():
                    if not key.startswith('_'):
                        st.caption(f"• {key.replace('_', ' ').title()}: {value}")
                        
        st.markdown("---")


def render_forecast_view(place_id):
    """Render forecast visualization"""
    st.markdown("### 📈 Demand Forecasts")
    
    # Get top items for forecasting
    top_items = st.session_state.data_processor.get_top_items(place_id, n=10)
    
    if len(top_items) == 0:
        st.warning("No items with sufficient history for forecasting.")
        return
        
    # Item selector - showing only names for user-friendliness
    item_name_to_id = {}
    item_names = []
    
    for _, row in top_items.iterrows():
        item_id = row.item_id
        item_name = row.get('title', f'Item {item_id}')
        # Ensure unique names
        if item_name in item_name_to_id:
            item_name = f"{item_name} (#{item_id})"
        item_name_to_id[item_name] = item_id
        item_names.append(item_name)
    
    selected_item_label = st.selectbox(
        "Select item to view forecast",
        options=item_names,
        help="Choose a product to see its demand forecast"
    )
    
    selected_item_id = item_name_to_id[selected_item_label]
    
    # Generate forecast
    with st.spinner("Generating forecast..."):
        forecast = st.session_state.forecast_engine.forecast_item(
            place_id=place_id,
            item_id=selected_item_id,
            horizon=4
        )
        
    # Get historical data for visualization
    history = st.session_state.data_processor.get_item_history(
        place_id, 
        selected_item_id, 
        weeks=12
    )
    
    if len(history) > 0:
        # Create visualization
        fig = create_forecast_chart(history, forecast)
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecast details
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Demand Type",
                forecast.get('demand_type', 'Unknown'),
                delta=None
            )
            
        with col2:
            st.metric(
                "Model Used",
                forecast.get('model_used', 'Unknown').title(),
                delta=None
            )
            
        with col3:
            if forecast['forecast']:
                next_week = forecast['forecast'][0]['predicted_demand']
                st.metric(
                    "Next Week Forecast",
                    f"{next_week} units",
                    delta=None
                )
                
        # Forecast table
        if forecast['forecast']:
            st.markdown("**Weekly Forecast:**")
            forecast_df = pd.DataFrame(forecast['forecast'])
            forecast_df['week_start'] = pd.to_datetime(forecast_df['week_start']).dt.strftime('%Y-%m-%d')
            st.dataframe(forecast_df, use_container_width=True, hide_index=True)


def create_forecast_chart(history, forecast):
    """Create a Plotly forecast chart"""
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=history['week_start'],
        y=history['demand'],
        mode='lines+markers',
        name='Historical Demand',
        line=dict(color='#2d5a3d', width=2),
        marker=dict(size=6)
    ))
    
    # Forecast data
    if forecast['forecast']:
        forecast_dates = [f['week_start'] for f in forecast['forecast']]
        forecast_values = [f['predicted_demand'] for f in forecast['forecast']]
        lower_bounds = [f.get('lower', f['predicted_demand']) for f in forecast['forecast']]
        upper_bounds = [f.get('upper', f['predicted_demand']) for f in forecast['forecast']]
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=forecast_dates + forecast_dates[::-1],
            y=upper_bounds + lower_bounds[::-1],
            fill='toself',
            fillcolor='rgba(45, 90, 61, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval',
            showlegend=True
        ))
        
        # Forecast line
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_values,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#f57c00', width=2, dash='dash'),
            marker=dict(size=8, symbol='diamond')
        ))
        
    fig.update_layout(
        title=f"Demand Forecast ({forecast.get('model_used', 'Unknown').title()} Model)",
        xaxis_title="Week",
        yaxis_title="Demand (units)",
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig


def render_analytics_view(place_id):
    """Render analytics and insights"""
    st.markdown("### 📊 Analytics & Insights")
    
    place_data = st.session_state.data_processor.get_place_data(place_id)
    stats = place_data.get('summary_stats', {})
    classification = place_data.get('demand_classification')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Location Statistics**")
        st.metric("Total Active Items", stats.get('total_items', 0))
        st.metric("Total Demand (Historical)", f"{stats.get('total_demand', 0):,.0f} units")
        st.metric("Weeks of Data", stats.get('weeks_active', 0))
        
    with col2:
        st.markdown("**Demand Pattern Distribution**")
        dist = stats.get('demand_type_distribution', {})
        if dist:
            dist_df = pd.DataFrame([
                {'Pattern': k, 'Count': v}
                for k, v in dist.items()
            ])
            
            fig = px.pie(
                dist_df, 
                values='Count', 
                names='Pattern',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No demand classification data available")
            
    # Top items table
    st.markdown("**Top Items by Demand**")
    top_items = st.session_state.data_processor.get_top_items(place_id, n=10)
    if len(top_items) > 0:
        display_cols = ['item_id', 'title', 'total_demand'] if 'title' in top_items.columns else ['item_id', 'total_demand']
        st.dataframe(top_items[display_cols], use_container_width=True, hide_index=True)


def render_demo_mode():
    """Render demo mode with sample data"""
    st.markdown("""
    <div class="main-header">
        <h1>🍃 FreshFlow AI - Demo Mode</h1>
        <p>Explore the system capabilities with sample data</p>
        <div class="location-badge">
            📍 Demo Location
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("📊 **Demo Mode**: You're viewing sample data to explore the system's capabilities. Select a real location from the sidebar to see actual recommendations.")
    
    # Sample KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Recommendations", 12, delta="3 critical")
    with col2:
        st.metric("Items to Reorder", 5, delta="2 high priority")
    with col3:
        st.metric("Prep Adjustments", 8, delta="Updated for today")
    with col4:
        st.metric("Demand Factor", "122%", delta="+22% vs average")
        
    st.markdown("---")
    
    # Sample recommendations
    st.markdown("### 🎯 Sample AI Recommendations")
    
    sample_recs = [
        {
            'item': 'Organic Chicken Breast',
            'action': 'Order 150 units',
            'reason': 'Based on Prophet forecast: 180 units expected over 4 weeks. Current safety stock below threshold.',
            'risk': 'high',
            'type': '📦 Reorder'
        },
        {
            'item': 'Greek Yogurt',
            'action': 'Apply 20% discount',
            'reason': 'Recent demand (45 units in 4 weeks) is 60% below average. Risk of expiry.',
            'risk': 'medium',
            'type': '🏷️ Markdown'
        },
        {
            'item': 'Fresh Salmon Fillet',
            'action': 'Prep 25 units for today',
            'reason': 'Friday peak demand (+39%). Expected high traffic due to weekend shoppers.',
            'risk': 'low',
            'type': '👨‍🍳 Prep'
        }
    ]
    
    for rec in sample_recs:
        with st.container():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{rec['type']} - {rec['item']}**")
                st.caption(rec['reason'])
            with col2:
                risk_colors = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢'}
                st.markdown(f"{risk_colors[rec['risk']]} {rec['risk'].title()}")
            st.success(f"**Action:** {rec['action']}")
            st.markdown("---")
            
    # Feature showcase
    st.markdown("### ✨ Key Features")
    
    feature_cols = st.columns(3)
    with feature_cols[0]:
        st.markdown("""
        **🎯 AI-Powered Decisions**
        - Demand forecasting
        - Automatic model selection
        - Confidence intervals
        """)
        
    with feature_cols[1]:
        st.markdown("""
        **📍 Location Personalization**
        - Place-specific recommendations
        - Local event awareness
        - Historical patterns
        """)
        
    with feature_cols[2]:
        st.markdown("""
        **📊 Explainable AI**
        - Clear action rationale
        - Confidence levels
        - Impact estimates
        """)


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    """Main application entry point"""
    # Render sidebar and get selected place
    selected_place_id = render_sidebar()
    
    if selected_place_id is None:
        st.error("Please select a location to continue.")
        return
        
    # Demo mode
    if selected_place_id == -1:
        render_demo_mode()
        return
        
    # Load recommendations for selected place
    try:
        with st.spinner("Loading recommendations..."):
            recommendations, context = get_place_recommendations(selected_place_id)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Try selecting 'Demo Location' to explore the system.")
        return
        
    # Render main content
    render_header(selected_place_id, context)
    render_kpis(recommendations, context)
    
    st.markdown("---")
    
    render_context_panel(context)
    
    st.markdown("---")
    
    # Create tabs for main views
    tab1, tab2, tab3 = st.tabs(["🎯 Recommendations", "📈 Forecasts", "📊 Analytics"])
    
    with tab1:
        render_recommendations(recommendations)
        
    with tab2:
        render_forecast_view(selected_place_id)
        
    with tab3:
        render_analytics_view(selected_place_id)


if __name__ == "__main__":
    main()
