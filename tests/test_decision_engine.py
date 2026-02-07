"""
Test Script for Inventory Decision Engine
==========================================
Demonstrates the decision engine with example inputs and outputs.
"""

import sys
import io
import pandas as pd
from pathlib import Path
from inventory_decision_engine import InventoryDecisionEngine
import json

# Configure UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def test_decision_engine():
    """Test the decision engine with real data."""
    
    print("=" * 80)
    print("INVENTORY DECISION ENGINE - TEST SUITE")
    print("=" * 80)
    
    # Setup paths
    data_dir = Path(__file__).parent
    outputs_dir = data_dir / "outputs"
    
    # Load cleaned data
    print("\n📂 Loading cleaned data...")
    cleaned_data = {}
    
    cleaned_data_dir = outputs_dir / "cleaned_data"
    if cleaned_data_dir.exists():
        for csv_file in cleaned_data_dir.glob("cleaned_*.csv"):
            table_name = csv_file.stem.replace("cleaned_", "")
            cleaned_data[table_name] = pd.read_csv(csv_file)
            print(f"   ✓ Loaded {table_name}")
    
    # Initialize engine
    print("\n🔧 Initializing decision engine...")
    engine = InventoryDecisionEngine(cleaned_data)
    
    # Load analytics
    analytics_dir = outputs_dir / "analytics"
    if analytics_dir.exists():
        engine.load_analytics(str(analytics_dir))
        print("   ✓ Loaded analytics data")
    
    # Test 1: Prep Quantity Calculation
    print("\n" + "=" * 80)
    print("TEST 1: Prep Quantity Calculation")
    print("=" * 80)
    
    if 'demand' in engine.analytics_data and len(engine.analytics_data['demand']) > 0:
        # Get top 5 items by demand
        top_items = engine.analytics_data['demand'].groupby('item_id')['demand_quantity'].sum().nlargest(5)
        
        for item_id in top_items.index:
            prep_rec = engine.calculate_prep_quantity(item_id, current_stock=0, days_ahead=1)
            print(f"\n{engine._format_prep_recommendation(prep_rec)}")
    
    # Test 2: Expiry Risk Assessment
    print("\n" + "=" * 80)
    print("TEST 2: Expiry Risk Assessment")
    print("=" * 80)
    
    if 'demand' in engine.analytics_data and len(engine.analytics_data['demand']) > 0:
        # Get items with low demand (potential expiry risk)
        item_demand = engine.analytics_data['demand'].groupby('item_id')['demand_quantity'].sum()
        low_demand_items = item_demand[item_demand < 20].head(5)
        
        for item_id in low_demand_items.index:
            expiry_assessment = engine.assess_expiry_risk(item_id, current_stock=30)
            if expiry_assessment['risk_level'] in ['high', 'critical']:
                print(f"\n{engine._format_expiry_recommendation(expiry_assessment)}")
    
    # Test 3: Stock Alerts
    print("\n" + "=" * 80)
    print("TEST 3: Stock Alerts")
    print("=" * 80)
    
    alerts = engine.generate_stock_alerts()
    print(f"\nGenerated {len(alerts)} stock alerts")
    
    for alert in alerts[:5]:  # Show first 5
        print(f"\n{engine._format_stock_alert(alert)}")
    
    # Test 4: ML Forecast Integration
    print("\n" + "=" * 80)
    print("TEST 4: ML Forecast Integration")
    print("=" * 80)
    
    if 'demand' in engine.analytics_data and len(engine.analytics_data['demand']) > 0:
        sample_item = engine.analytics_data['demand']['item_id'].iloc[0]
        
        # Set ML forecast
        engine.set_ml_forecast(sample_item, {
            'predicted_demand': 25.5,
            'confidence': 0.85,
            'model_type': 'LSTM'
        })
        
        print(f"\n📊 ML Forecast Set for Item {sample_item}:")
        print(f"   Predicted Demand: 25.5 units/day")
        print(f"   Confidence: 85%")
        print(f"   Model Type: LSTM")
        
        # Calculate prep with ML
        prep_rec_ml = engine.calculate_prep_quantity(sample_item, use_ml=True)
        print(f"\n{engine._format_prep_recommendation(prep_rec_ml)}")
        
        # Compare with rule-based only
        prep_rec_rules = engine.calculate_prep_quantity(sample_item, use_ml=False)
        print(f"\n📊 Comparison:")
        print(f"   Rule-based only: {prep_rec_rules['prep_quantity']:.1f} units")
        print(f"   With ML forecast: {prep_rec_ml['prep_quantity']:.1f} units")
        print(f"   Difference: {prep_rec_ml['prep_quantity'] - prep_rec_rules['prep_quantity']:.1f} units")
    
    # Test 5: Generate All Recommendations
    print("\n" + "=" * 80)
    print("TEST 5: Generate All Recommendations")
    print("=" * 80)
    
    print("\n🔄 Generating comprehensive recommendations...")
    all_recs = engine.generate_all_recommendations()
    
    if len(all_recs) > 0:
        print(f"\n✓ Generated {len(all_recs)} total recommendations")
        
        # Save to CSV
        output_file = outputs_dir / "decision_engine_recommendations.csv"
        all_recs.to_csv(output_file, index=False)
        print(f"✓ Saved to {output_file}")
        
        # Summary by type
        if 'recommendation_type' in all_recs.columns:
            summary = all_recs['recommendation_type'].value_counts()
            print(f"\n📊 Recommendations by Type:")
            for rec_type, count in summary.items():
                print(f"   {rec_type}: {count}")
    
    print("\n" + "=" * 80)
    print("TEST SUITE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    test_decision_engine()
