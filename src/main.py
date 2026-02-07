"""
Main Data Engineering Pipeline
===============================
Orchestrates the complete data engineering and analytics pipeline.
Run this script to execute all steps from discovery to decision outputs.
"""

import sys
import io
from pathlib import Path
from datetime import datetime
import json

# Configure UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Import all modules
from data_discovery import DataDiscovery
from data_quality_report import DataQualityReport
from data_model import DataModelDocumenter
from data_cleaning import DataCleaningPipeline
from inventory_analytics import InventoryAnalytics
from decision_outputs import DecisionOutputs


def main():
    """Execute complete data engineering pipeline."""
    
    print("=" * 80)
    print("INVENTORY MANAGEMENT DATA ENGINEERING PIPELINE")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Setup paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent  # Go up to project root
    data_dir = project_root / "data"  # Use the data directory at project root
    output_dir = project_root / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    # ============================================================================
    # STEP 1: DATA DISCOVERY
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 1: DATA DISCOVERY")
    print("=" * 80)
    
    discovery = DataDiscovery(data_dir)
    discovery_results = discovery.discover_all()
    discovery.summary = discovery_results['summary']
    discovery.print_summary_report()
    
    # Save discovery results
    discovery_file = output_dir / "01_discovery_results.json"
    with open(discovery_file, 'w') as f:
        json.dump(discovery_results, f, indent=2, default=str)
    print(f"\n✓ Discovery results saved to {discovery_file}")
    
    # ============================================================================
    # STEP 2: DATA QUALITY ASSESSMENT
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 2: DATA QUALITY ASSESSMENT")
    print("=" * 80)
    
    quality_report = DataQualityReport(discovery_results)
    quality_results = quality_report.generate_report()
    quality_report.print_report()
    
    # Save quality report
    quality_file = output_dir / "02_quality_report.json"
    quality_report.save_report(str(quality_file))
    
    # ============================================================================
    # STEP 3: DATA MODEL DOCUMENTATION
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 3: DATA MODEL DOCUMENTATION")
    print("=" * 80)
    
    model_doc = DataModelDocumenter(discovery_results)
    model_results = model_doc.generate_model_documentation()
    model_doc.print_model_summary()
    
    # Save model documentation
    model_file = output_dir / "03_data_model.json"
    model_doc.save_documentation(str(model_file))
    
    # ============================================================================
    # STEP 4: DATA CLEANING
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 4: DATA CLEANING")
    print("=" * 80)
    
    cleaning_pipeline = DataCleaningPipeline(data_dir)
    
    # Clean key tables for analytics
    key_tables = [
        'fct_orders.csv',
        'fct_order_items.csv',
        'dim_skus.csv',
        'dim_menu_items.csv',
        'dim_places.csv',
        'dim_bill_of_materials.csv',
        'fct_inventory_reports.csv',
        'dim_campaigns.csv',
        'fct_campaigns.csv',
        'fct_bonus_codes.csv'
    ]
    
    cleaned_data = cleaning_pipeline.clean_all_tables(key_tables)
    
    # Save cleaned data
    cleaned_dir = output_dir / "cleaned_data"
    cleaning_pipeline.save_cleaned_data(str(cleaned_dir))
    
    # Save cleaning summary
    cleaning_summary = cleaning_pipeline.get_cleaning_summary()
    cleaning_file = output_dir / "04_cleaning_summary.json"
    with open(cleaning_file, 'w') as f:
        json.dump(cleaning_summary, f, indent=2, default=str)
    print(f"\n✓ Cleaning summary saved to {cleaning_file}")
    
    # ============================================================================
    # STEP 5: INVENTORY ANALYTICS FOUNDATION
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 5: INVENTORY ANALYTICS FOUNDATION")
    print("=" * 80)
    
    analytics = InventoryAnalytics(cleaned_data)
    
    # Calculate demand per SKU
    print("\n📊 Calculating demand per SKU...")
    try:
        demand_daily = analytics.calculate_demand_per_sku(period='daily')
        print(f"✓ Calculated daily demand for {len(demand_daily)} item-period combinations")
    except Exception as e:
        print(f"⚠ Error calculating demand: {str(e)}")
    
    # Calculate stock levels
    print("\n📦 Calculating stock levels...")
    try:
        stock_levels = analytics.calculate_stock_levels_over_time()
        print(f"✓ Tracked stock levels for {len(stock_levels)} SKUs")
    except Exception as e:
        print(f"⚠ Error calculating stock levels: {str(e)}")
    
    # Calculate expiry risk
    print("\n⏰ Calculating expiry risk indicators...")
    try:
        expiry_risk = analytics.calculate_expiry_risk_indicators()
        print(f"✓ Calculated expiry risk for {len(expiry_risk)} items")
    except Exception as e:
        print(f"⚠ Error calculating expiry risk: {str(e)}")
    
    # Calculate waste metrics
    print("\n🗑️  Calculating waste metrics...")
    try:
        waste_metrics = analytics.calculate_waste_metrics()
        print(f"✓ Calculated waste metrics for {len(waste_metrics)} items")
    except Exception as e:
        print(f"⚠ Error calculating waste metrics: {str(e)}")
    
    # Save analytics tables
    analytics_dir = output_dir / "analytics"
    analytics_dir.mkdir(exist_ok=True)
    import pandas as pd
    for table_name, table_data in analytics.analytics_tables.items():
        if isinstance(table_data, pd.DataFrame) and len(table_data) > 0:
            output_file = analytics_dir / f"analytics_{table_name}.csv"
            table_data.to_csv(output_file, index=False)
            print(f"✓ Saved {table_name} to {output_file}")
    
    # ============================================================================
    # STEP 6: DECISION-ORIENTED OUTPUTS
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 6: DECISION-ORIENTED OUTPUTS")
    print("=" * 80)
    
    decisions = DecisionOutputs(analytics)
    
    # Generate prep recommendations
    print("\n🍳 Generating prep quantity recommendations...")
    try:
        prep_recs = decisions.recommend_prep_quantities(forecast_days=7)
        if len(prep_recs) > 0:
            print(f"✓ Generated recommendations for {len(prep_recs)} items")
    except Exception as e:
        print(f"⚠ Error generating prep recommendations: {str(e)}")
    
    # Identify expiry risks
    print("\n⚠️  Identifying expiry risks...")
    try:
        expiry_risks = decisions.identify_expiry_risks(risk_threshold=60.0)
        if len(expiry_risks) > 0:
            print(f"✓ Identified {len(expiry_risks)} items at risk")
    except Exception as e:
        print(f"⚠ Error identifying expiry risks: {str(e)}")
    
    # Generate stock alerts
    print("\n🔔 Generating stock alerts...")
    try:
        stock_alerts = decisions.generate_stock_alerts()
        print(f"✓ Generated stock alerts")
    except Exception as e:
        print(f"⚠ Error generating stock alerts: {str(e)}")
    
    # Prepare forecasting inputs
    print("\n🤖 Preparing forecasting inputs...")
    try:
        forecasting_inputs = decisions.prepare_forecasting_inputs()
        print(f"✓ Prepared forecasting inputs")
    except Exception as e:
        print(f"⚠ Error preparing forecasting inputs: {str(e)}")
    
    # Save decision outputs
    decisions_dir = output_dir / "decisions"
    decisions.save_decisions(str(decisions_dir))
    
    # Generate executive summary
    exec_summary = decisions.generate_executive_summary()
    exec_file = output_dir / "07_executive_summary.json"
    with open(exec_file, 'w') as f:
        json.dump(exec_summary, f, indent=2, default=str)
    print(f"\n✓ Executive summary saved to {exec_file}")
    
    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n📁 Output Directory: {output_dir}")
    print("\nGenerated Files:")
    print("  • 01_discovery_results.json - Data discovery and profiling")
    print("  • 02_quality_report.json - Data quality assessment")
    print("  • 03_data_model.json - Data model and relationships")
    print("  • 04_cleaning_summary.json - Data cleaning summary")
    print("  • cleaned_data/ - Cleaned CSV files")
    print("  • analytics/ - Analytics foundation tables")
    print("  • decisions/ - Decision-oriented outputs")
    print("  • 07_executive_summary.json - Executive summary")
    
    print("\n" + "=" * 80)
    print("BUSINESS VALUE DELIVERED")
    print("=" * 80)
    print("✅ Data Quality Assessment: Know which tables are reliable")
    print("✅ Data Model Documentation: Understand relationships")
    print("✅ Clean Analytics-Ready Data: Foundation for AI/ML models")
    print("✅ Demand Analytics: Daily/weekly demand per SKU")
    print("✅ Stock Level Tracking: Current inventory status")
    print("✅ Expiry Risk Indicators: Prevent waste proactively")
    print("✅ Prep Recommendations: Optimize daily preparation")
    print("✅ Stock Alerts: Prevent stockouts and overstock")
    print("✅ Forecasting Inputs: Ready for AI demand forecasting")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    import pandas as pd  # Import here to avoid issues
    main()
