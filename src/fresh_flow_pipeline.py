"""
Fresh Flow Markets - Inventory Intelligence Pipeline
======================================================
Main orchestrator for the enhanced inventory management system.

This is the primary entry point for running the complete pipeline.
It coordinates all phases from data loading to recommendation generation.

Phases:
1. Data Foundation - Load and validate data
2. Forecasting Core - Generate demand forecasts
3. Context Adjustments - Apply event and weather adjustments
4. Inventory Intelligence - Calculate health scores and recommendations
5. Output & Integration - Generate unified outputs
6. Business Explanation - Add plain English explanations

Usage:
    python fresh_flow_pipeline.py

The pipeline will:
- Load all CSV files from the current directory
- Validate data quality
- Generate demand forecasts
- Calculate inventory health scores
- Produce actionable recommendations
- Export results to outputs/ directory

Author: Fresh Flow Markets AI Team
Version: 2.0
"""

import sys
import io
from pathlib import Path
from datetime import datetime, date, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# Configure UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Import our modules
from utils.logger import get_logger, LogContext
from utils.constants import FORECAST_CONFIG, RECOMMENDATION_CONFIG
from services.data_loader import DataLoader, aggregate_daily_demand
from services.forecaster import DemandForecaster, combine_forecast_results
from services.context_adjustments import (
    ContextAdjuster, Event, WeatherCondition,
    create_sample_events, create_sample_weather
)
from services.inventory_health import (
    InventoryHealthScorer, health_scores_to_dataframe
)
from services.recommendation_engine import (
    RecommendationEngine, recommendations_to_dataframe
)
from services.output_generator import OutputGenerator, OutputPackage
from models.explanation import ExplanationGenerator

logger = get_logger(__name__)


def print_header():
    """Print the pipeline header."""
    print("=" * 80)
    print("🍃 FRESH FLOW MARKETS - INVENTORY INTELLIGENCE SYSTEM")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)
    print()


def print_phase_header(phase_num: int, phase_name: str):
    """Print a phase header."""
    print()
    print("=" * 80)
    print(f"PHASE {phase_num}: {phase_name}")
    print("=" * 80)


def run_pipeline(data_dir: str = None, output_dir: str = None) -> dict:
    """
    Run the complete inventory intelligence pipeline.
    
    Parameters
    ----------
    data_dir : str, optional
        Directory containing CSV files. Defaults to script directory.
    output_dir : str, optional
        Directory for outputs. Defaults to './outputs'.
    
    Returns
    -------
    dict
        Summary of pipeline execution
    """
    print_header()
    
    # Setup paths
    script_dir = Path(__file__).parent
    data_dir = Path(data_dir) if data_dir else script_dir
    output_dir = Path(output_dir) if output_dir else script_dir / "outputs"
    
    pipeline_summary = {
        "started_at": datetime.now().isoformat(),
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "phases": {}
    }
    
    # =========================================================================
    # PHASE 1: DATA FOUNDATION
    # =========================================================================
    print_phase_header(1, "DATA FOUNDATION")
    
    with LogContext(logger, "Phase 1: Data Foundation"):
        print("📂 Loading and validating data files...")
        
        loader = DataLoader(data_dir, validate_schemas=True)
        load_result = loader.load_all()
        
        # Print summary
        print(f"\n✓ Loaded {load_result.summary['total_files']} files")
        print(f"✓ Total records: {load_result.summary['total_rows']:,}")
        
        # Check for critical files
        if 'warnings' in load_result.summary:
            for warning in load_result.summary['warnings']:
                print(f"⚠️ {warning}")
        
        # Get sales data and aggregate to daily demand
        try:
            orders_df, order_items_df = loader.get_sales_data()
            daily_demand = aggregate_daily_demand(orders_df, order_items_df)
            print(f"✓ Aggregated daily demand: {len(daily_demand):,} rows")
            print(f"  - Items: {daily_demand['item_id'].nunique()}")
            print(f"  - Date range: {daily_demand['date'].min()} to {daily_demand['date'].max()}")
        except Exception as e:
            logger.error(f"Error aggregating demand: {e}")
            daily_demand = None
        
        pipeline_summary["phases"]["data_foundation"] = {
            "files_loaded": load_result.summary['total_files'],
            "total_records": load_result.summary['total_rows'],
            "demand_rows": len(daily_demand) if daily_demand is not None else 0
        }
    
    # =========================================================================
    # PHASE 2: FORECASTING CORE
    # =========================================================================
    print_phase_header(2, "FORECASTING CORE")
    
    with LogContext(logger, "Phase 2: Forecasting Core"):
        print("📈 Generating demand forecasts...")
        
        if daily_demand is not None and len(daily_demand) > 0:
            forecaster = DemandForecaster(
                horizon=FORECAST_CONFIG["default_forecast_horizon"],
                seasonal_periods=FORECAST_CONFIG["seasonal_period"]
            )
            
            forecast_results = forecaster.forecast_all(daily_demand)
            forecasts_df = combine_forecast_results(forecast_results)
            
            # Count methods used
            method_counts = {}
            for result in forecast_results:
                method_counts[result.method] = method_counts.get(result.method, 0) + 1
            
            print(f"\n✓ Generated forecasts for {len(forecast_results)} items")
            print(f"  - Methods used:")
            for method, count in method_counts.items():
                print(f"    • {method}: {count} items")
            
            pipeline_summary["phases"]["forecasting"] = {
                "items_forecast": len(forecast_results),
                "methods_used": method_counts
            }
        else:
            forecasts_df = None
            forecast_results = []
            print("⚠️ No demand data available for forecasting")
    
    # =========================================================================
    # PHASE 3: CONTEXT ADJUSTMENTS
    # =========================================================================
    print_phase_header(3, "CONTEXT ADJUSTMENTS")
    
    with LogContext(logger, "Phase 3: Context Adjustments"):
        print("🌤️ Applying event and weather adjustments...")
        
        if forecasts_df is not None and len(forecasts_df) > 0:
            adjuster = ContextAdjuster(
                apply_day_of_week=True,
                apply_weather=True,
                apply_events=True
            )
            
            # Create sample events for demo
            # In production, these would come from a calendar system
            start_date = date.today()
            sample_events = create_sample_events(start_date, num_days=14)
            for event in sample_events[:5]:  # Add a few events
                adjuster.add_event(event)
            
            # Create sample weather for demo
            sample_weather = create_sample_weather(start_date, num_days=14)
            for weather in sample_weather[:7]:  # Add a week of weather
                adjuster.add_weather(weather)
            
            # Apply adjustments
            adjusted_forecasts = adjuster.apply_adjustments(forecasts_df)
            
            # Calculate impact
            total_original = forecasts_df['forecast'].sum()
            total_adjusted = adjusted_forecasts['adjusted_forecast'].sum()
            adjustment_pct = ((total_adjusted - total_original) / total_original) * 100
            
            print(f"\n✓ Applied context adjustments")
            print(f"  - Events added: {len(adjuster.events)}")
            print(f"  - Weather conditions: {len(adjuster.weather_conditions)}")
            print(f"  - Total demand adjustment: {adjustment_pct:+.1f}%")
            
            pipeline_summary["phases"]["context_adjustments"] = {
                "events_applied": len(adjuster.events),
                "weather_conditions": len(adjuster.weather_conditions),
                "demand_adjustment_pct": round(adjustment_pct, 2)
            }
        else:
            adjusted_forecasts = forecasts_df
    
    # =========================================================================
    # PHASE 4: INVENTORY INTELLIGENCE
    # =========================================================================
    print_phase_header(4, "INVENTORY INTELLIGENCE")
    
    with LogContext(logger, "Phase 4: Inventory Intelligence"):
        print("🎯 Calculating health scores and generating recommendations...")
        
        # Get inventory data
        inventory_df = loader.get_inventory_data()
        menu_df = loader.get_menu_data()
        
        # Create item name mapping
        item_names = {}
        if len(menu_df) > 0 and 'id' in menu_df.columns and 'title' in menu_df.columns:
            item_names = dict(zip(menu_df['id'], menu_df['title']))
        
        # Calculate health scores
        scorer = InventoryHealthScorer()
        
        if adjusted_forecasts is not None and len(adjusted_forecasts) > 0:
            health_scores = scorer.calculate_health_scores(
                forecast_df=adjusted_forecasts,
                inventory_df=inventory_df
            )
            health_df = health_scores_to_dataframe(health_scores)
            
            # Risk summary
            risk_counts = health_df['risk_level'].value_counts().to_dict() \
                if 'risk_level' in health_df.columns else {}
            
            print(f"\n✓ Calculated health scores for {len(health_scores)} items")
            print(f"  - HIGH risk: {risk_counts.get('HIGH', 0)}")
            print(f"  - MEDIUM risk: {risk_counts.get('MEDIUM', 0)}")
            print(f"  - LOW risk: {risk_counts.get('LOW', 0)}")
            
            # Generate recommendations
            engine = RecommendationEngine()
            recommendations = engine.generate_recommendations(
                health_scores=health_scores,
                forecast_df=adjusted_forecasts,
                inventory_df=inventory_df,
                item_names=item_names
            )
            recommendations_df = recommendations_to_dataframe(recommendations)
            
            # Recommendation summary
            urgency_counts = recommendations_df['urgency'].value_counts().to_dict() \
                if 'urgency' in recommendations_df.columns else {}
            
            print(f"\n✓ Generated {len(recommendations)} recommendations")
            print(f"  - CRITICAL: {urgency_counts.get('CRITICAL', 0)}")
            print(f"  - HIGH: {urgency_counts.get('HIGH', 0)}")
            print(f"  - MEDIUM: {urgency_counts.get('MEDIUM', 0)}")
            
            pipeline_summary["phases"]["inventory_intelligence"] = {
                "items_scored": len(health_scores),
                "risk_counts": risk_counts,
                "recommendations_count": len(recommendations),
                "urgency_counts": urgency_counts
            }
        else:
            health_df = None
            recommendations_df = None
            print("⚠️ No forecast data available for health scoring")
    
    # =========================================================================
    # PHASE 5: OUTPUT & INTEGRATION
    # =========================================================================
    print_phase_header(5, "OUTPUT & INTEGRATION")
    
    with LogContext(logger, "Phase 5: Output & Integration"):
        print("📤 Generating unified outputs...")
        
        generator = OutputGenerator(output_dir=str(output_dir))
        
        # Generate output package
        package = generator.generate_output_package(
            forecasts=forecasts_df if forecasts_df is not None else pd.DataFrame(),
            adjusted_forecasts=adjusted_forecasts,
            health_scores=health_df if health_df is not None else pd.DataFrame(),
            recommendations=recommendations_df if recommendations_df is not None else pd.DataFrame(),
            historical_demand=daily_demand,
            item_names=item_names
        )
        
        # Export all
        exported = generator.export_all(package)
        
        print(f"\n✓ Exported {len(exported)} output files")
        for name, path in exported.items():
            print(f"  - {name}: {path}")
        
        pipeline_summary["phases"]["output_integration"] = {
            "files_exported": len(exported),
            "output_directory": str(output_dir)
        }
    
    # =========================================================================
    # PHASE 6: BUSINESS EXPLANATION
    # =========================================================================
    print_phase_header(6, "BUSINESS EXPLANATION")
    
    with LogContext(logger, "Phase 6: Business Explanation"):
        print("💬 Generating executive summary with plain English explanations...")
        
        # Print executive summary
        summary = package.executive_summary
        
        print("\n" + "=" * 60)
        print("📊 EXECUTIVE SUMMARY")
        print("=" * 60)
        
        if 'forecast_summary' in summary:
            fs = summary['forecast_summary']
            print(f"\nForecast Overview:")
            print(f"  • {fs.get('items_forecasted', 0)} items forecasted")
            print(f"  • {fs.get('total_expected_demand', 0):,.0f} total units expected")
        
        if 'risk_summary' in summary:
            rs = summary['risk_summary']
            print(f"\nRisk Overview:")
            print(f"  • {rs.get('high_risk_items', 0)} items at HIGH risk")
            print(f"  • {rs.get('medium_risk_items', 0)} items at MEDIUM risk")
            print(f"  • Average health score: {rs.get('average_health_score', 0):.0f}/100")
        
        if 'action_summary' in summary:
            acts = summary['action_summary']
            print(f"\nAction Summary:")
            print(f"  • {acts.get('total_recommendations', 0)} total recommendations")
            print(f"  • {acts.get('critical_actions', 0)} CRITICAL actions")
        
        if 'key_insights' in summary:
            print(f"\nKey Insights:")
            for insight in summary['key_insights']:
                print(f"  {insight}")
        
        pipeline_summary["phases"]["business_explanation"] = {
            "summary_generated": True
        }
    
    # =========================================================================
    # COMPLETION
    # =========================================================================
    print()
    print("=" * 80)
    print("✅ PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Outputs saved to: {output_dir}")
    print()
    
    pipeline_summary["completed_at"] = datetime.now().isoformat()
    pipeline_summary["status"] = "success"
    
    # Save pipeline summary
    summary_path = output_dir / "pipeline_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(pipeline_summary, f, indent=2, default=str)
    
    return pipeline_summary


# Need to import pandas for empty DataFrames
import pandas as pd


if __name__ == "__main__":
    try:
        summary = run_pipeline()
        print("Pipeline executed successfully!")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"\n❌ Pipeline failed with error: {e}")
        raise
