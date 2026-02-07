"""
Decision-Oriented Outputs
===========================
Creates action-ready outputs for inventory management:
- Recommended daily prep quantities
- SKUs at risk of expiration
- Overstock/understock alerts
- Inputs for demand forecasting models
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class DecisionOutputs:
    """Generate decision-oriented, action-ready outputs."""
    
    def __init__(self, analytics: 'InventoryAnalytics'):
        """
        Initialize with analytics foundation.
        
        Parameters:
        -----------
        analytics : InventoryAnalytics
            Instance of InventoryAnalytics with computed metrics
        """
        self.analytics = analytics
        self.decisions = {}
        
    def recommend_prep_quantities(self, 
                                  forecast_days: int = 7,
                                  safety_stock_multiplier: float = 1.5) -> pd.DataFrame:
        """
        Recommend daily prep quantities based on historical demand.
        
        Business Value:
        - Reduces waste by avoiding over-preparation
        - Prevents stockouts by ensuring adequate stock
        - Optimizes kitchen labor and ingredient usage
        
        Parameters:
        -----------
        forecast_days : int
            Number of days to forecast ahead
        safety_stock_multiplier : float
            Multiplier for safety stock (1.5 = 50% buffer)
        """
        # Get historical demand
        if 'demand_per_sku' not in self.analytics.analytics_tables:
            demand = self.analytics.calculate_demand_per_sku(period='daily')
        else:
            demand = self.analytics.analytics_tables['demand_per_sku']
        
        recommendations = []
        
        if 'item_id' in demand.columns:
            # Calculate average daily demand per item
            daily_demand = demand.groupby('item_id').agg({
                'demand_quantity': ['mean', 'std', 'max', 'count']
            }).reset_index()
            daily_demand.columns = ['item_id', 'avg_daily_demand', 'demand_std', 'max_daily_demand', 'days_with_sales']
            
            # Calculate recommended prep quantity
            daily_demand['recommended_daily_prep'] = daily_demand.apply(
                lambda row: self._calculate_prep_quantity(
                    row['avg_daily_demand'],
                    row['demand_std'],
                    row['max_daily_demand'],
                    safety_stock_multiplier
                ),
                axis=1
            )
            
            # Add confidence level
            daily_demand['confidence'] = daily_demand.apply(
                lambda row: 'high' if row['days_with_sales'] > 20
                           else 'medium' if row['days_with_sales'] > 10
                           else 'low',
                axis=1
            )
            
            # Add business context
            daily_demand['recommendation_reason'] = daily_demand.apply(
                lambda row: self._explain_prep_recommendation(row),
                axis=1
            )
            
            recommendations = daily_demand
        
        self.decisions['prep_recommendations'] = recommendations
        
        print(f"✓ Generated prep recommendations for {len(recommendations)} items")
        return recommendations
    
    def _calculate_prep_quantity(self, 
                                 avg_demand: float,
                                 demand_std: float,
                                 max_demand: float,
                                 safety_multiplier: float) -> float:
        """Calculate recommended prep quantity."""
        if pd.isna(avg_demand) or avg_demand == 0:
            return 0
        
        # Base quantity: average demand
        base = avg_demand
        
        # Add safety stock based on variability
        if not pd.isna(demand_std) and demand_std > 0:
            safety_stock = demand_std * safety_multiplier
        else:
            # If no std, use 20% of average as buffer
            safety_stock = avg_demand * 0.2
        
        # Cap at reasonable maximum (2x average or max seen)
        recommended = base + safety_stock
        max_reasonable = max(avg_demand * 2, max_demand) if not pd.isna(max_demand) else avg_demand * 2
        
        return min(recommended, max_reasonable)
    
    def _explain_prep_recommendation(self, row: pd.Series) -> str:
        """Generate business-friendly explanation."""
        avg = row.get('avg_daily_demand', 0)
        rec = row.get('recommended_daily_prep', 0)
        conf = row.get('confidence', 'low')
        
        if rec == 0:
            return "No sales history - start with minimal prep and monitor"
        
        if conf == 'high':
            return f"Based on {row.get('days_with_sales', 0)} days of sales data. Prep {rec:.1f} units daily (avg: {avg:.1f})"
        elif conf == 'medium':
            return f"Limited data ({row.get('days_with_sales', 0)} days). Prep {rec:.1f} units but monitor closely"
        else:
            return f"Very limited data. Start with {rec:.1f} units and adjust based on actual sales"
    
    def identify_expiry_risks(self, risk_threshold: float = 60.0) -> pd.DataFrame:
        """
        Identify SKUs/items at risk of expiration.
        
        Business Value:
        - Prevents waste by flagging items before they expire
        - Enables proactive promotions or menu adjustments
        - Reduces food cost impact
        
        Parameters:
        -----------
        risk_threshold : float
            Risk score threshold (0-100) above which to flag
        """
        # Get expiry risk data
        if 'expiry_risk' not in self.analytics.analytics_tables:
            expiry_risk = self.analytics.calculate_expiry_risk_indicators()
        else:
            expiry_risk = self.analytics.analytics_tables['expiry_risk']
        
        if len(expiry_risk) == 0:
            return pd.DataFrame()
        
        # Filter high-risk items
        if 'expiry_risk_score' in expiry_risk.columns:
            high_risk = expiry_risk[expiry_risk['expiry_risk_score'] >= risk_threshold].copy()
            
            # Add actionable recommendations
            high_risk['action_recommended'] = high_risk.apply(
                lambda row: self._recommend_expiry_action(row),
                axis=1
            )
            
            # Sort by risk score
            high_risk = high_risk.sort_values('expiry_risk_score', ascending=False)
            
            self.decisions['expiry_risks'] = high_risk
            
            print(f"✓ Identified {len(high_risk)} items at risk of expiration")
            return high_risk
        
        return pd.DataFrame()
    
    def _recommend_expiry_action(self, row: pd.Series) -> str:
        """Generate actionable recommendation for expiry risk."""
        days_since = row.get('days_since_last_sale', 0)
        demand = row.get('total_demand', 0)
        risk_score = row.get('expiry_risk_score', 0)
        
        if risk_score >= 80:
            return "URGENT: Consider promotion or remove from menu. No sales in 60+ days."
        elif risk_score >= 60:
            if days_since > 30:
                return "High risk: Create limited-time promotion to move inventory"
            else:
                return "Monitor closely: Consider reducing order quantity"
        else:
            return "Low risk: Continue monitoring"
    
    def generate_stock_alerts(self, 
                             low_stock_threshold_pct: float = 0.2,
                             overstock_threshold_pct: float = 2.0) -> Dict[str, pd.DataFrame]:
        """
        Generate overstock and understock alerts.
        
        Business Value:
        - Prevents stockouts (lost sales, customer dissatisfaction)
        - Reduces overstock (waste, tied-up capital)
        - Optimizes inventory investment
        
        Parameters:
        -----------
        low_stock_threshold_pct : float
            Alert if stock < threshold% of average demand
        overstock_threshold_pct : float
            Alert if stock > threshold% of average demand
        """
        alerts = {
            'understock': pd.DataFrame(),
            'overstock': pd.DataFrame()
        }
        
        # Get stock levels
        stock_levels = self.analytics.calculate_stock_levels_over_time()
        
        # Get demand
        if 'demand_per_sku' not in self.analytics.analytics_tables:
            demand = self.analytics.calculate_demand_per_sku(period='daily')
        else:
            demand = self.analytics.analytics_tables['demand_per_sku']
        
        # Calculate average daily demand
        if 'item_id' in demand.columns:
            avg_demand = demand.groupby('item_id')['demand_quantity'].mean().reset_index()
            avg_demand.columns = ['item_id', 'avg_daily_demand']
            
            # Merge with stock (simplified - would need proper SKU mapping)
            # For now, create alerts based on demand patterns
            
            # Understock: items with high demand but low/no stock
            # Overstock: items with low demand but high stock
            
            # This would require proper item_id -> sku_id mapping
            # For demonstration, we'll create alerts based on demand patterns
            
            # Items with high demand (potential understock risk)
            high_demand = avg_demand[avg_demand['avg_daily_demand'] > 10].copy()
            high_demand['alert_type'] = 'potential_understock'
            high_demand['alert_message'] = high_demand.apply(
                lambda row: f"High demand item (avg {row['avg_daily_demand']:.1f}/day). Ensure adequate stock.",
                axis=1
            )
            
            # Items with low/no demand (potential overstock risk)
            low_demand = avg_demand[avg_demand['avg_daily_demand'] < 2].copy()
            low_demand['alert_type'] = 'potential_overstock'
            low_demand['alert_message'] = low_demand.apply(
                lambda row: f"Low demand item (avg {row['avg_daily_demand']:.1f}/day). Review stock levels.",
                axis=1
            )
            
            alerts['understock'] = high_demand
            alerts['overstock'] = low_demand
        
        self.decisions['stock_alerts'] = alerts
        
        print(f"✓ Generated {len(alerts['understock'])} understock alerts and {len(alerts['overstock'])} overstock alerts")
        return alerts
    
    def prepare_forecasting_inputs(self) -> Dict[str, pd.DataFrame]:
        """
        Prepare clean inputs for demand forecasting models.
        
        Business Value:
        - Enables AI/ML forecasting models
        - Provides time-series data in standard format
        - Includes features for model training
        
        Returns:
        --------
        dict with keys:
        - 'time_series': Daily/weekly time series data
        - 'features': Feature matrix for ML models
        - 'metadata': Item/location metadata
        """
        forecasting_inputs = {}
        
        # Get demand data
        if 'demand_per_sku' not in self.analytics.analytics_tables:
            demand_daily = self.analytics.calculate_demand_per_sku(period='daily')
            demand_weekly = self.analytics.calculate_demand_per_sku(period='weekly')
        else:
            demand_daily = self.analytics.analytics_tables['demand_per_sku']
            demand_weekly = self.analytics.calculate_demand_per_sku(period='weekly')
        
        # Prepare time series format
        if 'item_id' in demand_daily.columns and 'period' in demand_daily.columns:
            # Pivot to time series format
            ts_data = demand_daily.pivot_table(
                index='period',
                columns='item_id',
                values='demand_quantity',
                fill_value=0
            )
            
            forecasting_inputs['time_series_daily'] = ts_data
            forecasting_inputs['time_series_weekly'] = demand_weekly
        
        # Prepare feature matrix
        features = []
        if 'fct_order_items.csv' in self.analytics.data:
            order_items = self.analytics.data['fct_order_items.csv'].copy()
            
            # Aggregate features per item
            if 'item_id' in order_items.columns:
                item_features = order_items.groupby('item_id').agg({
                    'quantity': ['sum', 'mean', 'std', 'count'],
                    'price': ['mean', 'std'],
                    'cost': ['sum', 'mean']
                }).reset_index()
                
                # Flatten column names
                item_features.columns = ['item_id'] + [
                    f"{col[0]}_{col[1]}" for col in item_features.columns[1:]
                ]
                
                features = item_features
        
        forecasting_inputs['features'] = pd.DataFrame(features) if features else pd.DataFrame()
        
        # Metadata
        metadata = {}
        if 'dim_menu_items.csv' in self.analytics.data:
            menu_items = self.analytics.data['dim_menu_items.csv'].copy()
            if 'id' in menu_items.columns:
                metadata['menu_items'] = menu_items[['id', 'title', 'price', 'status']].copy()
        
        forecasting_inputs['metadata'] = metadata
        
        self.decisions['forecasting_inputs'] = forecasting_inputs
        
        print("✓ Prepared forecasting inputs (time series, features, metadata)")
        return forecasting_inputs
    
    def generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary of all decisions and recommendations."""
        summary = {
            'generated_at': datetime.now().isoformat(),
            'prep_recommendations_count': len(self.decisions.get('prep_recommendations', pd.DataFrame())),
            'expiry_risks_count': len(self.decisions.get('expiry_risks', pd.DataFrame())),
            'stock_alerts': {
                'understock_count': len(self.decisions.get('stock_alerts', {}).get('understock', pd.DataFrame())),
                'overstock_count': len(self.decisions.get('stock_alerts', {}).get('overstock', pd.DataFrame()))
            },
            'key_insights': []
        }
        
        # Add key insights
        if 'prep_recommendations' in self.decisions and len(self.decisions['prep_recommendations']) > 0:
            prep_recs = self.decisions['prep_recommendations']
            if 'recommended_daily_prep' in prep_recs.columns:
                total_prep = prep_recs['recommended_daily_prep'].sum()
                summary['key_insights'].append(
                    f"Total recommended daily prep quantity: {total_prep:.0f} units across all items"
                )
        
        if 'expiry_risks' in self.decisions and len(self.decisions['expiry_risks']) > 0:
            high_risk_count = len(self.decisions['expiry_risks'])
            summary['key_insights'].append(
                f"{high_risk_count} items identified at high risk of expiration - action required"
            )
        
        return summary
    
    def save_decisions(self, output_dir: str):
        """Save all decision outputs to CSV files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        for decision_name, decision_data in self.decisions.items():
            if isinstance(decision_data, pd.DataFrame):
                if len(decision_data) > 0:
                    output_file = output_path / f"decision_{decision_name}.csv"
                    decision_data.to_csv(output_file, index=False)
                    print(f"✓ Saved {decision_name} to {output_file}")
            elif isinstance(decision_data, dict):
                for sub_name, sub_data in decision_data.items():
                    if isinstance(sub_data, pd.DataFrame) and len(sub_data) > 0:
                        output_file = output_path / f"decision_{decision_name}_{sub_name}.csv"
                        sub_data.to_csv(output_file, index=False)
                        print(f"✓ Saved {decision_name}.{sub_name} to {output_file}")


if __name__ == "__main__":
    # Example usage
    pass
