"""
Inventory Decision Engine
=========================
Rule-based + ML-driven decision engine for inventory management.
Provides actionable recommendations with business impact estimates.

Architecture:
- Deterministic rules first (explainable)
- ML forecasts as inputs, not final decisions
- Human-readable outputs
- Streamlit-ready
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class InventoryDecisionEngine:
    """
    Core decision engine for inventory management.
    Combines rule-based logic with ML forecast inputs.
    """
    
    def __init__(self, 
                 cleaned_data: Dict[str, pd.DataFrame],
                 analytics_data: Optional[Dict[str, pd.DataFrame]] = None):
        """
        Initialize decision engine with cleaned data.
        
        Parameters:
        -----------
        cleaned_data : dict
            Dictionary of cleaned DataFrames
        analytics_data : dict, optional
            Pre-computed analytics tables
        """
        self.cleaned_data = cleaned_data
        self.analytics_data = analytics_data or {}
        self.decisions = []
        self.ml_forecasts = {}  # Store ML forecast inputs
        
    def load_analytics(self, analytics_dir: str):
        """Load pre-computed analytics from directory."""
        analytics_path = Path(analytics_dir)
        
        if (analytics_path / "analytics_demand_per_sku.csv").exists():
            self.analytics_data['demand'] = pd.read_csv(
                analytics_path / "analytics_demand_per_sku.csv"
            )
        
        if (analytics_path / "analytics_stock_levels.csv").exists():
            self.analytics_data['stock_levels'] = pd.read_csv(
                analytics_path / "analytics_stock_levels.csv"
            )
        
        if (analytics_path / "analytics_waste_metrics.csv").exists():
            self.analytics_data['waste_metrics'] = pd.read_csv(
                analytics_path / "analytics_waste_metrics.csv"
            )
    
    def set_ml_forecast(self, item_id: int, forecast: Dict[str, Any]):
        """
        Set ML forecast for an item.
        
        Parameters:
        -----------
        item_id : int
            Menu item ID
        forecast : dict
            Forecast data with keys: 'predicted_demand', 'confidence', 'model_type'
        """
        self.ml_forecasts[item_id] = forecast
    
    def calculate_prep_quantity(self,
                               item_id: int,
                               current_stock: float = 0,
                               days_ahead: int = 1,
                               use_ml: bool = False) -> Dict[str, Any]:
        """
        Calculate recommended prep quantity for an item.
        
        Rule-based logic:
        1. Use historical average demand
        2. Add safety stock based on variability
        3. Consider current stock levels
        4. Optionally incorporate ML forecast
        
        Parameters:
        -----------
        item_id : int
            Menu item ID
        current_stock : float
            Current stock level
        days_ahead : int
            Number of days to plan for
        use_ml : bool
            Whether to use ML forecast if available
        
        Returns:
        --------
        dict with recommendation details
        """
        # Get historical demand
        historical_demand = self._get_historical_demand(item_id)
        
        if historical_demand is None:
            return {
                'item_id': item_id,
                'recommendation': 'insufficient_data',
                'prep_quantity': 0,
                'reason': 'No historical demand data available',
                'rule_used': 'N/A',
                'business_impact': 'Cannot calculate - need historical data'
            }
        
        avg_daily_demand = historical_demand.get('avg_daily', 0)
        demand_std = historical_demand.get('std_daily', 0)
        max_daily_demand = historical_demand.get('max_daily', 0)
        
        # Get ML forecast if available and requested
        ml_forecast = None
        if use_ml and item_id in self.ml_forecasts:
            ml_forecast = self.ml_forecasts[item_id]
            predicted_demand = ml_forecast.get('predicted_demand', avg_daily_demand)
            confidence = ml_forecast.get('confidence', 0.5)
            
            # Blend ML forecast with historical (weighted by confidence)
            blended_demand = (predicted_demand * confidence + 
                            avg_daily_demand * (1 - confidence))
            avg_daily_demand = blended_demand
        
        # Calculate base prep quantity
        base_prep = avg_daily_demand * days_ahead
        
        # Add safety stock (1.5x standard deviation or 20% buffer)
        if demand_std > 0:
            safety_stock = demand_std * 1.5
        else:
            safety_stock = base_prep * 0.2
        
        # Adjust for current stock
        net_prep_needed = base_prep + safety_stock - current_stock
        
        # Ensure non-negative
        recommended_prep = max(0, net_prep_needed)
        
        # Cap at reasonable maximum (2x average or max seen)
        max_reasonable = max(avg_daily_demand * 2, max_daily_demand) * days_ahead
        recommended_prep = min(recommended_prep, max_reasonable)
        
        # Round to reasonable precision
        recommended_prep = round(recommended_prep, 1)
        
        # Determine confidence level
        if historical_demand.get('days_with_sales', 0) > 20:
            confidence_level = 'high'
        elif historical_demand.get('days_with_sales', 0) > 10:
            confidence_level = 'medium'
        else:
            confidence_level = 'low'
        
        # Calculate business impact
        waste_risk = self._estimate_waste_risk(recommended_prep, avg_daily_demand * days_ahead)
        stockout_risk = self._estimate_stockout_risk(recommended_prep, avg_daily_demand * days_ahead)
        
        rule_used = "Historical average + safety stock (1.5σ)"
        if ml_forecast:
            rule_used += f" + ML forecast (confidence: {confidence:.0%})"
        if current_stock > 0:
            rule_used += f" - current stock adjustment"
        
        return {
            'item_id': item_id,
            'recommendation': 'prep',
            'prep_quantity': recommended_prep,
            'base_demand': round(avg_daily_demand * days_ahead, 1),
            'safety_stock': round(safety_stock, 1),
            'current_stock': current_stock,
            'confidence': confidence_level,
            'rule_used': rule_used,
            'business_impact': {
                'waste_risk': waste_risk,
                'stockout_risk': stockout_risk,
                'expected_waste_reduction': f"{waste_risk:.0%}",
                'expected_stockout_prevention': f"{stockout_risk:.0%}"
            },
            'ml_forecast_used': ml_forecast is not None
        }
    
    def assess_expiry_risk(self, item_id: int, current_stock: float) -> Dict[str, Any]:
        """
        Assess expiry risk for an item.
        
        Rules:
        1. Days since last sale > 30 = high risk
        2. Low demand + high stock = high risk
        3. No sales in 60+ days = critical risk
        
        Returns:
        --------
        dict with expiry risk assessment and recommendations
        """
        # Get demand data
        historical_demand = self._get_historical_demand(item_id)
        
        if historical_demand is None:
            return {
                'item_id': item_id,
                'expiry_risk_score': 0,
                'risk_level': 'unknown',
                'recommendations': []
            }
        
        days_since_sale = historical_demand.get('days_since_last_sale', 999)
        avg_daily_demand = historical_demand.get('avg_daily', 0)
        total_demand = historical_demand.get('total_demand', 0)
        
        # Calculate risk score (0-100)
        risk_score = 0
        
        # Days since sale component (0-50 points)
        if days_since_sale > 90:
            risk_score += 50
        elif days_since_sale > 60:
            risk_score += 35
        elif days_since_sale > 30:
            risk_score += 20
        elif days_since_sale > 14:
            risk_score += 10
        
        # Demand component (0-30 points)
        if total_demand == 0:
            risk_score += 30
        elif total_demand < 10:
            risk_score += 20
        elif total_demand < 50:
            risk_score += 10
        
        # Stock level component (0-20 points)
        if current_stock > 0:
            days_of_stock = current_stock / avg_daily_demand if avg_daily_demand > 0 else 999
            if days_of_stock > 30:
                risk_score += 20
            elif days_of_stock > 14:
                risk_score += 10
        
        # Determine risk level
        if risk_score >= 70:
            risk_level = 'critical'
        elif risk_score >= 50:
            risk_level = 'high'
        elif risk_score >= 30:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        # Generate recommendations
        recommendations = []
        
        if risk_score >= 70:
            recommendations.append({
                'action': 'discount',
                'priority': 'urgent',
                'discount_percent': 30,
                'reason': f'Critical expiry risk: No sales in {days_since_sale} days, {current_stock:.1f} units in stock',
                'expected_impact': 'Reduce waste by 70-90%, generate revenue from slow-moving inventory',
                'rule_used': 'Critical risk threshold (score >= 70)'
            })
            recommendations.append({
                'action': 'reduce_prep',
                'priority': 'high',
                'reason': 'Stop ordering until current stock is cleared',
                'expected_impact': 'Prevent additional waste accumulation',
                'rule_used': 'Critical risk - halt new prep'
            })
        elif risk_score >= 50:
            recommendations.append({
                'action': 'discount',
                'priority': 'high',
                'discount_percent': 20,
                'reason': f'High expiry risk: {days_since_sale} days since last sale, {current_stock:.1f} units in stock',
                'expected_impact': 'Reduce waste by 50-70%, improve cash flow',
                'rule_used': 'High risk threshold (score >= 50)'
            })
            recommendations.append({
                'action': 'bundle',
                'priority': 'medium',
                'reason': 'Bundle with fast-moving items to increase visibility',
                'expected_impact': 'Increase sales velocity by 30-50%',
                'rule_used': 'Bundling strategy for slow movers'
            })
        elif risk_score >= 30:
            recommendations.append({
                'action': 'monitor',
                'priority': 'medium',
                'reason': f'Moderate risk: Monitor closely, consider small discount if no improvement',
                'expected_impact': 'Early intervention prevents waste',
                'rule_used': 'Medium risk - proactive monitoring'
            })
        
        return {
            'item_id': item_id,
            'expiry_risk_score': risk_score,
            'risk_level': risk_level,
            'days_since_last_sale': days_since_sale,
            'current_stock': current_stock,
            'avg_daily_demand': round(avg_daily_demand, 2),
            'recommendations': recommendations,
            'rule_used': f'Risk scoring: days_since_sale ({days_since_sale}) + demand ({total_demand}) + stock ({current_stock})'
        }
    
    def generate_stock_alerts(self) -> List[Dict[str, Any]]:
        """
        Generate stock alerts (understock/overstock).
        
        Rules:
        - Understock: High demand + low stock
        - Overstock: Low demand + high stock
        
        Returns:
        --------
        List of alert dictionaries
        """
        alerts = []
        
        # Get stock levels
        stock_levels = self.analytics_data.get('stock_levels', pd.DataFrame())
        if stock_levels.empty and 'dim_skus.csv' in self.cleaned_data:
            stock_levels = self.cleaned_data['dim_skus.csv'].copy()
        
        # Get demand data
        demand_data = self.analytics_data.get('demand', pd.DataFrame())
        
        if demand_data.empty:
            return alerts
        
        # Calculate average daily demand per item
        if 'item_id' in demand_data.columns:
            avg_demand = demand_data.groupby('item_id').agg({
                'demand_quantity': ['mean', 'sum', 'count']
            }).reset_index()
            avg_demand.columns = ['item_id', 'avg_daily_demand', 'total_demand', 'days_with_sales']
            
            for _, row in avg_demand.iterrows():
                item_id = row['item_id']
                avg_daily = row['avg_daily_demand']
                total_demand = row['total_demand']
                
                # Get current stock (simplified - would need proper SKU mapping)
                current_stock = 0  # Placeholder
                
                # Understock alert
                if avg_daily > 10 and current_stock < avg_daily * 2:
                    alerts.append({
                        'item_id': item_id,
                        'alert_type': 'understock',
                        'severity': 'high' if current_stock < avg_daily else 'medium',
                        'current_stock': current_stock,
                        'avg_daily_demand': round(avg_daily, 2),
                        'days_of_stock': round(current_stock / avg_daily, 1) if avg_daily > 0 else 0,
                        'recommendation': {
                            'action': 'reorder',
                            'priority': 'high' if current_stock < avg_daily else 'medium',
                            'reorder_quantity': round(avg_daily * 7, 1),  # 7 days supply
                            'reason': f'High demand item ({avg_daily:.1f}/day) running low on stock',
                            'expected_impact': 'Prevent stockouts, maintain sales velocity',
                            'rule_used': f'Understock rule: demand ({avg_daily:.1f}) > threshold AND stock ({current_stock}) < 2x demand'
                        }
                    })
                
                # Overstock alert
                if avg_daily < 2 and current_stock > avg_daily * 14:
                    alerts.append({
                        'item_id': item_id,
                        'alert_type': 'overstock',
                        'severity': 'high' if current_stock > avg_daily * 30 else 'medium',
                        'current_stock': current_stock,
                        'avg_daily_demand': round(avg_daily, 2),
                        'days_of_stock': round(current_stock / avg_daily, 1) if avg_daily > 0 else 999,
                        'recommendation': {
                            'action': 'reduce_prep',
                            'priority': 'high' if current_stock > avg_daily * 30 else 'medium',
                            'reason': f'Low demand item ({avg_daily:.1f}/day) with excessive stock ({current_stock:.1f} units)',
                            'expected_impact': 'Reduce waste risk, free up capital',
                            'rule_used': f'Overstock rule: demand ({avg_daily:.1f}) < threshold AND stock ({current_stock}) > 14x demand'
                        }
                    })
        
        return alerts
    
    def generate_all_recommendations(self) -> pd.DataFrame:
        """
        Generate all recommendations for all items.
        
        Returns:
        --------
        DataFrame with all recommendations
        """
        all_recommendations = []
        
        # Get all items with demand
        demand_data = self.analytics_data.get('demand', pd.DataFrame())
        if demand_data.empty:
            return pd.DataFrame()
        
        if 'item_id' in demand_data.columns:
            unique_items = demand_data['item_id'].unique()
            
            for item_id in unique_items[:1000]:  # Limit for performance
                # Prep quantity recommendation
                prep_rec = self.calculate_prep_quantity(item_id)
                prep_rec['recommendation_type'] = 'prep_quantity'
                all_recommendations.append(prep_rec)
                
                # Expiry risk assessment
                expiry_assessment = self.assess_expiry_risk(item_id, current_stock=0)
                if expiry_assessment['risk_level'] in ['high', 'critical']:
                    expiry_assessment['recommendation_type'] = 'expiry_risk'
                    all_recommendations.append(expiry_assessment)
        
        # Add stock alerts
        alerts = self.generate_stock_alerts()
        for alert in alerts:
            alert['recommendation_type'] = 'stock_alert'
            all_recommendations.append(alert)
        
        return pd.DataFrame(all_recommendations)
    
    def format_recommendation_for_display(self, recommendation: Dict[str, Any]) -> str:
        """
        Format a recommendation for human-readable display.
        
        Parameters:
        -----------
        recommendation : dict
            Recommendation dictionary
        
        Returns:
        --------
        Formatted string
        """
        rec_type = recommendation.get('recommendation_type', 'unknown')
        
        if rec_type == 'prep_quantity':
            return self._format_prep_recommendation(recommendation)
        elif rec_type == 'expiry_risk':
            return self._format_expiry_recommendation(recommendation)
        elif rec_type == 'stock_alert':
            return self._format_stock_alert(recommendation)
        else:
            return str(recommendation)
    
    def _format_prep_recommendation(self, rec: Dict[str, Any]) -> str:
        """Format prep quantity recommendation."""
        lines = [
            f"📦 PREP RECOMMENDATION - Item ID: {rec['item_id']}",
            f"   Recommended Quantity: {rec['prep_quantity']:.1f} units",
            f"   Base Demand: {rec.get('base_demand', 0):.1f} units",
            f"   Safety Stock: {rec.get('safety_stock', 0):.1f} units",
            f"   Current Stock: {rec.get('current_stock', 0):.1f} units",
            f"   Confidence: {rec.get('confidence', 'unknown').upper()}",
            f"   Rule Used: {rec.get('rule_used', 'N/A')}",
            f"   Business Impact:",
            f"     - Waste Risk: {rec.get('business_impact', {}).get('waste_risk', 0):.0%}",
            f"     - Stockout Risk: {rec.get('business_impact', {}).get('stockout_risk', 0):.0%}"
        ]
        return "\n".join(lines)
    
    def _format_expiry_recommendation(self, rec: Dict[str, Any]) -> str:
        """Format expiry risk recommendation."""
        lines = [
            f"⚠️ EXPIRY RISK ALERT - Item ID: {rec['item_id']}",
            f"   Risk Level: {rec['risk_level'].upper()} (Score: {rec['expiry_risk_score']}/100)",
            f"   Days Since Last Sale: {rec.get('days_since_last_sale', 'N/A')}",
            f"   Current Stock: {rec.get('current_stock', 0):.1f} units",
            f"   Average Daily Demand: {rec.get('avg_daily_demand', 0):.2f} units",
            f"   Rule Used: {rec.get('rule_used', 'N/A')}",
            f"   RECOMMENDED ACTIONS:"
        ]
        
        for action in rec.get('recommendations', []):
            lines.append(f"     • {action['action'].upper()} (Priority: {action['priority']})")
            lines.append(f"       Reason: {action['reason']}")
            lines.append(f"       Expected Impact: {action['expected_impact']}")
            if 'discount_percent' in action:
                lines.append(f"       Discount: {action['discount_percent']}%")
        
        return "\n".join(lines)
    
    def _format_stock_alert(self, alert: Dict[str, Any]) -> str:
        """Format stock alert."""
        alert_type = alert.get('alert_type', 'unknown')
        icon = "🔴" if alert_type == 'understock' else "🟡"
        
        lines = [
            f"{icon} STOCK ALERT - Item ID: {alert['item_id']}",
            f"   Type: {alert_type.upper()}",
            f"   Severity: {alert.get('severity', 'unknown').upper()}",
            f"   Current Stock: {alert.get('current_stock', 0):.1f} units",
            f"   Average Daily Demand: {alert.get('avg_daily_demand', 0):.2f} units",
            f"   Days of Stock: {alert.get('days_of_stock', 0):.1f} days"
        ]
        
        rec = alert.get('recommendation', {})
        if rec:
            lines.append(f"   RECOMMENDATION:")
            lines.append(f"     Action: {rec.get('action', 'N/A').upper()}")
            lines.append(f"     Priority: {rec.get('priority', 'N/A').upper()}")
            lines.append(f"     Reason: {rec.get('reason', 'N/A')}")
            lines.append(f"     Expected Impact: {rec.get('expected_impact', 'N/A')}")
            if 'reorder_quantity' in rec:
                lines.append(f"     Reorder Quantity: {rec['reorder_quantity']:.1f} units")
        
        return "\n".join(lines)
    
    # Helper methods
    
    def _get_historical_demand(self, item_id: int) -> Optional[Dict[str, Any]]:
        """Get historical demand statistics for an item."""
        demand_data = self.analytics_data.get('demand', pd.DataFrame())
        
        if demand_data.empty or 'item_id' not in demand_data.columns:
            return None
        
        item_demand = demand_data[demand_data['item_id'] == item_id]
        
        if len(item_demand) == 0:
            return None
        
        # Calculate statistics
        total_demand = item_demand['demand_quantity'].sum()
        avg_daily = item_demand['demand_quantity'].mean()
        std_daily = item_demand['demand_quantity'].std()
        max_daily = item_demand['demand_quantity'].max()
        days_with_sales = len(item_demand)
        
        # Get last sale date (if available)
        if 'period' in item_demand.columns:
            try:
                last_period = item_demand['period'].max()
                # Try to parse as date
                if isinstance(last_period, str):
                    last_date = pd.to_datetime(last_period, errors='coerce')
                    if pd.notna(last_date):
                        days_since = (datetime.now() - last_date).days
                    else:
                        days_since = 999
                else:
                    days_since = 999
            except:
                days_since = 999
        else:
            days_since = 999
        
        return {
            'total_demand': total_demand,
            'avg_daily': avg_daily,
            'std_daily': std_daily if not pd.isna(std_daily) else 0,
            'max_daily': max_daily,
            'days_with_sales': days_with_sales,
            'days_since_last_sale': days_since
        }
    
    def _estimate_waste_risk(self, prep_quantity: float, expected_demand: float) -> float:
        """Estimate waste risk percentage."""
        if expected_demand == 0:
            return 1.0  # 100% waste risk if no demand
        
        excess = prep_quantity - expected_demand
        if excess <= 0:
            return 0.0
        
        waste_ratio = excess / prep_quantity
        return min(1.0, waste_ratio)
    
    def _estimate_stockout_risk(self, prep_quantity: float, expected_demand: float) -> float:
        """Estimate stockout risk percentage."""
        if expected_demand == 0:
            return 0.0
        
        shortage = expected_demand - prep_quantity
        if shortage <= 0:
            return 0.0
        
        stockout_ratio = shortage / expected_demand
        return min(1.0, stockout_ratio)


# Example usage and testing functions

def example_usage():
    """Example usage of the decision engine."""
    print("=" * 80)
    print("INVENTORY DECISION ENGINE - EXAMPLE USAGE")
    print("=" * 80)
    
    # Load cleaned data (example)
    data_dir = Path(__file__).parent
    cleaned_data = {}
    
    # Load key tables
    for file in ['fct_orders.csv', 'fct_order_items.csv', 'dim_menu_items.csv', 'dim_skus.csv']:
        file_path = data_dir / file
        if file_path.exists():
            cleaned_data[file] = pd.read_csv(file_path)
    
    # Initialize engine
    engine = InventoryDecisionEngine(cleaned_data)
    
    # Load analytics if available
    analytics_dir = data_dir / "outputs" / "analytics"
    if analytics_dir.exists():
        engine.load_analytics(str(analytics_dir))
    
    # Example 1: Calculate prep quantity
    print("\n" + "-" * 80)
    print("EXAMPLE 1: Calculate Prep Quantity")
    print("-" * 80)
    
    # Get a sample item ID
    if 'demand' in engine.analytics_data and len(engine.analytics_data['demand']) > 0:
        sample_item = engine.analytics_data['demand']['item_id'].iloc[0]
        
        prep_rec = engine.calculate_prep_quantity(sample_item, current_stock=5, days_ahead=1)
        print(engine._format_prep_recommendation(prep_rec))
    
    # Example 2: Assess expiry risk
    print("\n" + "-" * 80)
    print("EXAMPLE 2: Assess Expiry Risk")
    print("-" * 80)
    
    if 'demand' in engine.analytics_data and len(engine.analytics_data['demand']) > 0:
        sample_item = engine.analytics_data['demand']['item_id'].iloc[0]
        
        expiry_assessment = engine.assess_expiry_risk(sample_item, current_stock=50)
        print(engine._format_expiry_recommendation(expiry_assessment))
    
    # Example 3: Generate stock alerts
    print("\n" + "-" * 80)
    print("EXAMPLE 3: Generate Stock Alerts")
    print("-" * 80)
    
    alerts = engine.generate_stock_alerts()
    for alert in alerts[:3]:  # Show first 3
        print(engine._format_stock_alert(alert))
        print()
    
    print("\n" + "=" * 80)
    print("Example usage complete!")
    print("=" * 80)


if __name__ == "__main__":
    example_usage()
