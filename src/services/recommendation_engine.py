"""
Recommendation Engine Service
==============================
Generate actionable, explainable recommendations for inventory decisions.

Design Principles:
- Every recommendation has an explanation in plain English
- Recommendations are prioritized by business impact
- Logic is deterministic and rule-based (no black-box ML)
- All thresholds are configurable

Recommendation Types:
1. Order Recommendations: When and how much to reorder
2. Promotion Suggestions: Mark down slow movers or near-expiry items
3. Waste Risk Alerts: Flag items at risk of expiration
4. Bundle Suggestions: Combine slow movers with fast movers

Output Format:
- Each recommendation includes:
  - action: What to do
  - item_id: Which item
  - urgency: HIGH, MEDIUM, LOW
  - impact: Estimated business impact
  - explanation: Plain English reasoning
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum

from utils.logger import get_logger
from utils.constants import RECOMMENDATION_CONFIG, HEALTH_SCORE_CONFIG

logger = get_logger(__name__)


class ActionType(Enum):
    """Types of actions that can be recommended."""
    REORDER = "reorder"
    DISCOUNT = "discount"
    BUNDLE = "bundle"
    REDUCE_ORDER = "reduce_order"
    INCREASE_ORDER = "increase_order"
    MONITOR = "monitor"
    REMOVE = "remove"
    PRIORITIZE_SALES = "prioritize_sales"


class Urgency(Enum):
    """Urgency levels for recommendations."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class Recommendation:
    """
    A single actionable recommendation.
    
    Attributes
    ----------
    item_id : Any
        Identifier of the affected item
    item_name : str
        Human-readable name of the item
    action : ActionType
        Type of action recommended
    urgency : Urgency
        How urgent is this action
    quantity : Optional[float]
        Quantity involved (e.g., reorder amount)
    discount_pct : Optional[float]
        Discount percentage if applicable
    estimated_impact : float
        Estimated business impact (revenue saved/gained)
    explanation : str
        Plain English explanation
    supporting_data : Dict[str, Any]
        Data that supports this recommendation
    """
    item_id: Any
    item_name: str
    action: ActionType
    urgency: Urgency
    quantity: Optional[float] = None
    discount_pct: Optional[float] = None
    estimated_impact: float = 0.0
    explanation: str = ""
    supporting_data: Dict[str, Any] = field(default_factory=dict)


class RecommendationEngine:
    """
    Generate inventory recommendations based on health scores and forecasts.
    
    This engine applies business rules to generate actionable recommendations.
    All rules are explicit and documented.
    
    Rule Categories:
    1. Stock Management Rules
       - Reorder when days_of_supply < safety_threshold
       - Reduce orders when days_of_supply > overstock_threshold
    
    2. Expiry Management Rules
       - Discount at 60% shelf life remaining
       - Bundle at 40% shelf life remaining
       - Prioritize sales at < 3 days to expiry
    
    3. Demand-Based Rules
       - Increase orders for growing demand
       - Decrease orders for declining demand
       - Remove items with no demand for 14+ days
    
    Usage
    -----
    >>> engine = RecommendationEngine()
    >>> recommendations = engine.generate_recommendations(
    ...     health_scores,
    ...     forecast_df,
    ...     inventory_df
    ... )
    >>> for rec in recommendations:
    ...     print(f"{rec.urgency.value}: {rec.explanation}")
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the recommendation engine.
        
        Parameters
        ----------
        config : dict, optional
            Custom configuration. Uses defaults if not provided.
        """
        self.config = config or RECOMMENDATION_CONFIG
        
        logger.info("RecommendationEngine initialized")
    
    def generate_recommendations(
        self,
        health_scores: List['HealthScore'],
        forecast_df: pd.DataFrame,
        inventory_df: pd.DataFrame,
        item_names: Optional[Dict[Any, str]] = None
    ) -> List[Recommendation]:
        """
        Generate all recommendations based on current state.
        
        Parameters
        ----------
        health_scores : List[HealthScore]
            Health assessments for items
        forecast_df : pd.DataFrame
            Demand forecasts
        inventory_df : pd.DataFrame
            Current inventory levels
        item_names : Dict[Any, str], optional
            Mapping of item_id to display name
        
        Returns
        -------
        List[Recommendation]
            Prioritized list of recommendations
        """
        item_names = item_names or {}
        all_recommendations = []
        
        for health in health_scores:
            item_id = health.item_id
            item_name = item_names.get(item_id, f"Item {item_id}")
            
            # Get item-specific data
            item_forecast = forecast_df[forecast_df['item_id'] == item_id] \
                if 'item_id' in forecast_df.columns else pd.DataFrame()
            
            item_inventory = self._get_item_inventory(inventory_df, item_id)
            
            # Generate recommendations based on rules
            recs = []
            
            # 1. Stock Level Rules
            recs.extend(self._apply_stock_rules(
                health, item_forecast, item_inventory, item_id, item_name
            ))
            
            # 2. Expiry Rules
            recs.extend(self._apply_expiry_rules(
                health, item_inventory, item_id, item_name
            ))
            
            # 3. Demand Trend Rules
            recs.extend(self._apply_trend_rules(
                health, item_forecast, item_id, item_name
            ))
            
            # 4. Waste Reduction Rules
            recs.extend(self._apply_waste_rules(
                health, item_inventory, item_id, item_name
            ))
            
            all_recommendations.extend(recs)
        
        # Sort by urgency and impact
        all_recommendations.sort(
            key=lambda r: (
                self._urgency_order(r.urgency),
                -r.estimated_impact
            )
        )
        
        # Log summary
        critical = sum(1 for r in all_recommendations if r.urgency == Urgency.CRITICAL)
        high = sum(1 for r in all_recommendations if r.urgency == Urgency.HIGH)
        
        logger.info(
            f"Generated {len(all_recommendations)} recommendations: "
            f"{critical} CRITICAL, {high} HIGH"
        )
        
        return all_recommendations
    
    def _get_item_inventory(
        self,
        inventory_df: pd.DataFrame,
        item_id: Any
    ) -> Dict[str, Any]:
        """Get inventory data for an item."""
        if len(inventory_df) == 0:
            return {}
        
        # Try different column names for item matching
        for col in ['item_id', 'id', 'sku_id']:
            if col in inventory_df.columns:
                match = inventory_df[inventory_df[col] == item_id]
                if len(match) > 0:
                    return match.iloc[0].to_dict()
        
        return {}
    
    def _apply_stock_rules(
        self,
        health: 'HealthScore',
        forecast_df: pd.DataFrame,
        inventory: Dict,
        item_id: Any,
        item_name: str
    ) -> List[Recommendation]:
        """Apply stock level rules."""
        recs = []
        
        stock_factors = health.factors.get('stock_level', {})
        status = stock_factors.get('status', 'unknown')
        days_of_supply = stock_factors.get('days_of_supply', 0)
        current_stock = stock_factors.get('current_stock', 0)
        avg_daily = stock_factors.get('avg_daily_demand', 0)
        
        # Rule: Understock - need to reorder
        if status == 'understock' or (days_of_supply and days_of_supply < 3):
            # Calculate reorder quantity
            target_days = 10  # Target 10 days of supply
            reorder_qty = max(0, (target_days * avg_daily) - current_stock)
            
            # Apply safety stock
            safety_multiplier = self.config.get('safety_stock_multiplier', 1.2)
            reorder_qty = reorder_qty * safety_multiplier
            
            if reorder_qty > 0:
                urgency = Urgency.CRITICAL if days_of_supply < 1 else Urgency.HIGH
                
                recs.append(Recommendation(
                    item_id=item_id,
                    item_name=item_name,
                    action=ActionType.REORDER,
                    urgency=urgency,
                    quantity=round(reorder_qty, 0),
                    estimated_impact=round(reorder_qty * 10, 2),  # Assume $10 per unit
                    explanation=(
                        f"Reorder {reorder_qty:.0f} units of {item_name}. "
                        f"Current stock ({current_stock:.0f}) only covers "
                        f"{days_of_supply:.1f} days of expected demand "
                        f"({avg_daily:.1f} units/day). "
                        f"Target is 10+ days of supply to avoid stockouts."
                    ),
                    supporting_data={
                        'current_stock': current_stock,
                        'days_of_supply': days_of_supply,
                        'avg_daily_demand': avg_daily,
                        'target_days': target_days
                    }
                ))
        
        # Rule: Overstock - reduce future orders
        elif status == 'overstock' or (days_of_supply and days_of_supply > 21):
            overstock_amount = current_stock - (14 * avg_daily)  # Excess over 14 days
            
            if overstock_amount > 0:
                recs.append(Recommendation(
                    item_id=item_id,
                    item_name=item_name,
                    action=ActionType.REDUCE_ORDER,
                    urgency=Urgency.MEDIUM,
                    quantity=round(overstock_amount, 0),
                    estimated_impact=round(overstock_amount * 5, 2),  # Tied-up capital
                    explanation=(
                        f"Consider reducing orders for {item_name}. "
                        f"Current stock ({current_stock:.0f} units) represents "
                        f"{days_of_supply:.0f} days of supply. "
                        f"This ties up capital and increases waste risk. "
                        f"Recommend reducing next order by {overstock_amount:.0f} units."
                    ),
                    supporting_data={
                        'current_stock': current_stock,
                        'days_of_supply': days_of_supply,
                        'excess_amount': overstock_amount
                    }
                ))
        
        return recs
    
    def _apply_expiry_rules(
        self,
        health: 'HealthScore',
        inventory: Dict,
        item_id: Any,
        item_name: str
    ) -> List[Recommendation]:
        """Apply expiry management rules."""
        recs = []
        
        expiry_factors = health.factors.get('days_to_expiry', {})
        days_to_expiry = expiry_factors.get('days_to_expiry')
        status = expiry_factors.get('status', 'unknown')
        
        if days_to_expiry is None:
            return recs  # Non-perishable, no expiry rules
        
        stock = health.factors.get('stock_level', {}).get('current_stock', 0)
        
        # Rule: Critical expiry - prioritize sales
        if days_to_expiry <= 3 and stock > 0:
            discount_pct = self.config.get('promotion_discount_pct', 0.20) * 2  # 40% off
            
            recs.append(Recommendation(
                item_id=item_id,
                item_name=item_name,
                action=ActionType.PRIORITIZE_SALES,
                urgency=Urgency.CRITICAL,
                quantity=stock,
                discount_pct=discount_pct,
                estimated_impact=round(stock * 8, 2),  # Value at risk
                explanation=(
                    f"URGENT: {item_name} expires in {days_to_expiry} days! "
                    f"{stock:.0f} units at risk of waste. "
                    f"Recommend {discount_pct*100:.0f}% discount to clear inventory. "
                    f"Consider featuring prominently in store."
                ),
                supporting_data={
                    'days_to_expiry': days_to_expiry,
                    'stock_at_risk': stock
                }
            ))
        
        # Rule: Warning expiry - offer discount
        elif days_to_expiry <= 7 and stock > 0:
            discount_pct = self.config.get('promotion_discount_pct', 0.20)
            
            recs.append(Recommendation(
                item_id=item_id,
                item_name=item_name,
                action=ActionType.DISCOUNT,
                urgency=Urgency.HIGH,
                quantity=stock,
                discount_pct=discount_pct,
                estimated_impact=round(stock * 5, 2),
                explanation=(
                    f"{item_name} expires in {days_to_expiry} days. "
                    f"{stock:.0f} units should be discounted by {discount_pct*100:.0f}% "
                    f"to accelerate sales and reduce waste risk."
                ),
                supporting_data={
                    'days_to_expiry': days_to_expiry,
                    'stock_remaining': stock
                }
            ))
        
        return recs
    
    def _apply_trend_rules(
        self,
        health: 'HealthScore',
        forecast_df: pd.DataFrame,
        item_id: Any,
        item_name: str
    ) -> List[Recommendation]:
        """Apply demand trend rules."""
        recs = []
        
        trend_factors = health.factors.get('demand_trend', {})
        trend_pct = trend_factors.get('trend_pct', 0)
        status = trend_factors.get('status', 'unknown')
        
        avg_daily = health.factors.get('stock_level', {}).get('avg_daily_demand', 0)
        
        # Rule: Severe decline - consider removal
        if status == 'severe_decline' and avg_daily < 1:
            recs.append(Recommendation(
                item_id=item_id,
                item_name=item_name,
                action=ActionType.REMOVE,
                urgency=Urgency.MEDIUM,
                estimated_impact=0,
                explanation=(
                    f"Consider removing {item_name} from inventory. "
                    f"Demand has dropped {abs(trend_pct):.0f}% "
                    f"with current sales under 1 unit/day. "
                    f"This SKU may not justify inventory costs."
                ),
                supporting_data={
                    'trend_pct': trend_pct,
                    'avg_daily_demand': avg_daily
                }
            ))
        
        # Rule: Strong growth - increase orders
        elif trend_pct > 20:
            increase_pct = min(trend_pct / 2, 30)  # Cap at 30% increase
            
            recs.append(Recommendation(
                item_id=item_id,
                item_name=item_name,
                action=ActionType.INCREASE_ORDER,
                urgency=Urgency.MEDIUM,
                estimated_impact=round(avg_daily * 5, 2),
                explanation=(
                    f"{item_name} demand is growing {trend_pct:.0f}%. "
                    f"Consider increasing order quantity by {increase_pct:.0f}% "
                    f"to capture this growth and prevent stockouts."
                ),
                supporting_data={
                    'trend_pct': trend_pct,
                    'recommended_increase_pct': increase_pct
                }
            ))
        
        return recs
    
    def _apply_waste_rules(
        self,
        health: 'HealthScore',
        inventory: Dict,
        item_id: Any,
        item_name: str
    ) -> List[Recommendation]:
        """
        Apply waste reduction rules.
        
        Waste Risk = Low Demand + High Stock + Near Expiry
        """
        recs = []
        
        stock_factors = health.factors.get('stock_level', {})
        expiry_factors = health.factors.get('days_to_expiry', {})
        turnover_factors = health.factors.get('turnover_rate', {})
        
        days_of_supply = stock_factors.get('days_of_supply', 0)
        days_to_expiry = expiry_factors.get('days_to_expiry')
        turnover_status = turnover_factors.get('status', 'unknown')
        current_stock = stock_factors.get('current_stock', 0)
        
        # Rule: High waste risk - bundle suggestion
        if (turnover_status == 'slow_moving' and 
            days_of_supply and days_of_supply > 14 and
            current_stock > 0):
            
            recs.append(Recommendation(
                item_id=item_id,
                item_name=item_name,
                action=ActionType.BUNDLE,
                urgency=Urgency.MEDIUM,
                quantity=current_stock,
                estimated_impact=round(current_stock * 3, 2),
                explanation=(
                    f"{item_name} is a slow mover with {days_of_supply:.0f} days of stock. "
                    f"Consider bundling with fast-moving items or creating a combo deal "
                    f"to accelerate sales and reduce carrying costs."
                ),
                supporting_data={
                    'turnover_status': turnover_status,
                    'days_of_supply': days_of_supply
                }
            ))
        
        return recs
    
    def _urgency_order(self, urgency: Urgency) -> int:
        """Map urgency to sort order."""
        order = {
            Urgency.CRITICAL: 0,
            Urgency.HIGH: 1,
            Urgency.MEDIUM: 2,
            Urgency.LOW: 3
        }
        return order.get(urgency, 99)


def recommendations_to_dataframe(
    recommendations: List[Recommendation]
) -> pd.DataFrame:
    """
    Convert recommendations to a DataFrame for reporting.
    
    Parameters
    ----------
    recommendations : List[Recommendation]
        List of recommendations
    
    Returns
    -------
    pd.DataFrame
        DataFrame with recommendation details
    """
    if len(recommendations) == 0:
        return pd.DataFrame()
    
    records = []
    for rec in recommendations:
        records.append({
            'item_id': rec.item_id,
            'item_name': rec.item_name,
            'action': rec.action.value,
            'urgency': rec.urgency.value,
            'quantity': rec.quantity,
            'discount_pct': rec.discount_pct,
            'estimated_impact': rec.estimated_impact,
            'explanation': rec.explanation
        })
    
    return pd.DataFrame(records)


def calculate_recommended_order_quantity(
    current_stock: float,
    avg_daily_demand: float,
    lead_time_days: int = 2,
    safety_stock_days: int = 3,
    target_days_supply: int = 14
) -> Tuple[float, str]:
    """
    Calculate recommended order quantity using reorder point logic.
    
    Formula:
    - Reorder Point = (Lead Time × Daily Demand) + Safety Stock
    - Order Quantity = Target Supply - Current Stock
    
    Parameters
    ----------
    current_stock : float
        Current inventory level
    avg_daily_demand : float
        Average daily demand
    lead_time_days : int
        Days until order arrives
    safety_stock_days : int
        Buffer days of stock
    target_days_supply : int
        Target days of inventory
    
    Returns
    -------
    Tuple[float, str]
        (order quantity, explanation)
    """
    if avg_daily_demand <= 0:
        return 0, "No demand - no reorder needed"
    
    # Calculate reorder point
    reorder_point = (lead_time_days + safety_stock_days) * avg_daily_demand
    
    # Calculate target stock
    target_stock = target_days_supply * avg_daily_demand
    
    # Calculate order quantity
    if current_stock <= reorder_point:
        order_qty = target_stock - current_stock
        order_qty = max(0, order_qty)
        
        explanation = (
            f"Order {order_qty:.0f} units. "
            f"Current stock ({current_stock:.0f}) is at or below "
            f"reorder point ({reorder_point:.0f}). "
            f"This order brings inventory to {target_days_supply} days of supply."
        )
    else:
        order_qty = 0
        days_until_reorder = (current_stock - reorder_point) / avg_daily_demand
        
        explanation = (
            f"No order needed. Stock is above reorder point. "
            f"Next order in approximately {days_until_reorder:.0f} days."
        )
    
    return order_qty, explanation
