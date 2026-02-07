"""
Inventory Health Scoring Service
=================================
Calculate health scores and risk levels for inventory items.

Design Principles:
- Transparent scoring: every component is documented
- Configurable weights: business can adjust priorities
- Clear risk levels: LOW, MEDIUM, HIGH with thresholds
- Defensive: handles missing data gracefully

Health Score Components:
1. Stock Level (30%): Current stock vs forecasted demand
2. Days to Expiry (35%): Time until product expires
3. Demand Trend (20%): Is demand increasing or decreasing?
4. Turnover Rate (15%): How fast is inventory moving?

Score Range: 0-100 (higher = healthier inventory)
Risk Levels:
- LOW (70-100): Inventory is well-managed
- MEDIUM (40-69): Needs attention
- HIGH (0-39): Immediate action required
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta

from utils.logger import get_logger
from utils.constants import HEALTH_SCORE_CONFIG, RECOMMENDATION_CONFIG

logger = get_logger(__name__)


@dataclass
class HealthScore:
    """
    Health assessment for a single inventory item.
    
    Attributes
    ----------
    item_id : Any
        Identifier of the item
    health_score : float
        Overall health score (0-100)
    risk_level : str
        Risk classification (LOW, MEDIUM, HIGH)
    components : Dict[str, float]
        Individual component scores
    factors : Dict[str, Any]
        Input factors used for calculation
    explanation : str
        Human-readable explanation of the score
    """
    item_id: Any
    health_score: float
    risk_level: str
    components: Dict[str, float] = field(default_factory=dict)
    factors: Dict[str, Any] = field(default_factory=dict)
    explanation: str = ""


class InventoryHealthScorer:
    """
    Calculate inventory health scores and identify risks.
    
    This class evaluates inventory health based on:
    - How much stock vs expected demand
    - How close to expiration
    - Demand trends (growing/declining)
    - Historical turnover rates
    
    Usage
    -----
    >>> scorer = InventoryHealthScorer()
    >>> scores = scorer.calculate_health_scores(
    ...     forecasts_df,
    ...     inventory_df,
    ...     expiry_df
    ... )
    >>> high_risk = [s for s in scores if s.risk_level == 'HIGH']
    
    Scoring Logic Documentation:
    
    1. Stock Level Score (30% weight)
       - Perfect (100): Stock covers 7-14 days of demand
       - Good (80): Stock covers 5-7 or 14-21 days
       - Fair (50): Stock covers 3-5 or 21-30 days
       - Poor (20): Less than 3 days or more than 30 days
       - Critical (0): Zero stock with demand
    
    2. Expiry Score (35% weight)
       - Perfect (100): > 14 days to expiry or non-perishable
       - Good (70): 8-14 days to expiry
       - Warning (40): 4-7 days to expiry
       - Critical (10): <= 3 days to expiry
    
    3. Demand Trend Score (20% weight)
       - Stable/Growing (100): Trend >= 0
       - Slight decline (70): -10% to 0
       - Moderate decline (40): -25% to -10%
       - Severe decline (10): < -25%
    
    4. Turnover Score (15% weight)
       - Excellent (100): Turnover > 4x per month
       - Good (70): 2-4x per month
       - Fair (40): 1-2x per month
       - Poor (10): < 1x per month
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the health scorer.
        
        Parameters
        ----------
        config : dict, optional
            Custom configuration. Uses defaults if not provided.
        """
        self.config = config or HEALTH_SCORE_CONFIG
        self.weights = self.config.get('weights', {
            'stock_level': 0.30,
            'days_to_expiry': 0.35,
            'demand_trend': 0.20,
            'turnover_rate': 0.15
        })
        
        self.risk_thresholds = self.config.get('risk_thresholds', {
            'low': 70,
            'medium': 40
        })
        
        logger.info(f"HealthScorer initialized with weights: {self.weights}")
    
    def calculate_health_scores(
        self,
        forecast_df: pd.DataFrame,
        inventory_df: pd.DataFrame,
        expiry_df: Optional[pd.DataFrame] = None,
        item_column: str = 'item_id',
        forecast_column: str = 'forecast',
        stock_column: str = 'quantity',
        expiry_column: str = 'days_to_expiry'
    ) -> List[HealthScore]:
        """
        Calculate health scores for all inventory items.
        
        Parameters
        ----------
        forecast_df : pd.DataFrame
            Demand forecasts per item
        inventory_df : pd.DataFrame
            Current inventory levels
        expiry_df : pd.DataFrame, optional
            Expiry information per item
        item_column : str
            Column name for item identifier
        forecast_column : str
            Column name for forecast values
        stock_column : str
            Column name for stock quantity
        expiry_column : str
            Column name for days to expiry
        
        Returns
        -------
        List[HealthScore]
            Health scores for each item
        """
        scores = []
        
        # Get unique items from forecast
        if len(forecast_df) == 0:
            logger.warning("Empty forecast data provided")
            return scores
        
        items = forecast_df[item_column].unique()
        logger.info(f"Calculating health scores for {len(items)} items")
        
        # Aggregate forecast to get total expected demand
        demand_summary = self._aggregate_demand(
            forecast_df, item_column, forecast_column
        )
        
        # Calculate demand trends
        trends = self._calculate_trends(forecast_df, item_column, forecast_column)
        
        for item_id in items:
            # Get item data
            item_demand = demand_summary.get(item_id, {})
            item_trend = trends.get(item_id, 0)
            
            # Get inventory level
            item_inventory = self._get_item_inventory(
                inventory_df, item_id, item_column, stock_column
            )
            
            # Get expiry info
            item_expiry = self._get_item_expiry(
                expiry_df, item_id, item_column, expiry_column
            )
            
            # Calculate component scores
            components = {}
            factors = {}
            
            # 1. Stock Level Score
            components['stock_level'], factors['stock_level'] = self._score_stock_level(
                item_inventory, item_demand
            )
            
            # 2. Expiry Score
            components['days_to_expiry'], factors['days_to_expiry'] = self._score_expiry(
                item_expiry
            )
            
            # 3. Demand Trend Score
            components['demand_trend'], factors['demand_trend'] = self._score_trend(
                item_trend
            )
            
            # 4. Turnover Score
            components['turnover_rate'], factors['turnover_rate'] = self._score_turnover(
                item_inventory, item_demand
            )
            
            # Calculate weighted total
            health_score = sum(
                components[key] * self.weights[key]
                for key in components
            )
            
            # Determine risk level
            risk_level = self._determine_risk_level(health_score)
            
            # Generate explanation
            explanation = self._generate_explanation(
                components, factors, health_score, risk_level
            )
            
            scores.append(HealthScore(
                item_id=item_id,
                health_score=round(health_score, 1),
                risk_level=risk_level,
                components=components,
                factors=factors,
                explanation=explanation
            ))
        
        # Log summary
        high_risk = sum(1 for s in scores if s.risk_level == 'HIGH')
        medium_risk = sum(1 for s in scores if s.risk_level == 'MEDIUM')
        
        logger.info(
            f"Health scoring complete: {high_risk} HIGH risk, "
            f"{medium_risk} MEDIUM risk, {len(scores) - high_risk - medium_risk} LOW risk"
        )
        
        return scores
    
    def _aggregate_demand(
        self,
        forecast_df: pd.DataFrame,
        item_column: str,
        forecast_column: str
    ) -> Dict[Any, Dict[str, float]]:
        """Aggregate forecast to get demand summary per item."""
        summary = {}
        
        for item_id, group in forecast_df.groupby(item_column):
            total_demand = group[forecast_column].sum()
            days = len(group)
            avg_daily = total_demand / days if days > 0 else 0
            
            summary[item_id] = {
                'total_demand': total_demand,
                'avg_daily_demand': avg_daily,
                'days': days
            }
        
        return summary
    
    def _calculate_trends(
        self,
        forecast_df: pd.DataFrame,
        item_column: str,
        forecast_column: str
    ) -> Dict[Any, float]:
        """
        Calculate demand trend as percentage change.
        
        Compare last 3 days average to first 3 days average.
        """
        trends = {}
        
        for item_id, group in forecast_df.groupby(item_column):
            if len(group) < 6:
                trends[item_id] = 0  # Not enough data
                continue
            
            values = group[forecast_column].values
            first_avg = np.mean(values[:3])
            last_avg = np.mean(values[-3:])
            
            if first_avg > 0:
                trend_pct = (last_avg - first_avg) / first_avg * 100
            else:
                trend_pct = 0
            
            trends[item_id] = trend_pct
        
        return trends
    
    def _get_item_inventory(
        self,
        inventory_df: pd.DataFrame,
        item_id: Any,
        item_column: str,
        stock_column: str
    ) -> float:
        """Get current inventory level for an item."""
        if inventory_df is None or len(inventory_df) == 0:
            return 0
        
        # Try to match by item_id or id
        if item_column in inventory_df.columns:
            match = inventory_df[inventory_df[item_column] == item_id]
        elif 'id' in inventory_df.columns:
            match = inventory_df[inventory_df['id'] == item_id]
        else:
            return 0
        
        if len(match) > 0 and stock_column in match.columns:
            return float(match[stock_column].iloc[0])
        
        return 0
    
    def _get_item_expiry(
        self,
        expiry_df: Optional[pd.DataFrame],
        item_id: Any,
        item_column: str,
        expiry_column: str
    ) -> Optional[int]:
        """Get days to expiry for an item."""
        if expiry_df is None or len(expiry_df) == 0:
            return None  # No expiry info = assume non-perishable
        
        if item_column in expiry_df.columns:
            match = expiry_df[expiry_df[item_column] == item_id]
        elif 'id' in expiry_df.columns:
            match = expiry_df[expiry_df['id'] == item_id]
        else:
            return None
        
        if len(match) > 0 and expiry_column in match.columns:
            value = match[expiry_column].iloc[0]
            if pd.notna(value):
                return int(value)
        
        return None
    
    def _score_stock_level(
        self,
        current_stock: float,
        demand_info: Dict[str, float]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Score based on stock level vs expected demand.
        
        Returns (score, factors dict)
        """
        avg_daily = demand_info.get('avg_daily_demand', 0)
        
        factors = {
            'current_stock': current_stock,
            'avg_daily_demand': avg_daily
        }
        
        if avg_daily <= 0:
            # No demand expected
            if current_stock > 0:
                factors['days_of_supply'] = float('inf')
                factors['status'] = 'potential_overstock'
                return 50, factors  # Medium score - might be overstock
            else:
                factors['days_of_supply'] = 0
                factors['status'] = 'no_demand_no_stock'
                return 100, factors  # Perfect - no demand, no stock
        
        days_of_supply = current_stock / avg_daily
        factors['days_of_supply'] = round(days_of_supply, 1)
        
        # Score based on days of supply
        if 7 <= days_of_supply <= 14:
            score = 100
            factors['status'] = 'optimal'
        elif 5 <= days_of_supply < 7 or 14 < days_of_supply <= 21:
            score = 80
            factors['status'] = 'good'
        elif 3 <= days_of_supply < 5 or 21 < days_of_supply <= 30:
            score = 50
            factors['status'] = 'fair'
        elif days_of_supply < 3:
            score = 20 if days_of_supply > 0 else 0
            factors['status'] = 'understock'
        else:  # > 30 days
            score = 30
            factors['status'] = 'overstock'
        
        return score, factors
    
    def _score_expiry(
        self,
        days_to_expiry: Optional[int]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Score based on days until expiry.
        
        Returns (score, factors dict)
        """
        factors = {
            'days_to_expiry': days_to_expiry
        }
        
        if days_to_expiry is None:
            factors['status'] = 'non_perishable'
            return 100, factors  # Non-perishable = no expiry risk
        
        expiry_thresholds = self.config.get('expiry_thresholds', {
            'critical': 3,
            'warning': 7,
            'acceptable': 14
        })
        
        if days_to_expiry <= expiry_thresholds['critical']:
            score = 10
            factors['status'] = 'critical'
        elif days_to_expiry <= expiry_thresholds['warning']:
            score = 40
            factors['status'] = 'warning'
        elif days_to_expiry <= expiry_thresholds['acceptable']:
            score = 70
            factors['status'] = 'acceptable'
        else:
            score = 100
            factors['status'] = 'good'
        
        return score, factors
    
    def _score_trend(
        self,
        trend_pct: float
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Score based on demand trend.
        
        Returns (score, factors dict)
        """
        factors = {
            'trend_pct': round(trend_pct, 1)
        }
        
        if trend_pct >= 0:
            score = 100
            factors['status'] = 'stable_or_growing'
        elif trend_pct >= -10:
            score = 70
            factors['status'] = 'slight_decline'
        elif trend_pct >= -25:
            score = 40
            factors['status'] = 'moderate_decline'
        else:
            score = 10
            factors['status'] = 'severe_decline'
        
        return score, factors
    
    def _score_turnover(
        self,
        current_stock: float,
        demand_info: Dict[str, float]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Score based on inventory turnover rate.
        
        Turnover = (Monthly Demand) / (Average Stock)
        Higher turnover = faster moving = healthier
        """
        avg_daily = demand_info.get('avg_daily_demand', 0)
        monthly_demand = avg_daily * 30
        
        factors = {
            'monthly_demand': round(monthly_demand, 1),
            'current_stock': current_stock
        }
        
        if current_stock <= 0:
            if avg_daily > 0:
                factors['turnover'] = 'stockout'
                factors['status'] = 'no_stock'
                return 0, factors  # Bad - demand but no stock
            else:
                factors['turnover'] = 'n/a'
                factors['status'] = 'inactive'
                return 50, factors  # Neutral - no demand, no stock
        
        turnover = monthly_demand / current_stock
        factors['turnover'] = round(turnover, 2)
        
        if turnover > 4:
            score = 100
            factors['status'] = 'excellent'
        elif turnover > 2:
            score = 70
            factors['status'] = 'good'
        elif turnover > 1:
            score = 40
            factors['status'] = 'fair'
        else:
            score = 10
            factors['status'] = 'slow_moving'
        
        return score, factors
    
    def _determine_risk_level(self, health_score: float) -> str:
        """Map health score to risk level."""
        if health_score >= self.risk_thresholds['low']:
            return 'LOW'
        elif health_score >= self.risk_thresholds['medium']:
            return 'MEDIUM'
        else:
            return 'HIGH'
    
    def _generate_explanation(
        self,
        components: Dict[str, float],
        factors: Dict[str, Dict[str, Any]],
        health_score: float,
        risk_level: str
    ) -> str:
        """Generate a human-readable explanation of the health score."""
        parts = []
        
        parts.append(f"Overall health score: {health_score:.0f}/100 ({risk_level} risk).")
        
        # Stock level explanation
        stock_factors = factors.get('stock_level', {})
        if 'days_of_supply' in stock_factors:
            dos = stock_factors['days_of_supply']
            if dos == float('inf'):
                parts.append("Stock present but no expected demand.")
            elif dos < 3:
                parts.append(f"Only {dos:.1f} days of supply remaining - reorder soon.")
            elif dos > 21:
                parts.append(f"High stock ({dos:.1f} days of supply) - consider promotions.")
            else:
                parts.append(f"Stock level is healthy ({dos:.1f} days of supply).")
        
        # Expiry explanation
        expiry_factors = factors.get('days_to_expiry', {})
        if expiry_factors.get('days_to_expiry') is not None:
            days = expiry_factors['days_to_expiry']
            if days <= 3:
                parts.append(f"URGENT: Only {days} days until expiry!")
            elif days <= 7:
                parts.append(f"Warning: {days} days until expiry.")
        
        # Trend explanation
        trend_factors = factors.get('demand_trend', {})
        trend_pct = trend_factors.get('trend_pct', 0)
        if trend_pct < -10:
            parts.append(f"Demand is declining ({trend_pct:.0f}%).")
        elif trend_pct > 10:
            parts.append(f"Demand is growing ({trend_pct:.0f}%).")
        
        return ' '.join(parts)


def health_scores_to_dataframe(scores: List[HealthScore]) -> pd.DataFrame:
    """
    Convert health scores to a DataFrame for reporting.
    
    Parameters
    ----------
    scores : List[HealthScore]
        List of health score results
    
    Returns
    -------
    pd.DataFrame
        DataFrame with health metrics per item
    """
    if len(scores) == 0:
        return pd.DataFrame()
    
    records = []
    for score in scores:
        record = {
            'item_id': score.item_id,
            'health_score': score.health_score,
            'risk_level': score.risk_level,
            'explanation': score.explanation
        }
        
        # Add component scores
        for key, value in score.components.items():
            record[f'score_{key}'] = value
        
        # Add key factors
        stock_factors = score.factors.get('stock_level', {})
        record['days_of_supply'] = stock_factors.get('days_of_supply', None)
        record['stock_status'] = stock_factors.get('status', 'unknown')
        
        expiry_factors = score.factors.get('days_to_expiry', {})
        record['days_to_expiry'] = expiry_factors.get('days_to_expiry', None)
        
        records.append(record)
    
    df = pd.DataFrame(records)
    
    # Sort by risk level and health score
    risk_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
    df['_risk_order'] = df['risk_level'].map(risk_order)
    df = df.sort_values(['_risk_order', 'health_score']).drop(columns=['_risk_order'])
    
    return df.reset_index(drop=True)
