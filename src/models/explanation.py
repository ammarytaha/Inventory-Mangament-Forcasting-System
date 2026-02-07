"""
Business Explanation Layer
===========================
Generate plain English explanations for all system outputs.

Design Principles:
- No technical jargon
- Focus on WHY, not just WHAT
- Actionable language
- Suitable for non-technical stakeholders

Every explanation answers three questions:
1. What is happening? (The situation)
2. Why does it matter? (The impact)
3. What should I do? (The action)

This module is the "last mile" of the system - it translates
technical analysis into business-friendly language.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, date

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BusinessExplanation:
    """
    A complete business explanation for a recommendation or insight.
    
    Attributes
    ----------
    headline : str
        Short, attention-grabbing summary (max 10 words)
    situation : str
        What is happening? (2-3 sentences)
    impact : str
        Why does it matter? (1-2 sentences with $ or % where possible)
    action : str
        What should be done? (Clear, specific action)
    supporting_facts : List[str]
        Bullet points with key data
    confidence : str
        How confident is this recommendation? (HIGH, MEDIUM, LOW)
    """
    headline: str
    situation: str
    impact: str
    action: str
    supporting_facts: List[str] = field(default_factory=list)
    confidence: str = "MEDIUM"
    
    def to_string(self) -> str:
        """Format as readable text."""
        parts = [
            f"📌 {self.headline}",
            "",
            f"Situation: {self.situation}",
            f"Impact: {self.impact}",
            f"Action: {self.action}",
        ]
        
        if self.supporting_facts:
            parts.append("")
            parts.append("Key Facts:")
            for fact in self.supporting_facts:
                parts.append(f"  • {fact}")
        
        parts.append(f"Confidence: {self.confidence}")
        
        return "\n".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "headline": self.headline,
            "situation": self.situation,
            "impact": self.impact,
            "action": self.action,
            "supporting_facts": self.supporting_facts,
            "confidence": self.confidence
        }


class ExplanationGenerator:
    """
    Generate business-friendly explanations for all system outputs.
    
    This class transforms technical data into language that:
    - A store manager can understand
    - A business executive can act on
    - A hackathon judge can follow
    
    Usage
    -----
    >>> generator = ExplanationGenerator()
    >>> explanation = generator.explain_recommendation(recommendation)
    >>> print(explanation.to_string())
    
    Templates:
    The generator uses templates that follow a consistent pattern:
    - Active voice ("Reorder X" not "X should be reordered")
    - Specific numbers ("15 units" not "some units")
    - Business outcomes ("avoid stockout" not "prevent inventory depletion")
    """
    
    # Template strings for different scenarios
    TEMPLATES = {
        'reorder': {
            'headline': "Reorder {item_name} Now",
            'situation': (
                "{item_name} is running low. Current stock ({current_stock:.0f} units) "
                "will only last {days_of_supply:.1f} days based on expected demand of "
                "{avg_daily:.1f} units per day."
            ),
            'impact': (
                "Without action, you'll run out of stock in {days_of_supply:.0f} days. "
                "Stockouts cost you sales and can frustrate customers."
            ),
            'action': "Order {order_qty:.0f} units today to maintain healthy stock levels."
        },
        
        'discount': {
            'headline': "Discount {item_name} Before It Expires",
            'situation': (
                "{item_name} has only {days_to_expiry} days until expiration. "
                "You have {stock:.0f} units that need to move quickly."
            ),
            'impact': (
                "If unsold, you'll lose ${potential_loss:.0f} worth of inventory. "
                "A {discount_pct:.0f}% discount could save most of this value."
            ),
            'action': (
                "Apply a {discount_pct:.0f}% discount immediately and feature "
                "prominently in store."
            )
        },
        
        'reduce_order': {
            'headline': "Reduce Orders for {item_name}",
            'situation': (
                "You're overstocked on {item_name}. Current inventory "
                "({current_stock:.0f} units) is enough for {days_of_supply:.0f} days - "
                "much more than the target of 14 days."
            ),
            'impact': (
                "Excess inventory ties up ${capital_tied:.0f} in capital and "
                "increases waste risk."
            ),
            'action': (
                "Skip the next order or reduce it by {reduce_by:.0f} units."
            )
        },
        
        'bundle': {
            'headline': "Create a Bundle with {item_name}",
            'situation': (
                "{item_name} is a slow mover with {days_of_supply:.0f} days of "
                "inventory but only {avg_daily:.1f} units sold per day."
            ),
            'impact': (
                "Slow movers increase carrying costs and waste risk. "
                "Bundling can boost sales velocity."
            ),
            'action': (
                "Create a combo deal pairing {item_name} with a fast seller "
                "at a small discount."
            )
        },
        
        'prioritize_sales': {
            'headline': "URGENT: Clear {item_name} Immediately",
            'situation': (
                "{item_name} expires in just {days_to_expiry} days. "
                "You have {stock:.0f} units at risk of becoming waste."
            ),
            'impact': (
                "You'll lose ${at_risk:.0f} if this inventory isn't sold. "
                "This is a time-critical issue."
            ),
            'action': (
                "Offer {discount_pct:.0f}% off, place at checkout, "
                "and consider giving to staff if unsold by end of day."
            )
        },
        
        'increase_order': {
            'headline': "Increase Orders for Growing {item_name}",
            'situation': (
                "Demand for {item_name} is growing at {trend_pct:.0f}% week-over-week. "
                "Current stock may not keep up."
            ),
            'impact': (
                "Growing demand is good! But stockouts mean missed sales. "
                "Each lost sale could cost you ${sale_value:.0f}."
            ),
            'action': (
                "Increase your next order by {increase_pct:.0f}% to capture this growth."
            )
        },
        
        'remove': {
            'headline': "Consider Removing {item_name}",
            'situation': (
                "{item_name} hasn't sold well - only {avg_daily:.1f} units per day, "
                "down {trend_pct:.0f}% recently."
            ),
            'impact': (
                "This item takes up shelf space and capital that could "
                "be used for better sellers."
            ),
            'action': (
                "Review whether to discontinue this item. "
                "Clear remaining stock with a markdown, then replace."
            )
        },
        
        'monitor': {
            'headline': "Monitor {item_name}",
            'situation': (
                "{item_name} is performing within normal ranges. "
                "Stock levels and demand are balanced."
            ),
            'impact': "No immediate action needed, but stay aware of changes.",
            'action': "Continue monitoring. Review again in one week."
        }
    }
    
    def __init__(self):
        """Initialize the explanation generator."""
        logger.info("ExplanationGenerator initialized")
    
    def explain_recommendation(
        self,
        action_type: str,
        item_name: str,
        data: Dict[str, Any]
    ) -> BusinessExplanation:
        """
        Generate a complete explanation for a recommendation.
        
        Parameters
        ----------
        action_type : str
            Type of action (reorder, discount, bundle, etc.)
        item_name : str
            Name of the item
        data : dict
            Data to fill into templates
        
        Returns
        -------
        BusinessExplanation
            Complete explanation with all components
        """
        # Get template for this action type
        template = self.TEMPLATES.get(action_type, self.TEMPLATES['monitor'])
        
        # Add item_name to data
        data = {**data, 'item_name': item_name}
        
        # Fill templates (with safe defaults)
        headline = self._safe_format(template['headline'], data)
        situation = self._safe_format(template['situation'], data)
        impact = self._safe_format(template['impact'], data)
        action = self._safe_format(template['action'], data)
        
        # Generate supporting facts
        facts = self._generate_supporting_facts(action_type, data)
        
        # Determine confidence
        confidence = self._determine_confidence(data)
        
        return BusinessExplanation(
            headline=headline,
            situation=situation,
            impact=impact,
            action=action,
            supporting_facts=facts,
            confidence=confidence
        )
    
    def _safe_format(self, template: str, data: Dict[str, Any]) -> str:
        """Format template with safe defaults for missing values."""
        # Provide defaults for common fields
        defaults = {
            'item_name': 'This item',
            'current_stock': 0,
            'days_of_supply': 0,
            'avg_daily': 0,
            'days_to_expiry': 0,
            'stock': 0,
            'discount_pct': 20,
            'potential_loss': 0,
            'order_qty': 0,
            'capital_tied': 0,
            'reduce_by': 0,
            'trend_pct': 0,
            'sale_value': 10,
            'at_risk': 0,
            'increase_pct': 10
        }
        
        # Merge defaults with provided data
        merged = {**defaults, **data}
        
        try:
            return template.format(**merged)
        except (KeyError, ValueError) as e:
            logger.warning(f"Template formatting error: {e}")
            return template
    
    def _generate_supporting_facts(
        self,
        action_type: str,
        data: Dict[str, Any]
    ) -> List[str]:
        """Generate bullet point facts from data."""
        facts = []
        
        if 'current_stock' in data:
            facts.append(f"Current stock: {data['current_stock']:.0f} units")
        
        if 'avg_daily' in data and data['avg_daily'] > 0:
            facts.append(f"Average daily demand: {data['avg_daily']:.1f} units")
        
        if 'days_of_supply' in data and data['days_of_supply'] < float('inf'):
            facts.append(f"Days of supply: {data['days_of_supply']:.1f} days")
        
        if 'days_to_expiry' in data and data['days_to_expiry'] is not None:
            facts.append(f"Days until expiry: {data['days_to_expiry']}")
        
        if 'health_score' in data:
            facts.append(f"Health score: {data['health_score']:.0f}/100")
        
        if 'trend_pct' in data and abs(data['trend_pct']) > 5:
            direction = "up" if data['trend_pct'] > 0 else "down"
            facts.append(f"Demand trend: {abs(data['trend_pct']):.0f}% {direction}")
        
        return facts
    
    def _determine_confidence(self, data: Dict[str, Any]) -> str:
        """Determine confidence level based on data quality."""
        # Look for indicators of data quality
        data_points = data.get('data_points', 0)
        
        if data_points >= 30:
            return "HIGH"
        elif data_points >= 14:
            return "MEDIUM"
        else:
            return "LOW"


def generate_demand_explanation(
    item_name: str,
    historical_avg: float,
    forecast_avg: float,
    trend_pct: float,
    adjustment_factors: Optional[Dict[str, float]] = None
) -> str:
    """
    Generate a plain English explanation of demand changes.
    
    Parameters
    ----------
    item_name : str
        Name of the item
    historical_avg : float
        Historical average daily demand
    forecast_avg : float
        Forecasted average daily demand
    trend_pct : float
        Percentage change in trend
    adjustment_factors : dict, optional
        Factors that adjusted the forecast (events, weather)
    
    Returns
    -------
    str
        Plain English explanation
    """
    parts = []
    
    # Describe the forecast
    if abs(forecast_avg - historical_avg) < 0.1 * historical_avg:
        parts.append(
            f"Demand for {item_name} is expected to stay steady at about "
            f"{forecast_avg:.0f} units per day."
        )
    elif forecast_avg > historical_avg:
        increase = ((forecast_avg - historical_avg) / historical_avg) * 100
        parts.append(
            f"Demand for {item_name} is expected to increase by {increase:.0f}% "
            f"to {forecast_avg:.0f} units per day."
        )
    else:
        decrease = ((historical_avg - forecast_avg) / historical_avg) * 100
        parts.append(
            f"Demand for {item_name} is expected to decrease by {decrease:.0f}% "
            f"to {forecast_avg:.0f} units per day."
        )
    
    # Explain why (adjustments)
    if adjustment_factors:
        reasons = []
        for factor, value in adjustment_factors.items():
            if value > 1.1:
                reasons.append(f"{factor} is expected to boost demand")
            elif value < 0.9:
                reasons.append(f"{factor} may reduce demand")
        
        if reasons:
            parts.append("This is because " + " and ".join(reasons) + ".")
    
    return " ".join(parts)


def generate_risk_explanation(
    item_name: str,
    health_score: float,
    risk_level: str,
    primary_risk: str,
    risk_factors: Dict[str, Any]
) -> str:
    """
    Generate a plain English explanation of inventory risk.
    
    Parameters
    ----------
    item_name : str
        Name of the item
    health_score : float
        Overall health score (0-100)
    risk_level : str
        Risk classification (LOW, MEDIUM, HIGH)
    primary_risk : str
        The main risk factor
    risk_factors : dict
        All risk factors with values
    
    Returns
    -------
    str
        Plain English explanation
    """
    if risk_level == 'LOW':
        return (
            f"{item_name} is in good shape with a health score of {health_score:.0f}/100. "
            f"Stock levels are balanced with expected demand. No immediate action needed."
        )
    
    elif risk_level == 'MEDIUM':
        if primary_risk == 'overstock':
            return (
                f"{item_name} has more stock than needed (score: {health_score:.0f}/100). "
                f"Consider slowing down orders to avoid waste and free up capital."
            )
        elif primary_risk == 'understock':
            return (
                f"{item_name} is getting low (score: {health_score:.0f}/100). "
                f"Keep an eye on it and prepare to reorder soon."
            )
        elif primary_risk == 'expiry':
            days = risk_factors.get('days_to_expiry', 'a few')
            return (
                f"{item_name} has {days} days until expiry (score: {health_score:.0f}/100). "
                f"Consider a promotion to move inventory faster."
            )
        else:
            return (
                f"{item_name} needs attention (score: {health_score:.0f}/100). "
                f"Review stock levels and demand patterns."
            )
    
    else:  # HIGH risk
        if primary_risk == 'expiry':
            days = risk_factors.get('days_to_expiry', 'very few')
            stock = risk_factors.get('stock', 0)
            return (
                f"⚠️ URGENT: {item_name} expires in {days} days with {stock:.0f} units "
                f"at risk (score: {health_score:.0f}/100). "
                f"Immediate discount or clearance needed to avoid waste!"
            )
        elif primary_risk == 'stockout':
            days = risk_factors.get('days_of_supply', 0)
            return (
                f"⚠️ URGENT: {item_name} will run out in {days:.0f} days "
                f"(score: {health_score:.0f}/100). "
                f"Reorder immediately to avoid stockout!"
            )
        else:
            return (
                f"⚠️ {item_name} is at HIGH risk (score: {health_score:.0f}/100). "
                f"Review immediately and take corrective action."
            )


def generate_action_explanation(
    item_name: str,
    action: str,
    quantity: Optional[float],
    urgency: str,
    reason: str,
    expected_outcome: str
) -> str:
    """
    Generate a plain English explanation of a recommended action.
    
    Parameters
    ----------
    item_name : str
        Name of the item
    action : str
        Action type (reorder, discount, bundle, etc.)
    quantity : float, optional
        Quantity involved
    urgency : str
        Urgency level
    reason : str
        Why this action is recommended
    expected_outcome : str
        What will happen if action is taken
    
    Returns
    -------
    str
        Plain English explanation
    """
    # Build action statement
    if action == 'reorder':
        action_stmt = f"Reorder {quantity:.0f} units of {item_name}"
    elif action == 'discount':
        action_stmt = f"Apply a discount to {item_name}"
    elif action == 'bundle':
        action_stmt = f"Create a bundle or combo with {item_name}"
    elif action == 'reduce_order':
        action_stmt = f"Reduce your next order of {item_name}"
    elif action == 'increase_order':
        action_stmt = f"Increase your next order of {item_name}"
    elif action == 'remove':
        action_stmt = f"Consider discontinuing {item_name}"
    elif action == 'prioritize_sales':
        action_stmt = f"Prioritize selling {item_name} today"
    else:
        action_stmt = f"Review {item_name}"
    
    # Build urgency prefix
    if urgency == 'CRITICAL':
        urgency_prefix = "🔴 TAKE ACTION NOW: "
    elif urgency == 'HIGH':
        urgency_prefix = "🟠 Act soon: "
    elif urgency == 'MEDIUM':
        urgency_prefix = "🟡 Consider: "
    else:
        urgency_prefix = ""
    
    # Combine into full explanation
    explanation = f"{urgency_prefix}{action_stmt}. {reason} {expected_outcome}"
    
    return explanation


def explain_forecast_method(method: str, data_points: int) -> str:
    """
    Explain why a particular forecasting method was used.
    
    This helps stakeholders understand and trust the forecasts.
    """
    if method == 'holt_winters':
        return (
            f"This forecast uses Holt-Winters smoothing, which is good at capturing "
            f"trends and weekly patterns. We had {data_points} days of data to work with, "
            f"which is enough for reliable predictions."
        )
    elif method == 'moving_average':
        return (
            f"This forecast uses a simple moving average because we only had "
            f"{data_points} days of data. With more history, we could use "
            f"more sophisticated methods. The forecast is less certain but still useful."
        )
    else:
        return (
            f"Limited data ({data_points} days) means this forecast has high uncertainty. "
            f"Use it as a rough guide and monitor actual sales closely."
        )
