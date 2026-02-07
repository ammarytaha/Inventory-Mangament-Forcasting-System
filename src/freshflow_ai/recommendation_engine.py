"""
FreshFlow AI - Recommendation Engine
======================================

AI-powered recommendation engine that generates actionable inventory
decisions based on forecasts, inventory levels, and business context.

Recommendation Types:
- REORDER: Items needing replenishment
- DISCOUNT: Near-expiry items requiring markdown
- BUNDLE: Items to combine for promotions
- PREP_ADJUST: Kitchen prep quantity adjustments
- ALERT: Critical inventory situations

Each recommendation includes:
- Priority/Risk level
- Specific action with quantities
- Business rationale
- Expected impact
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

from .config import Config, DEFAULT_CONFIG
from .data_processor import DataProcessor
from .forecaster import ForecastEngine

logger = logging.getLogger(__name__)


class RecommendationType(Enum):
    """Types of inventory recommendations"""
    REORDER = "reorder"
    DISCOUNT = "discount"
    BUNDLE = "bundle"
    PREP_ADJUST = "prep_adjust"
    ALERT = "alert"
    HOLD = "hold"


class RiskLevel(Enum):
    """Risk severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Recommendation:
    """A single inventory recommendation"""
    recommendation_id: str
    place_id: int
    item_id: int
    item_name: str
    recommendation_type: RecommendationType
    risk_level: RiskLevel
    action: str
    quantity: Optional[int] = None
    discount_percent: Optional[float] = None
    rationale: str = ""
    expected_impact: str = ""
    confidence: float = 0.8
    created_at: datetime = field(default_factory=datetime.now)
    valid_until: Optional[datetime] = None
    additional_data: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert recommendation to dictionary"""
        return {
            'recommendation_id': self.recommendation_id,
            'place_id': self.place_id,
            'item_id': self.item_id,
            'item_name': self.item_name,
            'recommendation_type': self.recommendation_type.value,
            'risk_level': self.risk_level.value,
            'action': self.action,
            'quantity': self.quantity,
            'discount_percent': self.discount_percent,
            'rationale': self.rationale,
            'expected_impact': self.expected_impact,
            'confidence': self.confidence,
            'created_at': self.created_at.isoformat(),
            'valid_until': self.valid_until.isoformat() if self.valid_until else None,
            **self.additional_data
        }


class RecommendationEngine:
    """
    AI-powered recommendation engine for inventory decisions.
    
    Generates personalized, actionable recommendations for each location
    based on forecasts, inventory levels, and contextual factors.
    
    Usage:
        engine = RecommendationEngine(data_processor, forecast_engine)
        recs = engine.generate_recommendations(place_id=94025)
    """
    
    def __init__(
        self,
        data_processor: Optional[DataProcessor] = None,
        forecast_engine: Optional[ForecastEngine] = None,
        config: Optional[Config] = None
    ):
        """
        Initialize the RecommendationEngine.
        
        Args:
            data_processor: DataProcessor instance
            forecast_engine: ForecastEngine instance
            config: Configuration object
        """
        self.config = config or DEFAULT_CONFIG
        self.data_processor = data_processor or DataProcessor(self.config)
        self.forecast_engine = forecast_engine or ForecastEngine(self.data_processor, self.config)
        self._recommendation_counter = 0
        
    def _generate_id(self) -> str:
        """Generate unique recommendation ID"""
        self._recommendation_counter += 1
        return f"REC-{datetime.now().strftime('%Y%m%d')}-{self._recommendation_counter:04d}"
    
    def generate_recommendations(
        self,
        place_id: int,
        forecast_horizon: int = 4,
        top_n_items: int = 50,
        context: Optional[Dict] = None
    ) -> List[Recommendation]:
        """
        Generate all recommendations for a location.
        
        This is the main entry point for getting location-specific
        AI recommendations.
        
        Args:
            place_id: Location identifier
            forecast_horizon: Weeks to forecast ahead
            top_n_items: Number of top items to analyze
            context: Additional context (holidays, events, weather)
            
        Returns:
            List of Recommendation objects sorted by priority
        """
        logger.info(f"Generating recommendations for place {place_id}")
        
        # Get place data and forecasts
        place_data = self.data_processor.get_place_data(place_id)
        forecasts = self.forecast_engine.forecast_place(
            place_id, 
            horizon=forecast_horizon,
            top_n_items=top_n_items
        )
        
        all_recommendations = []
        
        # Generate different types of recommendations
        all_recommendations.extend(
            self._generate_reorder_recommendations(place_id, place_data, forecasts)
        )
        
        all_recommendations.extend(
            self._generate_discount_recommendations(place_id, place_data, forecasts)
        )
        
        all_recommendations.extend(
            self._generate_prep_recommendations(place_id, place_data, forecasts, context)
        )
        
        all_recommendations.extend(
            self._generate_bundle_recommendations(place_id, place_data)
        )
        
        all_recommendations.extend(
            self._generate_alert_recommendations(place_id, place_data, forecasts)
        )
        
        # Sort by priority (critical first)
        priority_order = {
            RiskLevel.CRITICAL: 0,
            RiskLevel.HIGH: 1,
            RiskLevel.MEDIUM: 2,
            RiskLevel.LOW: 3
        }
        
        all_recommendations.sort(
            key=lambda r: (priority_order.get(r.risk_level, 4), -r.confidence)
        )
        
        logger.info(f"Generated {len(all_recommendations)} recommendations")
        return all_recommendations
    
    def _get_item_name(self, place_data: Dict, item_id: int) -> str:
        """Get item name from place data"""
        items = place_data.get('items')
        if items is not None:
            match = items[items['id'] == item_id]
            if len(match) > 0:
                return str(match.iloc[0].get('title', f'Item {item_id}'))
        return f'Item {item_id}'
    
    def _generate_reorder_recommendations(
        self,
        place_id: int,
        place_data: Dict,
        forecasts: List[Dict]
    ) -> List[Recommendation]:
        """Generate reorder/replenishment recommendations"""
        recommendations = []
        
        for forecast in forecasts:
            item_id = forecast.get('item_id')
            if not item_id:
                continue
                
            # Calculate reorder point
            reorder_info = self.forecast_engine.calculate_reorder_point(
                place_id, item_id, forecast
            )
            
            # Get safety stock
            safety_info = self.forecast_engine.calculate_safety_stock(place_id, item_id)
            
            # Calculate recommended order quantity
            if forecast['forecast']:
                weekly_forecast = sum(f['predicted_demand'] for f in forecast['forecast'])
            else:
                weekly_forecast = 0
                
            # Simple reorder logic (would be enhanced with actual inventory levels)
            if weekly_forecast > 0:
                order_qty = max(
                    reorder_info['reorder_point'],
                    int(weekly_forecast * 1.2)  # Order 20% buffer
                )
                
                # Determine risk level based on demand type
                demand_type = forecast.get('demand_type', 'Unknown')
                if demand_type in ['Lumpy', 'Erratic']:
                    risk_level = RiskLevel.HIGH
                elif demand_type == 'Intermittent':
                    risk_level = RiskLevel.MEDIUM
                else:
                    risk_level = RiskLevel.LOW
                    
                item_name = self._get_item_name(place_data, item_id)
                
                rec = Recommendation(
                    recommendation_id=self._generate_id(),
                    place_id=place_id,
                    item_id=item_id,
                    item_name=item_name,
                    recommendation_type=RecommendationType.REORDER,
                    risk_level=risk_level,
                    action=f"Order {order_qty} units",
                    quantity=order_qty,
                    rationale=f"Based on {forecast['model_used']} forecast: {weekly_forecast} units expected over {len(forecast['forecast'])} weeks. "
                             f"Reorder point: {reorder_info['reorder_point']} units. Safety stock: {safety_info['safety_stock']} units.",
                    expected_impact=f"Prevents stockout risk for the next {len(forecast['forecast'])} weeks",
                    confidence=0.85 if demand_type == 'Smooth' else 0.70,
                    valid_until=datetime.now() + timedelta(days=7),
                    additional_data={
                        'reorder_point': reorder_info['reorder_point'],
                        'safety_stock': safety_info['safety_stock'],
                        'weekly_forecast': weekly_forecast,
                        'demand_type': demand_type,
                        'forecast_model': forecast['model_used']
                    }
                )
                recommendations.append(rec)
                
        return recommendations
    
    def _generate_discount_recommendations(
        self,
        place_id: int,
        place_data: Dict,
        forecasts: List[Dict]
    ) -> List[Recommendation]:
        """Generate discount/markdown recommendations for slow-moving items"""
        recommendations = []
        
        # Get demand classification
        classification = place_data.get('demand_classification')
        if classification is None or len(classification) == 0:
            return recommendations
            
        # Find items with low demand or intermittent patterns
        slow_movers = classification[
            classification['demand_type'].isin(['Intermittent', 'Lumpy', 'Insufficient Data'])
        ]
        
        for _, row in slow_movers.head(10).iterrows():
            item_id = int(row['item_id'])
            demand_type = row['demand_type']
            
            # Calculate days since last significant demand
            history = self.data_processor.get_item_history(place_id, item_id, weeks=8)
            
            if len(history) == 0:
                continue
                
            recent_demand = history['demand'].tail(4).sum()
            avg_demand = history['demand'].mean()
            
            # Recommend discount if recent demand is below average
            if recent_demand < avg_demand * 0.5 and avg_demand > 0:
                # Determine discount level
                if recent_demand == 0:
                    discount = self.config.business.discount_tiers['aggressive']
                    risk_level = RiskLevel.HIGH
                elif recent_demand < avg_demand * 0.25:
                    discount = self.config.business.discount_tiers['moderate']
                    risk_level = RiskLevel.MEDIUM
                else:
                    discount = self.config.business.discount_tiers['light']
                    risk_level = RiskLevel.LOW
                    
                item_name = self._get_item_name(place_data, item_id)
                
                rec = Recommendation(
                    recommendation_id=self._generate_id(),
                    place_id=place_id,
                    item_id=item_id,
                    item_name=item_name,
                    recommendation_type=RecommendationType.DISCOUNT,
                    risk_level=risk_level,
                    action=f"Apply {int(discount*100)}% discount",
                    discount_percent=discount,
                    rationale=f"Recent demand ({recent_demand:.0f} units in 4 weeks) is significantly below "
                             f"average ({avg_demand:.1f} units/week). Demand pattern: {demand_type}.",
                    expected_impact=f"Expected to increase movement by 30-50% and reduce waste risk",
                    confidence=0.75,
                    valid_until=datetime.now() + timedelta(days=14),
                    additional_data={
                        'recent_demand': recent_demand,
                        'average_demand': avg_demand,
                        'demand_type': demand_type
                    }
                )
                recommendations.append(rec)
                
        return recommendations
    
    def _generate_prep_recommendations(
        self,
        place_id: int,
        place_data: Dict,
        forecasts: List[Dict],
        context: Optional[Dict] = None
    ) -> List[Recommendation]:
        """Generate kitchen prep quantity recommendations"""
        recommendations = []
        today = datetime.now()
        
        # Get weekly factor for today
        day_of_week = today.weekday()
        weekly_factor = self.config.get_weekly_factor(day_of_week)
        
        # Check for holiday impact
        is_holiday, holiday_name, holiday_impact = self.config.get_holiday_impact(today)
        
        # Context adjustments
        context_factor = 1.0
        context_notes = []
        
        if is_holiday:
            context_factor *= holiday_impact
            context_notes.append(f"{holiday_name} (impact: {holiday_impact:.0%})")
            
        if context:
            if context.get('special_event'):
                event_impact = context.get('event_impact', 1.2)
                context_factor *= event_impact
                context_notes.append(f"Special event: {context.get('special_event')}")
            if context.get('weather_impact'):
                context_factor *= context.get('weather_impact', 1.0)
                context_notes.append(f"Weather adjustment")
                
        # Generate prep recommendations for top items
        for forecast in forecasts[:15]:  # Top 15 items
            item_id = forecast.get('item_id')
            if not item_id or not forecast['forecast']:
                continue
                
            # Get weekly forecast and adjust to daily
            weekly_forecast = forecast['forecast'][0]['predicted_demand']
            
            # Adjust for day of week
            daily_estimate = (weekly_forecast / 7) * weekly_factor * context_factor
            
            # Round to practical prep quantities
            prep_qty = self._round_to_prep_unit(daily_estimate)
            
            if prep_qty > 0:
                item_name = self._get_item_name(place_data, item_id)
                
                # Build rationale
                rationale_parts = [
                    f"Weekly forecast: {weekly_forecast} units.",
                    f"Day factor ({['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][day_of_week]}): {weekly_factor:.0%}."
                ]
                if context_notes:
                    rationale_parts.append("Adjustments: " + ", ".join(context_notes))
                    
                rec = Recommendation(
                    recommendation_id=self._generate_id(),
                    place_id=place_id,
                    item_id=item_id,
                    item_name=item_name,
                    recommendation_type=RecommendationType.PREP_ADJUST,
                    risk_level=RiskLevel.LOW,
                    action=f"Prep {prep_qty} units for today",
                    quantity=prep_qty,
                    rationale=" ".join(rationale_parts),
                    expected_impact="Optimal prep to minimize waste while meeting demand",
                    confidence=0.80,
                    valid_until=datetime.now() + timedelta(days=1),
                    additional_data={
                        'weekly_forecast': weekly_forecast,
                        'day_factor': weekly_factor,
                        'context_factor': context_factor,
                        'day_of_week': day_of_week,
                        'context_notes': context_notes
                    }
                )
                recommendations.append(rec)
                
        return recommendations
    
    def _round_to_prep_unit(self, quantity: float) -> int:
        """Round quantity to practical kitchen prep units"""
        if quantity < 1:
            return 0
        elif quantity < 5:
            return round(quantity)
        elif quantity < 20:
            return round(quantity / 5) * 5  # Round to nearest 5
        elif quantity < 50:
            return round(quantity / 10) * 10  # Round to nearest 10
        else:
            return round(quantity / 25) * 25  # Round to nearest 25
            
    def _generate_bundle_recommendations(
        self,
        place_id: int,
        place_data: Dict
    ) -> List[Recommendation]:
        """Generate product bundle recommendations for promotions"""
        recommendations = []
        
        weekly = place_data.get('weekly_demand')
        if weekly is None or len(weekly) == 0:
            return recommendations
            
        # Find frequently co-occurring items (simplified approach)
        # In a full implementation, this would use association rules mining
        
        # Get items with similar demand patterns
        item_stats = weekly.groupby('item_id').agg({
            'demand': ['mean', 'std', 'count']
        }).reset_index()
        item_stats.columns = ['item_id', 'mean_demand', 'std_demand', 'weeks_active']
        
        # Find slow movers that could be bundled with fast movers
        slow_items = item_stats[item_stats['mean_demand'] < item_stats['mean_demand'].median()]
        fast_items = item_stats[item_stats['mean_demand'] >= item_stats['mean_demand'].quantile(0.75)]
        
        if len(slow_items) > 0 and len(fast_items) > 0:
            # Suggest bundling top slow mover with top fast mover
            slow_item = slow_items.iloc[0]
            fast_item = fast_items.iloc[0]
            
            slow_name = self._get_item_name(place_data, int(slow_item['item_id']))
            fast_name = self._get_item_name(place_data, int(fast_item['item_id']))
            
            rec = Recommendation(
                recommendation_id=self._generate_id(),
                place_id=place_id,
                item_id=int(slow_item['item_id']),
                item_name=slow_name,
                recommendation_type=RecommendationType.BUNDLE,
                risk_level=RiskLevel.LOW,
                action=f"Create combo with '{fast_name}'",
                rationale=f"'{slow_name}' has lower demand ({slow_item['mean_demand']:.1f}/week) - "
                         f"bundling with popular '{fast_name}' ({fast_item['mean_demand']:.1f}/week) can increase movement.",
                expected_impact="Potential 20-40% increase in slow-mover sales while boosting basket size",
                confidence=0.65,
                valid_until=datetime.now() + timedelta(days=30),
                additional_data={
                    'bundle_with_item_id': int(fast_item['item_id']),
                    'bundle_with_name': fast_name,
                    'slow_mover_demand': slow_item['mean_demand'],
                    'fast_mover_demand': fast_item['mean_demand']
                }
            )
            recommendations.append(rec)
            
        return recommendations
    
    def _generate_alert_recommendations(
        self,
        place_id: int,
        place_data: Dict,
        forecasts: List[Dict]
    ) -> List[Recommendation]:
        """Generate critical alert recommendations"""
        recommendations = []
        
        # Alert for items with high forecast uncertainty
        for forecast in forecasts:
            item_id = forecast.get('item_id')
            if not item_id:
                continue
                
            # Check for high uncertainty in forecast
            if forecast['forecast']:
                first_forecast = forecast['forecast'][0]
                predicted = first_forecast['predicted_demand']
                lower = first_forecast.get('lower', predicted)
                upper = first_forecast.get('upper', predicted)
                
                # If confidence interval is very wide, alert
                if upper and lower and predicted > 0:
                    spread = (upper - lower) / predicted
                    if spread > 1.5:  # More than 150% spread
                        item_name = self._get_item_name(place_data, item_id)
                        
                        rec = Recommendation(
                            recommendation_id=self._generate_id(),
                            place_id=place_id,
                            item_id=item_id,
                            item_name=item_name,
                            recommendation_type=RecommendationType.ALERT,
                            risk_level=RiskLevel.HIGH,
                            action="Review manually - high forecast uncertainty",
                            rationale=f"Forecast range is very wide: {lower}-{upper} units (predicted: {predicted}). "
                                     f"Demand pattern may be changing or data quality issue.",
                            expected_impact="Manual review can prevent over/under stocking",
                            confidence=0.6,
                            valid_until=datetime.now() + timedelta(days=3),
                            additional_data={
                                'predicted': predicted,
                                'lower_bound': lower,
                                'upper_bound': upper,
                                'spread_ratio': spread,
                                'demand_type': forecast.get('demand_type')
                            }
                        )
                        recommendations.append(rec)
                        
        return recommendations
    
    def get_recommendations_summary(
        self,
        recommendations: List[Recommendation]
    ) -> Dict:
        """
        Get a summary of recommendations for dashboard display.
        
        Args:
            recommendations: List of Recommendation objects
            
        Returns:
            Summary dictionary with counts and key metrics
        """
        summary = {
            'total_recommendations': len(recommendations),
            'by_type': {},
            'by_risk': {},
            'critical_actions': [],
            'top_priorities': []
        }
        
        # Count by type
        for rec in recommendations:
            type_key = rec.recommendation_type.value
            risk_key = rec.risk_level.value
            
            summary['by_type'][type_key] = summary['by_type'].get(type_key, 0) + 1
            summary['by_risk'][risk_key] = summary['by_risk'].get(risk_key, 0) + 1
            
            # Track critical actions
            if rec.risk_level == RiskLevel.CRITICAL:
                summary['critical_actions'].append({
                    'item': rec.item_name,
                    'action': rec.action,
                    'rationale': rec.rationale
                })
                
        # Top 5 priorities
        summary['top_priorities'] = [
            {
                'item': rec.item_name,
                'type': rec.recommendation_type.value,
                'action': rec.action,
                'risk': rec.risk_level.value
            }
            for rec in recommendations[:5]
        ]
        
        return summary
    
    def export_recommendations(
        self,
        recommendations: List[Recommendation],
        format: str = 'dataframe'
    ) -> Any:
        """
        Export recommendations to various formats.
        
        Args:
            recommendations: List of Recommendation objects
            format: Output format ('dataframe', 'dict', 'csv')
            
        Returns:
            Recommendations in requested format
        """
        data = [rec.to_dict() for rec in recommendations]
        
        if format == 'dict':
            return data
        elif format == 'dataframe':
            return pd.DataFrame(data)
        elif format == 'csv':
            df = pd.DataFrame(data)
            return df.to_csv(index=False)
        else:
            raise ValueError(f"Unknown format: {format}")
