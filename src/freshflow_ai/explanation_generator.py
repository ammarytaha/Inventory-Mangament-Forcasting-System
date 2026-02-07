"""
FreshFlow AI - Explanation Generator
=====================================

Generates human-readable explanations for AI recommendations.
Follows the principle of explainable AI - every recommendation
should be understandable by non-technical users.

Features:
- Plain English explanations
- Visual risk indicators
- Confidence level explanations
- Action-impact summaries
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

from .recommendation_engine import Recommendation, RecommendationType, RiskLevel
from .config import Config, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class ExplanationGenerator:
    """
    Generates human-readable explanations for AI recommendations.
    
    Every recommendation includes:
    - What to do (clear action)
    - Why (business rationale)
    - Expected outcome (impact)
    - Confidence level (how sure we are)
    
    Usage:
        generator = ExplanationGenerator()
        explanation = generator.explain_recommendation(recommendation)
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the ExplanationGenerator.
        
        Args:
            config: Configuration object
        """
        self.config = config or DEFAULT_CONFIG
        
        # Risk level descriptions
        self.risk_descriptions = {
            RiskLevel.CRITICAL: {
                'icon': '🔴',
                'label': 'Critical',
                'description': 'Requires immediate attention - potential significant loss',
                'action_urgency': 'Act today'
            },
            RiskLevel.HIGH: {
                'icon': '🟠',
                'label': 'High Priority',
                'description': 'Should be addressed within 24-48 hours',
                'action_urgency': 'Act within 1-2 days'
            },
            RiskLevel.MEDIUM: {
                'icon': '🟡',
                'label': 'Medium',
                'description': 'Plan to address this week',
                'action_urgency': 'Act this week'
            },
            RiskLevel.LOW: {
                'icon': '🟢',
                'label': 'Low',
                'description': 'Routine optimization opportunity',
                'action_urgency': 'When convenient'
            }
        }
        
        # Recommendation type templates
        self.type_templates = {
            RecommendationType.REORDER: {
                'icon': '📦',
                'action_verb': 'Order',
                'category': 'Replenishment'
            },
            RecommendationType.DISCOUNT: {
                'icon': '🏷️',
                'action_verb': 'Discount',
                'category': 'Markdown'
            },
            RecommendationType.BUNDLE: {
                'icon': '🎁',
                'action_verb': 'Bundle',
                'category': 'Promotion'
            },
            RecommendationType.PREP_ADJUST: {
                'icon': '👨‍🍳',
                'action_verb': 'Prepare',
                'category': 'Kitchen Prep'
            },
            RecommendationType.ALERT: {
                'icon': '⚠️',
                'action_verb': 'Review',
                'category': 'Alert'
            },
            RecommendationType.HOLD: {
                'icon': '✋',
                'action_verb': 'Hold',
                'category': 'No Action'
            }
        }
        
    def explain_recommendation(
        self,
        recommendation: Recommendation,
        detail_level: str = 'standard'
    ) -> Dict:
        """
        Generate a complete explanation for a recommendation.
        
        Args:
            recommendation: The recommendation to explain
            detail_level: 'brief', 'standard', or 'detailed'
            
        Returns:
            Dictionary with explanation components
        """
        rec_type = self.type_templates.get(recommendation.recommendation_type, {})
        risk_info = self.risk_descriptions.get(recommendation.risk_level, {})
        
        explanation = {
            'summary': self._generate_summary(recommendation, rec_type, risk_info),
            'headline': self._generate_headline(recommendation, rec_type),
            'action': self._format_action(recommendation, rec_type),
            'why': self._explain_why(recommendation),
            'impact': self._explain_impact(recommendation),
            'confidence': self._explain_confidence(recommendation),
            'risk': {
                'level': recommendation.risk_level.value,
                'icon': risk_info.get('icon', '⚪'),
                'label': risk_info.get('label', 'Unknown'),
                'description': risk_info.get('description', ''),
                'urgency': risk_info.get('action_urgency', 'Unknown')
            },
            'metadata': {
                'recommendation_id': recommendation.recommendation_id,
                'type': recommendation.recommendation_type.value,
                'category': rec_type.get('category', 'Other'),
                'created_at': recommendation.created_at.isoformat(),
                'valid_until': recommendation.valid_until.isoformat() if recommendation.valid_until else None
            }
        }
        
        # Add detailed breakdown if requested
        if detail_level == 'detailed':
            explanation['detailed_breakdown'] = self._generate_detailed_breakdown(recommendation)
            
        return explanation
    
    def _generate_summary(
        self,
        rec: Recommendation,
        rec_type: Dict,
        risk_info: Dict
    ) -> str:
        """Generate a one-line summary"""
        icon = rec_type.get('icon', '📋')
        risk_icon = risk_info.get('icon', '⚪')
        
        return f"{icon} {rec.item_name}: {rec.action} {risk_icon}"
    
    def _generate_headline(self, rec: Recommendation, rec_type: Dict) -> str:
        """Generate a descriptive headline"""
        category = rec_type.get('category', 'Action')
        
        headlines = {
            RecommendationType.REORDER: f"Replenishment needed for {rec.item_name}",
            RecommendationType.DISCOUNT: f"Price reduction recommended for {rec.item_name}",
            RecommendationType.BUNDLE: f"Bundle opportunity with {rec.item_name}",
            RecommendationType.PREP_ADJUST: f"Prep quantity update for {rec.item_name}",
            RecommendationType.ALERT: f"Attention required: {rec.item_name}",
            RecommendationType.HOLD: f"No action needed: {rec.item_name}"
        }
        
        return headlines.get(rec.recommendation_type, f"{category}: {rec.item_name}")
    
    def _format_action(self, rec: Recommendation, rec_type: Dict) -> Dict:
        """Format the recommended action"""
        action = {
            'text': rec.action,
            'verb': rec_type.get('action_verb', 'Do'),
            'specifics': {}
        }
        
        if rec.quantity is not None:
            action['specifics']['quantity'] = rec.quantity
            action['specifics']['quantity_text'] = f"{rec.quantity} units"
            
        if rec.discount_percent is not None:
            action['specifics']['discount'] = rec.discount_percent
            action['specifics']['discount_text'] = f"{int(rec.discount_percent * 100)}% off"
            
        return action
    
    def _explain_why(self, rec: Recommendation) -> Dict:
        """Generate the 'why' explanation"""
        why = {
            'summary': rec.rationale,
            'factors': []
        }
        
        # Extract factors from additional data
        data = rec.additional_data
        
        if 'demand_type' in data:
            why['factors'].append({
                'factor': 'Demand Pattern',
                'value': data['demand_type'],
                'explanation': self._explain_demand_type(data['demand_type'])
            })
            
        if 'weekly_forecast' in data:
            why['factors'].append({
                'factor': 'Forecasted Demand',
                'value': f"{data['weekly_forecast']} units/week",
                'explanation': "Based on historical patterns and trend analysis"
            })
            
        if 'safety_stock' in data:
            why['factors'].append({
                'factor': 'Safety Stock',
                'value': f"{data['safety_stock']} units",
                'explanation': "Buffer to prevent stockouts during demand variability"
            })
            
        if 'day_factor' in data:
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_idx = data.get('day_of_week', 0)
            why['factors'].append({
                'factor': 'Day Pattern',
                'value': f"{data['day_factor']:.0%}",
                'explanation': f"{day_names[day_idx]} typically sees {data['day_factor']:.0%} of average demand"
            })
            
        if 'context_notes' in data and data['context_notes']:
            why['factors'].append({
                'factor': 'Special Conditions',
                'value': ', '.join(data['context_notes']),
                'explanation': "External factors affecting demand"
            })
            
        return why
    
    def _explain_demand_type(self, demand_type: str) -> str:
        """Explain what a demand type means"""
        explanations = {
            'Smooth': "Regular, predictable demand - easy to forecast accurately",
            'Erratic': "Variable demand levels - requires wider safety buffers",
            'Intermittent': "Sporadic demand with gaps - common for specialty items",
            'Lumpy': "Unpredictable both in timing and quantity - highest uncertainty",
            'Insufficient Data': "Not enough history for reliable patterns"
        }
        return explanations.get(demand_type, "Demand pattern classification")
    
    def _explain_impact(self, rec: Recommendation) -> Dict:
        """Explain the expected impact"""
        impact = {
            'summary': rec.expected_impact,
            'metrics': []
        }
        
        # Add quantified impacts where possible
        if rec.recommendation_type == RecommendationType.REORDER:
            impact['metrics'].append({
                'metric': 'Stockout Prevention',
                'description': 'Ensures product availability for forecasted demand'
            })
            
        elif rec.recommendation_type == RecommendationType.DISCOUNT:
            if rec.discount_percent:
                estimated_lift = 30 + (rec.discount_percent * 100)  # Rough estimate
                impact['metrics'].append({
                    'metric': 'Sales Velocity',
                    'description': f"Expected {estimated_lift:.0f}% increase in sales velocity"
                })
                impact['metrics'].append({
                    'metric': 'Waste Reduction',
                    'description': "Reduced risk of expiry-related waste"
                })
                
        elif rec.recommendation_type == RecommendationType.PREP_ADJUST:
            impact['metrics'].append({
                'metric': 'Prep Efficiency',
                'description': 'Optimized prep quantity to match expected demand'
            })
            impact['metrics'].append({
                'metric': 'Waste Minimization',
                'description': 'Reduced over-prep waste while ensuring availability'
            })
            
        return impact
    
    def _explain_confidence(self, rec: Recommendation) -> Dict:
        """Explain the confidence level"""
        confidence_pct = int(rec.confidence * 100)
        
        if rec.confidence >= 0.85:
            level = 'High'
            explanation = "Strong historical patterns support this recommendation"
        elif rec.confidence >= 0.7:
            level = 'Good'
            explanation = "Reasonable confidence based on available data"
        elif rec.confidence >= 0.5:
            level = 'Moderate'
            explanation = "Some uncertainty - consider reviewing manually"
        else:
            level = 'Low'
            explanation = "Limited data - treat as suggestion for review"
            
        return {
            'percentage': confidence_pct,
            'level': level,
            'explanation': explanation,
            'visual': '●' * (confidence_pct // 20) + '○' * (5 - confidence_pct // 20)
        }
    
    def _generate_detailed_breakdown(self, rec: Recommendation) -> Dict:
        """Generate a detailed breakdown for power users"""
        breakdown = {
            'data_inputs': [],
            'model_details': {},
            'calculation_steps': []
        }
        
        data = rec.additional_data
        
        # List all data inputs
        for key, value in data.items():
            breakdown['data_inputs'].append({
                'field': key.replace('_', ' ').title(),
                'value': value
            })
            
        # Model details if available
        if 'forecast_model' in data:
            breakdown['model_details'] = {
                'model': data['forecast_model'],
                'description': self._describe_model(data['forecast_model'])
            }
            
        return breakdown
    
    def _describe_model(self, model_name: str) -> str:
        """Describe the forecasting model used"""
        descriptions = {
            'prophet': "Facebook Prophet - excellent for capturing seasonality and trends",
            'croston': "Croston's method - specialized for intermittent demand patterns",
            'lightgbm': "LightGBM - machine learning model for complex patterns",
            'moving_average': "Moving average - simple, robust baseline method",
            'insufficient_data': "Not enough data for sophisticated modeling"
        }
        return descriptions.get(model_name, f"Model: {model_name}")
    
    def generate_dashboard_cards(
        self,
        recommendations: List[Recommendation],
        max_cards: int = 5
    ) -> List[Dict]:
        """
        Generate card-style explanations for dashboard display.
        
        Args:
            recommendations: List of recommendations
            max_cards: Maximum number of cards to generate
            
        Returns:
            List of card dictionaries for UI rendering
        """
        cards = []
        
        for rec in recommendations[:max_cards]:
            explanation = self.explain_recommendation(rec, detail_level='brief')
            
            card = {
                'id': rec.recommendation_id,
                'icon': self.type_templates.get(rec.recommendation_type, {}).get('icon', '📋'),
                'risk_icon': self.risk_descriptions.get(rec.risk_level, {}).get('icon', '⚪'),
                'item_name': rec.item_name,
                'headline': explanation['headline'],
                'action': rec.action,
                'urgency': explanation['risk']['urgency'],
                'confidence_visual': explanation['confidence']['visual'],
                'why_preview': rec.rationale[:100] + '...' if len(rec.rationale) > 100 else rec.rationale,
                'category': explanation['metadata']['category'],
                'risk_level': rec.risk_level.value,
                'can_expand': True
            }
            cards.append(card)
            
        return cards
    
    def generate_action_summary(
        self,
        recommendations: List[Recommendation]
    ) -> str:
        """
        Generate a text summary of all recommended actions.
        
        Args:
            recommendations: List of recommendations
            
        Returns:
            Formatted text summary
        """
        if not recommendations:
            return "✅ No immediate actions required. Inventory is well-balanced."
            
        summary_parts = []
        
        # Count by type
        by_type = {}
        for rec in recommendations:
            type_name = rec.recommendation_type.value
            by_type[type_name] = by_type.get(type_name, 0) + 1
            
        # Count by risk
        critical_count = sum(1 for r in recommendations if r.risk_level == RiskLevel.CRITICAL)
        high_count = sum(1 for r in recommendations if r.risk_level == RiskLevel.HIGH)
        
        # Header
        summary_parts.append(f"📊 **Action Summary** - {len(recommendations)} recommendations\n")
        
        # Urgency notice
        if critical_count > 0:
            summary_parts.append(f"🔴 **{critical_count} critical items** require immediate attention\n")
        if high_count > 0:
            summary_parts.append(f"🟠 **{high_count} high-priority items** should be addressed today\n")
            
        # By category
        summary_parts.append("\n**By Category:**")
        for type_name, count in by_type.items():
            icon = self.type_templates.get(RecommendationType(type_name), {}).get('icon', '📋')
            summary_parts.append(f"  {icon} {type_name.title()}: {count}")
            
        # Top 3 actions
        summary_parts.append("\n**Top Actions:**")
        for i, rec in enumerate(recommendations[:3], 1):
            icon = self.risk_descriptions.get(rec.risk_level, {}).get('icon', '⚪')
            summary_parts.append(f"  {i}. {icon} {rec.item_name}: {rec.action}")
            
        return '\n'.join(summary_parts)
    
    def format_for_print(
        self,
        recommendation: Recommendation
    ) -> str:
        """
        Format a recommendation for text/print output.
        
        Args:
            recommendation: The recommendation to format
            
        Returns:
            Formatted text string
        """
        explanation = self.explain_recommendation(recommendation, detail_level='standard')
        
        lines = [
            "=" * 60,
            explanation['headline'],
            "=" * 60,
            "",
            f"📋 Action: {recommendation.action}",
            f"⏰ Urgency: {explanation['risk']['urgency']}",
            f"📊 Confidence: {explanation['confidence']['visual']} ({explanation['confidence']['percentage']}%)",
            "",
            "📝 Why this recommendation:",
            f"   {recommendation.rationale}",
            "",
            "💡 Expected Impact:",
            f"   {recommendation.expected_impact}",
            "",
            "=" * 60
        ]
        
        return '\n'.join(lines)
