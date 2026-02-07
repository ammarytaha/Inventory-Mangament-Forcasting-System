"""
Output Generator Service
=========================
Unified output generation for the inventory management system.

Capabilities:
- Combine forecasts, adjustments, health scores, and recommendations
- Export to CSV and JSON formats
- Generate visualization-ready data
- Create executive summaries

Output Structure:
outputs/
├── forecasts/
│   ├── forecast_all_items.csv
│   └── forecast_summary.json
├── analytics/
│   ├── health_scores.csv
│   └── demand_summary.csv
├── decisions/
│   ├── recommendations_all.csv
│   ├── recommendations_critical.csv
│   └── executive_summary.json
└── visualizations/
    ├── demand_vs_forecast.json
    └── risk_indicators.json
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, date

from utils.logger import get_logger
from utils.constants import OUTPUT_CONFIG

logger = get_logger(__name__)


@dataclass
class OutputPackage:
    """
    Complete output package from the inventory system.
    
    Attributes
    ----------
    forecasts : pd.DataFrame
        All forecasts with adjustments
    health_scores : pd.DataFrame
        Inventory health assessments
    recommendations : pd.DataFrame
        Action recommendations
    executive_summary : Dict[str, Any]
        High-level summary for stakeholders
    visualization_data : Dict[str, Any]
        Data prepared for visualization
    metadata : Dict[str, Any]
        Processing metadata
    """
    forecasts: pd.DataFrame = field(default_factory=pd.DataFrame)
    health_scores: pd.DataFrame = field(default_factory=pd.DataFrame)
    recommendations: pd.DataFrame = field(default_factory=pd.DataFrame)
    executive_summary: Dict[str, Any] = field(default_factory=dict)
    visualization_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class OutputGenerator:
    """
    Generate and export unified outputs from the inventory system.
    
    This class handles:
    - Combining data from all pipeline stages
    - Formatting for different audiences
    - Export to multiple formats
    - Generating visualization-ready data
    
    Usage
    -----
    >>> generator = OutputGenerator(output_dir="./outputs")
    >>> package = generator.generate_output_package(
    ...     forecasts, health_scores, recommendations
    ... )
    >>> generator.export_all(package)
    """
    
    def __init__(
        self,
        output_dir: str = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize the output generator.
        
        Parameters
        ----------
        output_dir : str
            Base directory for outputs
        config : dict, optional
            Custom configuration
        """
        self.config = config or OUTPUT_CONFIG
        
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(self.config.get('output_base_dir', 'outputs'))
        
        # Create output directories
        self._create_directories()
        
        logger.info(f"OutputGenerator initialized: output_dir={self.output_dir}")
    
    def _create_directories(self) -> None:
        """Create output directory structure."""
        subdirs = [
            self.config.get('analytics_subdir', 'analytics'),
            self.config.get('decisions_subdir', 'decisions'),
            self.config.get('forecasts_subdir', 'forecasts'),
            'visualizations'
        ]
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        for subdir in subdirs:
            (self.output_dir / subdir).mkdir(exist_ok=True)
    
    def generate_output_package(
        self,
        forecasts: pd.DataFrame,
        adjusted_forecasts: Optional[pd.DataFrame],
        health_scores: pd.DataFrame,
        recommendations: pd.DataFrame,
        historical_demand: Optional[pd.DataFrame] = None,
        item_names: Optional[Dict[Any, str]] = None
    ) -> OutputPackage:
        """
        Generate a complete output package.
        
        Parameters
        ----------
        forecasts : pd.DataFrame
            Raw forecast data
        adjusted_forecasts : pd.DataFrame, optional
            Context-adjusted forecasts
        health_scores : pd.DataFrame
            Health score data
        recommendations : pd.DataFrame
            Recommendation data
        historical_demand : pd.DataFrame, optional
            Historical demand for visualization
        item_names : Dict, optional
            Item ID to name mapping
        
        Returns
        -------
        OutputPackage
            Complete output package
        """
        package = OutputPackage()
        
        # Combine forecasts
        if adjusted_forecasts is not None and len(adjusted_forecasts) > 0:
            package.forecasts = adjusted_forecasts
        else:
            package.forecasts = forecasts
        
        package.health_scores = health_scores
        package.recommendations = recommendations
        
        # Add item names if available
        if item_names and len(package.forecasts) > 0:
            package.forecasts['item_name'] = package.forecasts['item_id'].map(item_names)
        
        # Generate executive summary
        package.executive_summary = self._generate_executive_summary(
            forecasts, health_scores, recommendations
        )
        
        # Generate visualization data
        package.visualization_data = self._generate_visualization_data(
            forecasts, adjusted_forecasts, health_scores, 
            recommendations, historical_demand
        )
        
        # Add metadata
        package.metadata = {
            'generated_at': datetime.now().isoformat(),
            'forecast_rows': len(forecasts),
            'health_scores_count': len(health_scores),
            'recommendations_count': len(recommendations),
            'output_version': '2.0'
        }
        
        return package
    
    def _generate_executive_summary(
        self,
        forecasts: pd.DataFrame,
        health_scores: pd.DataFrame,
        recommendations: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Generate executive summary for stakeholders.
        
        This summary is designed to be presented to non-technical
        business stakeholders.
        """
        summary = {
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'report_title': 'Fresh Flow Markets - Inventory Intelligence Report'
        }
        
        # Forecast summary
        if len(forecasts) > 0:
            total_forecast = forecasts['forecast'].sum() if 'forecast' in forecasts.columns else 0
            items_forecast = forecasts['item_id'].nunique() if 'item_id' in forecasts.columns else 0
            
            summary['forecast_summary'] = {
                'items_forecasted': int(items_forecast),
                'total_expected_demand': round(float(total_forecast), 0),
                'forecast_horizon_days': int(forecasts['date'].nunique()) if 'date' in forecasts.columns else 0
            }
        
        # Risk summary
        if len(health_scores) > 0:
            risk_counts = health_scores['risk_level'].value_counts().to_dict() \
                if 'risk_level' in health_scores.columns else {}
            
            summary['risk_summary'] = {
                'high_risk_items': risk_counts.get('HIGH', 0),
                'medium_risk_items': risk_counts.get('MEDIUM', 0),
                'low_risk_items': risk_counts.get('LOW', 0),
                'average_health_score': round(
                    health_scores['health_score'].mean(), 1
                ) if 'health_score' in health_scores.columns else 0
            }
        
        # Action summary
        if len(recommendations) > 0:
            urgency_counts = recommendations['urgency'].value_counts().to_dict() \
                if 'urgency' in recommendations.columns else {}
            action_counts = recommendations['action'].value_counts().to_dict() \
                if 'action' in recommendations.columns else {}
            
            summary['action_summary'] = {
                'total_recommendations': len(recommendations),
                'critical_actions': urgency_counts.get('CRITICAL', 0),
                'high_priority_actions': urgency_counts.get('HIGH', 0),
                'top_actions_by_type': action_counts
            }
            
            # Top critical items
            if 'urgency' in recommendations.columns:
                critical = recommendations[recommendations['urgency'] == 'CRITICAL']
                if len(critical) > 0:
                    summary['critical_items'] = critical.head(5)[
                        ['item_name', 'action', 'explanation']
                    ].to_dict('records') if 'item_name' in critical.columns else []
        
        # Key insights (plain English)
        summary['key_insights'] = self._generate_key_insights(
            health_scores, recommendations
        )
        
        return summary
    
    def _generate_key_insights(
        self,
        health_scores: pd.DataFrame,
        recommendations: pd.DataFrame
    ) -> List[str]:
        """Generate plain English insights."""
        insights = []
        
        if len(health_scores) > 0 and 'risk_level' in health_scores.columns:
            high_risk = (health_scores['risk_level'] == 'HIGH').sum()
            
            if high_risk > 0:
                insights.append(
                    f"⚠️ {high_risk} items require immediate attention due to "
                    f"high inventory risk."
                )
            else:
                insights.append(
                    "✅ All items are at low or medium risk levels."
                )
        
        if len(recommendations) > 0:
            if 'urgency' in recommendations.columns:
                critical = (recommendations['urgency'] == 'CRITICAL').sum()
                if critical > 0:
                    insights.append(
                        f"🔴 {critical} CRITICAL actions needed today to prevent "
                        f"stockouts or waste."
                    )
            
            if 'action' in recommendations.columns:
                reorders = (recommendations['action'] == 'reorder').sum()
                discounts = (recommendations['action'] == 'discount').sum()
                
                if reorders > 0:
                    insights.append(
                        f"📦 {reorders} items need reordering to maintain stock levels."
                    )
                
                if discounts > 0:
                    insights.append(
                        f"🏷️ {discounts} items recommended for discount to reduce "
                        f"waste risk."
                    )
            
            if 'estimated_impact' in recommendations.columns:
                total_impact = recommendations['estimated_impact'].sum()
                insights.append(
                    f"💰 Estimated value at risk or opportunity: ${total_impact:,.0f}"
                )
        
        return insights
    
    def _generate_visualization_data(
        self,
        forecasts: pd.DataFrame,
        adjusted_forecasts: Optional[pd.DataFrame],
        health_scores: pd.DataFrame,
        recommendations: pd.DataFrame,
        historical_demand: Optional[pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Generate data structures optimized for visualization.
        
        These structures can be consumed by charting libraries.
        """
        viz_data = {}
        
        # Forecast vs Adjusted Forecast
        if adjusted_forecasts is not None and len(adjusted_forecasts) > 0:
            viz_data['forecast_comparison'] = self._prep_forecast_comparison(
                forecasts, adjusted_forecasts
            )
        
        # Historical vs Forecast
        if historical_demand is not None and len(historical_demand) > 0:
            viz_data['demand_timeline'] = self._prep_demand_timeline(
                historical_demand, forecasts
            )
        
        # Risk Distribution
        if len(health_scores) > 0:
            viz_data['risk_distribution'] = self._prep_risk_distribution(health_scores)
        
        # Recommendation Priority Matrix
        if len(recommendations) > 0:
            viz_data['action_matrix'] = self._prep_action_matrix(recommendations)
        
        return viz_data
    
    def _prep_forecast_comparison(
        self,
        forecasts: pd.DataFrame,
        adjusted_forecasts: pd.DataFrame
    ) -> Dict[str, Any]:
        """Prepare forecast comparison data."""
        if 'item_id' not in forecasts.columns:
            return {}
        
        comparison = {
            'items': [],
            'dates': [],
            'base_forecast': [],
            'adjusted_forecast': []
        }
        
        # Aggregate by date
        if 'date' in forecasts.columns and 'forecast' in forecasts.columns:
            base_by_date = forecasts.groupby('date')['forecast'].sum()
            
            for date_val, forecast_val in base_by_date.items():
                comparison['dates'].append(str(date_val))
                comparison['base_forecast'].append(float(forecast_val))
        
        if 'date' in adjusted_forecasts.columns:
            adj_col = 'adjusted_forecast' if 'adjusted_forecast' in adjusted_forecasts.columns \
                else 'forecast'
            adj_by_date = adjusted_forecasts.groupby('date')[adj_col].sum()
            
            for date_val, adj_val in adj_by_date.items():
                comparison['adjusted_forecast'].append(float(adj_val))
        
        return comparison
    
    def _prep_demand_timeline(
        self,
        historical: pd.DataFrame,
        forecasts: pd.DataFrame
    ) -> Dict[str, List]:
        """Prepare historical + forecast timeline."""
        timeline = {
            'dates': [],
            'values': [],
            'type': []  # 'historical' or 'forecast'
        }
        
        # Add historical data
        if 'date' in historical.columns and 'demand' in historical.columns:
            hist_by_date = historical.groupby('date')['demand'].sum().sort_index()
            
            for date_val, demand_val in hist_by_date.items():
                timeline['dates'].append(str(date_val))
                timeline['values'].append(float(demand_val))
                timeline['type'].append('historical')
        
        # Add forecast data
        if 'date' in forecasts.columns and 'forecast' in forecasts.columns:
            forecast_by_date = forecasts.groupby('date')['forecast'].sum().sort_index()
            
            for date_val, forecast_val in forecast_by_date.items():
                timeline['dates'].append(str(date_val))
                timeline['values'].append(float(forecast_val))
                timeline['type'].append('forecast')
        
        return timeline
    
    def _prep_risk_distribution(
        self,
        health_scores: pd.DataFrame
    ) -> Dict[str, Any]:
        """Prepare risk distribution data."""
        if 'risk_level' not in health_scores.columns:
            return {}
        
        risk_counts = health_scores['risk_level'].value_counts().to_dict()
        
        return {
            'categories': ['LOW', 'MEDIUM', 'HIGH'],
            'counts': [
                risk_counts.get('LOW', 0),
                risk_counts.get('MEDIUM', 0),
                risk_counts.get('HIGH', 0)
            ],
            'colors': ['#28a745', '#ffc107', '#dc3545']
        }
    
    def _prep_action_matrix(
        self,
        recommendations: pd.DataFrame
    ) -> Dict[str, Any]:
        """Prepare action priority matrix."""
        if 'action' not in recommendations.columns:
            return {}
        
        matrix = {
            'actions': [],
            'urgency': [],
            'count': [],
            'total_impact': []
        }
        
        for (action, urgency), group in recommendations.groupby(['action', 'urgency']):
            matrix['actions'].append(action)
            matrix['urgency'].append(urgency)
            matrix['count'].append(len(group))
            
            if 'estimated_impact' in group.columns:
                matrix['total_impact'].append(float(group['estimated_impact'].sum()))
            else:
                matrix['total_impact'].append(0)
        
        return matrix
    
    def export_all(
        self,
        package: OutputPackage,
        formats: List[str] = None
    ) -> Dict[str, str]:
        """
        Export all outputs to files.
        
        Parameters
        ----------
        package : OutputPackage
            The output package to export
        formats : List[str], optional
            Export formats ('csv', 'json'). Default: both
        
        Returns
        -------
        Dict[str, str]
            Mapping of output type to file path
        """
        formats = formats or ['csv', 'json']
        exported = {}
        
        forecasts_dir = self.output_dir / self.config.get('forecasts_subdir', 'forecasts')
        analytics_dir = self.output_dir / self.config.get('analytics_subdir', 'analytics')
        decisions_dir = self.output_dir / self.config.get('decisions_subdir', 'decisions')
        viz_dir = self.output_dir / 'visualizations'
        
        # Export forecasts
        if len(package.forecasts) > 0 and 'csv' in formats:
            path = forecasts_dir / 'forecast_all_items.csv'
            package.forecasts.to_csv(path, index=False)
            exported['forecasts_csv'] = str(path)
            logger.info(f"Exported forecasts to {path}")
        
        # Export health scores
        if len(package.health_scores) > 0 and 'csv' in formats:
            path = analytics_dir / 'health_scores.csv'
            package.health_scores.to_csv(path, index=False)
            exported['health_scores_csv'] = str(path)
            logger.info(f"Exported health scores to {path}")
        
        # Export recommendations
        if len(package.recommendations) > 0 and 'csv' in formats:
            # All recommendations
            path = decisions_dir / 'recommendations_all.csv'
            package.recommendations.to_csv(path, index=False)
            exported['recommendations_csv'] = str(path)
            
            # Critical only
            if 'urgency' in package.recommendations.columns:
                critical = package.recommendations[
                    package.recommendations['urgency'].isin(['CRITICAL', 'HIGH'])
                ]
                if len(critical) > 0:
                    crit_path = decisions_dir / 'recommendations_critical.csv'
                    critical.to_csv(crit_path, index=False)
                    exported['recommendations_critical_csv'] = str(crit_path)
            
            logger.info(f"Exported recommendations to {path}")
        
        # Export executive summary
        if 'json' in formats:
            path = decisions_dir / 'executive_summary.json'
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(package.executive_summary, f, indent=2, default=str)
            exported['executive_summary_json'] = str(path)
            logger.info(f"Exported executive summary to {path}")
        
        # Export visualization data
        if 'json' in formats and package.visualization_data:
            path = viz_dir / 'visualization_data.json'
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(package.visualization_data, f, indent=2, default=str)
            exported['visualization_json'] = str(path)
            logger.info(f"Exported visualization data to {path}")
        
        # Export metadata
        package.metadata['exported_files'] = exported
        meta_path = self.output_dir / 'run_metadata.json'
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(package.metadata, f, indent=2, default=str)
        
        logger.info(f"Export complete: {len(exported)} files written")
        
        return exported


class SimplePlotter:
    """
    Simple visualization hooks for demo purposes.
    
    This class provides basic plotting without heavy dependencies.
    Uses matplotlib if available, otherwise outputs data for external plotting.
    
    Usage
    -----
    >>> plotter = SimplePlotter()
    >>> plotter.plot_demand_forecast(historical_df, forecast_df, item_id=123)
    >>> plotter.plot_risk_distribution(health_scores_df)
    """
    
    def __init__(self):
        """Initialize the plotter."""
        self.has_matplotlib = False
        try:
            import matplotlib.pyplot as plt
            self.plt = plt
            self.has_matplotlib = True
        except ImportError:
            logger.warning("matplotlib not available - plotting disabled")
    
    def plot_demand_forecast(
        self,
        historical: pd.DataFrame,
        forecast: pd.DataFrame,
        item_id: Any,
        save_path: Optional[str] = None
    ) -> Optional[Any]:
        """
        Plot historical demand vs forecast for an item.
        
        Parameters
        ----------
        historical : pd.DataFrame
            Historical demand data
        forecast : pd.DataFrame
            Forecast data with lower/upper bounds
        item_id : Any
            Item to plot
        save_path : str, optional
            Path to save the plot
        
        Returns
        -------
        Figure or None
        """
        if not self.has_matplotlib:
            logger.warning("Cannot plot: matplotlib not available")
            return None
        
        # Filter to item
        hist = historical[historical['item_id'] == item_id] \
            if 'item_id' in historical.columns else historical
        fore = forecast[forecast['item_id'] == item_id] \
            if 'item_id' in forecast.columns else forecast
        
        fig, ax = self.plt.subplots(figsize=(12, 6))
        
        # Plot historical
        if len(hist) > 0 and 'date' in hist.columns:
            ax.plot(
                hist['date'], hist['demand'],
                'b-', label='Historical Demand', linewidth=2
            )
        
        # Plot forecast
        if len(fore) > 0 and 'date' in fore.columns:
            ax.plot(
                fore['date'], fore['forecast'],
                'r--', label='Forecast', linewidth=2
            )
            
            # Confidence band
            if 'lower_bound' in fore.columns and 'upper_bound' in fore.columns:
                ax.fill_between(
                    fore['date'],
                    fore['lower_bound'],
                    fore['upper_bound'],
                    alpha=0.2, color='red', label='Confidence Band'
                )
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Demand (units)')
        ax.set_title(f'Demand Forecast - Item {item_id}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self.plt.tight_layout()
        
        if save_path:
            self.plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        
        return fig
    
    def plot_risk_distribution(
        self,
        health_scores: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> Optional[Any]:
        """
        Plot distribution of inventory risk levels.
        
        Parameters
        ----------
        health_scores : pd.DataFrame
            Health score data with risk_level column
        save_path : str, optional
            Path to save the plot
        
        Returns
        -------
        Figure or None
        """
        if not self.has_matplotlib:
            return None
        
        if 'risk_level' not in health_scores.columns:
            return None
        
        risk_counts = health_scores['risk_level'].value_counts()
        
        fig, ax = self.plt.subplots(figsize=(8, 6))
        
        colors = {'LOW': '#28a745', 'MEDIUM': '#ffc107', 'HIGH': '#dc3545'}
        bar_colors = [colors.get(level, 'gray') for level in risk_counts.index]
        
        ax.bar(risk_counts.index, risk_counts.values, color=bar_colors)
        ax.set_xlabel('Risk Level')
        ax.set_ylabel('Number of Items')
        ax.set_title('Inventory Risk Distribution')
        
        # Add value labels
        for i, (level, count) in enumerate(risk_counts.items()):
            ax.text(i, count + 0.5, str(count), ha='center', fontweight='bold')
        
        self.plt.tight_layout()
        
        if save_path:
            self.plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_recommendation_summary(
        self,
        recommendations: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> Optional[Any]:
        """
        Plot recommendation summary by action type.
        """
        if not self.has_matplotlib:
            return None
        
        if 'action' not in recommendations.columns:
            return None
        
        action_counts = recommendations['action'].value_counts()
        
        fig, ax = self.plt.subplots(figsize=(10, 6))
        
        ax.barh(action_counts.index, action_counts.values, color='steelblue')
        ax.set_xlabel('Count')
        ax.set_ylabel('Action Type')
        ax.set_title('Recommendations by Action Type')
        
        self.plt.tight_layout()
        
        if save_path:
            self.plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
