"""
Inventory Analytics Foundation
===============================
Creates analytics-ready datasets for inventory management:
- Daily/weekly demand per SKU
- Stock levels over time
- Expiry risk indicators
- Waste proxy metrics (overstock vs sales)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class InventoryAnalytics:
    """Build analytics foundation for inventory decisions."""
    
    def __init__(self, cleaned_data: Dict[str, pd.DataFrame]):
        """
        Initialize with cleaned data.
        
        Parameters:
        -----------
        cleaned_data : dict
            Dictionary of cleaned DataFrames keyed by file name
        """
        self.data = cleaned_data
        self.analytics_tables = {}
        
    def calculate_demand_per_sku(self, 
                                 period: str = 'daily',
                                 start_date: Optional[datetime] = None,
                                 end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Calculate demand (quantity sold) per SKU by period.
        
        Parameters:
        -----------
        period : str
            'daily', 'weekly', or 'monthly'
        start_date : datetime, optional
            Start date for analysis
        end_date : datetime, optional
            End date for analysis
        """
        # Load required tables
        if 'fct_order_items.csv' not in self.data:
            raise ValueError("fct_order_items.csv not found in cleaned data")
        
        if 'dim_bill_of_materials.csv' not in self.data:
            raise ValueError("dim_bill_of_materials.csv not found in cleaned data")
        
        order_items = self.data['fct_order_items.csv'].copy()
        bom = self.data['dim_bill_of_materials.csv'].copy()
        
        # Get orders for date filtering
        if 'fct_orders.csv' in self.data:
            orders = self.data['fct_orders.csv'].copy()
            
            # Merge order items with orders to get dates
            if 'created' in orders.columns and 'order_id' in order_items.columns:
                order_items = order_items.merge(
                    orders[['id', 'created', 'place_id']],
                    left_on='order_id',
                    right_on='id',
                    how='left',
                    suffixes=('', '_order')
                )
        
        # Filter by date if provided
        if start_date and 'created' in order_items.columns:
            order_items = order_items[order_items['created'] >= start_date]
        if end_date and 'created' in order_items.columns:
            order_items = order_items[order_items['created'] <= end_date]
        
        # Map menu items to SKUs via Bill of Materials
        # First, get composite items (parent SKUs)
        if len(bom) > 0 and 'parent_sku_id' in bom.columns:
            # For composite items, we need to explode the BOM
            # For now, we'll work with direct SKU references if available
            pass
            
            # If order_items has item_id, we need to map to SKUs
            # This mapping might require dim_menu_items -> dim_skus relationship
            # For now, we'll calculate demand at menu item level and note SKU mapping needed
        
        # Aggregate demand
        if 'created' in order_items.columns:
            order_items['date'] = pd.to_datetime(order_items['created']).dt.date
            
            if period == 'daily':
                order_items['period'] = order_items['date']
            elif period == 'weekly':
                order_items['period'] = pd.to_datetime(order_items['created']).dt.to_period('W').astype(str)
            elif period == 'monthly':
                order_items['period'] = pd.to_datetime(order_items['created']).dt.to_period('M').astype(str)
        else:
            # No date column, aggregate overall
            order_items['period'] = 'all_time'
        
        # Group by item and period
        if 'item_id' in order_items.columns and 'quantity' in order_items.columns:
            demand = order_items.groupby(['item_id', 'period']).agg({
                'quantity': 'sum',
                'price': 'mean',  # Average price
                'cost': 'sum'  # Total cost
            }).reset_index()
            
            demand.columns = ['item_id', 'period', 'demand_quantity', 'avg_price', 'total_cost']
            
            # Add place-level aggregation if available
            if 'place_id' in order_items.columns:
                demand_by_place = order_items.groupby(['item_id', 'place_id', 'period']).agg({
                    'quantity': 'sum'
                }).reset_index()
                demand_by_place.columns = ['item_id', 'place_id', 'period', 'demand_quantity']
                self.analytics_tables['demand_by_place'] = demand_by_place
        else:
            # Fallback: aggregate by order_item id
            demand = order_items.groupby('period').agg({
                'quantity': 'sum'
            }).reset_index()
            demand.columns = ['period', 'demand_quantity']
        
        self.analytics_tables['demand_per_sku'] = demand
        return demand
    
    def calculate_stock_levels_over_time(self) -> pd.DataFrame:
        """
        Calculate stock levels over time from inventory reports.
        If inventory reports are sparse, use SKU dimension table as snapshot.
        """
        stock_levels = []
        
        # Try to get time-series from inventory reports
        if 'fct_inventory_reports.csv' in self.data:
            inv_reports = self.data['fct_inventory_reports.csv'].copy()
            if len(inv_reports) > 0:
                # Process inventory reports
                # Note: fct_inventory_reports appeared empty in discovery
                pass
        
        # Fallback: Use dim_skus as current snapshot
        if 'dim_skus.csv' in self.data:
            skus = self.data['dim_skus.csv'].copy()
            
            # Create snapshot with current date
            current_date = datetime.now().date()
            
            stock_levels_df = pd.DataFrame({
                'sku_id': skus['id'] if 'id' in skus.columns else skus.index,
                'date': current_date,
                'quantity': skus['quantity'] if 'quantity' in skus.columns else 0,
                'low_stock_threshold': skus['low_stock_threshold'] if 'low_stock_threshold' in skus.columns else None,
                'unit': skus['unit'] if 'unit' in skus.columns else None,
                'sku_title': skus['title'] if 'title' in skus.columns else None
            })
            
            # Calculate stock status
            if 'low_stock_threshold' in skus.columns and 'quantity' in skus.columns:
                stock_levels_df['stock_status'] = stock_levels_df.apply(
                    lambda row: 'low_stock' if row['quantity'] <= row['low_stock_threshold'] 
                               else 'adequate' if row['quantity'] > row['low_stock_threshold'] * 2
                               else 'normal',
                    axis=1
                )
            else:
                stock_levels_df['stock_status'] = 'unknown'
        
        self.analytics_tables['stock_levels'] = stock_levels_df
        return stock_levels_df
    
    def calculate_expiry_risk_indicators(self) -> pd.DataFrame:
        """
        Calculate expiry risk indicators.
        Since we don't have expiry dates, we'll use:
        - Days since last sale (if item not selling, higher expiry risk)
        - Stock turnover rate
        - Overstock indicators
        """
        expiry_risk = []
        
        # Get SKU stock levels
        stock_levels = self.calculate_stock_levels_over_time()
        
        # Get demand data
        if 'demand_per_sku' in self.analytics_tables:
            demand = self.analytics_tables['demand_per_sku']
        else:
            demand = self.calculate_demand_per_sku()
        
        # Get order items for last sale date
        if 'fct_order_items.csv' in self.data:
            order_items = self.data['fct_order_items.csv'].copy()
            
            # Get orders for dates
            if 'fct_orders.csv' in self.data:
                orders = self.data['fct_orders.csv'].copy()
                if 'created' in orders.columns:
                    order_items = order_items.merge(
                        orders[['id', 'created']],
                        left_on='order_id',
                        right_on='id',
                        how='left'
                    )
        
        # Calculate metrics per SKU/item
        if 'item_id' in demand.columns:
            # Get last sale date per item
            if 'created' in order_items.columns and 'item_id' in order_items.columns:
                last_sale = order_items.groupby('item_id')['created'].max().reset_index()
                last_sale.columns = ['item_id', 'last_sale_date']
                
                # Calculate days since last sale
                last_sale['days_since_last_sale'] = (
                    datetime.now() - pd.to_datetime(last_sale['last_sale_date'])
                ).dt.days
                
                # Merge with stock levels (if we can map item_id to sku_id)
                # For now, we'll create separate analysis
                expiry_risk = last_sale.copy()
                
                # Add demand metrics
                demand_summary = demand.groupby('item_id').agg({
                    'demand_quantity': ['sum', 'mean', 'std']
                }).reset_index()
                demand_summary.columns = ['item_id', 'total_demand', 'avg_demand', 'demand_std']
                
                expiry_risk = expiry_risk.merge(demand_summary, on='item_id', how='left')
                
                # Calculate risk score
                # Higher risk if: low/no sales, high stock, long time since last sale
                expiry_risk['expiry_risk_score'] = expiry_risk.apply(
                    lambda row: self._calculate_expiry_risk_score(
                        row.get('days_since_last_sale', 0),
                        row.get('total_demand', 0)
                    ),
                    axis=1
                )
        
        self.analytics_tables['expiry_risk'] = expiry_risk
        return expiry_risk
    
    def _calculate_expiry_risk_score(self, days_since_sale: float, total_demand: float) -> float:
        """Calculate expiry risk score (0-100, higher = more risk)."""
        score = 0
        
        # Days since last sale component (0-50 points)
        if days_since_sale > 90:
            score += 50
        elif days_since_sale > 60:
            score += 35
        elif days_since_sale > 30:
            score += 20
        elif days_since_sale > 14:
            score += 10
        
        # Demand component (0-50 points)
        if total_demand == 0:
            score += 50
        elif total_demand < 10:
            score += 30
        elif total_demand < 50:
            score += 15
        
        return min(100, score)
    
    def calculate_waste_metrics(self) -> pd.DataFrame:
        """
        Calculate waste proxy metrics:
        - Overstock vs sales ratio
        - Slow-moving items
        - Excess inventory indicators
        """
        waste_metrics = []
        
        # Get stock levels
        stock_levels = self.calculate_stock_levels_over_time()
        
        # Get demand
        if 'demand_per_sku' not in self.analytics_tables:
            demand = self.calculate_demand_per_sku(period='monthly')
        else:
            demand = self.analytics_tables['demand_per_sku']
        
        # Calculate monthly demand average
        if 'item_id' in demand.columns:
            monthly_demand = demand.groupby('item_id').agg({
                'demand_quantity': ['sum', 'mean']
            }).reset_index()
            monthly_demand.columns = ['item_id', 'total_demand', 'avg_monthly_demand']
            
            # Merge with stock (if we can map)
            # For now, create metrics based on demand patterns
            
            # Identify slow-moving items
            monthly_demand['is_slow_moving'] = monthly_demand['avg_monthly_demand'] < 5
            monthly_demand['is_fast_moving'] = monthly_demand['avg_monthly_demand'] > 50
            
            # Calculate turnover rate proxy (demand / stock if available)
            # This would require proper SKU mapping
            
            waste_metrics = monthly_demand.copy()
            waste_metrics['waste_risk'] = waste_metrics.apply(
                lambda row: 'high' if row['is_slow_moving'] and row['total_demand'] < 20
                           else 'medium' if row['is_slow_moving']
                           else 'low',
                axis=1
            )
        
        self.analytics_tables['waste_metrics'] = waste_metrics
        return waste_metrics
    
    def generate_analytics_summary(self) -> Dict[str, Any]:
        """Generate summary of all analytics tables."""
        return {
            'tables_generated': list(self.analytics_tables.keys()),
            'summary': {
                'demand_periods': len(self.analytics_tables.get('demand_per_sku', pd.DataFrame())),
                'stock_levels_tracked': len(self.analytics_tables.get('stock_levels', pd.DataFrame())),
                'expiry_risks_identified': len(self.analytics_tables.get('expiry_risk', pd.DataFrame())),
                'waste_metrics_calculated': len(self.analytics_tables.get('waste_metrics', pd.DataFrame()))
            }
        }


if __name__ == "__main__":
    # Example usage
    pass
