"""
02_aggregation.py - Weekly Aggregation Pipeline

Creates weekly demand aggregation at place-item level.
Primary forecasting dataset as specified in readme2.md.

Usage:
    python 02_aggregation.py

Requires:
    - data/orders_clean.parquet
    - data/order_items_clean.parquet

Outputs:
    - data/weekly_place_item.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = Path(__file__).parent.parent / "data"
DATA_VERSION = "1.0.0"
GENERATED_AT = datetime.utcnow().isoformat() + "Z"

print("="*70)
print("INVENTORY FORECASTING - WEEKLY AGGREGATION PIPELINE")
print(f"Generated at: {GENERATED_AT}")
print("="*70)


def load_clean_data():
    """Load cleaned parquet files"""
    print("\n[1/4] Loading cleaned data...")
    
    orders = pd.read_parquet(DATA_PATH / "orders_clean.parquet")
    order_items = pd.read_parquet(DATA_PATH / "order_items_clean.parquet")
    places = pd.read_parquet(DATA_PATH / "dim_places_clean.parquet")
    
    print(f"    orders: {len(orders):,}")
    print(f"    order_items: {len(order_items):,}")
    print(f"    places: {len(places):,}")
    
    return orders, order_items, places


def merge_order_data(orders, order_items):
    """Merge order items with orders to get place_id"""
    print("\n[2/4] Merging order data...")
    
    # Merge to get place_id on order items
    merged = order_items.merge(
        orders[['id', 'place_id', 'created_dt']],
        left_on='order_id',
        right_on='id',
        how='inner',
        suffixes=('', '_order')
    )
    
    # Use order's created_dt as the transaction time
    merged['transaction_dt'] = merged['created_dt_order']
    
    print(f"    Merged records: {len(merged):,}")
    
    return merged


def create_weekly_aggregation(merged, places):
    """Create weekly aggregation at place-item level"""
    print("\n[3/4] Creating weekly aggregation...")
    
    # Extract week start (Monday)
    merged['week_start'] = merged['transaction_dt'].dt.to_period('W-MON').dt.start_time
    
    # Aggregate by place_id, item_id, week_start
    weekly = merged.groupby(['place_id', 'item_id', 'week_start']).agg(
        demand=('quantity', 'sum'),
        transactions=('id', 'count'),
        avg_price=('price', 'mean'),
        min_transaction=('transaction_dt', 'min'),
        max_transaction=('transaction_dt', 'max')
    ).reset_index()
    
    # Calculate days with transactions in the week
    weekly['days_active'] = (
        (merged.groupby(['place_id', 'item_id', 'week_start'])['transaction_dt']
         .apply(lambda x: x.dt.date.nunique())
         .reset_index(name='days_active')['days_active'])
    )
    
    # Merge with places to get active flag
    weekly = weekly.merge(
        places[['id', 'active_flag']],
        left_on='place_id',
        right_on='id',
        how='left'
    )
    weekly['is_active'] = weekly['active_flag'].fillna(False)
    weekly = weekly.drop(columns=['id', 'active_flag'], errors='ignore')
    
    # Add data version
    weekly['data_version'] = DATA_VERSION
    
    # Ensure week_start is timezone-aware (UTC)
    weekly['week_start'] = pd.to_datetime(weekly['week_start']).dt.tz_localize('UTC')
    
    print(f"    Total weekly records: {len(weekly):,}")
    print(f"    Unique place-item combinations: {weekly.groupby(['place_id', 'item_id']).ngroups:,}")
    print(f"    Date range: {weekly['week_start'].min()} to {weekly['week_start'].max()}")
    
    return weekly


def save_aggregation(weekly):
    """Save weekly aggregation to parquet"""
    print("\n[4/4] Saving weekly aggregation...")
    
    # Select final columns
    final_cols = [
        'place_id', 'item_id', 'week_start',
        'demand', 'transactions', 'avg_price',
        'is_active', 'days_active', 'data_version'
    ]
    weekly_final = weekly[final_cols].copy()
    
    # Sort for efficient querying
    weekly_final = weekly_final.sort_values(['place_id', 'item_id', 'week_start'])
    
    weekly_final.to_parquet(DATA_PATH / "weekly_place_item.parquet", index=False)
    print(f"    Saved: weekly_place_item.parquet ({len(weekly_final):,} rows)")
    
    return weekly_final


def main():
    # Load data
    orders, order_items, places = load_clean_data()
    
    # Merge
    merged = merge_order_data(orders, order_items)
    
    # Aggregate
    weekly = create_weekly_aggregation(merged, places)
    
    # Save
    weekly_final = save_aggregation(weekly)
    
    print("\n" + "="*70)
    print("WEEKLY AGGREGATION COMPLETE")
    print("="*70)
    
    # Summary stats
    print("\nSummary Statistics:")
    print(f"    Total weeks: {weekly_final['week_start'].nunique()}")
    print(f"    Mean demand per week: {weekly_final['demand'].mean():.2f}")
    print(f"    Median demand per week: {weekly_final['demand'].median():.2f}")
    print(f"    Zero demand weeks: {(weekly_final['demand'] == 0).sum():,}")
    
    return len(weekly_final)


if __name__ == "__main__":
    row_count = main()
