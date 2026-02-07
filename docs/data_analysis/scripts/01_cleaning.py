"""
01_cleaning.py - Data Cleaning Pipeline

Production data cleaning for inventory forecasting system.
Follows specifications from readme2.md.

Usage:
    python 01_cleaning.py

Outputs:
    - data/orders_clean.parquet
    - data/order_items_clean.parquet
    - data/dim_items_clean.parquet
    - data/dim_places_clean.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuration
RAW_DATA_PATH = Path(__file__).parent.parent.parent  # Points to Inventory Management folder
OUTPUT_PATH = Path(__file__).parent.parent / "data"
DATA_VERSION = "1.0.0"
GENERATED_AT = datetime.utcnow().isoformat() + "Z"

# Cutoff for inactive places (90 days from last order in dataset)
LAST_ORDER_DATE = pd.Timestamp("2024-02-16")
INACTIVE_CUTOFF = LAST_ORDER_DATE - timedelta(days=90)

print("="*70)
print("INVENTORY FORECASTING - DATA CLEANING PIPELINE")
print(f"Generated at: {GENERATED_AT}")
print(f"Data version: {DATA_VERSION}")
print("="*70)


def load_raw_data():
    """Load raw CSV files"""
    print("\n[1/5] Loading raw data...")
    
    orders = pd.read_csv(RAW_DATA_PATH / "fct_orders.csv", low_memory=False)
    order_items = pd.read_csv(RAW_DATA_PATH / "fct_order_items.csv", low_memory=False)
    items = pd.read_csv(RAW_DATA_PATH / "dim_items.csv", low_memory=False)
    places = pd.read_csv(RAW_DATA_PATH / "dim_places.csv", low_memory=False)
    
    print(f"    orders: {len(orders):,} rows")
    print(f"    order_items: {len(order_items):,} rows")
    print(f"    items: {len(items):,} rows")
    print(f"    places: {len(places):,} rows")
    
    return orders, order_items, items, places


def clean_orders(orders):
    """Clean orders table"""
    print("\n[2/5] Cleaning orders...")
    
    initial_count = len(orders)
    
    # Convert Unix timestamp to UTC datetime
    orders['created_dt'] = pd.to_datetime(orders['created'], unit='s', utc=True)
    
    # Remove demo transactions
    if 'demo_mode' in orders.columns:
        orders = orders[orders['demo_mode'] != 1]
        print(f"    After removing demo_mode=1: {len(orders):,}")
    
    # Remove cancelled orders
    if 'status' in orders.columns:
        orders = orders[~orders['status'].isin(['cancelled', 'Cancelled', 'CANCELLED'])]
        print(f"    After removing cancelled: {len(orders):,}")
    
    # Remove duplicates
    orders = orders.drop_duplicates(subset=['id'])
    print(f"    After deduplication: {len(orders):,}")
    
    # Remove PII fields
    pii_cols = ['customer_name', 'external_id', 'account_id', 'customer_mobile_phone']
    cols_to_drop = [c for c in pii_cols if c in orders.columns]
    orders = orders.drop(columns=cols_to_drop, errors='ignore')
    
    # Select required columns
    required_cols = ['id', 'place_id', 'created_dt', 'total_amount', 'status', 'type', 'demo_mode']
    available_cols = [c for c in required_cols if c in orders.columns]
    orders_clean = orders[available_cols].copy()
    
    # Ensure proper types
    orders_clean['id'] = orders_clean['id'].astype('int64')
    orders_clean['place_id'] = orders_clean['place_id'].astype('float64')  # Has NaN
    
    print(f"    Removed {initial_count - len(orders_clean):,} rows ({(initial_count - len(orders_clean))/initial_count*100:.1f}%)")
    print(f"    Final orders: {len(orders_clean):,}")
    
    return orders_clean


def clean_order_items(order_items, valid_order_ids):
    """Clean order items table"""
    print("\n[3/5] Cleaning order items...")
    
    initial_count = len(order_items)
    
    # Convert Unix timestamp to UTC datetime
    order_items['created_dt'] = pd.to_datetime(order_items['created'], unit='s', utc=True)
    
    # Remove orphan order items (not in valid orders)
    order_items = order_items[order_items['order_id'].isin(valid_order_ids)]
    print(f"    After removing orphans: {len(order_items):,}")
    
    # Filter quantity >= 0 (remove returns/negative adjustments)
    order_items = order_items[order_items['quantity'] >= 0]
    print(f"    After filtering quantity>=0: {len(order_items):,}")
    
    # Remove duplicates
    order_items = order_items.drop_duplicates(subset=['id'])
    print(f"    After deduplication: {len(order_items):,}")
    
    # Select required columns
    required_cols = ['id', 'order_id', 'item_id', 'quantity', 'price', 'created_dt', 
                     'status', 'discount_amount', 'add_on_ids']
    available_cols = [c for c in required_cols if c in order_items.columns]
    items_clean = order_items[available_cols].copy()
    
    # Fill missing discount_amount with 0
    if 'discount_amount' in items_clean.columns:
        items_clean['discount_amount'] = items_clean['discount_amount'].fillna(0)
    
    print(f"    Removed {initial_count - len(items_clean):,} rows ({(initial_count - len(items_clean))/initial_count*100:.1f}%)")
    print(f"    Final order_items: {len(items_clean):,}")
    
    return items_clean


def clean_items(items):
    """Clean items dimension table"""
    print("\n[4/5] Cleaning items...")
    
    initial_count = len(items)
    
    # Remove duplicates
    items = items.drop_duplicates(subset=['id'])
    
    # Select required columns
    required_cols = ['id', 'title', 'section_id', 'status', 'price']
    available_cols = [c for c in required_cols if c in items.columns]
    items_clean = items[available_cols].copy()
    
    print(f"    Final items: {len(items_clean):,}")
    
    return items_clean


def clean_places(places, orders_clean):
    """Clean places dimension table with activity analysis"""
    print("\n[5/5] Cleaning places...")
    
    initial_count = len(places)
    
    # Remove duplicates
    places = places.drop_duplicates(subset=['id'])
    
    # Calculate first/last order per place
    place_activity = orders_clean.groupby('place_id').agg(
        first_order=('created_dt', 'min'),
        last_order=('created_dt', 'max'),
        order_count=('id', 'count')
    ).reset_index()
    
    # Select required columns from places
    required_cols = ['id', 'title', 'country', 'inventory_management']
    available_cols = [c for c in required_cols if c in places.columns]
    places_clean = places[available_cols].copy()
    
    # Merge activity data
    places_clean = places_clean.merge(place_activity, left_on='id', right_on='place_id', how='left')
    places_clean = places_clean.drop(columns=['place_id'], errors='ignore')
    
    # Add active flag (had orders in last 90 days)
    places_clean['active_flag'] = (places_clean['last_order'] >= INACTIVE_CUTOFF.tz_localize('UTC')).fillna(False)
    
    active_places = places_clean[places_clean['active_flag']]['id'].tolist()
    
    print(f"    Total places: {len(places_clean):,}")
    print(f"    Active places (orders in last 90 days): {len(active_places):,}")
    
    return places_clean, active_places


def save_data(orders_clean, order_items_clean, items_clean, places_clean):
    """Save cleaned data to parquet"""
    print("\n[SAVE] Writing parquet files...")
    
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    orders_clean.to_parquet(OUTPUT_PATH / "orders_clean.parquet", index=False)
    print(f"    Saved: orders_clean.parquet ({len(orders_clean):,} rows)")
    
    order_items_clean.to_parquet(OUTPUT_PATH / "order_items_clean.parquet", index=False)
    print(f"    Saved: order_items_clean.parquet ({len(order_items_clean):,} rows)")
    
    items_clean.to_parquet(OUTPUT_PATH / "dim_items_clean.parquet", index=False)
    print(f"    Saved: dim_items_clean.parquet ({len(items_clean):,} rows)")
    
    places_clean.to_parquet(OUTPUT_PATH / "dim_places_clean.parquet", index=False)
    print(f"    Saved: dim_places_clean.parquet ({len(places_clean):,} rows)")


def main():
    # Load raw data
    orders, order_items, items, places = load_raw_data()
    
    # Clean orders
    orders_clean = clean_orders(orders)
    valid_order_ids = set(orders_clean['id'].tolist())
    
    # Clean order items
    order_items_clean = clean_order_items(order_items, valid_order_ids)
    
    # Clean items
    items_clean = clean_items(items)
    
    # Clean places
    places_clean, active_places = clean_places(places, orders_clean)
    
    # Filter orders and items to active places only
    orders_clean = orders_clean[orders_clean['place_id'].isin(active_places)]
    valid_order_ids = set(orders_clean['id'].tolist())
    order_items_clean = order_items_clean[order_items_clean['order_id'].isin(valid_order_ids)]
    
    print(f"\n[FILTER] After filtering to active places only:")
    print(f"    orders_clean: {len(orders_clean):,}")
    print(f"    order_items_clean: {len(order_items_clean):,}")
    
    # Save cleaned data
    save_data(orders_clean, order_items_clean, items_clean, places_clean)
    
    print("\n" + "="*70)
    print("DATA CLEANING COMPLETE")
    print("="*70)
    
    return {
        'orders_rows': len(orders_clean),
        'order_items_rows': len(order_items_clean),
        'items_rows': len(items_clean),
        'places_rows': len(places_clean),
        'active_places': len(active_places)
    }


if __name__ == "__main__":
    stats = main()
