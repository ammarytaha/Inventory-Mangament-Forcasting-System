"""
03_feature_engineering.py - Feature Engineering Pipeline

Creates leakage-safe features for forecasting models.
Implements demand classification using SBC methodology.

Usage:
    python 03_feature_engineering.py

Requires:
    - data/weekly_place_item.parquet

Outputs:
    - data/features_place_item_week.parquet
    - data/demand_classification.csv
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

# Temporal split dates (as per readme2.md)
TRAIN_END = pd.Timestamp("2023-06-30", tz='UTC')
VAL_END = pd.Timestamp("2023-10-31", tz='UTC')
TEST_END = pd.Timestamp("2024-02-29", tz='UTC')

# SBC Classification thresholds
ADI_THRESHOLD = 1.32
CV2_THRESHOLD = 0.49

# Danish holidays (2021-2024)
DANISH_HOLIDAYS = [
    # 2021
    "2021-01-01", "2021-04-01", "2021-04-02", "2021-04-04", "2021-04-05",
    "2021-05-13", "2021-05-23", "2021-05-24", "2021-06-05",
    "2021-12-24", "2021-12-25", "2021-12-26", "2021-12-31",
    # 2022
    "2022-01-01", "2022-04-14", "2022-04-15", "2022-04-17", "2022-04-18",
    "2022-05-13", "2022-05-26", "2022-06-05", "2022-06-06",
    "2022-12-24", "2022-12-25", "2022-12-26", "2022-12-31",
    # 2023
    "2023-01-01", "2023-04-06", "2023-04-07", "2023-04-09", "2023-04-10",
    "2023-05-05", "2023-05-18", "2023-05-28", "2023-05-29",
    "2023-12-24", "2023-12-25", "2023-12-26", "2023-12-31",
    # 2024
    "2024-01-01", "2024-03-28", "2024-03-29", "2024-03-31", "2024-04-01",
    "2024-05-09", "2024-05-19", "2024-05-20",
    "2024-12-24", "2024-12-25", "2024-12-26", "2024-12-31",
]
DANISH_HOLIDAYS = pd.to_datetime(DANISH_HOLIDAYS).tz_localize('UTC')

print("="*70)
print("INVENTORY FORECASTING - FEATURE ENGINEERING PIPELINE")
print(f"Generated at: {GENERATED_AT}")
print("="*70)


def load_weekly_data():
    """Load weekly aggregated data"""
    print("\n[1/6] Loading weekly data...")
    
    weekly = pd.read_parquet(DATA_PATH / "weekly_place_item.parquet")
    print(f"    Loaded: {len(weekly):,} rows")
    
    return weekly


def expand_to_full_calendar(weekly):
    """Expand data to include all weeks (with zero-fill for missing)"""
    print("\n[2/6] Expanding to full calendar...")
    
    # Get all unique place-item combinations
    place_items = weekly[['place_id', 'item_id']].drop_duplicates()
    
    # Get all weeks in range
    min_week = weekly['week_start'].min()
    max_week = weekly['week_start'].max()
    all_weeks = pd.date_range(start=min_week, end=max_week, freq='W-MON', tz='UTC')
    
    # Create full cartesian product
    place_items['key'] = 1
    weeks_df = pd.DataFrame({'week_start': all_weeks, 'key': 1})
    full_grid = place_items.merge(weeks_df, on='key').drop(columns=['key'])
    
    print(f"    Full grid: {len(full_grid):,} rows")
    
    # Merge with actual data
    expanded = full_grid.merge(
        weekly,
        on=['place_id', 'item_id', 'week_start'],
        how='left'
    )
    
    # Fill missing demand with 0
    expanded['demand'] = expanded['demand'].fillna(0)
    expanded['transactions'] = expanded['transactions'].fillna(0)
    expanded['avg_price'] = expanded.groupby(['place_id', 'item_id'])['avg_price'].ffill().bfill()
    expanded['is_active'] = expanded['is_active'].fillna(True)
    expanded['days_active'] = expanded['days_active'].fillna(0)
    expanded['data_version'] = DATA_VERSION
    
    print(f"    Expanded: {len(expanded):,} rows")
    
    return expanded


def add_lag_features(df):
    """Add lag-based features (leakage-safe)"""
    print("\n[3/6] Adding lag features...")
    
    df = df.sort_values(['place_id', 'item_id', 'week_start'])
    
    # Group for lag calculations
    group_cols = ['place_id', 'item_id']
    
    # Lag features
    df['lag_1w'] = df.groupby(group_cols)['demand'].shift(1)
    df['lag_2w'] = df.groupby(group_cols)['demand'].shift(2)
    df['lag_4w'] = df.groupby(group_cols)['demand'].shift(4)
    df['lag_52w'] = df.groupby(group_cols)['demand'].shift(52)
    
    # Rolling statistics (using past data only - shift + rolling)
    df['roll_mean_4w'] = df.groupby(group_cols)['demand'].transform(
        lambda x: x.shift(1).rolling(4, min_periods=1).mean()
    )
    df['roll_std_4w'] = df.groupby(group_cols)['demand'].transform(
        lambda x: x.shift(1).rolling(4, min_periods=1).std()
    )
    df['roll_max_4w'] = df.groupby(group_cols)['demand'].transform(
        lambda x: x.shift(1).rolling(4, min_periods=1).max()
    )
    df['roll_min_4w'] = df.groupby(group_cols)['demand'].transform(
        lambda x: x.shift(1).rolling(4, min_periods=1).min()
    )
    
    # Fill NaN with 0 for initial periods
    lag_cols = ['lag_1w', 'lag_2w', 'lag_4w', 'lag_52w', 
                'roll_mean_4w', 'roll_std_4w', 'roll_max_4w', 'roll_min_4w']
    df[lag_cols] = df[lag_cols].fillna(0)
    
    print(f"    Added {len(lag_cols)} lag/rolling features")
    
    return df


def add_demand_pattern_features(df):
    """Add demand pattern features"""
    print("\n[4/6] Adding demand pattern features...")
    
    df = df.sort_values(['place_id', 'item_id', 'week_start'])
    group_cols = ['place_id', 'item_id']
    
    # Zero demand ratio (historical only - using expanding window)
    df['non_zero_cum'] = df.groupby(group_cols)['demand'].transform(
        lambda x: (x.shift(1) > 0).cumsum()
    )
    df['week_num'] = df.groupby(group_cols).cumcount()
    df['zero_demand_ratio'] = 1 - (df['non_zero_cum'] / df['week_num'].replace(0, 1))
    df['zero_demand_ratio'] = df['zero_demand_ratio'].fillna(1)
    
    # Days since demand
    df['had_demand'] = (df['demand'] > 0).astype(int)
    df['demand_gap'] = df.groupby(group_cols)['had_demand'].transform(
        lambda x: x.groupby((x != x.shift()).cumsum()).cumcount() + 1
    )
    df['days_since_demand'] = df.groupby(group_cols)['demand_gap'].transform(
        lambda x: x.where(x.shift(1) == 0, 0).cummax()
    )
    
    # Demand streak (consecutive weeks with demand)
    df['demand_streak'] = df.groupby(group_cols)['had_demand'].transform(
        lambda x: x.groupby((x != x.shift()).cumsum()).cumsum()
    )
    
    # Rolling CV
    df['cv_rolling'] = df['roll_std_4w'] / df['roll_mean_4w'].replace(0, 1)
    df['cv_rolling'] = df['cv_rolling'].fillna(0)
    
    # Price features
    df['avg_price_week'] = df['avg_price']
    df['price_lag'] = df.groupby(group_cols)['avg_price'].shift(1)
    df['price_change_flag'] = (df['avg_price'] != df['price_lag']).astype(int)
    df['price_change_flag'] = df['price_change_flag'].fillna(0)
    
    # Clean up temporary columns
    df = df.drop(columns=['non_zero_cum', 'week_num', 'had_demand', 'demand_gap', 'price_lag'], errors='ignore')
    
    print("    Added demand pattern features")
    
    return df


def add_calendar_features(df):
    """Add calendar features"""
    print("\n[5/6] Adding calendar features...")
    
    # Basic calendar
    df['day_of_week'] = df['week_start'].dt.dayofweek  # Monday = 0
    df['week_of_year'] = df['week_start'].dt.isocalendar().week.astype(int)
    df['month'] = df['week_start'].dt.month
    df['is_weekend'] = 0  # Week aggregation, so N/A but included for consistency
    
    # Holiday flag (is there a holiday in this week?)
    def week_has_holiday(week_start):
        week_end = week_start + pd.Timedelta(days=6)
        return any((DANISH_HOLIDAYS >= week_start) & (DANISH_HOLIDAYS <= week_end))
    
    df['is_holiday'] = df['week_start'].apply(week_has_holiday).astype(int)
    
    # Train/val/test split
    def get_split(dt):
        if dt <= TRAIN_END:
            return 'train'
        elif dt <= VAL_END:
            return 'val'
        else:
            return 'test'
    
    df['train_val_test_flag'] = df['week_start'].apply(get_split)
    
    print(f"    Train rows: {(df['train_val_test_flag'] == 'train').sum():,}")
    print(f"    Val rows: {(df['train_val_test_flag'] == 'val').sum():,}")
    print(f"    Test rows: {(df['train_val_test_flag'] == 'test').sum():,}")
    
    return df


def classify_demand(df):
    """Classify demand patterns using SBC methodology"""
    print("\n[6/6] Classifying demand patterns...")
    
    # Calculate per place-item
    classification = df.groupby(['place_id', 'item_id']).agg(
        total_weeks=('week_start', 'count'),
        non_zero_weeks=('demand', lambda x: (x > 0).sum()),
        total_demand=('demand', 'sum'),
        mean_demand=('demand', lambda x: x[x > 0].mean() if (x > 0).any() else 0),
        std_demand=('demand', lambda x: x[x > 0].std() if (x > 0).sum() > 1 else 0)
    ).reset_index()
    
    # Calculate ADI (average demand interval)
    classification['demand_frequency'] = classification['non_zero_weeks'] / classification['total_weeks']
    classification['adi'] = 1 / classification['demand_frequency'].replace(0, np.inf)
    
    # Calculate CV² (coefficient of variation squared)
    classification['cv'] = classification['std_demand'] / classification['mean_demand'].replace(0, 1)
    classification['cv2'] = classification['cv'] ** 2
    
    # Classify using SBC methodology
    def sbc_classify(row):
        if row['non_zero_weeks'] < 2:
            return 'Insufficient Data'
        
        if row['adi'] <= ADI_THRESHOLD:
            if row['cv2'] <= CV2_THRESHOLD:
                return 'Smooth'
            else:
                return 'Erratic'
        else:
            if row['cv2'] <= CV2_THRESHOLD:
                return 'Intermittent'
            else:
                return 'Lumpy'
    
    classification['demand_type'] = classification.apply(sbc_classify, axis=1)
    
    # Weeks active (first to last demand)
    first_last = df[df['demand'] > 0].groupby(['place_id', 'item_id']).agg(
        first_demand=('week_start', 'min'),
        last_demand=('week_start', 'max')
    ).reset_index()
    first_last['weeks_active'] = ((first_last['last_demand'] - first_last['first_demand']).dt.days / 7 + 1).astype(int)
    
    classification = classification.merge(
        first_last[['place_id', 'item_id', 'weeks_active']],
        on=['place_id', 'item_id'],
        how='left'
    )
    classification['weeks_active'] = classification['weeks_active'].fillna(0).astype(int)
    
    # Select output columns
    output_cols = ['place_id', 'item_id', 'demand_type', 'adi', 'cv2', 'non_zero_weeks', 'weeks_active']
    classification_output = classification[output_cols].copy()
    
    # Print summary
    print("\n    Demand Classification Summary:")
    for dtype, count in classification_output['demand_type'].value_counts().items():
        pct = count / len(classification_output) * 100
        print(f"        {dtype:20s}: {count:>6,} ({pct:>5.1f}%)")
    
    # Save
    classification_output.to_csv(DATA_PATH / "demand_classification.csv", index=False)
    print(f"\n    Saved: demand_classification.csv ({len(classification_output):,} rows)")
    
    return classification_output


def add_demand_type_to_features(df, classification):
    """Merge demand type into features"""
    df = df.merge(
        classification[['place_id', 'item_id', 'demand_type']],
        on=['place_id', 'item_id'],
        how='left'
    )
    df['demand_type'] = df['demand_type'].fillna('Insufficient Data')
    return df


def save_features(df):
    """Save feature dataset"""
    print("\n[SAVE] Saving features...")
    
    # Select final columns
    final_cols = [
        # Keys
        'place_id', 'item_id', 'week_start',
        # Target
        'demand',
        # Lag features
        'lag_1w', 'lag_2w', 'lag_4w', 'lag_52w',
        # Rolling features
        'roll_mean_4w', 'roll_std_4w', 'roll_max_4w', 'roll_min_4w',
        # Demand pattern features
        'zero_demand_ratio', 'days_since_demand', 'demand_streak', 'cv_rolling',
        # Price features
        'avg_price_week', 'price_change_flag',
        # Calendar features
        'day_of_week', 'week_of_year', 'month', 'is_weekend', 'is_holiday',
        # Classification
        'demand_type',
        # Split
        'train_val_test_flag'
    ]
    
    df_final = df[final_cols].copy()
    df_final = df_final.sort_values(['place_id', 'item_id', 'week_start'])
    
    df_final.to_parquet(DATA_PATH / "features_place_item_week.parquet", index=False)
    print(f"    Saved: features_place_item_week.parquet ({len(df_final):,} rows)")
    
    return df_final


def main():
    # Load data
    weekly = load_weekly_data()
    
    # Expand to full calendar
    expanded = expand_to_full_calendar(weekly)
    
    # Add features
    df = add_lag_features(expanded)
    df = add_demand_pattern_features(df)
    df = add_calendar_features(df)
    
    # Classify demand
    classification = classify_demand(df)
    
    # Add demand type to features
    df = add_demand_type_to_features(df, classification)
    
    # Save
    features = save_features(df)
    
    print("\n" + "="*70)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*70)
    
    return len(features)


if __name__ == "__main__":
    row_count = main()
