# Developer Quick Start Guide

> **Forecasting Model Developer Handoff Package**  
> **Version:** 1.0.0

---

## 🚀 Quick Start

```python
import pandas as pd

# Load the main forecasting dataset
features = pd.read_parquet('data/features_place_item_week.parquet')

# Split by train/val/test
train = features[features['train_val_test_flag'] == 'train']
val = features[features['train_val_test_flag'] == 'val']
test = features[features['train_val_test_flag'] == 'test']

# Get demand classification
classification = pd.read_csv('data/demand_classification.csv')

print(f"Train: {len(train):,} rows")
print(f"Val: {len(val):,} rows")
print(f"Test: {len(test):,} rows")
```

---

## 📁 Folder Structure

```
new_developer_items/
├── data/
│   ├── orders_clean.parquet         # Cleaned orders
│   ├── order_items_clean.parquet    # Cleaned line items
│   ├── dim_items_clean.parquet      # Product dimension
│   ├── dim_places_clean.parquet     # Location dimension
│   ├── weekly_place_item.parquet    # Weekly aggregates
│   ├── features_place_item_week.parquet  # ⭐ MAIN DATASET
│   └── demand_classification.csv    # SBC classification
├── schema/
│   ├── orders_schema.json
│   ├── order_items_schema.json
│   ├── weekly_place_item_schema.json
│   └── features_schema.json
├── scripts/
│   ├── 01_cleaning.py
│   ├── 02_aggregation.py
│   ├── 03_feature_engineering.py
│   └── validate_checks.py
├── manifest.json
├── DATA_README.md
└── README_FOR_DEVELOPER.md          # ⬅️ You are here
```

---

## 🎯 Primary Forecast Dataset

**File:** `data/features_place_item_week.parquet`

| Column | Type | Description |
|--------|------|-------------|
| `place_id` | int | Location ID |
| `item_id` | int | Product ID |
| `week_start` | datetime | Week start (Monday, UTC) |
| `demand` | int | **TARGET** - Weekly quantity |
| `lag_1w` ... `lag_52w` | float | Lag features |
| `roll_mean_4w` ... | float | Rolling statistics |
| `demand_type` | str | SBC classification |
| `train_val_test_flag` | str | Split assignment |

---

## ⚠️ Critical Rules

### 1. NEVER Use Future Data
All features use **past data only**. Lag features are shifted, rolling stats exclude current period.

### 2. Respect Train/Val/Test Split
```python
# ❌ DON'T: Train on all data
model.fit(features)

# ✅ DO: Use only train split
train = features[features['train_val_test_flag'] == 'train']
model.fit(train)
```

### 3. Handle Demand Types Differently

```python
# Route to appropriate model
smooth = features[features['demand_type'] == 'Smooth']      # → ETS/Prophet
intermittent = features[features['demand_type'] == 'Intermittent']  # → Croston
lumpy = features[features['demand_type'] == 'Lumpy']        # → ML ensemble
```

---

## 📊 Model Selection by Demand Type

| Type | % Items | Recommended | Alternative |
|------|---------|-------------|-------------|
| **Smooth** | 4% | ETS, Prophet | LightGBM |
| **Erratic** | 2% | LightGBM | XGBoost |
| **Intermittent** | 47% | Croston | SBA |
| **Lumpy** | 17% | ISBTS | ML ensemble |
| **Insufficient** | 30% | Safety stock rules | — |

---

## 📈 Evaluation Metrics

**Primary:** WAPE (Weighted Absolute Percentage Error)

```python
def wape(y_true, y_pred):
    return np.abs(y_true - y_pred).sum() / y_true.sum()
```

**Also track:**
- Bias: `(y_pred.sum() - y_true.sum()) / y_true.sum()`
- RMSE: `np.sqrt(((y_true - y_pred) ** 2).mean())`

---

## 🔧 Regenerating Data

To regenerate the dataset from raw CSVs:

```bash
cd new_developer_items/scripts
python 01_cleaning.py           # ~2 min
python 02_aggregation.py        # ~1 min
python 03_feature_engineering.py # ~3 min
python validate_checks.py       # Validates output
```

---

## 📚 Additional Documentation

| Document | Contents |
|----------|----------|
| `DATA_README.md` | Full data specification |
| `manifest.json` | Row counts, versions, metadata |
| `schema/*.json` | JSON schemas for validation |
| `../readme2.md` | Complete analytical report |

---

## ✅ Pre-Flight Checklist

Before model training:

- [ ] Loaded `features_place_item_week.parquet`
- [ ] Filtered to `train_val_test_flag == 'train'`
- [ ] Filtered by `demand_type` for model routing
- [ ] Verified no future leakage (run `validate_checks.py`)
- [ ] Selected appropriate model for demand type
- [ ] Set up WAPE as primary metric

---

## 🆘 Common Issues

### Memory Errors
The features file is ~150MB. If memory is tight:
```python
# Filter to specific demand type
df = pd.read_parquet('data/features_place_item_week.parquet',
                     filters=[('demand_type', '==', 'Smooth')])
```

### Missing Lag Values
First 52 weeks have NaN for `lag_52w`. These are filled with 0:
```python
# Already filled, but verify
assert features['lag_52w'].isna().sum() == 0
```

---

**Good luck with your forecasting model! 🎯**
