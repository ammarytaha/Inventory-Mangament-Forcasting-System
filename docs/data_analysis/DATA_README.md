# Data Documentation

> **Version:** 1.0.0  
> **Generated:** 2026-02-07  
> **Timezone:** UTC

---

## 1. Overview

This package contains cleaned, aggregated, and feature-engineered datasets for inventory forecasting. All data has been processed according to the specifications in `readme2.md`.

## 2. Temporal Coverage

| Attribute | Value |
|-----------|-------|
| **Start Date** | 2021-02-12 |
| **End Date** | 2024-02-16 |
| **Duration** | ~3 years (1,099 days) |
| **Timezone** | UTC |

## 3. Train/Validation/Test Splits

| Split | Start | End | Purpose |
|-------|-------|-----|---------|
| **Train** | 2021-02-12 | 2023-06-30 | Model training |
| **Validation** | 2023-07-01 | 2023-10-31 | Hyperparameter tuning |
| **Test** | 2023-11-01 | 2024-02-16 | Final evaluation |

> ⚠️ **CRITICAL:** Never use test data for training or hyperparameter selection.

## 4. Filters Applied

The following filters were applied during data cleaning:

1. ❌ **Demo transactions** - `demo_mode = 1` removed
2. ❌ **Cancelled orders** - Status = cancelled removed
3. ❌ **Inactive places** - No orders in last 90 days (before 2023-11-18)
4. ❌ **Orphan items** - Order items without valid order_id
5. ❌ **Negative quantities** - quantity < 0 removed
6. ❌ **PII fields** - customer_name, external_id, account_id removed

## 5. Forecasting Unit Definition

| Attribute | Value |
|-----------|-------|
| **Primary Key** | `place_id` × `item_id` × `week_start` |
| **Time Granularity** | Weekly (Monday start) |
| **Target Variable** | `demand` (sum of quantity) |

## 6. Minimum History Requirements

For reliable forecasting, items should meet these thresholds:

| Requirement | Value | Rationale |
|-------------|-------|-----------|
| **Weeks of history** | ≥ 52 | Full year for seasonality |
| **Non-zero demand periods** | ≥ 26 | Sufficient signal |

Use `demand_classification.csv` to filter items meeting these requirements.

## 7. Demand Classification (SBC)

Items are classified using Syntetos-Boylan Classification:

| Class | ADI | CV² | % of Items | Recommended Model |
|-------|-----|-----|------------|-------------------|
| **Smooth** | ≤1.32 | ≤0.49 | ~4% | ETS, Prophet |
| **Erratic** | ≤1.32 | >0.49 | ~2% | ML (LightGBM) |
| **Intermittent** | >1.32 | ≤0.49 | ~47% | Croston |
| **Lumpy** | >1.32 | >0.49 | ~17% | ISBTS |
| **Insufficient** | - | - | ~30% | Rule-based |

- **ADI** = Average Demand Interval (inverse of demand frequency)
- **CV²** = Coefficient of Variation squared

## 8. Feature Descriptions

### Lag Features
| Feature | Description |
|---------|-------------|
| `lag_1w` | Demand 1 week prior |
| `lag_2w` | Demand 2 weeks prior |
| `lag_4w` | Demand 4 weeks prior |
| `lag_52w` | Demand same week last year |

### Rolling Statistics (4-week window, excludes current)
| Feature | Description |
|---------|-------------|
| `roll_mean_4w` | 4-week rolling mean |
| `roll_std_4w` | 4-week rolling std dev |
| `roll_max_4w` | 4-week rolling max |
| `roll_min_4w` | 4-week rolling min |

### Demand Pattern Features
| Feature | Description |
|---------|-------------|
| `zero_demand_ratio` | Historical % of zero-demand weeks |
| `days_since_demand` | Weeks since last non-zero demand |
| `demand_streak` | Consecutive weeks with demand |
| `cv_rolling` | Rolling coefficient of variation |

### Calendar Features
| Feature | Description |
|---------|-------------|
| `day_of_week` | 0=Monday, 6=Sunday |
| `week_of_year` | 1-52 |
| `month` | 1-12 |
| `is_weekend` | Always 0 (weekly data) |
| `is_holiday` | 1 if Danish holiday in week |

### Price Features
| Feature | Description |
|---------|-------------|
| `avg_price_week` | Average price during week |
| `price_change_flag` | 1 if price changed from prior week |

## 9. Known Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| No promotion data | Cannot model promo lifts | Use price as proxy |
| Rapid growth (2024) | Historical patterns may not hold | Weight recent data |
| 47% intermittent | Traditional models won't work | Use Croston/ML |

## 10. Contact

For questions about this data package, refer to:
- `readme2.md` - Full analytical specification
- `README_FOR_DEVELOPER.md` - Quick start guide
