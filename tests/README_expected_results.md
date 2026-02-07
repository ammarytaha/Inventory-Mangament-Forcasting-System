# Expected Model Results and Validation Guide

## Dataset Description

This dataset contains mock inventory forecasting data built upon original analyst data, preserving real item names and place names. The data includes:

- **Historical demand data** (from original analyst dataset)
- **Synthetic future demand** (52 weeks forward)
- **Simulated forecasts** using pattern-aware methods
- **Inventory simulation** with expiry dates and shelf life
- **Risk classification** based on expiry and overstock
- **Business recommendations** with discount suggestions

## Data Files

### 1. mock_data_for_model_testing.parquet

Contains the input features for forecasting models:

- `place_id`, `item_id`, `place_title`, `item_title`
- `week_start`: Week start date (Monday)
- `demand`: Historical and synthetic demand
- `avg_price_week`: Average price per week
- `transactions`: Number of transactions
- `is_active`, `days_active`: Activity indicators
- `demand_type`: SBC classification (Smooth, Erratic, Intermittent, Lumpy)
- `is_future`: Boolean flag (False = historical, True = synthetic)
- `data_source`: 'historical' or 'synthetic'
- `train_val_test_flag`: Data split assignment

### 2. expected_model_results_and_actions.parquet

Contains expected model outputs and business actions:

- `place_id`, `item_id`, `place_title`, `item_title`
- `week_start`: Week start date
- `demand`: Actual demand (for validation)
- `demand_type`: SBC classification
- `forecast_demand`: Expected forecast value
- `lower_confidence_bound`, `upper_confidence_bound`: 95% confidence interval
- `inventory_level`: Current inventory level
- `shelf_life_days`: Product shelf life
- `production_date`, `expiry_date`: Production and expiry dates
- `days_to_expiry`: Days until expiry from week_start
- `safety_stock`, `reorder_point`: Inventory management parameters
- `risk_flag`: Risk classification (Critical, High Risk, Low Risk, Safe)
- `recommended_action`: Business action recommendation
- `suggested_discount_percent`: Suggested discount percentage
- `urgency_score`: Urgency score (1-10 scale)
- `is_future`: Boolean flag
- `train_val_test_flag`: Data split assignment

## Forecasting Assumptions

### Demand Type Classification (SBC - Syntetos-Boylan Classification)

The dataset uses SBC classification based on:
- **ADI (Average Demand Interval)**: Threshold = 1.32 weeks
- **CV² (Coefficient of Variation squared)**: Threshold = 0.49

**Categories:**
- **Smooth**: Low ADI, Low CV² → Regular, predictable demand
- **Erratic**: Low ADI, High CV² → Regular intervals, variable sizes
- **Intermittent**: High ADI, Low CV² → Irregular intervals, consistent sizes
- **Lumpy**: High ADI, High CV² → Irregular intervals, variable sizes

### Expected Forecast Methods by Demand Type

#### 1. Smooth Demand (Exponential Smoothing / ETS)

**Expected Behavior:**
- Forecast should follow trend and seasonal patterns
- Low forecast error (MAPE < 15%)
- Confidence intervals should be tight
- Forecast should adapt to recent trends

**Validation Criteria:**
- WAPE < 20%
- Forecast should be within ±2 standard deviations of historical mean
- Trend continuation should be smooth

#### 2. Erratic Demand (Exponential Smoothing with Higher Uncertainty)

**Expected Behavior:**
- Forecast should capture mean level
- Higher uncertainty than smooth demand
- Confidence intervals should be wider
- May benefit from ensemble methods

**Validation Criteria:**
- WAPE < 30%
- Confidence intervals should cover 90-95% of actuals
- Forecast should not overreact to outliers

#### 3. Intermittent Demand (Croston Method)

**Expected Behavior:**
- Forecast = Expected demand size / Expected interval
- Forecast should account for zero-demand periods
- Lower forecast values than smooth demand
- Higher uncertainty

**Validation Criteria:**
- WAPE < 40% (higher tolerance for intermittent)
- Forecast should not predict demand in every period
- Should handle zero-demand periods correctly
- Croston forecast ≈ mean_demand_size / mean_interval

**Example:**
- If mean demand size = 10 units
- Mean interval = 3 weeks
- Expected Croston forecast ≈ 10/3 ≈ 3.33 units/week

#### 4. Lumpy Demand (Croston-SBA or ML Ensemble)

**Expected Behavior:**
- Most challenging demand pattern
- May require machine learning approaches
- High uncertainty
- Forecast should capture spike patterns

**Validation Criteria:**
- WAPE < 50% (highest tolerance)
- Should identify spike periods
- Confidence intervals should be very wide
- May require specialized models (ISBTS, ML ensemble)

## Risk Classification Logic

Risk classification is based on two factors:

1. **Days to Expiry**: Time until product expires
2. **Overstock Ratio**: inventory_level / forecast_demand

### Risk Levels

- **Critical**: 
  - Days to expiry < 7 days AND
  - Overstock ratio > 2.0
  
- **High Risk**:
  - Days to expiry between 7-14 days, OR
  - Overstock ratio > 1.5 AND days to expiry < 21 days
  
- **Low Risk**:
  - Overstock ratio > 1.2, OR
  - Days to expiry between 14-30 days
  
- **Safe**:
  - All other cases (aligned inventory, safe expiry window)

### Risk Thresholds

```python
CRITICAL_DAYS = 7
HIGH_RISK_DAYS = 14
OVERSTOCK_CRITICAL_MULTIPLIER = 2.0
OVERSTOCK_HIGH_MULTIPLIER = 1.5
```

## Action Recommendation Logic

### Discount Recommendations

- **Critical Risk**: 30-50% discount (based on days to expiry)
- **High Risk**: 15-30% discount
- **Low Risk**: 5-15% discount (monitoring recommended)
- **Safe**: No discount (0%)

### Urgency Score (1-10 scale)

- **Critical**: 9-10 (immediate action required)
- **High Risk**: 6-7 (action needed soon)
- **Low Risk**: 3-4 (monitor)
- **Safe**: 1 (no action)

### Recommended Actions

- **Critical**: Deep discount, priority promotion, stock transfer consideration
- **High Risk**: Moderate discount, marketing highlight
- **Low Risk**: Monitor inventory, mild promotion if levels rise
- **Safe**: No action required

## Validation Acceptance Criteria

### Data Quality Checks

1. ✅ **No negative demand**: All demand values ≥ 0
2. ✅ **No negative inventory**: All inventory_level ≥ 0
3. ✅ **Valid expiry dates**: expiry_date > production_date
4. ✅ **All rows have risk classification**: risk_flag is not null
5. ✅ **All forecasts are non-negative**: forecast_demand ≥ 0

### Forecast Accuracy Targets

| Demand Type | WAPE Target | MAPE Target | CI Coverage Target |
|------------|-------------|-------------|-------------------|
| Smooth | < 20% | < 15% | 90-95% |
| Erratic | < 30% | < 25% | 85-95% |
| Intermittent | < 40% | < 35% | 80-95% |
| Lumpy | < 50% | < 45% | 75-95% |

### Model Output Validation

When comparing your model outputs to expected results:

1. **Forecast Values**: Should be within ±20% of expected forecast_demand for smooth/erratic, ±40% for intermittent/lumpy
2. **Confidence Intervals**: Should cover 90-95% of actual demand for smooth demand, 75-95% for lumpy demand
3. **Risk Classification**: Should match expected risk_flag for at least 80% of items
4. **Trend Direction**: Forecast trend should match expected trend direction

## Instructions for Developer Testing

### Step 1: Load the Data

```python
import pandas as pd

# Load input data
input_data = pd.read_parquet('mock_data_for_model_testing.parquet')

# Load expected results
expected_results = pd.read_parquet('expected_model_results_and_actions.parquet')

# Split by train/val/test
train = input_data[input_data['train_val_test_flag'] == 'train']
val = input_data[input_data['train_val_test_flag'] == 'val']
test = input_data[input_data['train_val_test_flag'] == 'test']
```

### Step 2: Train Your Models

Train separate models for each demand type:

```python
# Smooth demand
smooth_data = train[train['demand_type'] == 'Smooth']
# Use ETS, Prophet, or similar

# Intermittent demand
intermittent_data = train[train['demand_type'] == 'Intermittent']
# Use Croston method

# Lumpy demand
lumpy_data = train[train['demand_type'] == 'Lumpy']
# Use ML ensemble or ISBTS
```

### Step 3: Generate Forecasts

Generate forecasts for the test set (future data):

```python
# Your model predictions
your_forecasts = your_model.predict(test)

# Compare with expected
expected_forecasts = expected_results[
    expected_results['is_future'] == True
][['place_id', 'item_id', 'week_start', 'forecast_demand']]
```

### Step 4: Validate Results

```python
# Merge predictions
comparison = expected_forecasts.merge(
    your_forecasts,
    on=['place_id', 'item_id', 'week_start'],
    suffixes=('_expected', '_yours')
)

# Calculate WAPE
wape = (abs(comparison['forecast_demand_expected'] - 
            comparison['forecast_demand_yours']).sum() / 
        comparison['forecast_demand_expected'].sum()) * 100

print(f"WAPE: {wape:.2f}%")
```

### Step 5: Risk Classification Validation

```python
# Compare risk classifications
risk_comparison = expected_results[
    expected_results['is_future'] == True
][['place_id', 'item_id', 'week_start', 'risk_flag']]

# Your risk classification logic should match expected risk_flag
# Accuracy should be > 80%
```

### Step 6: Action Recommendations Validation

```python
# Compare recommended actions
action_comparison = expected_results[
    expected_results['is_future'] == True
][['place_id', 'item_id', 'week_start', 
   'recommended_action', 'suggested_discount_percent', 'urgency_score']]

# Your recommendations should align with risk levels
# Critical items should have urgency_score >= 7
# Discount percentages should match risk level ranges
```

## Data Generation Methodology

The synthetic data was generated using:

1. **Pattern Learning**: Statistical patterns extracted from historical data:
   - Mean demand and standard deviation
   - Trend slope (linear regression)
   - Seasonal indices (7-week pattern)
   - Demand intermittency statistics
   - Price behavior

2. **Demand Extension**: Pattern-aware simulation:
   - Smooth/Erratic: Holt linear trend with seasonal adjustment
   - Intermittent: Croston-based interval simulation
   - Lumpy: Spike-based simulation with historical spike sizes

3. **Inventory Simulation**: 
   - Shelf life assignment (perishable: 3-14 days, semi-perishable: 30-90 days, long-life: 180-365 days)
   - Safety stock calculation: 1.65 * std(demand) * sqrt(lead_time)
   - Reorder point: mean(demand) + safety_stock
   - Inventory level simulation with reorder logic

4. **Forecast Simulation**:
   - Smooth/Erratic: Exponential smoothing forecast
   - Intermittent: Croston forecast (mean_size / mean_interval)
   - Lumpy: Croston-SBA with bias correction

5. **Risk Classification**: Based on expiry and overstock thresholds

6. **Recommendations**: Generated based on risk level and urgency

## Notes

- All dates are in UTC timezone
- Week start is always Monday
- Random seed = 42 for reproducibility
- Historical data is from original analyst dataset
- Future data (52 weeks) is synthetic but pattern-preserving

## Contact

For questions about the dataset or validation criteria, contact the Data Engineering Team.

---
*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
