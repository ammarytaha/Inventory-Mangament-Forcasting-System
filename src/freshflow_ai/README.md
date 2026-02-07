# 🍃 FreshFlow AI - Inventory Decision Engine

> **AI-Powered Inventory Management for Fresh Flow Markets**

An intelligent decision support system that helps restaurant and grocery operations managers reduce waste, prevent stockouts, and make confident inventory decisions.

---

## 🎯 Challenge Addressed

**Fresh Flow Markets: Inventory Management**

Restaurant and grocery owners face a relentless balancing act:
- **Over-stocking** → Waste and expired inventory eating away at profits
- **Under-stocking** → Stockouts, lost revenue, and frustrated customers

FreshFlow AI provides intelligent, data-driven systems to replace gut instinct with accurate predictions.

---

## ✨ Key Features

### 1. 🎯 AI-Powered Recommendations
- **Reorder Alerts**: Know exactly when and how much to order
- **Markdown Suggestions**: Optimal discount levels for slow-moving items
- **Prep Quantities**: Daily kitchen prep recommendations
- **Bundle Promotions**: Strategic product combinations

### 2. 📍 Location-Based Personalization
- Manager selects their specific location
- Recommendations tailored to each store's patterns
- Local event and holiday awareness
- Historical performance analysis per location

### 3. 📈 Multi-Model Forecasting
- **Automatic model selection** based on demand patterns
- Supports: Prophet, Croston, LightGBM, Moving Average
- Confidence intervals for all forecasts
- Backtesting validation

### 4. 🌍 Context-Aware Adjustments
- Weekly seasonality (Friday +39%, Sunday -18%)
- Danish holiday calendar integration
- Special event awareness
- Weather impact (future enhancement)

### 5. 📖 Explainable AI
- Every recommendation includes clear rationale
- Confidence levels with visual indicators
- Impact estimates for business decisions
- No black boxes - full transparency

---

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r freshflow_ai/requirements.txt

# Run the dashboard
streamlit run freshflow_dashboard.py
```

### Usage

1. **Select Your Location**: Choose your store from the sidebar
2. **Review Recommendations**: See AI-generated action items
3. **Explore Forecasts**: View demand predictions with confidence intervals
4. **Check Context**: See how external factors affect your location
5. **Take Action**: Follow the prioritized recommendations

---

## 📁 Project Structure

```
freshflow_ai/
├── __init__.py              # Package initialization
├── config.py                # Configuration management
├── data_processor.py        # Data loading and transformation
├── forecaster.py            # Multi-model forecasting engine
├── recommendation_engine.py # AI recommendation generation
├── context_engine.py        # External factors handling
├── explanation_generator.py # Human-readable explanations
└── requirements.txt         # Python dependencies

freshflow_dashboard.py       # Main Streamlit dashboard
```

---

## 🔧 Module Overview

### Config (`config.py`)
- Centralized configuration for all components
- Supports environment-based settings
- Holiday calendar and weekly patterns
- Customizable business rules

### Data Processor (`data_processor.py`)
- Loads raw CSV or pre-processed Parquet files
- Place-level filtering for personalization
- Weekly aggregation at place-item level
- Demand classification (SBC methodology)

### Forecaster (`forecaster.py`)
- Automatic model selection by demand type:
  - **Smooth**: Prophet, ETS
  - **Erratic**: LightGBM, XGBoost
  - **Intermittent**: Croston, SBA
  - **Lumpy**: ML ensemble
- Safety stock calculations
- Reorder point recommendations

### Recommendation Engine (`recommendation_engine.py`)
- Generates actionable recommendations:
  - REORDER: Replenishment needs
  - DISCOUNT: Markdown opportunities
  - BUNDLE: Promotion combinations
  - PREP_ADJUST: Kitchen prep quantities
  - ALERT: Critical situations

### Context Engine (`context_engine.py`)
- Holiday calendar (Danish national holidays)
- Weekly demand patterns
- Local event management
- Forecast adjustments based on context

### Explanation Generator (`explanation_generator.py`)
- Plain English explanations
- Risk level descriptions
- Confidence visualizations
- Dashboard-ready card formats

---

## 📊 Data Requirements

The system works with the following data structure:

### Required Files
- `fct_orders.csv`: Order transactions
- `fct_order_items.csv`: Order line items
- `dim_items.csv`: Product catalog
- `dim_places.csv`: Location master

### Optional (Pre-processed)
- `features_place_item_week.parquet`: Feature-engineered dataset
- `demand_classification.csv`: SBC classification results

See the `Data Analysis/` folder for complete data specifications.

---

## 🎨 Dashboard Features

### Main Views

1. **Recommendations Tab**
   - Prioritized action items
   - Risk-level badges
   - Expandable explanations

2. **Forecasts Tab**
   - Item-level demand forecasting
   - Historical vs predicted visualization
   - Confidence intervals

3. **Analytics Tab**
   - Location statistics
   - Demand pattern distribution
   - Top items ranking

### Context Panel
- Current day factor
- Active events/holidays
- Context-based suggestions

---

## 📈 Demand Classification (SBC)

Items are classified using Syntetos-Boylan methodology:

| Type | Characteristics | Model |
|------|-----------------|-------|
| Smooth | Regular, stable | Prophet/ETS |
| Erratic | Variable levels | LightGBM |
| Intermittent | Sporadic | Croston |
| Lumpy | Unpredictable | ML ensemble |

---

## 🔮 Future Enhancements

- [ ] Real-time inventory integration
- [ ] Weather API integration
- [ ] Automated model retraining
- [ ] Mobile-responsive design
- [ ] Multi-language support
- [ ] Supplier lead time optimization

---

## 📝 API Usage

```python
from freshflow_ai import (
    Config,
    DataProcessor,
    ForecastEngine,
    RecommendationEngine,
    ContextEngine,
    ExplanationGenerator
)

# Initialize
config = Config.from_workspace('/path/to/data')
processor = DataProcessor(config)
forecaster = ForecastEngine(processor, config)
recommender = RecommendationEngine(processor, forecaster, config)

# Get recommendations for a location
recommendations = recommender.generate_recommendations(
    place_id=94025,
    forecast_horizon=4
)

# Get explanation
explainer = ExplanationGenerator(config)
for rec in recommendations[:5]:
    explanation = explainer.explain_recommendation(rec)
    print(explanation['summary'])
```

---

## 🏆 Competition Submission

This solution addresses the **Fresh Flow Markets: Inventory Management** challenge by providing:

1. **Accurate demand prediction** (daily, weekly, monthly)
2. **Prep quantity recommendations** to minimize waste
3. **Expiry-based prioritization** with markdown suggestions
4. **Promotional bundles** for near-expired items
5. **External factor integration** (holidays, weekends)

---

## 👥 Team

FreshFlow AI Team - Competition Submission 2026

---

## 📄 License

MIT License - See LICENSE file for details
