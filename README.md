# FreshFlow Inventory Management Solution

**Deloitte x AUC Hackathon - Fresh Flow Markets Use Case**

An intelligent inventory management and demand forecasting solution for Fresh Flow Markets, designed to minimize waste, prevent stockouts, and optimize kitchen prep operations through AI-driven insights.

---

## Team Members

| Name | Role | Contributions |
|------|------|---------------|
| **Youssef Ibrahim** | Developer | Built the core solution architecture, implemented the data engineering pipeline, AI forecasting engine, decision recommendation system, and the interactive dashboard |
| **Ammar Yasser** | Data Analyst | Conducted comprehensive data analysis, identified data patterns and relationships, created data quality reports, and provided analytical insights for feature engineering |
| **Ziad Tolba** | QA/Testing Lead | Designed and implemented test cases, built the testing validation framework, ensured accuracy of recommendations, and documented expected results |

---

## Project Description

Fresh Flow Markets faces the classic inventory management dilemma: **over-stocking leads to waste and expired inventory, while under-stocking causes stockouts and lost revenue**. Our solution provides an intelligent system that:

- **Accurately predicts demand** using AI/ML forecasting models
- **Recommends optimal prep quantities** to minimize kitchen waste
- **Prioritizes inventory based on expiration dates** for proactive waste prevention
- **Generates actionable alerts** for overstock/understock situations
- **Provides an interactive dashboard** for real-time inventory insights

---

## Features

### 1. Data Discovery & Quality Assessment
- Automatic profiling of all data sources
- Data quality scoring and validation
- Relationship mapping between fact and dimension tables

### 2. AI-Powered Demand Forecasting
- Time-series forecasting using Prophet models
- Contextual adjustments for seasonality, holidays, and external factors
- Confidence intervals for risk-aware planning

### 3. Intelligent Decision Engine
- Real-time inventory health assessment
- Expiry risk detection and prioritization
- Automated reorder point calculations

### 4. Interactive Dashboard
- Real-time inventory metrics visualization
- Demand forecasts with interactive charts
- Actionable recommendations with business explanations
- **Data source toggle**: Switch between ML forecasting and historical data
- **User-friendly selection**: Choose places and items by name, not IDs

### 5. Business-Ready Explanations
- Every recommendation includes plain-English explanations
- Impact quantification (cost savings, waste reduction)
- Confidence levels for decision support

---

## Technologies Used

- **Python 3.10+** - Core programming language
- **Pandas & NumPy** - Data manipulation and analysis
- **PyArrow** - Efficient data storage (Parquet files)
- **Prophet** - Time-series forecasting
- **Streamlit** - Interactive dashboard framework
- **Plotly** - Data visualization
- **Scikit-learn** - Machine learning utilities

---

## 📁 Project Structure

```
Inventory-Management/
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
│
├── src/                                # Source code
│   ├── main.py                         # Main pipeline entry point
│   ├── freshflow_universal_dashboard.py # Main AI Dashboard (Streamlit)
│   ├── freshflow_dashboard.py          # Alternative dashboard
│   ├── run_dashboard.py                # Dashboard runner script
│   ├── run_freshflow.py                # FreshFlow runner
│   ├── data_discovery.py               # Data discovery and profiling
│   ├── data_quality_report.py          # Quality assessment
│   ├── data_model.py                   # Data model documentation
│   ├── data_cleaning.py                # Data cleaning pipeline
│   ├── inventory_analytics.py          # Analytics engine
│   ├── inventory_decision_engine.py    # Decision engine
│   ├── decision_outputs.py             # Output generation
│   ├── fresh_flow_pipeline.py          # Complete pipeline
│   │
│   ├── dashboard/                      # Dashboard components
│   │   ├── app.py                      # Dashboard app module
│   │   ├── data_loader.py              # Data loading utilities
│   │   └── prophet_forecaster.py       # Prophet forecasting
│   │
│   ├── freshflow_ai/                   # AI/ML modules
│   │   ├── config.py                   # AI configuration
│   │   ├── context_engine.py           # Context analysis
│   │   ├── data_processor.py           # Data processing
│   │   ├── explanation_generator.py    # Generate explanations
│   │   ├── forecaster.py               # Forecasting models
│   │   └── recommendation_engine.py    # AI recommendations
│   │
│   ├── services/                       # Business logic services
│   │   ├── context_adjustments.py      # Context adjustments
│   │   ├── data_loader.py              # Data loader service
│   │   ├── forecaster.py               # Forecasting service
│   │   ├── inventory_health.py         # Inventory health checks
│   │   ├── output_generator.py         # Output generation
│   │   └── recommendation_engine.py    # Recommendations
│   │
│   ├── models/                         # Data models
│   │   └── explanation.py              # Explanation models
│   │
│   ├── utils/                          # Utility functions
│   │   ├── constants.py                # Constants
│   │   ├── logger.py                   # Logging utilities
│   │   └── validators.py               # Data validators
│   │
│   ├── outputs/                        # Generated pipeline outputs
│   │   ├── analytics/                  # Analytics results
│   │   ├── decisions/                  # Decision outputs
│   │   ├── forecasts/                  # Forecast results
│   │   └── visualizations/             # Charts and graphs
│   │
│   └── api/                            # API endpoints (future)
│
├── tests/                              # Test files
│   ├── test_decision_engine.py         # Decision engine tests
│   ├── test_freshflow_solution.py      # Solution tests
│   ├── run_testing_validation.py       # Test runner
│   ├── README_expected_results.md      # Expected test results
│   └── TESTING_ACCURACY_REPORT.md      # Accuracy report
│
├── docs/                               # Documentation
│   ├── UX_DESIGN_NOTES.md              # UX design documentation
│   └── data_analysis/                  # Data analysis documentation
│       ├── DATA_README.md              # Data documentation
│       ├── README_FOR_DEVELOPER.md     # Developer guide
│       ├── data/                       # Processed data (parquet files)
│       │   ├── features_place_item_week.parquet
│       │   ├── weekly_place_item.parquet
│       │   ├── dim_places_clean.parquet
│       │   └── dim_items_clean.parquet
│       ├── schema/                     # Data schemas
│       └── scripts/                    # Analysis scripts
│
├── config/                             # Configuration files
│   └── settings.py                     # Application settings
│
├── data/                               # Raw data files
│   ├── dim_*.csv                       # Dimension tables
│   ├── fct_*.csv                       # Fact tables
│   └── most_ordered.csv
│
└── outputs/                            # Root-level outputs
    ├── 01_discovery_results.json
    ├── 02_quality_report.json
    ├── 03_data_model.json
    └── ...
```

---

## Installation

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd Inventory-Management
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activate (Windows CMD)
.\venv\Scripts\activate.bat

# Activate (Linux/Mac)
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Data Files
Ensure the following data files are in place:
- **Raw data**: `data/dim_*.csv` and `data/fct_*.csv`
- **Processed data**: `docs/data_analysis/data/*.parquet`

---

## Usage

### 🚀 Quick Start - Run the Interactive Dashboard

The main way to use the FreshFlow solution is through the **interactive AI dashboard**:

```bash
# Navigate to the src directory
cd src

# Run the main dashboard
streamlit run freshflow_universal_dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

#### Dashboard Features:
- **Place & Item Selection**: Choose locations and products by name
- **Data Source Toggle**: Switch between:
  - 🔮 **Forecasting Data** - ML-powered predictions with advanced features
  - 📈 **Weekly Demand Data** - Historical weekly aggregates
- **AI-Powered Insights**: Get intelligent inventory recommendations
- **Interactive Visualizations**: Explore demand patterns and forecasts

### Run the Complete Data Pipeline

To run the full data processing pipeline:

```bash
cd src
python main.py
```

This will execute:
1. Data discovery and profiling
2. Data quality assessment
3. Data model documentation
4. Data cleaning pipeline
5. Analytics calculation
6. Decision output generation

### Alternative Dashboard

For a simpler dashboard view:

```bash
cd src
streamlit run freshflow_dashboard.py
```

### Run Tests

```bash
cd tests
python run_testing_validation.py
```

### Run Individual Tests

```bash
cd tests
python test_decision_engine.py
python test_freshflow_solution.py
```

---

## Output Files

### Generated in `outputs/` directory:

| File | Description |
|------|-------------|
| `01_discovery_results.json` | Data profiling results |
| `02_quality_report.json` | Quality assessment scores |
| `03_data_model.json` | Relationship documentation |
| `04_cleaning_summary.json` | Cleaning operations log |
| `07_executive_summary.json` | Business-ready summary |
| `analytics/*.csv` | Analytics outputs |
| `decisions/*.csv` | Decision recommendations |
| `forecasts/*.csv` | Demand forecasts |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        FreshFlow Solution                        │
├─────────────────────────────────────────────────────────────────┤
│   ┌───────────┐    ┌───────────┐    ┌───────────────────────┐  │
│   │   Data    │    │  Quality  │    │     Data Model        │  │
│   │ Discovery │───▶│ Assessment│───▶│   Documentation       │  │
│   └───────────┘    └───────────┘    └───────────────────────┘  │
│         │                                      │                │
│         ▼                                      ▼                │
│   ┌───────────┐    ┌───────────┐    ┌───────────────────────┐  │
│   │   Data    │    │ Inventory │    │     Decision          │  │
│   │ Cleaning  │───▶│ Analytics │───▶│      Engine           │  │
│   └───────────┘    └───────────┘    └───────────────────────┘  │
│                                              │                  │
│                                              ▼                  │
│                          ┌───────────────────────────────────┐ │
│                          │     AI Forecasting Module         │ │
│                          │   (Prophet + Context Engine)      │ │
│                          └───────────────────────────────────┘ │
│                                              │                  │
│                                              ▼                  │
│                          ┌───────────────────────────────────┐ │
│                          │    Interactive Dashboard          │ │
│                          │        (Streamlit)                │ │
│                          └───────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Model

### Fact Tables
| Table | Description |
|-------|-------------|
| `fct_orders` | Order transactions |
| `fct_order_items` | Order line items |
| `fct_inventory_reports` | Inventory snapshots |
| `fct_campaigns` | Campaign usage |
| `fct_bonus_codes` | Bonus code redemptions |
| `fct_invoice_items` | Invoice line items |

### Dimension Tables
| Table | Description |
|-------|-------------|
| `dim_places` | Locations/stores |
| `dim_users` | Customers |
| `dim_menu_items` | Menu items/products |
| `dim_skus` | Stock keeping units |
| `dim_stock_categories` | SKU categories |
| `dim_bill_of_materials` | Bill of materials |

### Key Relationships
```
fct_orders → dim_places (place_id)
fct_orders → dim_users (user_id)
fct_order_items → fct_orders (order_id)
fct_order_items → dim_menu_items (item_id)
dim_menu_items → dim_skus (via BOM)
dim_skus → dim_stock_categories (stock_category_id)
```

---

## Business Value

| Impact Area | How We Deliver |
|-------------|----------------|
| **Reduce Waste** | Expiry risk alerts, prep optimization, overstock detection |
| **Prevent Stockouts** | Understock alerts, demand forecasting, safety stock calculations |
| **Optimize Operations** | Data-driven prep recommendations, inventory investment optimization |
| **Enable AI/ML** | Clean time series data, pre-computed features, structured star schema |

---

## Code Quality

- **Modular Design**: Each module is independent and reusable
- **Clear Comments**: Business context explained throughout
- **Error Handling**: Graceful handling of missing data
- **Type Hints**: Better code documentation
- **Clean Code**: Readable, maintainable Python following PEP 8

---

## Additional Documentation

For more detailed information, see the `docs/` directory:
- [Data Analysis Documentation](docs/data_analysis/DATA_README.md)
- [Developer Guide](docs/data_analysis/README_FOR_DEVELOPER.md)
- [UX Design Notes](docs/UX_DESIGN_NOTES.md)
- [Testing Accuracy Report](tests/TESTING_ACCURACY_REPORT.md)
- [Expected Test Results](tests/README_expected_results.md)

---

## License

This project was developed for the Deloitte x AUC Hackathon 2026.

---

**Team FreshFlow - Built for measurable business impact**
