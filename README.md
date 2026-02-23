# Agent Mark1 - Marketing Analytics Agent

An intelligent marketing analytics agent with deterministic business logic for data-driven marketing insights and channel optimization.

## Objective

**Increase Leads from high-conversion channels to increase Demo count and raise overall Demo Rate.**

Core principles:
- No random scaling
- No equal distribution  
- Only scale channels above average Demo Rate

---

## Phase Overview

| Phase | Name | Purpose | Location |
|-------|------|---------|----------|
| **PHASE 1** | Strategy Pipeline | LOCKED business rules and strategy simulation | `core/strategy/` |
| **PHASE 2** | Channel Data Preparation | ETL and channel performance aggregation | `pipelines/` |
| **PHASE 3** | Channel Growth Intelligence | Growth/Bottleneck detection, sensitivity simulation | `core/channel/` |
| **PHASE 4** | Enterprise Channel Optimization | LP-based demo maximization under capacity/budget/scaling constraints | `app/engine/enterprise_optimizer.py` |

---

## Project Structure

```
agent_mark1/
├── core/                           # Deterministic Business Logic (LOCKED)
│   ├── __init__.py                 # Module exports
│   ├── strategy/                   # PHASE 1: Strategy Pipeline
│   │   ├── __init__.py
│   │   ├── business_rules.py       # KPIs, Objectives, Constraints
│   │   ├── calculations.py         # Step 1-6 calculation functions
│   │   └── pipeline.py             # Strategy orchestration + CLI
│   └── channel/                    # PHASE 3: Channel Growth Intelligence
│       ├── __init__.py
│       ├── constants.py            # Fixed thresholds + dataclasses
│       ├── calculations.py         # Step 3.1-3.7 functions
│       └── pipeline.py             # Channel analysis + CLI
│
├── pipelines/                      # PHASE 2: Data Preparation
│   ├── channel_intelligence.py     # Deterministic data preparation for channel analysis
│   ├── fix_and_merge_demo.py       # handles data preparation

│
├── data/                           # Data storage
│   ├── raw/                        # Original data files
│   ├── interim/                    # Intermediate processing
│   └── processed/                  # Final clean data
│
├── app/                            # AI Application Layer (Phase 4 engine, API, LLM, RAG)
├── tests/                          # Unit tests
└── notebooks/                      # Analysis notebooks
```

---

## PHASE 1: Strategy Pipeline (LOCKED)

**Location:** `core/strategy/`

**Purpose:** LOCKED business logic for marketing strategy analysis following a strict 6-step thinking order.

### Files

| File | Description |
|------|-------------|
| `business_rules.py` | **LOCKED** constants: OBJECTIVE, FUNNEL, KPI, CAPACITY, ASSUMPTIONS, THINKING_ORDER |
| `calculations.py` | Deterministic calculation functions for all 6 steps |
| `pipeline.py` | Strategy orchestration with verbose output and recommendations |

### KPI Definitions (LOCKED)

| KPI | Formula | Type |
|-----|---------|------|
| **Demo Rate** | Demo / Lead | Primary |
| Lead Share | Channel Leads / Total Leads | Secondary |
| Demo Contribution | Channel Demos / Total Demos | Secondary |
| **Channel Impact Score** | Demo Rate x Lead Share | Derived |

### Engine Thinking Order (LOCKED)

1. Calculate overall Demo Rate
2. Analyze channel Demo Rate  
3. Rank channels by conversion
4. Calculate channel impact
5. Simulate reallocation
6. Check capacity constraint

### Capacity Constraint

```
Sales Capacity = 10 x number_of_sales x 22 (working days)
```

### Usage

```python
# Programmatic usage
from core.strategy import run_strategy_pipeline
import pandas as pd

df = pd.read_csv('data/processed/lead_demo_clean.csv')
result = run_strategy_pipeline(df, number_of_sales=5, verbose=True)

# Access results
print(result.overall_metrics.demo_rate_pct)  # 9.31%
print(result.highest_demo_rate_channel)       # ('Call Out', ChannelMetrics)
print(result.business_answers)                # Dict of Q&A
```

```bash
# CLI usage
python -m core.strategy.pipeline
```

---

## PHASE 2: Channel Data Preparation

**Location:** `pipelines/`

**Purpose:** ETL pipelines for data validation, channel aggregation, and classification.

### Files

| File | Description |
|------|-------------|
| `fix_and_merge_demo.py` | Merge Lead.xlsx with Demo.xlsx, create `lead_demo_clean.csv` |
| `channel_intelligence.py` | Main PHASE 2 pipeline: validation, baseline, aggregation, classification |
| `load_data.py` | Data loading utilities |
| `clean_lead_data.py` | Data cleaning and normalization |
| `aggregate_channel.py` | Channel-level aggregation |
| `feature_engineering.py` | Feature preparation for analysis |

### Channel Classification (LOCKED)

| Category | Criteria | Action |
|----------|----------|--------|
| **Scale** | Demo_Rate > Avg AND Lead_Share >= 10% | Increase leads |
| **Optimize** | Demo_Rate > Avg AND Lead_Share < 10% | Improve conversion |
| **Hidden Growth** | Demo_Rate < Avg AND Lead_Share >= 10% | Investigate |
| **Reduce** | Demo_Rate < Avg AND Lead_Share < 10% | Reduce allocation |

### Output Files

| File | Description |
|------|-------------|
| `lead_demo_clean.csv` | Merged Lead + Demo data |
| `channel_performance_summary.csv` | Channel-level metrics with classification |
| `channel_monthly_summary.csv` | Monthly breakdown |

### Usage

```bash
# Run PHASE 2 pipeline
python pipelines/fix_and_merge_demo.py
python pipelines/channel_intelligence.py
```

---

## PHASE 3: Channel Growth Intelligence

**Location:** `core/channel/`

**Purpose:** Identify Growth Lever Channels, Bottleneck Channels, and simulate sensitivity scenarios with trade-off detection.

### Files

| File | Description |
|------|-------------|
| `constants.py` | Fixed thresholds: GROWTH_LEAD_SHARE_THRESHOLD (10%), simulation parameters |
| `calculations.py` | Step 3.1-3.7 calculation functions |
| `pipeline.py` | Channel growth intelligence orchestration + CLI |

### Pipeline Steps

| Step | Name | Description |
|------|------|-------------|
| 3.1 | Baseline | Calculate total_leads, total_demo, overall_demo_rate |
| 3.2 | Channel Rankings | Rank by Demo_Rate, Impact_Score, Demo_Contribution |
| 3.3 | Growth Channels | Demo_Rate > Avg AND Lead_Share > 10% |
| 3.4 | Bottleneck Channels | Lead_Share > (1/N) AND Demo_Rate < Avg |
| 3.5 | Sensitivity Simulation | +10% growth, -20% bottleneck scenarios |
| 3.6 | Channel Mix Adjustment | Custom scenario simulation |
| 3.7 | Contact Type Overlay | Channel x Contact_Type analysis |

### Trade-Off Detection

Flags scenarios where Demo Rate improves but Total Demo decreases:

| Flag | Meaning |
|------|---------|
| `Aligned` | Rate and volume move in same direction |
| `Rate Optimization Trade-off` | Rate increases but Demo count decreases |
| `Volume vs Rate Trade-off` | Demo count increases but Rate decreases |

### Usage

```python
# Programmatic usage
from core.channel import run_channel_growth_intelligence, load_marketing_master

marketing = load_marketing_master()
result = run_channel_growth_intelligence(marketing, verbose=True)

# Access results
print(result['baseline'])                    # {'total_leads': 6134, ...}
print(result['growth_channels'])             # DataFrame
print(result['bottleneck_channels'])         # DataFrame
print(result['sensitivity_results'])         # Growth/Bottleneck scenarios
```

```bash
# CLI usage
python -m core.channel.pipeline
```

---

## PHASE 4: Enterprise Channel Optimization

**Location:** `app/engine/enterprise_optimizer.py`

**Purpose:** Optimize additional leads per channel to maximize total Demo count, under sales capacity, (optional) marketing budget, and per-channel scaling limits.

### Model Overview

- Decision variable: `x_i` = additional leads allocated to channel *i*
- Objective: Maximize total additional demos `Σ(x_i × Demo_Rate_i)`
- Constraints:
  - Capacity: `Σ x_i ≤ capacity_remaining` (from sales capacity formula)
  - Budget (optional): `Σ(x_i × Cost_per_Lead_i) ≤ marketing_budget` when a budget is provided
  - Channel scaling: `x_i ≤ Leads_i × Max_Scale_Percentage_i`
  - Non-negativity: `x_i ≥ 0`

### Behavior of Budget Constraint

- `marketing_budget=None` → **No budget constraint** (engine ignores budget, only capacity + scaling apply)
- `marketing_budget > 0` → Budget is enforced as a hard constraint
- `marketing_budget <= 0` (explicit) → Treated as "No Budget" (all additional allocations forced to 0)

### Inputs

- `channel_df` (from PHASE 2), with columns at minimum:
  - `Channel`, `Leads`, `Demo_Rate`
  - Optional: `Cost_per_Lead`, `Max_Scale_Percentage`
- `number_of_sales` → used in capacity formula:
  - `Sales Capacity = 10 × number_of_sales × 22`
- `marketing_budget` (optional float)
- `default_cost_per_lead`, `default_max_scale` used when columns are missing

### Key Outputs (EnterpriseOptimizationResult)

- `baseline` → current leads/demo/demo rate
- `optimization_result` → new totals, additional demos, new demo rate, solver status
- `allocation_plan` (DataFrame per channel) → current vs additional leads, additional demos, cost used, scaling utilization, `Scaling_Fully_Used` flag
- `constraints_status` → capacity/budget used & remaining, utilization %, `demo_per_capacity`, `demo_per_budget`
- Diagnostics & metadata:
  - `binding_constraint` (none/capacity/budget/scaling/multiple)
  - `executive_summary` (plain-language explanation of constraint
    situation)
  - `shadow_price_capacity`, `shadow_price_budget` + unit definitions
  - `defaults_used`, `model_assumptions`

### Usage (Programmatic)

```python
import pandas as pd
from app.engine import run_enterprise_optimization

df = pd.read_csv("data/processed/channel_performance_summary.csv")

# Example 1: No budget constraint
result_no_budget = run_enterprise_optimization(
    channel_df=df,
    number_of_sales=50,
    marketing_budget=None,
    verbose=False,
)

# Example 2: Budget and cost constraint
result_with_budget = run_enterprise_optimization(
    channel_df=df,
    number_of_sales=50,
    marketing_budget=50000.0,
    default_cost_per_lead=100.0,
    verbose=False,
)
```

```bash
# CLI usage (simple run)
python -m app.engine.enterprise_optimizer --sales 50 --budget 50000 \
    --cost-per-lead 100 --max-scale 0.30 \
    --data data/processed/channel_performance_summary.csv
```

---

## Data Flow

```
RAW DATA
data/raw/Lead.xlsx + data/raw/Demo.xlsx
         |
         v
PHASE 2: pipelines/fix_and_merge_demo.py
-> data/processed/lead_demo_clean.csv
         |
         v
PHASE 2: pipelines/channel_intelligence.py
-> data/processed/channel_performance_summary.csv
-> data/processed/channel_monthly_summary.csv
         |
         +------------------+-------------------+
         |                                      |
         v                                      v
PHASE 1: core/strategy/               PHASE 3: core/channel/
Strategy Pipeline                     Channel Growth Intelligence
-> Business Q&A                       -> Growth/Bottleneck Detection
-> Simulation Results                 -> Sensitivity Simulation
-> Recommendations                    -> Trade-Off Analysis
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run data preparation (PHASE 2)
python pipelines/fix_and_merge_demo.py
python pipelines/channel_intelligence.py

# 3. Run strategy analysis (PHASE 1)
python -m core.strategy.pipeline

# 4. Run channel growth intelligence (PHASE 3)
python -m core.channel.pipeline
```

---

## Key Metrics (Current Data)

| Metric | Value |
|--------|-------|
| Total Leads | 6,134 |
| Total Demos | 571 |
| Overall Demo Rate | 9.31% |
| Growth Channels | 2 (Call In, Get Demo) |
| Bottleneck Channels | 2 (Facebook, Unknown) |

---

## Import Examples

```python
# Import from core (backward compatible)
from core import OBJECTIVE, KPI, CAPACITY
from core import run_strategy_pipeline, get_recommendations

# Import from specific modules
from core.strategy import run_strategy_pipeline, StrategyPipelineResult
from core.channel import run_channel_growth_intelligence, SensitivityResult

# Import module namespaces
from core import strategy, channel
```

---

## Core Business Logic (LOCKED)

### Funnel Definitions

```
Lead -> Contacted -> Demo -> Conversion
  |                   |
  +--- Lost <---------+
```

- **Lead**: Unique Customer_Code
- **Demo**: Successfully scheduled demo (is_demo = 1)
- **Demo Rate**: Primary success metric

---

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=core tests/

# Run specific test file
pytest tests/test_strategy_pipeline.py -v
```

---

## Development

### Code Style

```bash
# Format code
black .
isort .

# Lint
flake8 core/ pipelines/ tests/

# Type check
mypy core/
```

---

## Phase Status

| Phase | Status | Description |
|-------|--------|-------------|
| PHASE 1 | [x] Complete | Strategy Pipeline in `core/strategy/` |
| PHASE 2 | [x] Complete | Channel Data Preparation in `pipelines/` |
| PHASE 3 | [x] Complete | Channel Growth Intelligence in `core/channel/` |

---

## Important Notes

**LOCKED Logic**: Files in `core/` contain approved business rules. Do NOT modify without stakeholder approval.

**Deterministic**: All `core/` functions are pure and deterministic - no AI, no randomness.

**Data Requirements**: Input DataFrame must have columns: `Customer_Code`, `Contact_Type`, `is_demo`

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
