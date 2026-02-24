"""
Strategy Module
===============
LOCKED business logic and strategy pipeline (PHASE 1).

Components:
- business_rules: Constants, KPIs, constraints, assumptions
- calculations: Deterministic calculation functions
- pipeline: Strategy orchestration following locked thinking order

Author: Backend Engineering Team
Created: 2026-02-18
"""

# Business Rules (LOCKED)
from domain.strategy.business_rules import (
    # Constants
    OBJECTIVE,
    FUNNEL,
    KPI,
    CAPACITY,
    ASSUMPTIONS,
    THINKING_ORDER,
    CORE_BUSINESS_QUESTIONS,
    # KPI Formulas
    calculate_demo_rate,
    calculate_lead_share,
    calculate_demo_contribution,
    calculate_channel_impact_score,
    calculate_sales_capacity,
)

# Calculations (Core Logic)
from domain.strategy.calculations import (
    # Data Structures
    OverallMetrics,
    ChannelMetrics,
    ChannelImpact,
    SimulationResult,
    CapacityCheck,
    # Step Functions
    calculate_overall_demo_rate,
    calculate_channel_metrics,
    rank_channels,
    calculate_channel_impact,
    simulate_increase,
    simulate_reduction,
    check_capacity,
    # Helpers
    get_highest_demo_rate_channel,
    get_high_volume_low_conversion_channels,
    get_scalable_channels,
    get_reducible_channels,
)

# Pipeline
from domain.strategy.pipeline import (
    StrategyPipelineResult,
    run_strategy_pipeline,
    get_recommendations,
)


__all__ = [
    # Business Rules
    "OBJECTIVE",
    "FUNNEL",
    "KPI",
    "CAPACITY",
    "ASSUMPTIONS",
    "THINKING_ORDER",
    "CORE_BUSINESS_QUESTIONS",
    # KPI Formulas
    "calculate_demo_rate",
    "calculate_lead_share",
    "calculate_demo_contribution",
    "calculate_channel_impact_score",
    "calculate_sales_capacity",
    # Data Structures
    "OverallMetrics",
    "ChannelMetrics",
    "ChannelImpact",
    "SimulationResult",
    "CapacityCheck",
    "StrategyPipelineResult",
    # Core Logic
    "calculate_overall_demo_rate",
    "calculate_channel_metrics",
    "rank_channels",
    "calculate_channel_impact",
    "simulate_increase",
    "simulate_reduction",
    "check_capacity",
    "get_highest_demo_rate_channel",
    "get_high_volume_low_conversion_channels",
    "get_scalable_channels",
    "get_reducible_channels",
    # Pipeline
    "run_strategy_pipeline",
    "get_recommendations",
]
