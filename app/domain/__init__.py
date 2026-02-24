"""
Core Business Logic Module (LOCKED)
====================================
This module contains LOCKED business rules and deterministic calculations.

DO NOT MODIFY without business approval.

Modules:
- strategy: PHASE 1 - Locked business rules, calculations, pipeline
- channel: PHASE 3 - Channel growth intelligence

Author: Backend Engineering Team
Created: 2026-02-18
"""

# Strategy Module (PHASE 1)
from . import strategy
from .strategy import (
    # Business Rules (LOCKED)
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
    # Data Structures
    OverallMetrics,
    ChannelMetrics,
    ChannelImpact,
    SimulationResult,
    CapacityCheck,
    StrategyPipelineResult,
    # Calculations
    calculate_overall_demo_rate,
    calculate_channel_metrics,
    rank_channels,
    calculate_channel_impact,
    simulate_increase,
    simulate_reduction,
    check_capacity,
    get_scalable_channels,
    get_reducible_channels,
    # Pipeline
    run_strategy_pipeline,
    get_recommendations,
)

# Channel Intelligence Module (PHASE 3)
from . import channel


__all__ = [
    # Modules
    "strategy",
    "channel",
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
    "get_scalable_channels",
    "get_reducible_channels",
    # Pipeline
    "run_strategy_pipeline",
    "get_recommendations",
]
