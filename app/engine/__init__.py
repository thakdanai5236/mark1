"""Engine module - Strategic logic and analytics engines."""

from .enterprise_optimizer import (
    run_enterprise_optimization,
    EnterpriseOptimizationResult,
    BaselineMetrics,
    OptimizationResult,
    ConstraintStatus,
    ChannelAllocation,
    result_to_dict,
    calculate_sales_capacity,
    DEFAULT_COST_PER_LEAD,
    DEFAULT_MAX_SCALE_PERCENTAGE,
)

__all__ = [
    "run_enterprise_optimization",
    "EnterpriseOptimizationResult",
    "BaselineMetrics",
    "OptimizationResult",
    "ConstraintStatus",
    "ChannelAllocation",
    "result_to_dict",
    "calculate_sales_capacity",
    "DEFAULT_COST_PER_LEAD",
    "DEFAULT_MAX_SCALE_PERCENTAGE",
]
