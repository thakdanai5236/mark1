"""
Channel Intelligence Module
===========================
Deterministic channel growth analysis and sensitivity simulation.

PHASE 3 of Marketing Analytics Agent.

Exports:
- constants: Fixed thresholds and data structures
- calculations: Step-by-step calculation functions
- pipeline: Main orchestration functions
"""

from domain.channels.constants import (
    # Configuration
    DATA_DIR,
    GROWTH_LEAD_SHARE_THRESHOLD,
    CONTACT_TYPE_MIN_LEADS,
    GROWTH_INCREASE_PCT,
    BOTTLENECK_DECREASE_PCT,
    # Data classes
    BaselineMetrics,
    SensitivityResult,
)

from domain.channels.calculations import (
    # Step 3.1
    calculate_baseline,
    # Step 3.2
    create_channel_rankings,
    # Step 3.3
    identify_growth_channels,
    # Step 3.4
    identify_bottleneck_channels,
    # Step 3.5
    simulate_growth_scenario,
    simulate_bottleneck_scenario,
    run_sensitivity_simulation,
    # Step 3.6
    simulate_channel_mix_adjustment,
    # Step 3.7
    analyze_contact_type_overlay,
)

from domain.channels.pipeline import (
    run_channel_growth_intelligence,
    load_marketing_master,
)


__all__ = [
    # Constants
    "DATA_DIR",
    "GROWTH_LEAD_SHARE_THRESHOLD",
    "CONTACT_TYPE_MIN_LEADS",
    "GROWTH_INCREASE_PCT",
    "BOTTLENECK_DECREASE_PCT",
    # Data classes
    "BaselineMetrics",
    "SensitivityResult",
    # Calculations
    "calculate_baseline",
    "create_channel_rankings",
    "identify_growth_channels",
    "identify_bottleneck_channels",
    "simulate_growth_scenario",
    "simulate_bottleneck_scenario",
    "run_sensitivity_simulation",
    "simulate_channel_mix_adjustment",
    "analyze_contact_type_overlay",
    # Pipeline
    "run_channel_growth_intelligence",
    "load_marketing_master",
]
