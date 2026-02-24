"""
Channel Intelligence Constants
==============================
Fixed thresholds and data structures for PHASE 3.

DO NOT MODIFY without business approval.

Author: Analytics Engineering Team
Created: 2026-02-18
"""

from pathlib import Path
from dataclasses import dataclass


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

DATA_DIR = Path("data/processed")


# =============================================================================
# FIXED THRESHOLDS (LOCKED)
# =============================================================================

# Growth channel criteria
GROWTH_LEAD_SHARE_THRESHOLD = 0.10  # Fixed at 10%

# Contact type analysis
CONTACT_TYPE_MIN_LEADS = 20  # Minimum leads for contact type analysis


# =============================================================================
# SIMULATION PARAMETERS (LOCKED)
# =============================================================================

GROWTH_INCREASE_PCT = 0.10  # +10% for growth channels
BOTTLENECK_DECREASE_PCT = 0.20  # -20% for bottleneck channels


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class BaselineMetrics:
    """Container for baseline metrics."""
    total_leads: int
    total_demo: int
    overall_demo_rate: float


@dataclass
class SensitivityResult:
    """
    Container for sensitivity simulation result.
    
    Tracks BOTH rate and volume changes to detect trade-offs:
    - Rate Optimization Trade-off: Rate ↑ but Demo ↓
    - Volume vs Rate Trade-off: Rate ↓ but Demo ↑
    - Aligned: Both move in same direction
    """
    channel: str
    scenario_type: str  # 'growth' or 'bottleneck'
    change_pct: float
    
    # Channel-level metrics
    original_leads: int
    new_leads: int
    original_demo: int
    new_demo: int
    
    # Overall metrics before/after
    original_overall_rate: float
    new_overall_rate: float
    rate_change: float
    
    # Trade-off tracking
    new_total_leads: int
    new_total_demo: int
    demo_delta: int  # Absolute change in total demo
    demo_rate_delta: float  # Change in overall demo rate
    trade_off_flag: str  # "Rate Optimization Trade-off" | "Volume vs Rate Trade-off" | "Aligned"
