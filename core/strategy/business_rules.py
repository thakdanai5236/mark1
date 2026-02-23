"""
Business Rules Configuration
============================
LOCKED business logic encoded as system configuration.

DO NOT MODIFY these definitions without business approval.
All values are derived from approved strategic framework.

Author: Backend Engineering Team
Created: 2026-02-18
"""

from dataclasses import dataclass
from typing import Dict, Callable
from enum import Enum


# =============================================================================
# OBJECTIVE CONSTANTS
# =============================================================================

@dataclass(frozen=True)
class Objective:
    """Strategic objective constants - LOCKED."""
    
    # Primary goal
    PRIMARY_GOAL: str = "Increase Demo count through high-conversion channel scaling"
    
    # Scaling constraints
    MIN_SCALE_INCREASE_PCT: float = 0.10  # 10%
    MAX_SCALE_INCREASE_PCT: float = 0.20  # 20%
    DEFAULT_REDUCTION_PCT: float = 0.20   # 20%
    
    # Decision rules
    SCALE_ONLY_ABOVE_AVERAGE: bool = True  # Only scale channels above average Demo Rate
    RANDOM_SCALING_FORBIDDEN: bool = True  # No random scaling allowed
    EQUAL_DISTRIBUTION_FORBIDDEN: bool = True  # No equal distribution allowed


OBJECTIVE = Objective()


# =============================================================================
# FUNNEL STAGE DEFINITIONS
# =============================================================================

class FunnelStage(Enum):
    """Funnel stage definitions - LOCKED."""
    LEAD = "lead"           # Unique Customer_Code
    CONTACTED = "contacted"  # Successfully contacted lead
    DEMO = "demo"           # Successfully scheduled demo
    LOST = "lost"           # No demo occurred


@dataclass(frozen=True)
class FunnelDefinitions:
    """Funnel definitions - LOCKED."""
    
    # Column mappings
    LEAD_ID_COLUMN: str = "Customer_Code"
    CHANNEL_COLUMN: str = "Contact_Type"
    DEMO_FLAG_COLUMN: str = "is_demo"
    DEMO_DATE_COLUMN: str = "Demo_Date"
    
    # Stage identification
    LEAD_DEFINITION: str = "Unique Customer_Code"
    CONTACTED_DEFINITION: str = "Successfully contacted lead"
    DEMO_DEFINITION: str = "Successfully scheduled demo (is_demo = 1)"
    LOST_DEFINITION: str = "No demo occurred (is_demo = 0)"


FUNNEL = FunnelDefinitions()


# =============================================================================
# KPI FORMULA DEFINITIONS
# =============================================================================

@dataclass(frozen=True)
class KPIDefinitions:
    """KPI formula definitions - LOCKED."""
    
    # Primary KPI
    PRIMARY_KPI: str = "Demo Rate"
    PRIMARY_KPI_FORMULA: str = "Demo / Lead"
    
    # Secondary KPIs
    SECONDARY_KPIS: tuple = (
        "Lead Share per Channel",
        "Contact Rate per Channel", 
        "Demo Contribution per Channel",
    )
    
    # Derived KPI
    DERIVED_KPI: str = "Channel Impact Score"
    DERIVED_KPI_FORMULA: str = "Demo Rate × Lead Share"


KPI = KPIDefinitions()


def calculate_demo_rate(demos: int, leads: int) -> float:
    """
    Calculate Demo Rate.
    Formula: Demo / Lead
    
    Args:
        demos: Number of demos
        leads: Number of leads
        
    Returns:
        Demo rate as decimal (0.0 to 1.0)
    """
    if leads == 0:
        return 0.0
    return demos / leads


def calculate_lead_share(channel_leads: int, total_leads: int) -> float:
    """
    Calculate Lead Share per Channel.
    Formula: Channel Leads / Total Leads
    
    Args:
        channel_leads: Number of leads in channel
        total_leads: Total number of leads
        
    Returns:
        Lead share as decimal (0.0 to 1.0)
    """
    if total_leads == 0:
        return 0.0
    return channel_leads / total_leads


def calculate_demo_contribution(channel_demos: int, total_demos: int) -> float:
    """
    Calculate Demo Contribution per Channel.
    Formula: Channel Demos / Total Demos
    
    Args:
        channel_demos: Number of demos in channel
        total_demos: Total number of demos
        
    Returns:
        Demo contribution as decimal (0.0 to 1.0)
    """
    if total_demos == 0:
        return 0.0
    return channel_demos / total_demos


def calculate_channel_impact_score(demo_rate: float, lead_share: float) -> float:
    """
    Calculate Channel Impact Score.
    Formula: Demo Rate × Lead Share
    
    Args:
        demo_rate: Channel demo rate
        lead_share: Channel lead share
        
    Returns:
        Impact score as decimal
    """
    return demo_rate * lead_share


# KPI Formula Registry
KPI_FORMULAS: Dict[str, Callable] = {
    "demo_rate": calculate_demo_rate,
    "lead_share": calculate_lead_share,
    "demo_contribution": calculate_demo_contribution,
    "channel_impact_score": calculate_channel_impact_score,
}


# =============================================================================
# CAPACITY CONSTRAINT
# =============================================================================

@dataclass(frozen=True)
class CapacityConstraint:
    """Capacity constraint configuration - LOCKED."""
    
    # Formula components
    LEADS_PER_SALES_PER_DAY: int = 10
    WORKING_DAYS_PER_MONTH: int = 22
    
    # Constraint rule
    FORMULA: str = "Sales Capacity = 10 × number_of_sales × 22"
    CONSTRAINT_RULE: str = "System must never propose lead increase beyond capacity"


CAPACITY = CapacityConstraint()


def calculate_sales_capacity(number_of_sales: int) -> int:
    """
    Calculate maximum sales capacity.
    Formula: 10 × number_of_sales × 22
    
    Args:
        number_of_sales: Number of sales personnel
        
    Returns:
        Maximum lead capacity per month
    """
    return (
        CAPACITY.LEADS_PER_SALES_PER_DAY 
        * number_of_sales 
        * CAPACITY.WORKING_DAYS_PER_MONTH
    )


# =============================================================================
# STRATEGIC ASSUMPTIONS
# =============================================================================

@dataclass(frozen=True)
class StrategicAssumptions:
    """Strategic assumptions - LOCKED."""
    
    # Core assumptions
    CONVERSION_VARIES_BY_CHANNEL: bool = True
    SCALING_HIGH_CONVERSION_INCREASES_DEMO_FASTER: bool = True
    RANDOM_SCALING_FORBIDDEN: bool = True
    
    # Decision flags
    ONLY_SCALE_ABOVE_AVERAGE: bool = True
    REDUCTION_APPLIES_TO_BELOW_AVERAGE: bool = True


ASSUMPTIONS = StrategicAssumptions()


# =============================================================================
# ENGINE THINKING ORDER
# =============================================================================

@dataclass(frozen=True)
class EngineThinkingOrder:
    """
    Engine thinking order - MUST BE HARD-CODED.
    This defines the strict sequence of operations.
    """
    
    STEP_1: str = "Calculate overall Demo Rate"
    STEP_2: str = "Analyze channel Demo Rate"
    STEP_3: str = "Rank channels by conversion"
    STEP_4: str = "Calculate channel impact"
    STEP_5: str = "Simulate reallocation"
    STEP_6: str = "Check capacity constraint"
    
    # Ordered list for iteration
    SEQUENCE: tuple = (
        "calculate_overall_demo_rate",
        "analyze_channel_demo_rate",
        "rank_channels_by_conversion",
        "calculate_channel_impact",
        "simulate_reallocation",
        "check_capacity_constraint",
    )


THINKING_ORDER = EngineThinkingOrder()


# =============================================================================
# CORE BUSINESS QUESTIONS
# =============================================================================

CORE_BUSINESS_QUESTIONS: tuple = (
    "1) Which channel has the highest Demo Rate?",
    "2) Which channel has high volume but low conversion?",
    "3) If we increase high-conversion channel by 10-20%, what is the projected demo count?",
    "4) If we reduce low-conversion channel by 20%, what is the effect on overall Demo Rate?",
)
