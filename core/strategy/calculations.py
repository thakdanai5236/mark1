"""
Core Logic Engine
=================
Deterministic calculation functions for marketing strategy.

All functions are pure and deterministic.
No AI or LLM code included.

Author: Backend Engineering Team
Created: 2026-02-18
"""

import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from core.strategy.business_rules import (
    FUNNEL,
    KPI,
    OBJECTIVE,
    CAPACITY,
    ASSUMPTIONS,
    calculate_demo_rate,
    calculate_lead_share,
    calculate_demo_contribution,
    calculate_channel_impact_score,
    calculate_sales_capacity,
)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class OverallMetrics:
    """Container for overall metrics."""
    total_leads: int
    total_demos: int
    demo_rate: float
    demo_rate_pct: float


@dataclass
class ChannelMetrics:
    """Container for channel-level metrics."""
    channel: str
    leads: int
    demos: int
    demo_rate: float
    demo_rate_pct: float
    lead_share: float
    lead_share_pct: float
    demo_contribution: float
    demo_contribution_pct: float
    is_above_average: bool


@dataclass
class ChannelImpact:
    """Container for channel impact analysis."""
    channel: str
    demo_rate: float
    lead_share: float
    impact_score: float
    rank: int


@dataclass
class SimulationResult:
    """Container for simulation results."""
    channel: str
    change_type: str  # 'increase' or 'reduction'
    change_pct: float
    current_leads: int
    projected_leads: int
    current_demos: int
    projected_demos: int
    current_demo_rate: float
    projected_overall_demo_rate: float
    demo_change: int
    is_within_capacity: bool
    capacity_limit: Optional[int] = None


@dataclass
class CapacityCheck:
    """Container for capacity check results."""
    current_leads: int
    capacity_limit: int
    available_capacity: int
    utilization_pct: float
    can_increase: bool
    max_additional_leads: int


# =============================================================================
# STEP 1: CALCULATE OVERALL DEMO RATE
# =============================================================================

def calculate_overall_demo_rate(df: pd.DataFrame) -> OverallMetrics:
    """
    Calculate overall Demo Rate from the dataset.
    
    Formula: Demo Rate = Demo / Lead
    
    Args:
        df: DataFrame with columns [Customer_Code, Contact_Type, is_demo]
        
    Returns:
        OverallMetrics with total leads, demos, and demo rate
    """
    total_leads = len(df)
    total_demos = df[FUNNEL.DEMO_FLAG_COLUMN].sum()
    demo_rate = calculate_demo_rate(total_demos, total_leads)
    
    return OverallMetrics(
        total_leads=total_leads,
        total_demos=int(total_demos),
        demo_rate=demo_rate,
        demo_rate_pct=round(demo_rate * 100, 2)
    )


# =============================================================================
# STEP 2: CALCULATE CHANNEL METRICS
# =============================================================================

def calculate_channel_metrics(
    df: pd.DataFrame,
    overall_metrics: OverallMetrics
) -> Dict[str, ChannelMetrics]:
    """
    Calculate metrics for each channel.
    
    Metrics per channel:
    - Lead count
    - Demo count  
    - Demo Rate
    - Lead Share
    - Demo Contribution
    - Is above average flag
    
    Args:
        df: DataFrame with lead data
        overall_metrics: Overall metrics for comparison
        
    Returns:
        Dictionary of channel name -> ChannelMetrics
    """
    channel_col = FUNNEL.CHANNEL_COLUMN
    demo_col = FUNNEL.DEMO_FLAG_COLUMN
    
    # Group by channel
    channel_stats = df.groupby(channel_col).agg(
        leads=(FUNNEL.LEAD_ID_COLUMN, 'count'),
        demos=(demo_col, 'sum')
    ).reset_index()
    
    result = {}
    
    for _, row in channel_stats.iterrows():
        channel = row[channel_col]
        leads = int(row['leads'])
        demos = int(row['demos'])
        
        # Calculate metrics
        demo_rate = calculate_demo_rate(demos, leads)
        lead_share = calculate_lead_share(leads, overall_metrics.total_leads)
        demo_contrib = calculate_demo_contribution(demos, overall_metrics.total_demos)
        
        # Check if above average
        is_above_avg = demo_rate > overall_metrics.demo_rate
        
        result[channel] = ChannelMetrics(
            channel=channel,
            leads=leads,
            demos=demos,
            demo_rate=demo_rate,
            demo_rate_pct=round(demo_rate * 100, 2),
            lead_share=lead_share,
            lead_share_pct=round(lead_share * 100, 2),
            demo_contribution=demo_contrib,
            demo_contribution_pct=round(demo_contrib * 100, 2),
            is_above_average=is_above_avg
        )
    
    return result


# =============================================================================
# STEP 3: RANK CHANNELS BY CONVERSION
# =============================================================================

def rank_channels(
    channel_metrics: Dict[str, ChannelMetrics],
    sort_by: str = "demo_rate",
    ascending: bool = False
) -> List[Tuple[str, ChannelMetrics]]:
    """
    Rank channels by specified metric.
    
    Default: Rank by Demo Rate (highest first)
    
    Args:
        channel_metrics: Dictionary of channel metrics
        sort_by: Metric to sort by ('demo_rate', 'leads', 'demos', 'impact_score')
        ascending: Sort order (False = highest first)
        
    Returns:
        List of (channel_name, metrics) tuples sorted by metric
    """
    sorted_channels = sorted(
        channel_metrics.items(),
        key=lambda x: getattr(x[1], sort_by),
        reverse=not ascending
    )
    
    return sorted_channels


def get_highest_demo_rate_channel(
    channel_metrics: Dict[str, ChannelMetrics]
) -> Tuple[str, ChannelMetrics]:
    """
    Get the channel with highest Demo Rate.
    
    Answers: "Which channel has the highest Demo Rate?"
    
    Args:
        channel_metrics: Dictionary of channel metrics
        
    Returns:
        Tuple of (channel_name, metrics)
    """
    ranked = rank_channels(channel_metrics, sort_by="demo_rate", ascending=False)
    return ranked[0] if ranked else (None, None)


def get_high_volume_low_conversion_channels(
    channel_metrics: Dict[str, ChannelMetrics],
    overall_demo_rate: float,
    min_lead_share: float = 0.05  # At least 5% of leads
) -> List[Tuple[str, ChannelMetrics]]:
    """
    Get channels with high volume but low conversion.
    
    Answers: "Which channel has high volume but low conversion?"
    
    Definition:
    - High volume: Lead share >= min_lead_share
    - Low conversion: Demo rate below average
    
    Args:
        channel_metrics: Dictionary of channel metrics
        overall_demo_rate: Average demo rate for comparison
        min_lead_share: Minimum lead share to be considered "high volume"
        
    Returns:
        List of (channel_name, metrics) tuples
    """
    result = []
    
    for channel, metrics in channel_metrics.items():
        is_high_volume = metrics.lead_share >= min_lead_share
        is_low_conversion = metrics.demo_rate < overall_demo_rate
        
        if is_high_volume and is_low_conversion:
            result.append((channel, metrics))
    
    # Sort by lead share (highest first)
    result.sort(key=lambda x: x[1].lead_share, reverse=True)
    
    return result


# =============================================================================
# STEP 4: CALCULATE CHANNEL IMPACT
# =============================================================================

def calculate_channel_impact(
    channel_metrics: Dict[str, ChannelMetrics]
) -> Dict[str, ChannelImpact]:
    """
    Calculate Channel Impact Score for each channel.
    
    Formula: Impact Score = Demo Rate × Lead Share
    
    Args:
        channel_metrics: Dictionary of channel metrics
        
    Returns:
        Dictionary of channel name -> ChannelImpact
    """
    impacts = {}
    
    for channel, metrics in channel_metrics.items():
        impact_score = calculate_channel_impact_score(
            metrics.demo_rate,
            metrics.lead_share
        )
        
        impacts[channel] = ChannelImpact(
            channel=channel,
            demo_rate=metrics.demo_rate,
            lead_share=metrics.lead_share,
            impact_score=impact_score,
            rank=0  # Will be set after sorting
        )
    
    # Assign ranks
    sorted_impacts = sorted(
        impacts.items(),
        key=lambda x: x[1].impact_score,
        reverse=True
    )
    
    for rank, (channel, impact) in enumerate(sorted_impacts, 1):
        impacts[channel].rank = rank
    
    return impacts


# =============================================================================
# STEP 5: SIMULATE REALLOCATION
# =============================================================================

def simulate_increase(
    channel: str,
    channel_metrics: Dict[str, ChannelMetrics],
    overall_metrics: OverallMetrics,
    increase_pct: float,
    capacity_limit: Optional[int] = None
) -> SimulationResult:
    """
    Simulate increasing leads in a specific channel.
    
    Answers: "If we increase high-conversion channel by X%, 
              what is the projected demo count?"
    
    Assumptions:
    - Channel maintains its Demo Rate after scaling
    - Increase applies to lead count
    
    Args:
        channel: Channel name to simulate
        channel_metrics: Dictionary of channel metrics
        overall_metrics: Current overall metrics
        increase_pct: Increase percentage (e.g., 0.10 for 10%)
        capacity_limit: Optional capacity limit
        
    Returns:
        SimulationResult with projected outcomes
    """
    if channel not in channel_metrics:
        raise ValueError(f"Channel '{channel}' not found in metrics")
    
    metrics = channel_metrics[channel]
    
    # Calculate projected values
    lead_increase = int(metrics.leads * increase_pct)
    projected_leads = metrics.leads + lead_increase
    
    # Projected demos (assuming same conversion rate)
    projected_channel_demos = int(projected_leads * metrics.demo_rate)
    demo_increase = projected_channel_demos - metrics.demos
    
    # Calculate new overall demo rate
    # Other channels remain unchanged
    other_demos = overall_metrics.total_demos - metrics.demos
    total_projected_demos = other_demos + projected_channel_demos
    
    other_leads = overall_metrics.total_leads - metrics.leads
    total_projected_leads = other_leads + projected_leads
    
    projected_overall_rate = calculate_demo_rate(
        total_projected_demos, 
        total_projected_leads
    )
    
    # Check capacity
    is_within_capacity = True
    if capacity_limit is not None:
        is_within_capacity = total_projected_leads <= capacity_limit
    
    return SimulationResult(
        channel=channel,
        change_type="increase",
        change_pct=increase_pct,
        current_leads=metrics.leads,
        projected_leads=projected_leads,
        current_demos=metrics.demos,
        projected_demos=projected_channel_demos,
        current_demo_rate=metrics.demo_rate,
        projected_overall_demo_rate=projected_overall_rate,
        demo_change=demo_increase,
        is_within_capacity=is_within_capacity,
        capacity_limit=capacity_limit
    )


def simulate_reduction(
    channel: str,
    channel_metrics: Dict[str, ChannelMetrics],
    overall_metrics: OverallMetrics,
    reduction_pct: float = 0.20  # Default 20% as per business rule
) -> SimulationResult:
    """
    Simulate reducing leads in a specific channel.
    
    Answers: "If we reduce low-conversion channel by X%, 
              what is the effect on overall Demo Rate?"
    
    Args:
        channel: Channel name to simulate
        channel_metrics: Dictionary of channel metrics
        overall_metrics: Current overall metrics
        reduction_pct: Reduction percentage (e.g., 0.20 for 20%)
        
    Returns:
        SimulationResult with projected outcomes
    """
    if channel not in channel_metrics:
        raise ValueError(f"Channel '{channel}' not found in metrics")
    
    metrics = channel_metrics[channel]
    
    # Calculate projected values
    lead_reduction = int(metrics.leads * reduction_pct)
    projected_leads = metrics.leads - lead_reduction
    
    # Projected demos (assuming same conversion rate)
    projected_channel_demos = int(projected_leads * metrics.demo_rate)
    demo_reduction = metrics.demos - projected_channel_demos
    
    # Calculate new overall demo rate
    other_demos = overall_metrics.total_demos - metrics.demos
    total_projected_demos = other_demos + projected_channel_demos
    
    other_leads = overall_metrics.total_leads - metrics.leads
    total_projected_leads = other_leads + projected_leads
    
    projected_overall_rate = calculate_demo_rate(
        total_projected_demos,
        total_projected_leads
    )
    
    return SimulationResult(
        channel=channel,
        change_type="reduction",
        change_pct=reduction_pct,
        current_leads=metrics.leads,
        projected_leads=projected_leads,
        current_demos=metrics.demos,
        projected_demos=projected_channel_demos,
        current_demo_rate=metrics.demo_rate,
        projected_overall_demo_rate=projected_overall_rate,
        demo_change=-demo_reduction,
        is_within_capacity=True,  # Reduction always within capacity
        capacity_limit=None
    )


# =============================================================================
# STEP 6: CHECK CAPACITY CONSTRAINT
# =============================================================================

def check_capacity(
    current_leads: int,
    number_of_sales: int
) -> CapacityCheck:
    """
    Check capacity constraint against sales team size.
    
    Formula: Sales Capacity = 10 × number_of_sales × 22
    
    Constraint: System must never propose lead increase beyond capacity.
    
    Args:
        current_leads: Current number of leads
        number_of_sales: Number of sales personnel
        
    Returns:
        CapacityCheck with capacity analysis
    """
    capacity_limit = calculate_sales_capacity(number_of_sales)
    available_capacity = capacity_limit - current_leads
    utilization_pct = (current_leads / capacity_limit * 100) if capacity_limit > 0 else 100
    
    return CapacityCheck(
        current_leads=current_leads,
        capacity_limit=capacity_limit,
        available_capacity=max(0, available_capacity),
        utilization_pct=round(utilization_pct, 2),
        can_increase=available_capacity > 0,
        max_additional_leads=max(0, available_capacity)
    )


def get_scalable_channels(
    channel_metrics: Dict[str, ChannelMetrics],
    overall_demo_rate: float
) -> List[Tuple[str, ChannelMetrics]]:
    """
    Get channels that can be scaled (above average Demo Rate).
    
    Business Rule: Only scale channels above average Demo Rate.
    
    Args:
        channel_metrics: Dictionary of channel metrics
        overall_demo_rate: Average demo rate threshold
        
    Returns:
        List of (channel_name, metrics) tuples that can be scaled
    """
    if not ASSUMPTIONS.ONLY_SCALE_ABOVE_AVERAGE:
        return list(channel_metrics.items())
    
    scalable = [
        (channel, metrics)
        for channel, metrics in channel_metrics.items()
        if metrics.demo_rate > overall_demo_rate
    ]
    
    # Sort by demo rate (highest first)
    scalable.sort(key=lambda x: x[1].demo_rate, reverse=True)
    
    return scalable


def get_reducible_channels(
    channel_metrics: Dict[str, ChannelMetrics],
    overall_demo_rate: float
) -> List[Tuple[str, ChannelMetrics]]:
    """
    Get channels that can be reduced (below average Demo Rate).
    
    Business Rule: Reduction applies to below average channels.
    
    Args:
        channel_metrics: Dictionary of channel metrics
        overall_demo_rate: Average demo rate threshold
        
    Returns:
        List of (channel_name, metrics) tuples that can be reduced
    """
    if not ASSUMPTIONS.REDUCTION_APPLIES_TO_BELOW_AVERAGE:
        return list(channel_metrics.items())
    
    reducible = [
        (channel, metrics)
        for channel, metrics in channel_metrics.items()
        if metrics.demo_rate < overall_demo_rate
    ]
    
    # Sort by demo rate (lowest first - worst performers)
    reducible.sort(key=lambda x: x[1].demo_rate)
    
    return reducible
