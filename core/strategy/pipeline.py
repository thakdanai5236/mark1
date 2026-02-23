"""
Strategy Pipeline
=================
Orchestrates the locked thinking order for marketing strategy analysis.

ENGINE THINKING ORDER (HARD-CODED):
1) Calculate overall Demo Rate
2) Analyze channel Demo Rate  
3) Rank channels by conversion
4) Calculate channel impact
5) Simulate reallocation
6) Check capacity constraint

Author: Backend Engineering Team
Created: 2026-02-18
"""

import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from core.strategy.business_rules import (
    OBJECTIVE,
    THINKING_ORDER,
    CORE_BUSINESS_QUESTIONS,
)
from core.strategy.calculations import (
    OverallMetrics,
    ChannelMetrics,
    ChannelImpact,
    SimulationResult,
    CapacityCheck,
    calculate_overall_demo_rate,
    calculate_channel_metrics,
    rank_channels,
    calculate_channel_impact,
    simulate_increase,
    simulate_reduction,
    check_capacity,
    get_highest_demo_rate_channel,
    get_high_volume_low_conversion_channels,
    get_scalable_channels,
    get_reducible_channels,
)


# =============================================================================
# PIPELINE RESULT CONTAINER
# =============================================================================

@dataclass
class StrategyPipelineResult:
    """Container for all pipeline results."""
    
    # Step 1: Overall metrics
    overall_metrics: OverallMetrics = None
    
    # Step 2: Channel metrics
    channel_metrics: Dict[str, ChannelMetrics] = field(default_factory=dict)
    
    # Step 3: Channel rankings
    channels_ranked_by_conversion: List[Tuple[str, ChannelMetrics]] = field(default_factory=list)
    highest_demo_rate_channel: Tuple[str, ChannelMetrics] = None
    high_volume_low_conversion: List[Tuple[str, ChannelMetrics]] = field(default_factory=list)
    
    # Step 4: Channel impact
    channel_impacts: Dict[str, ChannelImpact] = field(default_factory=dict)
    
    # Step 5: Simulations
    increase_simulations: List[SimulationResult] = field(default_factory=list)
    reduction_simulations: List[SimulationResult] = field(default_factory=list)
    
    # Step 6: Capacity
    capacity_check: CapacityCheck = None
    scalable_channels: List[Tuple[str, ChannelMetrics]] = field(default_factory=list)
    reducible_channels: List[Tuple[str, ChannelMetrics]] = field(default_factory=list)
    
    # Business question answers
    business_answers: Dict[str, str] = field(default_factory=dict)


# =============================================================================
# STRATEGY PIPELINE
# =============================================================================

def run_strategy_pipeline(
    df: pd.DataFrame,
    number_of_sales: int = 5,
    increase_pct_low: float = None,
    increase_pct_high: float = None,
    reduction_pct: float = None,
    verbose: bool = True
) -> StrategyPipelineResult:
    """
    Run the complete strategy analysis pipeline.
    
    MUST follow the locked thinking order:
    1) Calculate overall Demo Rate
    2) Analyze channel Demo Rate
    3) Rank channels by conversion
    4) Calculate channel impact
    5) Simulate reallocation
    6) Check capacity constraint
    
    Args:
        df: DataFrame with columns [Customer_Code, Contact_Type, is_demo]
        number_of_sales: Number of sales personnel for capacity calculation
        increase_pct_low: Low bound for increase simulation (default: 10%)
        increase_pct_high: High bound for increase simulation (default: 20%)
        reduction_pct: Reduction percentage for simulation (default: 20%)
        verbose: Print progress and results
        
    Returns:
        StrategyPipelineResult with all analysis results
    """
    # Use defaults from business rules if not specified
    increase_pct_low = increase_pct_low or OBJECTIVE.MIN_SCALE_INCREASE_PCT
    increase_pct_high = increase_pct_high or OBJECTIVE.MAX_SCALE_INCREASE_PCT
    reduction_pct = reduction_pct or OBJECTIVE.DEFAULT_REDUCTION_PCT
    
    result = StrategyPipelineResult()
    
    if verbose:
        print("=" * 70)
        print("  STRATEGY PIPELINE - LOCKED THINKING ORDER")
        print("=" * 70)
    
    # =========================================================================
    # STEP 1: Calculate overall Demo Rate
    # =========================================================================
    if verbose:
        print(f"\n[STEP 1] {THINKING_ORDER.STEP_1}")
        print("-" * 50)
    
    result.overall_metrics = calculate_overall_demo_rate(df)
    
    if verbose:
        print(f"  Total Leads: {result.overall_metrics.total_leads:,}")
        print(f"  Total Demos: {result.overall_metrics.total_demos:,}")
        print(f"  Demo Rate: {result.overall_metrics.demo_rate_pct}%")
    
    # =========================================================================
    # STEP 2: Analyze channel Demo Rate
    # =========================================================================
    if verbose:
        print(f"\n[STEP 2] {THINKING_ORDER.STEP_2}")
        print("-" * 50)
    
    result.channel_metrics = calculate_channel_metrics(df, result.overall_metrics)
    
    if verbose:
        print(f"  Channels analyzed: {len(result.channel_metrics)}")
        above_avg = sum(1 for m in result.channel_metrics.values() if m.is_above_average)
        below_avg = len(result.channel_metrics) - above_avg
        print(f"  Above average: {above_avg} channels")
        print(f"  Below average: {below_avg} channels")
    
    # =========================================================================
    # STEP 3: Rank channels by conversion
    # =========================================================================
    if verbose:
        print(f"\n[STEP 3] {THINKING_ORDER.STEP_3}")
        print("-" * 50)
    
    result.channels_ranked_by_conversion = rank_channels(
        result.channel_metrics, 
        sort_by="demo_rate"
    )
    
    result.highest_demo_rate_channel = get_highest_demo_rate_channel(
        result.channel_metrics
    )
    
    result.high_volume_low_conversion = get_high_volume_low_conversion_channels(
        result.channel_metrics,
        result.overall_metrics.demo_rate
    )
    
    if verbose:
        print("\n  Channel Rankings by Demo Rate:")
        for i, (channel, metrics) in enumerate(result.channels_ranked_by_conversion, 1):
            flag = "[+]" if metrics.is_above_average else "[-]"
            print(f"    {i}. {channel}: {metrics.demo_rate_pct}% {flag}")
        
        if result.highest_demo_rate_channel[0]:
            ch, m = result.highest_demo_rate_channel
            print(f"\n  Highest Demo Rate: {ch} ({m.demo_rate_pct}%)")
        
        if result.high_volume_low_conversion:
            print("\n  High Volume, Low Conversion:")
            for ch, m in result.high_volume_low_conversion:
                print(f"    - {ch}: {m.lead_share_pct}% leads, {m.demo_rate_pct}% rate")
    
    # =========================================================================
    # STEP 4: Calculate channel impact
    # =========================================================================
    if verbose:
        print(f"\n[STEP 4] {THINKING_ORDER.STEP_4}")
        print("-" * 50)
    
    result.channel_impacts = calculate_channel_impact(result.channel_metrics)
    
    if verbose:
        print("\n  Channel Impact Scores (Demo Rate x Lead Share):")
        sorted_impacts = sorted(
            result.channel_impacts.items(),
            key=lambda x: x[1].impact_score,
            reverse=True
        )
        for channel, impact in sorted_impacts:
            print(f"    #{impact.rank} {channel}: {impact.impact_score:.4f}")
    
    # =========================================================================
    # STEP 5: Simulate reallocation
    # =========================================================================
    if verbose:
        print(f"\n[STEP 5] {THINKING_ORDER.STEP_5}")
        print("-" * 50)
    
    # Get scalable and reducible channels
    result.scalable_channels = get_scalable_channels(
        result.channel_metrics,
        result.overall_metrics.demo_rate
    )
    
    result.reducible_channels = get_reducible_channels(
        result.channel_metrics,
        result.overall_metrics.demo_rate
    )
    
    # Simulate increases for scalable channels
    if verbose:
        print(f"\n  Simulating {increase_pct_low*100:.0f}%-{increase_pct_high*100:.0f}% increase on above-average channels:")
    
    for channel, _ in result.scalable_channels:
        # Simulate low bound
        sim_low = simulate_increase(
            channel,
            result.channel_metrics,
            result.overall_metrics,
            increase_pct_low
        )
        result.increase_simulations.append(sim_low)
        
        # Simulate high bound
        sim_high = simulate_increase(
            channel,
            result.channel_metrics,
            result.overall_metrics,
            increase_pct_high
        )
        result.increase_simulations.append(sim_high)
        
        if verbose:
            print(f"\n    {channel}:")
            print(f"      +{increase_pct_low*100:.0f}%: {sim_low.current_leads:,} -> {sim_low.projected_leads:,} leads, +{sim_low.demo_change} demos")
            print(f"      +{increase_pct_high*100:.0f}%: {sim_high.current_leads:,} -> {sim_high.projected_leads:,} leads, +{sim_high.demo_change} demos")
    
    # Simulate reductions for reducible channels
    if verbose:
        print(f"\n  Simulating {reduction_pct*100:.0f}% reduction on below-average channels:")
    
    for channel, _ in result.reducible_channels:
        sim = simulate_reduction(
            channel,
            result.channel_metrics,
            result.overall_metrics,
            reduction_pct
        )
        result.reduction_simulations.append(sim)
        
        if verbose:
            rate_change = (sim.projected_overall_demo_rate - result.overall_metrics.demo_rate) * 100
            direction = "+" if rate_change > 0 else ""
            print(f"    {channel}: {sim.current_leads:,} -> {sim.projected_leads:,} leads, overall rate {direction}{rate_change:.2f}%")
    
    # =========================================================================
    # STEP 6: Check capacity constraint
    # =========================================================================
    if verbose:
        print(f"\n[STEP 6] {THINKING_ORDER.STEP_6}")
        print("-" * 50)
    
    result.capacity_check = check_capacity(
        result.overall_metrics.total_leads,
        number_of_sales
    )
    
    if verbose:
        cap = result.capacity_check
        print(f"  Sales Team: {number_of_sales} persons")
        print(f"  Capacity Formula: 10 x {number_of_sales} x 22 = {cap.capacity_limit:,}")
        print(f"  Current Leads: {cap.current_leads:,}")
        print(f"  Utilization: {cap.utilization_pct}%")
        print(f"  Available Capacity: {cap.available_capacity:,} leads")
        print(f"  Can Increase: {'Yes' if cap.can_increase else 'No'}")
    
    # =========================================================================
    # Generate Business Answers
    # =========================================================================
    result.business_answers = _generate_business_answers(result)
    
    if verbose:
        print("\n" + "=" * 70)
        print("  BUSINESS QUESTIONS ANSWERED")
        print("=" * 70)
        for q, a in result.business_answers.items():
            print(f"\n  Q: {q}")
            print(f"  A: {a}")
    
    if verbose:
        print("\n" + "=" * 70)
        print("  PIPELINE COMPLETE")
        print("=" * 70 + "\n")
    
    return result


def _generate_business_answers(result: StrategyPipelineResult) -> Dict[str, str]:
    """
    Generate answers to core business questions.
    
    Questions:
    1) Highest Demo Rate channel?
    2) High volume but low conversion?
    3) If increase high-conversion channel by 10–20%, projected demo?
    4) If reduce low-conversion channel by 20%, effect on overall Demo Rate?
    
    Args:
        result: Pipeline result with all metrics
        
    Returns:
        Dictionary of question -> answer
    """
    answers = {}
    
    # Q1: Highest Demo Rate channel
    q1 = CORE_BUSINESS_QUESTIONS[0]
    if result.highest_demo_rate_channel[0]:
        ch, m = result.highest_demo_rate_channel
        answers[q1] = f"{ch} with {m.demo_rate_pct}% demo rate ({m.demos} demos from {m.leads} leads)"
    else:
        answers[q1] = "No data available"
    
    # Q2: High volume but low conversion
    q2 = CORE_BUSINESS_QUESTIONS[1]
    if result.high_volume_low_conversion:
        channels = [
            f"{ch} ({m.lead_share_pct}% leads, {m.demo_rate_pct}% rate)"
            for ch, m in result.high_volume_low_conversion
        ]
        answers[q2] = "; ".join(channels)
    else:
        answers[q2] = "No channels with high volume and low conversion found"
    
    # Q3: Projected demo from increase
    q3 = CORE_BUSINESS_QUESTIONS[2]
    if result.increase_simulations:
        projections = []
        for sim in result.increase_simulations:
            if sim.change_pct == OBJECTIVE.MAX_SCALE_INCREASE_PCT:  # Use 20% for answer
                projections.append(
                    f"{sim.channel} +{sim.change_pct*100:.0f}%: +{sim.demo_change} demos"
                )
        answers[q3] = "; ".join(projections) if projections else "No simulations available"
    else:
        answers[q3] = "No above-average channels to scale"
    
    # Q4: Effect of reduction on overall rate
    q4 = CORE_BUSINESS_QUESTIONS[3]
    if result.reduction_simulations:
        effects = []
        for sim in result.reduction_simulations:
            rate_change = (sim.projected_overall_demo_rate - result.overall_metrics.demo_rate) * 100
            direction = "+" if rate_change > 0 else ""
            effects.append(
                f"{sim.channel}: overall rate {direction}{rate_change:.2f}%"
            )
        answers[q4] = "; ".join(effects)
    else:
        answers[q4] = "No below-average channels to reduce"
    
    return answers


def get_recommendations(result: StrategyPipelineResult) -> List[str]:
    """
    Generate actionable recommendations based on pipeline results.
    
    Args:
        result: Pipeline result with all metrics
        
    Returns:
        List of recommendation strings
    """
    recommendations = []
    
    # Check capacity first
    if not result.capacity_check.can_increase:
        recommendations.append(
            f"[!] CAPACITY LIMIT REACHED: Cannot increase leads beyond {result.capacity_check.capacity_limit:,}"
        )
        return recommendations
    
    # Recommend scaling high-conversion channels
    for channel, metrics in result.scalable_channels[:3]:  # Top 3
        increase_capacity = min(
            int(metrics.leads * OBJECTIVE.MAX_SCALE_INCREASE_PCT),
            result.capacity_check.available_capacity
        )
        if increase_capacity > 0:
            projected_demos = int(increase_capacity * metrics.demo_rate)
            recommendations.append(
                f"[+] SCALE {channel}: +{increase_capacity:,} leads -> +{projected_demos} projected demos "
                f"(Demo Rate: {metrics.demo_rate_pct}%)"
            )
    
    # Recommend reducing low-conversion channels
    for channel, metrics in result.reducible_channels[:2]:  # Bottom 2
        reduction = int(metrics.leads * OBJECTIVE.DEFAULT_REDUCTION_PCT)
        lost_demos = int(reduction * metrics.demo_rate)
        recommendations.append(
            f"[-] REDUCE {channel}: -{reduction:,} leads -> -{lost_demos} demos "
            f"(Low Rate: {metrics.demo_rate_pct}%)"
        )
    
    return recommendations


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("Loading marketing master data...")
    
    try:
        # Load data
        df = pd.read_csv("data/processed/lead_demo_clean.csv")
        print(f"Data loaded: {len(df)} rows")
        
        # Run pipeline
        result = run_strategy_pipeline(df, number_of_sales=5, verbose=True)
        
        # Show recommendations
        print("\n" + "=" * 70)
        print("  RECOMMENDATIONS")
        print("=" * 70)
        for rec in get_recommendations(result):
            print(f"\n  {rec}")
        
    except FileNotFoundError as e:
        print(f"Error: Required data file not found - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        raise
