"""
Channel Intelligence Calculations
=================================
Step-by-step calculation functions for PHASE 3.

All functions are deterministic.
NO AI/LLM logic.
NO business logic modifications.

Author: Analytics Engineering Team
Created: 2026-02-18
"""

import pandas as pd
from typing import Dict, List, Any

from core.channel.constants import (
    GROWTH_LEAD_SHARE_THRESHOLD,
    CONTACT_TYPE_MIN_LEADS,
    GROWTH_INCREASE_PCT,
    BOTTLENECK_DECREASE_PCT,
    SensitivityResult,
)


# =============================================================================
# STEP 3.1 — BASELINE
# =============================================================================

def calculate_baseline(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Recompute baseline metrics from marketing_master.
    
    Must match Phase 2 baseline.
    
    Args:
        df: marketing_master DataFrame with Customer_Code, Demo_Flag
        
    Returns:
        Dictionary with total_leads, total_demo, overall_demo_rate
    """
    total_leads = len(df)
    total_demo = int(df["Demo_Flag"].sum())
    
    # Handle zero division safely
    if total_leads == 0:
        overall_demo_rate = 0.0
    else:
        overall_demo_rate = total_demo / total_leads
    
    return {
        "total_leads": total_leads,
        "total_demo": total_demo,
        "overall_demo_rate": float(overall_demo_rate),
    }


# =============================================================================
# STEP 3.2 — CHANNEL RANKING
# =============================================================================

def create_channel_rankings(channel_summary: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Create channel rankings by different metrics.
    
    Rankings:
    1) By Demo_Rate (descending)
    2) By Impact_Score (descending)
    3) By Demo_Contribution (descending)
    
    Args:
        channel_summary: channel_performance_summary DataFrame
        
    Returns:
        Dictionary with rankings DataFrames
    """
    # Ranking by Demo_Rate
    rank_by_demo_rate = channel_summary.copy()
    rank_by_demo_rate["Rank_Demo_Rate"] = (
        rank_by_demo_rate["Demo_Rate"]
        .rank(ascending=False, method="min")
        .astype(int)
    )
    rank_by_demo_rate = rank_by_demo_rate.sort_values("Rank_Demo_Rate")
    
    # Ranking by Impact_Score
    rank_by_impact = channel_summary.copy()
    rank_by_impact["Rank_Impact_Score"] = (
        rank_by_impact["Impact_Score"]
        .rank(ascending=False, method="min")
        .astype(int)
    )
    rank_by_impact = rank_by_impact.sort_values("Rank_Impact_Score")
    
    # Ranking by Demo_Contribution
    rank_by_contribution = channel_summary.copy()
    rank_by_contribution["Rank_Demo_Contribution"] = (
        rank_by_contribution["Demo_Contribution"]
        .rank(ascending=False, method="min")
        .astype(int)
    )
    rank_by_contribution = rank_by_contribution.sort_values("Rank_Demo_Contribution")
    
    return {
        "by_demo_rate": rank_by_demo_rate[["Channel", "Demo_Rate", "Rank_Demo_Rate"]].reset_index(drop=True),
        "by_impact_score": rank_by_impact[["Channel", "Impact_Score", "Rank_Impact_Score"]].reset_index(drop=True),
        "by_demo_contribution": rank_by_contribution[["Channel", "Demo_Contribution", "Rank_Demo_Contribution"]].reset_index(drop=True),
    }


# =============================================================================
# STEP 3.3 — IDENTIFY GROWTH LEVER
# =============================================================================

def identify_growth_channels(
    channel_summary: pd.DataFrame,
    overall_demo_rate: float
) -> pd.DataFrame:
    """
    Identify growth lever channels.
    
    Fixed criteria:
    - Demo_Rate > overall_demo_rate
    - Lead_Share > 0.10 (FIXED threshold)
    
    DO NOT compute percentile.
    DO NOT dynamically change threshold.
    
    Args:
        channel_summary: channel_performance_summary DataFrame
        overall_demo_rate: Overall demo rate from baseline
        
    Returns:
        DataFrame of growth channels
    """
    growth_mask = (
        (channel_summary["Demo_Rate"] > overall_demo_rate) &
        (channel_summary["Lead_Share"] > GROWTH_LEAD_SHARE_THRESHOLD)
    )
    
    growth_channels = channel_summary[growth_mask].copy()
    growth_channels = growth_channels.sort_values("Demo_Rate", ascending=False)
    growth_channels = growth_channels.reset_index(drop=True)
    
    return growth_channels


# =============================================================================
# STEP 3.4 — IDENTIFY BOTTLENECK
# =============================================================================

def identify_bottleneck_channels(
    channel_summary: pd.DataFrame,
    overall_demo_rate: float
) -> pd.DataFrame:
    """
    Identify bottleneck channels.
    
    Fixed criteria:
    - Lead_Share > (1 / number_of_channels)
    - Demo_Rate < overall_demo_rate
    
    Args:
        channel_summary: channel_performance_summary DataFrame
        overall_demo_rate: Overall demo rate from baseline
        
    Returns:
        DataFrame of bottleneck channels
    """
    number_of_channels = len(channel_summary)
    if number_of_channels > 0:
        average_lead_share = 1 / number_of_channels
    else:
        average_lead_share = 0.0
    
    bottleneck_mask = (
        (channel_summary["Lead_Share"] > average_lead_share) &
        (channel_summary["Demo_Rate"] < overall_demo_rate)
    )
    
    bottleneck_channels = channel_summary[bottleneck_mask].copy()
    bottleneck_channels = bottleneck_channels.sort_values("Lead_Share", ascending=False)
    bottleneck_channels = bottleneck_channels.reset_index(drop=True)
    
    return bottleneck_channels


# =============================================================================
# STEP 3.5 — SENSITIVITY SIMULATION
# =============================================================================

def simulate_growth_scenario(
    channel_summary: pd.DataFrame,
    growth_channels: pd.DataFrame,
    baseline: Dict[str, Any]
) -> List[SensitivityResult]:
    """
    Scenario A: Simulate +10% leads increase for each growth channel.
    
    For each Growth Channel:
    - Leads_new = Leads × 1.10
    - Demo_new = Leads_new × Demo_Rate
    - Other channels remain unchanged
    
    Computes BOTH:
    - Demo Rate change (demo_rate_delta)
    - Total Demo change (demo_delta)
    
    Args:
        channel_summary: Full channel summary
        growth_channels: Growth channels to simulate
        baseline: Baseline metrics
        
    Returns:
        List of SensitivityResult for each growth channel
    """
    results = []
    
    for _, row in growth_channels.iterrows():
        channel = row["Channel"]
        original_leads = int(row["Leads"])
        demo_rate = float(row["Demo_Rate"])
        original_demo = int(row["Demo"])
        
        # Simulate +10% increase
        new_leads = int(original_leads * (1 + GROWTH_INCREASE_PCT))
        new_channel_demo = int(new_leads * demo_rate)
        
        # Calculate new totals (other channels unchanged)
        leads_change = new_leads - original_leads
        demo_change = new_channel_demo - original_demo
        
        new_total_leads = baseline["total_leads"] + leads_change
        new_total_demo = baseline["total_demo"] + demo_change
        
        # Calculate new overall demo rate
        if new_total_leads > 0:
            new_overall_rate = new_total_demo / new_total_leads
        else:
            new_overall_rate = 0.0
        
        rate_change = new_overall_rate - baseline["overall_demo_rate"]
        demo_delta = new_total_demo - baseline["total_demo"]
        
        # Determine trade-off flag
        if rate_change > 0 and demo_delta < 0:
            trade_off_flag = "Rate Optimization Trade-off"
        elif rate_change < 0 and demo_delta > 0:
            trade_off_flag = "Volume vs Rate Trade-off"
        else:
            trade_off_flag = "Aligned"
        
        results.append(SensitivityResult(
            channel=channel,
            scenario_type="growth",
            change_pct=GROWTH_INCREASE_PCT,
            original_leads=original_leads,
            new_leads=new_leads,
            original_demo=original_demo,
            new_demo=new_channel_demo,
            original_overall_rate=baseline["overall_demo_rate"],
            new_overall_rate=float(new_overall_rate),
            rate_change=float(rate_change),
            new_total_leads=new_total_leads,
            new_total_demo=new_total_demo,
            demo_delta=demo_delta,
            demo_rate_delta=float(rate_change),
            trade_off_flag=trade_off_flag,
        ))
    
    return results


def simulate_bottleneck_scenario(
    channel_summary: pd.DataFrame,
    bottleneck_channels: pd.DataFrame,
    baseline: Dict[str, Any]
) -> List[SensitivityResult]:
    """
    Scenario B: Simulate -20% leads reduction for each bottleneck channel.
    
    For each Bottleneck Channel:
    - Leads_new = Leads × 0.80
    - Other channels remain unchanged
    
    Computes BOTH:
    - Demo Rate change (demo_rate_delta)
    - Total Demo change (demo_delta)
    
    Args:
        channel_summary: Full channel summary
        bottleneck_channels: Bottleneck channels to simulate
        baseline: Baseline metrics
        
    Returns:
        List of SensitivityResult for each bottleneck channel
    """
    results = []
    
    for _, row in bottleneck_channels.iterrows():
        channel = row["Channel"]
        original_leads = int(row["Leads"])
        demo_rate = float(row["Demo_Rate"])
        original_demo = int(row["Demo"])
        
        # Simulate -20% reduction
        new_leads = int(original_leads * (1 - BOTTLENECK_DECREASE_PCT))
        new_channel_demo = int(new_leads * demo_rate)
        
        # Calculate new totals (other channels unchanged)
        leads_change = new_leads - original_leads
        demo_change = new_channel_demo - original_demo
        
        new_total_leads = baseline["total_leads"] + leads_change
        new_total_demo = baseline["total_demo"] + demo_change
        
        # Calculate new overall demo rate
        if new_total_leads > 0:
            new_overall_rate = new_total_demo / new_total_leads
        else:
            new_overall_rate = 0.0
        
        rate_change = new_overall_rate - baseline["overall_demo_rate"]
        demo_delta = new_total_demo - baseline["total_demo"]
        
        # Determine trade-off flag
        if rate_change > 0 and demo_delta < 0:
            trade_off_flag = "Rate Optimization Trade-off"
        elif rate_change < 0 and demo_delta > 0:
            trade_off_flag = "Volume vs Rate Trade-off"
        else:
            trade_off_flag = "Aligned"
        
        results.append(SensitivityResult(
            channel=channel,
            scenario_type="bottleneck",
            change_pct=-BOTTLENECK_DECREASE_PCT,
            original_leads=original_leads,
            new_leads=new_leads,
            original_demo=original_demo,
            new_demo=new_channel_demo,
            original_overall_rate=baseline["overall_demo_rate"],
            new_overall_rate=float(new_overall_rate),
            rate_change=float(rate_change),
            new_total_leads=new_total_leads,
            new_total_demo=new_total_demo,
            demo_delta=demo_delta,
            demo_rate_delta=float(rate_change),
            trade_off_flag=trade_off_flag,
        ))
    
    return results


def run_sensitivity_simulation(
    channel_summary: pd.DataFrame,
    growth_channels: pd.DataFrame,
    bottleneck_channels: pd.DataFrame,
    baseline: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run both sensitivity scenarios.
    
    For each scenario, computes BOTH:
    - Demo Rate change (demo_rate_delta)
    - Total Demo change (demo_delta)
    
    Flags trade-offs when rate and volume move in opposite directions.
    
    Args:
        channel_summary: Full channel summary
        growth_channels: Growth channels
        bottleneck_channels: Bottleneck channels
        baseline: Baseline metrics
        
    Returns:
        Dictionary with growth and bottleneck simulation results
    """
    growth_results = simulate_growth_scenario(
        channel_summary, growth_channels, baseline
    )
    
    bottleneck_results = simulate_bottleneck_scenario(
        channel_summary, bottleneck_channels, baseline
    )
    
    # Convert to DataFrames for easier consumption
    growth_df = pd.DataFrame([vars(r) for r in growth_results]) if growth_results else pd.DataFrame()
    bottleneck_df = pd.DataFrame([vars(r) for r in bottleneck_results]) if bottleneck_results else pd.DataFrame()
    
    # Count trade-offs
    growth_tradeoffs = sum(1 for r in growth_results if r.trade_off_flag != "Aligned") if growth_results else 0
    bottleneck_tradeoffs = sum(1 for r in bottleneck_results if r.trade_off_flag != "Aligned") if bottleneck_results else 0
    
    return {
        "scenario_growth": {
            "description": f"+{GROWTH_INCREASE_PCT*100:.0f}% leads increase for growth channels",
            "results": growth_results,
            "summary": growth_df,
            "trade_off_count": growth_tradeoffs,
        },
        "scenario_reduction": {
            "description": f"-{BOTTLENECK_DECREASE_PCT*100:.0f}% leads reduction for bottleneck channels",
            "results": bottleneck_results,
            "summary": bottleneck_df,
            "trade_off_count": bottleneck_tradeoffs,
        },
    }


# =============================================================================
# STEP 3.6 — CHANNEL MIX ADJUSTMENT
# =============================================================================

def simulate_channel_mix_adjustment(
    channel_summary: pd.DataFrame,
    baseline: Dict[str, Any],
    adjustment_dict: Dict[str, float]
) -> Dict[str, Any]:
    """
    Simulate custom channel mix adjustments.
    
    adjustment_dict format:
    {"Channel A": +0.15, "Channel B": -0.10}
    
    Interpret as % change in leads.
    Apply independently.
    DO NOT normalize total leads.
    DO NOT enforce capacity constraint.
    
    Args:
        channel_summary: channel_performance_summary DataFrame
        baseline: Baseline metrics
        adjustment_dict: Dictionary of channel -> adjustment percentage
        
    Returns:
        Dictionary with simulation results
    """
    # Start with baseline totals
    total_leads_change = 0
    total_demo_change = 0
    
    channel_details = []
    
    for channel, adjustment_pct in adjustment_dict.items():
        # Find channel in summary
        channel_data = channel_summary[channel_summary["Channel"] == channel]
        
        if len(channel_data) == 0:
            channel_details.append({
                "channel": channel,
                "status": "not_found",
                "adjustment_pct": adjustment_pct,
                "leads_change": 0,
                "demo_change": 0,
            })
            continue
        
        row = channel_data.iloc[0]
        original_leads = int(row["Leads"])
        demo_rate = float(row["Demo_Rate"])
        original_demo = int(row["Demo"])
        
        # Apply adjustment
        new_leads = int(original_leads * (1 + adjustment_pct))
        new_demo = int(new_leads * demo_rate)
        
        leads_change = new_leads - original_leads
        demo_change = new_demo - original_demo
        
        total_leads_change += leads_change
        total_demo_change += demo_change
        
        channel_details.append({
            "channel": channel,
            "status": "adjusted",
            "adjustment_pct": float(adjustment_pct),
            "original_leads": original_leads,
            "new_leads": new_leads,
            "leads_change": leads_change,
            "original_demo": original_demo,
            "new_demo": new_demo,
            "demo_change": demo_change,
            "demo_rate": demo_rate,
        })
    
    # Calculate new totals
    new_total_leads = baseline["total_leads"] + total_leads_change
    new_total_demo = baseline["total_demo"] + total_demo_change
    
    # Calculate new overall demo rate
    if new_total_leads > 0:
        new_overall_rate = new_total_demo / new_total_leads
    else:
        new_overall_rate = 0.0
    
    rate_change = new_overall_rate - baseline["overall_demo_rate"]
    demo_delta = new_total_demo - baseline["total_demo"]
    
    # Determine trade-off flag
    if rate_change > 0 and demo_delta < 0:
        trade_off_flag = "Rate Optimization Trade-off"
    elif rate_change < 0 and demo_delta > 0:
        trade_off_flag = "Volume vs Rate Trade-off"
    else:
        trade_off_flag = "Aligned"
    
    return {
        "adjustments": adjustment_dict,
        "channel_details": pd.DataFrame(channel_details),
        "original_baseline": baseline,
        "new_totals": {
            "total_leads": new_total_leads,
            "total_demo": new_total_demo,
            "overall_demo_rate": float(new_overall_rate),
        },
        "impact": {
            "leads_change": total_leads_change,
            "demo_change": total_demo_change,
            "demo_delta": demo_delta,
            "rate_change": float(rate_change),
            "demo_rate_delta": float(rate_change),
            "trade_off_flag": trade_off_flag,
        },
    }


# =============================================================================
# STEP 3.7 — CONTACT_TYPE OVERLAY
# =============================================================================

def analyze_contact_type_overlay(
    df: pd.DataFrame,
    min_leads: int = CONTACT_TYPE_MIN_LEADS,
    top_n: int = 3
) -> pd.DataFrame:
    """
    Analyze Contact_Type within each Channel.
    
    Groups by Channel, Contact_Type.
    Computes Leads, Demo, Demo_Rate.
    Filters by minimum leads threshold.
    Returns top N combinations by Demo_Rate.
    
    Args:
        df: marketing_master DataFrame with Channel, Contact_Type, Demo_Flag
        min_leads: Minimum leads threshold (default: 20)
        top_n: Number of top combinations to return (default: 3)
        
    Returns:
        DataFrame with top contact type combinations
    """
    # Check if Contact_Type column exists
    if "Contact_Type" not in df.columns:
        return pd.DataFrame(columns=[
            "Channel", "Contact_Type", "Leads", "Demo", "Demo_Rate"
        ])
    
    # Group by Channel and Contact_Type
    overlay = df.groupby(["Channel", "Contact_Type"]).agg(
        Leads=("Customer_Code", "count"),
        Demo=("Demo_Flag", "sum")
    ).reset_index()
    
    # Calculate Demo_Rate (safe division)
    overlay["Demo_Rate"] = overlay.apply(
        lambda row: row["Demo"] / row["Leads"] if row["Leads"] > 0 else 0.0,
        axis=1
    ).astype(float)
    
    # Filter by minimum leads
    overlay_filtered = overlay[overlay["Leads"] >= min_leads].copy()
    
    # Sort by Demo_Rate descending and take top N
    overlay_filtered = overlay_filtered.sort_values(
        "Demo_Rate", ascending=False
    ).head(top_n).reset_index(drop=True)
    
    return overlay_filtered
