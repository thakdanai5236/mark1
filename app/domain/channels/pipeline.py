"""
Channel Intelligence Pipeline
=============================
Main orchestration for PHASE 3 Channel Growth Intelligence.

Author: Analytics Engineering Team
Created: 2026-02-18
"""

import pandas as pd
from typing import Dict, Any, Optional

from domain.channels.constants import (
    DATA_DIR,
    GROWTH_LEAD_SHARE_THRESHOLD,
    CONTACT_TYPE_MIN_LEADS,
)
from domain.channels.calculations import (
    calculate_baseline,
    create_channel_rankings,
    identify_growth_channels,
    identify_bottleneck_channels,
    run_sensitivity_simulation,
    simulate_channel_mix_adjustment,
    analyze_contact_type_overlay,
)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_channel_growth_intelligence(
    marketing_master: pd.DataFrame,
    channel_summary: Optional[pd.DataFrame] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run the complete Channel Growth Intelligence pipeline (PHASE 3).
    
    Steps:
    3.1 - Baseline
    3.2 - Channel Rankings
    3.3 - Growth Lever Identification
    3.4 - Bottleneck Identification
    3.5 - Sensitivity Simulation
    3.6 - (Available via simulate_channel_mix_adjustment)
    3.7 - Contact Type Overlay
    
    Args:
        marketing_master: Raw marketing data with Channel, Demo_Flag, Contact_Type
        channel_summary: Optional pre-loaded channel summary (loads from CSV if None)
        verbose: Print progress information
        
    Returns:
        Dictionary with all analysis results
    """
    if verbose:
        print("=" * 60)
        print("  PHASE 3: CHANNEL GROWTH INTELLIGENCE")
        print("=" * 60)
    
    # Load channel summary if not provided
    if channel_summary is None:
        channel_path = DATA_DIR / "channel_performance_summary.csv"
        channel_summary = pd.read_csv(channel_path)
        if verbose:
            print(f"\n  Loaded channel summary from {channel_path}")
    
    # Step 3.1: Baseline
    if verbose:
        print("\n[STEP 3.1] Calculating baseline...")
    baseline = calculate_baseline(marketing_master)
    if verbose:
        print(f"  Total Leads: {baseline['total_leads']:,}")
        print(f"  Total Demo: {baseline['total_demo']:,}")
        print(f"  Overall Demo Rate: {baseline['overall_demo_rate']:.4f}")
    
    # Step 3.2: Channel Rankings
    if verbose:
        print("\n[STEP 3.2] Creating channel rankings...")
    rankings = create_channel_rankings(channel_summary)
    if verbose:
        print(f"  Rankings created: {list(rankings.keys())}")
    
    # Step 3.3: Growth Channels
    if verbose:
        print("\n[STEP 3.3] Identifying growth channels...")
        print(f"  Criteria: Demo_Rate > {baseline['overall_demo_rate']:.4f} AND Lead_Share > {GROWTH_LEAD_SHARE_THRESHOLD}")
    growth_channels = identify_growth_channels(channel_summary, baseline["overall_demo_rate"])
    if verbose:
        print(f"  Growth channels found: {len(growth_channels)}")
        for _, row in growth_channels.iterrows():
            print(f"    - {row['Channel']}: {row['Demo_Rate']:.4f} rate, {row['Lead_Share']:.4f} share")
    
    # Step 3.4: Bottleneck Channels
    if verbose:
        print("\n[STEP 3.4] Identifying bottleneck channels...")
        number_of_channels = len(channel_summary)
        avg_share = 1 / number_of_channels if number_of_channels > 0 else 0
        print(f"  Criteria: Lead_Share > {avg_share:.4f} AND Demo_Rate < {baseline['overall_demo_rate']:.4f}")
    bottleneck_channels = identify_bottleneck_channels(channel_summary, baseline["overall_demo_rate"])
    if verbose:
        print(f"  Bottleneck channels found: {len(bottleneck_channels)}")
        for _, row in bottleneck_channels.iterrows():
            print(f"    - {row['Channel']}: {row['Demo_Rate']:.4f} rate, {row['Lead_Share']:.4f} share")
    
    # Step 3.5: Sensitivity Simulation
    if verbose:
        print("\n[STEP 3.5] Running sensitivity simulation...")
    sensitivity_results = run_sensitivity_simulation(
        channel_summary, growth_channels, bottleneck_channels, baseline
    )
    if verbose:
        growth_count = len(sensitivity_results['scenario_growth']['results'])
        bottleneck_count = len(sensitivity_results['scenario_reduction']['results'])
        total_tradeoffs = (
            sensitivity_results['scenario_growth']['trade_off_count'] +
            sensitivity_results['scenario_reduction']['trade_off_count']
        )
        print(f"  Growth scenarios: {growth_count}")
        print(f"  Bottleneck scenarios: {bottleneck_count}")
        print(f"  Trade-off scenarios: {total_tradeoffs}")
        
        # Show growth impact with trade-off flag
        for result in sensitivity_results['scenario_growth']['results']:
            flag = " [!] TRADE-OFF" if result.trade_off_flag != "Aligned" else ""
            print(f"    {result.channel}: +{result.change_pct*100:.0f}% -> rate {result.rate_change:+.6f}, demo {result.demo_delta:+,.0f}{flag}")
        
        # Show bottleneck impact with trade-off flag
        for result in sensitivity_results['scenario_reduction']['results']:
            flag = " [!] TRADE-OFF" if result.trade_off_flag != "Aligned" else ""
            print(f"    {result.channel}: {result.change_pct*100:.0f}% -> rate {result.rate_change:+.6f}, demo {result.demo_delta:+,.0f}{flag}")
    
    # Step 3.7: Contact Type Overlay
    if verbose:
        print("\n[STEP 3.7] Analyzing contact type overlay...")
    contact_type_insights = analyze_contact_type_overlay(marketing_master)
    if verbose:
        print(f"  Top {len(contact_type_insights)} combinations (min {CONTACT_TYPE_MIN_LEADS} leads):")
        for _, row in contact_type_insights.iterrows():
            print(f"    - {row['Channel']} x {row['Contact_Type']}: {row['Demo_Rate']:.4f}")
    
    if verbose:
        print("\n" + "=" * 60)
        print("  PHASE 3 COMPLETE")
        print("=" * 60 + "\n")
    
    return {
        "baseline": baseline,
        "rankings": rankings,
        "growth_channels": growth_channels,
        "bottleneck_channels": bottleneck_channels,
        "sensitivity_results": sensitivity_results,
        "contact_type_insights": contact_type_insights,
    }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_marketing_master(
    path: str = "data/processed/lead_demo_clean.csv"
) -> pd.DataFrame:
    """
    Load marketing master data.
    
    Args:
        path: Path to CSV file
        
    Returns:
        DataFrame with Channel, Demo_Flag columns
    """
    df = pd.read_csv(path)
    
    # Rename columns to match expected schema
    df = df.rename(columns={
        "Contact_Type": "Channel",
        "is_demo": "Demo_Flag",
    })
    
    # Fill null channels
    if df["Channel"].isnull().any():
        df["Channel"] = df["Channel"].fillna("Unknown")
    
    # Normalize channel names
    df["Channel"] = df["Channel"].astype(str).str.strip().str.title()
    
    return df


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("Loading marketing master data...")
    
    try:
        # Load data
        marketing_master = load_marketing_master()
        print(f"Marketing master loaded: {len(marketing_master)} rows")
        
        # Run pipeline
        results = run_channel_growth_intelligence(
            marketing_master,
            verbose=True
        )
        
        # Display summary
        print("\n" + "=" * 60)
        print("  RESULTS SUMMARY")
        print("=" * 60)
        print(f"\nGrowth Channels ({len(results['growth_channels'])}):")
        if len(results['growth_channels']) > 0:
            print(results['growth_channels'][["Channel", "Demo_Rate", "Lead_Share", "Impact_Score"]].to_string(index=False))
        
        print(f"\nBottleneck Channels ({len(results['bottleneck_channels'])}):")
        if len(results['bottleneck_channels']) > 0:
            print(results['bottleneck_channels'][["Channel", "Demo_Rate", "Lead_Share", "Impact_Score"]].to_string(index=False))
        
        print(f"\nTop Contact Type Combinations:")
        if len(results['contact_type_insights']) > 0:
            print(results['contact_type_insights'].to_string(index=False))
        
        # Demo of channel mix adjustment
        print("\n" + "=" * 60)
        print("  DEMO: CHANNEL MIX ADJUSTMENT")
        print("=" * 60)
        
        channel_summary = pd.read_csv("data/processed/channel_performance_summary.csv")
        adjustment = simulate_channel_mix_adjustment(
            channel_summary,
            results['baseline'],
            {"Call In": 0.15, "Facebook": -0.10}
        )
        
        print(f"\nAdjustments: {adjustment['adjustments']}")
        print(f"Leads Change: {adjustment['impact']['leads_change']:+,}")
        print(f"Demo Change: {adjustment['impact']['demo_change']:+,}")
        print(f"Demo Delta: {adjustment['impact']['demo_delta']:+,}")
        print(f"Rate Change: {adjustment['impact']['rate_change']:+.6f}")
        print(f"Trade-Off Flag: {adjustment['impact']['trade_off_flag']}")
        
        # Show sensitivity trade-off summary
        print("\n" + "=" * 60)
        print("  SENSITIVITY TRADE-OFF SUMMARY")
        print("=" * 60)
        sens = results['sensitivity_results']
        total_tradeoffs = (
            sens['scenario_growth']['trade_off_count'] +
            sens['scenario_reduction']['trade_off_count']
        )
        print(f"\nGrowth Trade-Offs: {sens['scenario_growth']['trade_off_count']}")
        print(f"Reduction Trade-Offs: {sens['scenario_reduction']['trade_off_count']}")
        print(f"Total Trade-Off Scenarios: {total_tradeoffs}")
        print("(Trade-off = Demo Rate improves but Total Demo decreases)")
        
    except FileNotFoundError as e:
        print(f"Error: Required data file not found - {e}")
    except Exception as e:
        print(f"Error: {e}")
        raise
