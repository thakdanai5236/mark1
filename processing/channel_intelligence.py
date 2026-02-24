"""
Channel Intelligence Data Preparation - PHASE 2
================================================
Deterministic data preparation for channel analysis.

NO AI/LLM logic.
NO business logic modifications.
Pandas only.

Author: Data Engineering Team
Created: 2026-02-18
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple


# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = Path("data/processed")

# Required columns
REQUIRED_COLUMNS = [
    "Customer_Code",
    "Channel",
    "Demo_Flag",
]

OPTIONAL_COLUMNS = [
    "Contact_Type",
    "Lead_Create_Time",
    "Demo_Date",
]


# =============================================================================
# VALIDATION (MANDATORY RULES)
# =============================================================================

def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate input data according to global validation rules.
    
    Rules:
    1) Customer_Code must be unique
    2) Channel must not contain null values
    3) Demo_Flag must only contain {0, 1}
    4) Normalize Channel: strip() and title()
    
    Args:
        df: Input DataFrame
        
    Returns:
        Validated and normalized DataFrame
        
    Raises:
        ValueError: If validation fails
    """
    df = df.copy()
    
    # Rule 1: Customer_Code must be unique
    if df["Customer_Code"].duplicated().any():
        duplicate_count = df["Customer_Code"].duplicated().sum()
        raise ValueError(
            f"Customer_Code is not unique. Found {duplicate_count} duplicate entries."
        )
    
    # Rule 2: Channel must not contain null values
    if df["Channel"].isnull().any():
        null_count = df["Channel"].isnull().sum()
        raise ValueError(
            f"Channel contains {null_count} null values. Null values are not allowed."
        )
    
    # Rule 3: Demo_Flag must only contain {0, 1}
    valid_flags = {0, 1}
    invalid_flags = set(df["Demo_Flag"].unique()) - valid_flags
    if invalid_flags:
        raise ValueError(
            f"Demo_Flag contains invalid values: {invalid_flags}. "
            f"Only {{0, 1}} are allowed."
        )
    
    # Rule 4: Normalize Channel - strip() and title()
    df["Channel"] = df["Channel"].astype(str).str.strip().str.title()
    
    return df


# =============================================================================
# STEP 2.1 — BASELINE SNAPSHOT
# =============================================================================

def calculate_baseline_snapshot(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute baseline metrics from validated data.
    
    Metrics:
    - total_leads: count(Customer_Code)
    - total_demo: sum(Demo_Flag)
    - overall_demo_rate: total_demo / total_leads
    
    Args:
        df: Validated DataFrame
        
    Returns:
        Dictionary with baseline metrics
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
# STEP 2.2 — CHANNEL AGGREGATION
# =============================================================================

def build_channel_summary(
    df: pd.DataFrame,
    baseline: Dict[str, float]
) -> pd.DataFrame:
    """
    Build channel performance summary with aggregation.
    
    Metrics per channel:
    - Leads: count(Customer_Code)
    - Demo: sum(Demo_Flag)
    - Lead_Share: Leads / total_leads
    - Demo_Rate: Demo / Leads
    - Demo_Contribution: Demo / total_demo
    
    Args:
        df: Validated DataFrame
        baseline: Baseline metrics from calculate_baseline_snapshot()
        
    Returns:
        Channel summary DataFrame
    """
    total_leads = baseline["total_leads"]
    total_demo = baseline["total_demo"]
    
    # Group by Channel
    channel_agg = df.groupby("Channel").agg(
        Leads=("Customer_Code", "count"),
        Demo=("Demo_Flag", "sum")
    ).reset_index()
    
    # Calculate Lead_Share (safe division)
    if total_leads > 0:
        channel_agg["Lead_Share"] = channel_agg["Leads"] / total_leads
    else:
        channel_agg["Lead_Share"] = 0.0
    
    # Calculate Demo_Rate (safe division per channel)
    channel_agg["Demo_Rate"] = channel_agg.apply(
        lambda row: row["Demo"] / row["Leads"] if row["Leads"] > 0 else 0.0,
        axis=1
    )
    
    # Calculate Demo_Contribution (safe division)
    if total_demo > 0:
        channel_agg["Demo_Contribution"] = channel_agg["Demo"] / total_demo
    else:
        channel_agg["Demo_Contribution"] = 0.0
    
    # Ensure numeric types (float)
    channel_agg["Lead_Share"] = channel_agg["Lead_Share"].astype(float)
    channel_agg["Demo_Rate"] = channel_agg["Demo_Rate"].astype(float)
    channel_agg["Demo_Contribution"] = channel_agg["Demo_Contribution"].astype(float)
    
    return channel_agg


# =============================================================================
# STEP 2.3 — IMPACT SCORE
# =============================================================================

def add_impact_score(channel_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Impact Score for each channel.
    
    Formula: Impact_Score = Lead_Share × Demo_Rate
    
    Args:
        channel_summary: Channel summary DataFrame
        
    Returns:
        DataFrame with Impact_Score column added
    """
    channel_summary = channel_summary.copy()
    channel_summary["Impact_Score"] = (
        channel_summary["Lead_Share"] * channel_summary["Demo_Rate"]
    ).astype(float)
    
    return channel_summary


# =============================================================================
# STEP 2.4 — CHANNEL CLASSIFICATION
# =============================================================================

def classify_channels(
    channel_summary: pd.DataFrame,
    overall_demo_rate: float
) -> pd.DataFrame:
    """
    Classify channels into strategic categories.
    
    Classification Logic:
    - number_of_channels = total unique channels
    - average_lead_share = 1 / number_of_channels
    - Volume_High = Lead_Share > average_lead_share
    - Conversion_High = Demo_Rate > overall_demo_rate
    
    Classification Matrix:
    - High Volume + High Conversion → "Scale"
    - High Volume + Low Conversion → "Optimize"
    - Low Volume + High Conversion → "Hidden Growth"
    - Low Volume + Low Conversion → "Reduce"
    
    Args:
        channel_summary: Channel summary with Impact_Score
        overall_demo_rate: Overall demo rate from baseline
        
    Returns:
        DataFrame with Strategy_Tag column added
    """
    channel_summary = channel_summary.copy()
    
    # Calculate thresholds
    number_of_channels = len(channel_summary)
    if number_of_channels > 0:
        average_lead_share = 1 / number_of_channels
    else:
        average_lead_share = 0.0
    
    # Classification flags
    volume_high = channel_summary["Lead_Share"] > average_lead_share
    conversion_high = channel_summary["Demo_Rate"] > overall_demo_rate
    
    # Apply classification matrix
    conditions = [
        (volume_high & conversion_high),       # Scale
        (volume_high & ~conversion_high),      # Optimize
        (~volume_high & conversion_high),      # Hidden Growth
        (~volume_high & ~conversion_high),     # Reduce
    ]
    
    choices = ["Scale", "Optimize", "Hidden Growth", "Reduce"]
    
    channel_summary["Strategy_Tag"] = np.select(conditions, choices, default="Unknown")
    
    return channel_summary


# =============================================================================
# STEP 2.5 — REALLOCATION BASE
# =============================================================================

def add_expected_demo(channel_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Expected Demo for reallocation planning.
    
    Formula: Expected_Demo = Leads × Demo_Rate
    
    Args:
        channel_summary: Channel summary DataFrame
        
    Returns:
        DataFrame with Expected_Demo column added
    """
    channel_summary = channel_summary.copy()
    channel_summary["Expected_Demo"] = (
        channel_summary["Leads"] * channel_summary["Demo_Rate"]
    ).astype(float)
    
    return channel_summary


# =============================================================================
# STEP 2.6 — DATE LAYER
# =============================================================================

def build_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build monthly channel summary with date layer.
    
    Creates:
    - Month extraction
    - Year_Month (format YYYY-MM)
    
    Groups by Year_Month, Channel with metrics:
    - Leads
    - Demo
    - Demo_Rate
    
    Args:
        df: Validated DataFrame with Lead_Create_Time column
        
    Returns:
        Monthly channel summary DataFrame
    """
    df = df.copy()
    
    # Check if Lead_Create_Time exists
    if "Lead_Create_Time" not in df.columns:
        # Return empty DataFrame with expected columns if no date column
        return pd.DataFrame(columns=[
            "Year_Month", "Channel", "Leads", "Demo", "Demo_Rate"
        ])
    
    # Convert Lead_Create_Time to datetime
    df["Lead_Create_Time"] = pd.to_datetime(df["Lead_Create_Time"], errors="coerce")
    
    # Create Year_Month
    df["Year_Month"] = df["Lead_Create_Time"].dt.strftime("%Y-%m")
    
    # Filter out rows with invalid dates
    df = df.dropna(subset=["Year_Month"])
    
    if len(df) == 0:
        return pd.DataFrame(columns=[
            "Year_Month", "Channel", "Leads", "Demo", "Demo_Rate"
        ])
    
    # Group by Year_Month and Channel
    monthly_agg = df.groupby(["Year_Month", "Channel"]).agg(
        Leads=("Customer_Code", "count"),
        Demo=("Demo_Flag", "sum")
    ).reset_index()
    
    # Calculate Demo_Rate (safe division)
    monthly_agg["Demo_Rate"] = monthly_agg.apply(
        lambda row: row["Demo"] / row["Leads"] if row["Leads"] > 0 else 0.0,
        axis=1
    ).astype(float)
    
    # Sort by Year_Month and Channel
    monthly_agg = monthly_agg.sort_values(["Year_Month", "Channel"]).reset_index(drop=True)
    
    return monthly_agg


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_channel_intelligence_pipeline(
    df: pd.DataFrame,
    save_output: bool = True,
    verbose: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    Run the complete Channel Intelligence pipeline.
    
    Steps:
    1. Validate data
    2. Calculate baseline snapshot
    3. Build channel summary (aggregation)
    4. Add Impact Score
    5. Classify channels
    6. Add Expected Demo
    7. Build monthly summary
    8. Save outputs
    
    Args:
        df: Input DataFrame with required columns
        save_output: Whether to save CSV files
        verbose: Print progress information
        
    Returns:
        Tuple of (channel_performance_summary, monthly_channel_summary, baseline)
    """
    if verbose:
        print("=" * 60)
        print("  PHASE 2: CHANNEL INTELLIGENCE DATA PREPARATION")
        print("=" * 60)
    
    # Step 0: Validate
    if verbose:
        print("\n[STEP 0] Validating data...")
    df_validated = validate_data(df)
    if verbose:
        print(f"  ✓ Validated {len(df_validated)} records")
    
    # Step 2.1: Baseline Snapshot
    if verbose:
        print("\n[STEP 2.1] Calculating baseline snapshot...")
    baseline = calculate_baseline_snapshot(df_validated)
    if verbose:
        print(f"  Total Leads: {baseline['total_leads']:,}")
        print(f"  Total Demo: {baseline['total_demo']:,}")
        print(f"  Overall Demo Rate: {baseline['overall_demo_rate']:.4f}")
    
    # Step 2.2: Channel Aggregation
    if verbose:
        print("\n[STEP 2.2] Building channel summary...")
    channel_summary = build_channel_summary(df_validated, baseline)
    if verbose:
        print(f"  ✓ Aggregated {len(channel_summary)} channels")
    
    # Step 2.3: Impact Score
    if verbose:
        print("\n[STEP 2.3] Adding Impact Score...")
    channel_summary = add_impact_score(channel_summary)
    if verbose:
        print("  ✓ Impact Score calculated")
    
    # Step 2.4: Channel Classification
    if verbose:
        print("\n[STEP 2.4] Classifying channels...")
    channel_summary = classify_channels(channel_summary, baseline["overall_demo_rate"])
    if verbose:
        tag_counts = channel_summary["Strategy_Tag"].value_counts().to_dict()
        for tag, count in tag_counts.items():
            print(f"  {tag}: {count} channels")
    
    # Step 2.5: Reallocation Base
    if verbose:
        print("\n[STEP 2.5] Adding Expected Demo...")
    channel_summary = add_expected_demo(channel_summary)
    if verbose:
        print("  ✓ Expected Demo calculated")
    
    # Step 2.6: Date Layer
    if verbose:
        print("\n[STEP 2.6] Building monthly summary...")
    monthly_summary = build_monthly_summary(df_validated)
    if verbose:
        if len(monthly_summary) > 0:
            print(f"  ✓ Monthly summary: {monthly_summary['Year_Month'].nunique()} months")
        else:
            print("  ⚠ No Lead_Create_Time data available for monthly summary")
    
    # Reorder columns for final output
    channel_performance_columns = [
        "Channel",
        "Leads",
        "Lead_Share",
        "Demo",
        "Demo_Rate",
        "Demo_Contribution",
        "Impact_Score",
        "Strategy_Tag",
        "Expected_Demo",
    ]
    channel_summary = channel_summary[channel_performance_columns]
    
    # Sort by Impact_Score descending
    channel_summary = channel_summary.sort_values(
        "Impact_Score", ascending=False
    ).reset_index(drop=True)
    
    # Save outputs
    if save_output:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        channel_path = OUTPUT_DIR / "channel_performance_summary.csv"
        monthly_path = OUTPUT_DIR / "monthly_channel_summary.csv"
        
        channel_summary.to_csv(channel_path, index=False)
        monthly_summary.to_csv(monthly_path, index=False)
        
        if verbose:
            print("\n" + "=" * 60)
            print("  OUTPUT FILES SAVED")
            print("=" * 60)
            print(f"  → {channel_path}")
            print(f"  → {monthly_path}")
    
    if verbose:
        print("\n" + "=" * 60)
        print("  PHASE 2 COMPLETE")
        print("=" * 60 + "\n")
    
    return channel_summary, monthly_summary, baseline


# =============================================================================
# DATA PREPARATION HELPER
# =============================================================================

def prepare_marketing_master(
    lead_path: str = "data/raw/Lead.xlsx",
    demo_path: str = "data/processed/lead_demo_clean.csv",
    fill_null_channel: str = "Unknown"
) -> pd.DataFrame:
    """
    Prepare marketing_master DataFrame from raw data.
    
    Maps existing columns to required schema:
    - Customer_Code → Customer_Code
    - Contact_Type → Channel
    - is_demo → Demo_Flag
    
    Args:
        lead_path: Path to Lead.xlsx for Lead_Create_Time
        demo_path: Path to processed lead_demo data
        fill_null_channel: Value to fill null Channel values (default: "Unknown")
        
    Returns:
        DataFrame with required columns for pipeline
    """
    # Load processed data
    df_demo = pd.read_csv(demo_path)
    
    # Load raw lead data for Lead_Create_Time
    df_lead = pd.read_excel(lead_path)
    
    # Identify Lead_Create_Time column (look for date column)
    date_columns = [col for col in df_lead.columns if "วันที่" in col or "date" in col.lower() or "time" in col.lower()]
    lead_create_col = date_columns[0] if date_columns else None
    
    # Build marketing_master
    marketing_master = df_demo.rename(columns={
        "Contact_Type": "Channel",
        "is_demo": "Demo_Flag",
    })
    
    # Fill null Channel values
    if marketing_master["Channel"].isnull().any():
        null_count = marketing_master["Channel"].isnull().sum()
        print(f"  Filling {null_count} null Channel values with '{fill_null_channel}'")
        marketing_master["Channel"] = marketing_master["Channel"].fillna(fill_null_channel)
    
    # Add Lead_Create_Time if available
    if lead_create_col and lead_create_col in df_lead.columns:
        # Get Customer_Code to Lead_Create_Time mapping
        lead_dates = df_lead[["Customer_Code", lead_create_col]].copy()
        lead_dates = lead_dates.rename(columns={lead_create_col: "Lead_Create_Time"})
        lead_dates = lead_dates.drop_duplicates(subset=["Customer_Code"])
        
        # Merge
        marketing_master = marketing_master.merge(
            lead_dates,
            on="Customer_Code",
            how="left"
        )
    
    return marketing_master


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("Preparing marketing_master from existing data...")
    
    try:
        # Prepare data
        marketing_master = prepare_marketing_master()
        print(f"Marketing master prepared: {len(marketing_master)} rows")
        print(f"Columns: {list(marketing_master.columns)}")
        
        # Run pipeline
        channel_summary, monthly_summary, baseline = run_channel_intelligence_pipeline(
            marketing_master,
            save_output=True,
            verbose=True
        )
        
        # Display results
        print("\n" + "=" * 60)
        print("  CHANNEL PERFORMANCE SUMMARY")
        print("=" * 60)
        print(channel_summary.to_string(index=False))
        
    except FileNotFoundError as e:
        print(f"Error: Required data file not found - {e}")
    except ValueError as e:
        print(f"Validation Error: {e}")
