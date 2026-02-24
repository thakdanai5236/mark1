"""
Pipeline: Fix and Merge Demo Data
================================
This script handles data preparation by:
1. Loading Lead.xlsx and Demo.xlsx
2. Translating Thai column names in Demo.xlsx to English
3. Merging datasets on Customer_Code
4. Creating is_demo flag
5. Saving cleaned data to CSV

Author: Data Engineering Team
Created: 2026-02-18
Updated: 2026-02-18 - Fixed Demo.xlsx header row and column mapping
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Tuple

# =============================================================================
# CONFIGURATION
# =============================================================================

# File paths
RAW_DATA_PATH = Path("data/raw")
PROCESSED_DATA_PATH = Path("data/processed")

LEAD_FILE = RAW_DATA_PATH / "Lead.xlsx"
DEMO_FILE = RAW_DATA_PATH / "Demo.xlsx"
OUTPUT_FILE = PROCESSED_DATA_PATH / "lead_demo_clean.csv"

# Demo.xlsx has Thai headers in the first row (row 0)
DEMO_HEADER_ROW = 0

# =============================================================================
# COLUMN MAPPING: Thai to English
# =============================================================================
# This mapping can be updated as needed without modifying the main logic
# Demo.xlsx is a CRM export with scheduled demo appointments

DEMO_COLUMN_MAPPING: Dict[str, str] = {
    # Key identifiers
    "ลำดับ": "Sequence",
    "รหัสลูกค้า": "Customer_Code",
    "ชื่อลูกค้า": "Customer_Name",
    
    # Date/Time columns
    "วันที่ติดต่อ": "Contact_Date",
    "วันที่บันทึกล่าสุด": "Last_Updated_Date",
    
    # Contact information
    "ประเภทการติดต่อ": "Contact_Type",
    "ครั้งที่": "Contact_Count",
    "เบอร์โทรศัพท์": "Phone",
    "อีเมล": "Email",
    "ที่อยู่": "Address",
    "ผู้ติดต่อ (นิติบุคคล)": "Contact_Person",
    
    # Project/Sales information
    "โครงการ": "Project",
    "ติดตาม": "Follow_Up",
    "คาดขาย": "Expected_Sale",
    "เกรด": "Grade",  # Contains "นัด DEMO" for demo appointments
    "พนักงานขาย": "Sales_Person",
    "รายละเอียด": "Details",
    
    # System columns
    "ผู้บันทึกล่าสุด (UserID)": "Last_Updated_By",
}


# =============================================================================
# FUNCTIONS
# =============================================================================

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Lead and Demo Excel files into DataFrames.
    
    Returns:
        Tuple of (lead_df, demo_df)
    """
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    # Load Lead.xlsx
    print(f"Loading Lead file: {LEAD_FILE}")
    lead_df = pd.read_excel(LEAD_FILE)
    print(f"  -> Loaded {len(lead_df):,} rows, {len(lead_df.columns)} columns")
    
    # Load Demo.xlsx (skip first row which is a URL)
    print(f"Loading Demo file: {DEMO_FILE}")
    demo_df = pd.read_excel(DEMO_FILE, header=DEMO_HEADER_ROW)
    print(f"  -> Loaded {len(demo_df):,} rows, {len(demo_df.columns)} columns")
    
    return lead_df, demo_df


def translate_demo_columns(demo_df: pd.DataFrame) -> pd.DataFrame:
    """
    Translate Thai column names in Demo DataFrame to English.
    Uses the DEMO_COLUMN_MAPPING dictionary.
    
    Args:
        demo_df: DataFrame with Thai column names
        
    Returns:
        DataFrame with English column names
    """
    print("\n" + "=" * 60)
    print("TRANSLATING DEMO COLUMNS")
    print("=" * 60)
    
    # Create a copy to avoid modifying original
    df = demo_df.copy()
    
    # Track translations
    translated = []
    not_mapped = []
    
    for col in df.columns:
        if col in DEMO_COLUMN_MAPPING:
            translated.append(f"  '{col}' -> '{DEMO_COLUMN_MAPPING[col]}'")
        else:
            not_mapped.append(col)
    
    # Apply mapping
    df = df.rename(columns=DEMO_COLUMN_MAPPING)
    
    print(f"Translated {len(translated)} columns")
    if not_mapped:
        print(f"Columns not in mapping (kept as-is): {len(not_mapped)}")
        for col in not_mapped[:5]:  # Show first 5
            print(f"  - {col}")
        if len(not_mapped) > 5:
            print(f"  ... and {len(not_mapped) - 5} more")
    
    return df


def validate_data(lead_df: pd.DataFrame, demo_df: pd.DataFrame) -> None:
    """
    Perform basic data validation and print statistics.
    
    Args:
        lead_df: Lead DataFrame
        demo_df: Demo DataFrame (with translated columns)
    """
    print("\n" + "=" * 60)
    print("DATA VALIDATION")
    print("=" * 60)
    
    # Check for duplicate Customer_Code in Lead
    lead_duplicates = lead_df['Customer_Code'].duplicated().sum()
    print(f"\nLead file:")
    print(f"  Total rows: {len(lead_df):,}")
    print(f"  Unique Customer_Code: {lead_df['Customer_Code'].nunique():,}")
    print(f"  Duplicate Customer_Code: {lead_duplicates:,}")
    
    if lead_duplicates > 0:
        print(f"  ⚠️  WARNING: Found {lead_duplicates} duplicate Customer_Code entries")
    
    # Check Demo data
    print(f"\nDemo file:")
    print(f"  Total rows: {len(demo_df):,}")
    
    if 'Customer_Code' in demo_df.columns:
        demo_with_code = demo_df['Customer_Code'].notna().sum()
        print(f"  Rows with Customer_Code: {demo_with_code:,}")
        print(f"  Unique Customer_Code: {demo_df['Customer_Code'].nunique():,}")
    else:
        print("  ⚠️  WARNING: Customer_Code column not found in Demo")
    
    # Check Grade column for "นัด DEMO" entries
    if 'Grade' in demo_df.columns:
        demo_scheduled = (demo_df['Grade'] == 'นัด DEMO').sum()
        print(f"  Rows with Grade='นัด DEMO': {demo_scheduled:,}")
    
    if 'Contact_Date' in demo_df.columns:
        demo_with_date = demo_df['Contact_Date'].notna().sum()
        print(f"  Rows with Contact_Date: {demo_with_date:,}")


def merge_data(lead_df: pd.DataFrame, demo_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge Lead and Demo data using LEFT JOIN on Customer_Code.
    Creates is_demo flag based on presence in Demo data (all Demo records are "นัด DEMO").
    
    Args:
        lead_df: Lead DataFrame
        demo_df: Demo DataFrame (with translated columns)
        
    Returns:
        Merged DataFrame with selected columns
    """
    print("\n" + "=" * 60)
    print("MERGING DATA")
    print("=" * 60)
    
    # Select necessary columns from Lead
    lead_cols = ['Customer_Code', 'Contact_Type']
    lead_selected = lead_df[lead_cols].copy()
    print(f"Selected from Lead: {lead_cols}")
    
    # Select necessary columns from Demo
    # Contact_Date is the date when demo was scheduled/contacted
    demo_cols = ['Customer_Code', 'Contact_Date']
    
    # Check if columns exist
    available_demo_cols = [col for col in demo_cols if col in demo_df.columns]
    if len(available_demo_cols) != len(demo_cols):
        missing = set(demo_cols) - set(available_demo_cols)
        print(f"⚠️  WARNING: Missing columns in Demo: {missing}")
    
    demo_selected = demo_df[available_demo_cols].copy()
    
    # Rename Contact_Date to Demo_Date for clarity in output
    if 'Contact_Date' in demo_selected.columns:
        demo_selected = demo_selected.rename(columns={'Contact_Date': 'Demo_Date'})
    
    print(f"Selected from Demo: {available_demo_cols}")
    
    # Remove rows without Customer_Code in demo (can't join without key)
    demo_selected = demo_selected[demo_selected['Customer_Code'].notna()]
    print(f"Demo rows with valid Customer_Code: {len(demo_selected):,}")
    
    # Drop duplicate Customer_Code in Demo (keep first occurrence - most recent contact)
    demo_selected = demo_selected.drop_duplicates(subset=['Customer_Code'], keep='first')
    print(f"Demo rows after deduplication: {len(demo_selected):,}")
    
    # Perform LEFT JOIN
    merged_df = lead_selected.merge(
        demo_selected,
        on='Customer_Code',
        how='left'
    )
    print(f"\nMerged dataset: {len(merged_df):,} rows")
    
    # Create is_demo flag
    # All records in Demo file are scheduled demos (Grade = "นัด DEMO")
    # So if a Customer_Code exists in demo_selected, they have a demo scheduled
    if 'Demo_Date' in merged_df.columns:
        merged_df['is_demo'] = merged_df['Demo_Date'].notna().astype(int)
    else:
        merged_df['is_demo'] = 0
    
    return merged_df


def calculate_statistics(merged_df: pd.DataFrame) -> None:
    """
    Calculate and print final statistics.
    
    Args:
        merged_df: Final merged DataFrame
    """
    print("\n" + "=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)
    
    total_leads = len(merged_df)
    total_demos = merged_df['is_demo'].sum()
    demo_rate = (total_demos / total_leads * 100) if total_leads > 0 else 0
    
    print(f"\nTotal Leads: {total_leads:,}")
    print(f"Total with Demo: {total_demos:,}")
    print(f"Demo Rate: {demo_rate:.2f}%")
    
    # Contact Type breakdown
    print("\nDemo Rate by Contact Type:")
    print("-" * 40)
    
    contact_stats = merged_df.groupby('Contact_Type').agg(
        total_leads=('Customer_Code', 'count'),
        demos=('is_demo', 'sum')
    ).reset_index()
    contact_stats['demo_rate'] = (contact_stats['demos'] / contact_stats['total_leads'] * 100).round(2)
    contact_stats = contact_stats.sort_values('total_leads', ascending=False)
    
    for _, row in contact_stats.iterrows():
        print(f"  {row['Contact_Type']}: {row['demos']}/{row['total_leads']} ({row['demo_rate']:.2f}%)")


def save_processed(merged_df: pd.DataFrame) -> None:
    """
    Save the processed DataFrame to CSV.
    
    Args:
        merged_df: Final merged DataFrame
    """
    print("\n" + "=" * 60)
    print("SAVING PROCESSED DATA")
    print("=" * 60)
    
    # Create output directory if it doesn't exist
    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    merged_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"Total rows: {len(merged_df):,}")
    print(f"Columns: {merged_df.columns.tolist()}")


def main() -> None:
    """
    Main pipeline execution.
    """
    print("\n" + "=" * 60)
    print("  PIPELINE: FIX AND MERGE DEMO DATA")
    print("=" * 60)
    
    # Step 1: Load data
    lead_df, demo_df = load_data()
    
    # Step 2: Translate Demo columns
    demo_df_translated = translate_demo_columns(demo_df)
    
    # Step 3: Validate data
    validate_data(lead_df, demo_df_translated)
    
    # Step 4: Merge data
    merged_df = merge_data(lead_df, demo_df_translated)
    
    # Step 5: Calculate statistics
    calculate_statistics(merged_df)
    
    # Step 6: Save processed data
    save_processed(merged_df)
    
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
