"""
PHASE 4: Enterprise Channel Optimization Engine
================================================

Linear Programming based optimizer for maximizing Demo count
under capacity, budget, and channel scaling constraints.

Mathematical Formulation:
-------------------------
Maximize: Σ (x_i × Demo_Rate_i)

Subject to:
    1) Σ x_i ≤ capacity_remaining           (Sales capacity)
    2) Σ (x_i × Cost_per_Lead_i) ≤ budget   (Marketing budget)
    3) x_i ≤ Leads_i × Max_Scale_Pct_i      (Channel scaling limit)
    4) x_i ≥ 0                              (Non-negativity)

Author: Optimization Engineering Team
Created: 2026-02-23
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import pandas as pd
import numpy as np

try:
    from pulp import (
        LpProblem,
        LpMaximize,
        LpVariable,
        LpStatus,
        lpSum,
        value,
        PULP_CBC_CMD,
    )
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False


# =============================================================================
# CONSTANTS
# =============================================================================

# Default values when columns are missing (explicit)
DEFAULT_COST_PER_LEAD = 0.0             # Cost per lead if missing
DEFAULT_MAX_SCALE_PERCENTAGE = 0.30     # 30% max increase

# Sales capacity formula: number_of_sales × demos_per_day × working_days
DEMOS_PER_SALES_PER_DAY = 10
WORKING_DAYS_PER_MONTH = 22


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class BaselineMetrics:
    """Current state before optimization."""
    current_leads: int
    current_demo: int
    current_demo_rate: float


@dataclass
class OptimizationResult:
    """Optimization output metrics."""
    total_additional_demo: float
    new_total_demo: float
    new_total_leads: float
    new_demo_rate: float
    solver_status: str


@dataclass
class ConstraintStatus:
    """Status of constraint utilization."""
    capacity_used: float
    budget_used: float
    capacity_remaining: float
    budget_remaining: float
    capacity_limit: float
    budget_limit: float
    capacity_utilization_pct: float
    budget_utilization_pct: float
    demo_per_capacity: Optional[float]
    demo_per_budget: Optional[float]


@dataclass
class ChannelAllocation:
    """Allocation plan for a single channel."""
    channel: str
    current_leads: int
    additional_leads: float
    new_leads: float
    demo_rate: float
    additional_demo: float
    cost_used: float
    max_allowed: float
    utilization_pct: float
    demo_per_cost: float
    roi_index: float
    scaling_fully_used: bool


@dataclass
class EnterpriseOptimizationResult:
    """Complete optimization result package."""
    baseline: BaselineMetrics
    optimization_result: OptimizationResult
    allocation_plan: pd.DataFrame
    constraints_status: ConstraintStatus
    channel_allocations: List[ChannelAllocation]
    feasible: bool
    message: str
    binding_constraint: str
    shadow_price_capacity: Optional[float]
    shadow_price_budget: Optional[float]
    executive_summary: str
    defaults_used: Dict[str, float]
    model_assumptions: Dict[str, bool]
    shadow_price_unit_definition: Dict[str, str]


# =============================================================================
# LP MODEL BUILDING
# =============================================================================

def calculate_sales_capacity(number_of_sales: int) -> int:
    """
    Calculate total sales capacity.
    
    Formula: number_of_sales × 10 demos/day × 22 working days
    
    Args:
        number_of_sales: Number of sales representatives
        
    Returns:
        Total monthly demo capacity
    """
    return number_of_sales * DEMOS_PER_SALES_PER_DAY * WORKING_DAYS_PER_MONTH


def prepare_channel_data(
    channel_df: pd.DataFrame,
    default_cost: float = DEFAULT_COST_PER_LEAD,
    default_max_scale: float = DEFAULT_MAX_SCALE_PERCENTAGE,
) -> pd.DataFrame:
    """
    Prepare channel data with required columns.
    
    Adds Cost_per_Lead and Max_Scale_Percentage if missing.
    
    Args:
        channel_df: Channel performance DataFrame
        default_cost: Default cost per lead
        default_max_scale: Default max scaling percentage
        
    Returns:
        DataFrame with all required columns
    """
    df = channel_df.copy()
    
    # Ensure required columns exist
    if "Cost_per_Lead" not in df.columns:
        df["Cost_per_Lead"] = default_cost

    if "Max_Scale_Percentage" not in df.columns:
        df["Max_Scale_Percentage"] = default_max_scale

    # Fill any NaN values explicitly with defaults
    df["Cost_per_Lead"] = df["Cost_per_Lead"].fillna(default_cost)
    df["Max_Scale_Percentage"] = df["Max_Scale_Percentage"].fillna(default_max_scale)
    
    # Ensure Demo_Rate is float
    df["Demo_Rate"] = df["Demo_Rate"].astype(float)
    
    return df


def build_lp_model(
    channel_df: pd.DataFrame,
    capacity_remaining: float,
    marketing_budget: Optional[float],
) -> Tuple[LpProblem, Dict[str, LpVariable]]:
    """
    Build the Linear Programming model.
    
    Decision Variables:
        x_i = additional leads to allocate to channel i
    
    Objective:
        Maximize Σ (x_i × Demo_Rate_i)
    
    Constraints:
        1) Σ x_i ≤ capacity_remaining
        2) Σ (x_i × Cost_per_Lead_i) ≤ marketing_budget
        3) x_i ≤ Leads_i × Max_Scale_Percentage_i
        4) x_i ≥ 0
    
    Args:
        channel_df: Prepared channel DataFrame
        capacity_remaining: Remaining sales capacity
        marketing_budget: Available marketing budget. If None, no budget
            constraint is applied.
        
    Returns:
        Tuple of (LpProblem, decision_variables_dict)
    """
    if not PULP_AVAILABLE:
        raise ImportError("PuLP is required. Install with: pip install pulp")
    
    # Create the LP problem
    model = LpProblem("Enterprise_Channel_Optimization", LpMaximize)
    
    # Decision variables: x_i = additional leads for channel i
    channels = channel_df["Channel"].tolist()
    x = {}
    
    for idx, row in channel_df.iterrows():
        channel = row["Channel"]
        max_additional = row["Leads"] * row["Max_Scale_Percentage"]
        
        # Create variable with upper bound from scaling constraint
        x[channel] = LpVariable(
            name=f"x_{channel.replace(' ', '_')}",
            lowBound=0,
            upBound=max_additional,
            cat="Continuous",
        )
    
    # Objective Function: Maximize total additional demos
    # Σ (x_i × Demo_Rate_i)
    model += lpSum([
        x[row["Channel"]] * row["Demo_Rate"]
        for _, row in channel_df.iterrows()
    ]), "Maximize_Additional_Demo"
    
    # Constraint 1: Capacity Constraint
    # Σ x_i ≤ capacity_remaining
    if capacity_remaining > 0:
        model += lpSum([x[ch] for ch in channels]) <= capacity_remaining, "Capacity_Constraint"
    else:
        # If no capacity, all x_i must be 0
        for ch in channels:
            model += x[ch] == 0, f"Zero_Capacity_{ch.replace(' ', '_')}"
    
    # Constraint 2: Budget Constraint (optional)
    # Σ (x_i × Cost_per_Lead_i) ≤ marketing_budget
    # If marketing_budget is None, we do NOT add a budget constraint
    if marketing_budget is not None:
        if marketing_budget > 0:
            model += lpSum([
                x[row["Channel"]] * row["Cost_per_Lead"]
                for _, row in channel_df.iterrows()
            ]) <= marketing_budget, "Budget_Constraint"
        else:
            # If explicit zero/negative budget, all x_i must be 0
            for ch in channels:
                model += x[ch] == 0, f"Zero_Budget_{ch.replace(' ', '_')}"
    
    # Constraint 3: Channel Scaling Limits (already in variable bounds)
    # x_i ≤ Leads_i × Max_Scale_Percentage_i
    # This is handled by the upBound parameter in LpVariable
    
    return model, x


# =============================================================================
# SOLVER
# =============================================================================

def solve_max_demo(
    model: LpProblem,
    time_limit: int = 60,
    verbose: bool = False,
) -> str:
    """
    Solve the LP model.
    
    Args:
        model: The LP problem to solve
        time_limit: Maximum solving time in seconds
        verbose: Print solver output
        
    Returns:
        Solver status string
    """
    # Use CBC solver (comes with PuLP)
    solver = PULP_CBC_CMD(
        msg=1 if verbose else 0,
        timeLimit=time_limit,
    )
    
    model.solve(solver)
    
    return LpStatus[model.status]


# =============================================================================
# SOLUTION EXTRACTION
# =============================================================================

def extract_solution(
    model: LpProblem,
    variables: Dict[str, LpVariable],
    channel_df: pd.DataFrame,
) -> Tuple[List[ChannelAllocation], float, float]:
    """
    Extract solution from solved model.
    
    Args:
        model: Solved LP model
        variables: Decision variables dict
        channel_df: Channel data DataFrame
        
    Returns:
        Tuple of (allocations, total_additional_demo, total_cost)
    """
    allocations = []
    total_additional_demo = 0.0
    total_cost = 0.0
    
    tol = 1e-5

    for _, row in channel_df.iterrows():
        channel = row["Channel"]
        x_val = value(variables[channel]) or 0.0
        
        demo_rate = row["Demo_Rate"]
        cost_per_lead = row["Cost_per_Lead"]
        current_leads = row["Leads"]
        max_allowed = current_leads * row["Max_Scale_Percentage"]
        
        additional_demo = x_val * demo_rate
        cost_used = x_val * cost_per_lead
        utilization = (x_val / max_allowed * 100) if max_allowed > 0 else 0.0

        # Efficiency metrics
        demo_per_cost = (demo_rate / cost_per_lead) if cost_per_lead > 0 else 0.0
        roi_index = (additional_demo / cost_used) if cost_used > 0 else 0.0

        scaling_fully_used = False
        if max_allowed > 0:
            scaling_fully_used = abs(x_val - max_allowed) <= tol * max(1.0, abs(max_allowed))
        
        allocation = ChannelAllocation(
            channel=channel,
            current_leads=int(current_leads),
            additional_leads=round(x_val, 2),
            new_leads=current_leads + x_val,
            demo_rate=demo_rate,
            additional_demo=round(additional_demo, 2),
            cost_used=round(cost_used, 2),
            max_allowed=round(max_allowed, 2),
            utilization_pct=round(utilization, 2),
            demo_per_cost=round(demo_per_cost, 6),
            roi_index=round(roi_index, 6),
            scaling_fully_used=scaling_fully_used,
        )
        allocations.append(allocation)
        
        total_additional_demo += additional_demo
        total_cost += cost_used
    
    return allocations, total_additional_demo, total_cost


def allocations_to_dataframe(allocations: List[ChannelAllocation]) -> pd.DataFrame:
    """
    Convert allocations list to DataFrame.
    
    Args:
        allocations: List of ChannelAllocation objects
        
    Returns:
        DataFrame with allocation plan
    """
    data = []
    for a in allocations:
        data.append({
            "Channel": a.channel,
            "Current_Leads": a.current_leads,
            "Additional_Leads": a.additional_leads,
            "New_Leads": a.new_leads,
            "Demo_Rate": a.demo_rate,
            "Additional_Demo": a.additional_demo,
            "Cost_Used": a.cost_used,
            "Max_Allowed": a.max_allowed,
            "Utilization_Pct": a.utilization_pct,
            "Demo_per_Cost": a.demo_per_cost,
            "ROI_Index": a.roi_index,
            "Scaling_Fully_Used": a.scaling_fully_used,
        })
    
    df = pd.DataFrame(data)
    
    # Sort by additional demo descending
    df = df.sort_values("Additional_Demo", ascending=False).reset_index(drop=True)
    
    return df


# =============================================================================
# RESULT SUMMARIZATION
# =============================================================================

def summarize_results(
    allocations: List[ChannelAllocation],
    total_additional_demo: float,
    total_cost: float,
    baseline: BaselineMetrics,
    capacity_remaining: float,
    marketing_budget: Optional[float],
    solver_status: str,
    shadow_price_capacity: Optional[float] = None,
    shadow_price_budget: Optional[float] = None,
    defaults_used: Optional[Dict[str, float]] = None,
    model_assumptions: Optional[Dict[str, bool]] = None,
    shadow_price_unit_definition: Optional[Dict[str, str]] = None,
) -> EnterpriseOptimizationResult:
    """
    Create comprehensive result summary.
    
    Args:
        allocations: Channel allocation list
        total_additional_demo: Total additional demos from optimization
        total_cost: Total cost used
        baseline: Baseline metrics
        capacity_remaining: Initial capacity remaining
        marketing_budget: Initial budget
        solver_status: Solver status string
        
    Returns:
        EnterpriseOptimizationResult
    """
    # Calculate new totals
    total_additional_leads = sum(a.additional_leads for a in allocations)
    new_total_leads = baseline.current_leads + total_additional_leads
    new_total_demo = baseline.current_demo + total_additional_demo
    new_demo_rate = new_total_demo / new_total_leads if new_total_leads > 0 else 0.0
    
    # Optimization result
    opt_result = OptimizationResult(
        total_additional_demo=round(total_additional_demo, 2),
        new_total_demo=round(new_total_demo, 2),
        new_total_leads=round(new_total_leads, 2),
        new_demo_rate=round(new_demo_rate, 6),
        solver_status=solver_status,
    )
    
    # Constraint status and utilization diagnostics
    raw_capacity_used = total_additional_leads
    raw_budget_used = total_cost
    capacity_limit = capacity_remaining
    # When marketing_budget is None, treat as "no limit" for diagnostics.
    # We still report the actual cost used but set budget_limit to 0.0 so
    # utilization stays at 0 and budget never appears binding.
    budget_limit = marketing_budget if marketing_budget is not None else 0.0

    capacity_utilization_pct = (
        raw_capacity_used / capacity_limit if capacity_limit > 0 else 0.0
    )
    budget_utilization_pct = (
        raw_budget_used / budget_limit if budget_limit > 0 else 0.0
    )

    demo_per_capacity = (
        total_additional_demo / raw_capacity_used if raw_capacity_used > 0 else None
    )
    demo_per_budget = (
        total_additional_demo / raw_budget_used if raw_budget_used > 0 else None
    )

    constraints = ConstraintStatus(
        capacity_used=round(raw_capacity_used, 2),
        budget_used=round(raw_budget_used, 2),
        capacity_remaining=round(capacity_limit - raw_capacity_used, 2),
        budget_remaining=round(budget_limit - raw_budget_used, 2),
        capacity_limit=capacity_limit,
        budget_limit=budget_limit,
        capacity_utilization_pct=round(capacity_utilization_pct, 6),
        budget_utilization_pct=round(budget_utilization_pct, 6),
        demo_per_capacity=round(demo_per_capacity, 6) if demo_per_capacity is not None else None,
        demo_per_budget=round(demo_per_budget, 6) if demo_per_budget is not None else None,
    )
    
    # Allocation DataFrame
    allocation_df = allocations_to_dataframe(allocations)

    # Determine feasibility
    feasible = solver_status == "Optimal"

    # Binding constraint diagnostics with scaled tolerance
    binding_tolerance = 1e-5

    def is_binding(lhs: float, rhs: float) -> bool:
        scaled_tol = binding_tolerance * max(1.0, abs(rhs))
        return abs(lhs - rhs) <= scaled_tol

    is_capacity_binding = (
        constraints.capacity_limit > 0
        and is_binding(constraints.capacity_used, constraints.capacity_limit)
    )
    is_budget_binding = (
        constraints.budget_limit > 0
        and is_binding(constraints.budget_used, constraints.budget_limit)
    )
    scaling_binding = any(
        a.max_allowed > 0 and is_binding(a.additional_leads, a.max_allowed)
        for a in allocations
    )

    binding_flags: List[str] = []
    if is_capacity_binding:
        binding_flags.append("capacity")
    if is_budget_binding:
        binding_flags.append("budget")
    if scaling_binding:
        binding_flags.append("scaling")

    if not binding_flags:
        binding_constraint = "none"
    elif len(binding_flags) == 1:
        binding_constraint = binding_flags[0]
    else:
        binding_constraint = "multiple"

    # Executive summary message (deterministic)
    if not feasible or solver_status != "Optimal":
        executive_summary = (
            "Optimization is not optimal or feasible under current constraints."
        )
    elif binding_constraint == "capacity":
        executive_summary = (
            "Growth is constrained by sales capacity. Increasing sales capacity "
            "would increase demo potential."
        )
    elif binding_constraint == "budget":
        executive_summary = (
            "Marketing budget is the limiting factor. Additional budget would "
            "increase demo output."
        )
    elif binding_constraint == "scaling":
        executive_summary = (
            "Channel scaling limits prevent further demo growth. Consider "
            "relaxing per-channel limits."
        )
    elif binding_constraint == "multiple":
        executive_summary = (
            "Multiple constraints simultaneously limit demo growth (capacity, "
            "budget, and/or channel scaling)."
        )
    else:
        executive_summary = (
            "Current configuration does not fully utilize capacity or budget; "
            "there is slack available."
        )

    if feasible:
        message = "Optimization completed successfully."
    elif solver_status == "Infeasible":
        message = "No feasible solution exists under current constraints."
    elif solver_status == "Unbounded":
        message = "Problem is unbounded. Check constraint definitions."
    else:
        message = f"Solver returned status: {solver_status}"

    defaults_used = defaults_used or {
        "Cost_per_Lead_default": DEFAULT_COST_PER_LEAD,
        "Max_Scale_Percentage_default": DEFAULT_MAX_SCALE_PERCENTAGE,
    }

    model_assumptions = model_assumptions or {
        "uses_sales_capacity_formula": True,
        "scaling_limits_enforced": True,
        "non_negative_allocations": True,
    }

    shadow_price_unit_definition = shadow_price_unit_definition or {
        "capacity": "Marginal additional demos per extra unit of sales capacity (lead)",
        "budget": "Marginal additional demos per extra budget unit",
    }

    return EnterpriseOptimizationResult(
        baseline=baseline,
        optimization_result=opt_result,
        allocation_plan=allocation_df,
        constraints_status=constraints,
        channel_allocations=allocations,
        feasible=feasible,
        message=message,
        binding_constraint=binding_constraint,
        shadow_price_capacity=shadow_price_capacity,
        shadow_price_budget=shadow_price_budget,
        executive_summary=executive_summary,
        defaults_used=defaults_used,
        model_assumptions=model_assumptions,
        shadow_price_unit_definition=shadow_price_unit_definition,
    )


# =============================================================================
# MAIN OPTIMIZATION FUNCTION
# =============================================================================

def run_enterprise_optimization(
    channel_df: pd.DataFrame,
    number_of_sales: int,
    marketing_budget: Optional[float] = None,
    current_total_leads: Optional[int] = None,
    current_total_demo: Optional[int] = None,
    default_cost_per_lead: float = DEFAULT_COST_PER_LEAD,
    default_max_scale: float = DEFAULT_MAX_SCALE_PERCENTAGE,
    verbose: bool = False,
) -> EnterpriseOptimizationResult:
    """
    Run the enterprise channel optimization.
    
    This is the main entry point for the optimizer.
    
    Args:
        channel_df: Channel performance DataFrame with columns:
            - Channel: Channel name
            - Leads: Current lead count
            - Demo_Rate: Conversion rate (float, e.g., 0.1 = 10%)
            - Cost_per_Lead: (optional) Cost per lead
            - Max_Scale_Percentage: (optional) Max scaling allowed
        number_of_sales: Number of sales representatives
        marketing_budget: Available marketing budget. If None, budget is
            treated as unconstrained (no budget limit).
        current_total_leads: Override total leads (default: sum from df)
        current_total_demo: Override total demo (default: sum from df)
        default_cost_per_lead: Default cost when not in data
        default_max_scale: Default max scale when not in data
        verbose: Print optimization progress
        
    Returns:
        EnterpriseOptimizationResult with complete optimization details
    """
    if not PULP_AVAILABLE:
        raise ImportError(
            "PuLP is required for optimization. "
            "Install with: pip install pulp"
        )
    
    if verbose:
        print("=" * 60)
        print("  PHASE 4: ENTERPRISE CHANNEL OPTIMIZATION")
        print("=" * 60)
    
    # Prepare data
    df = prepare_channel_data(
        channel_df,
        default_cost=default_cost_per_lead,
        default_max_scale=default_max_scale,
    )
    
    # Calculate baseline
    total_leads = current_total_leads if current_total_leads is not None else int(df["Leads"].sum())
    total_demo = current_total_demo if current_total_demo is not None else int(df["Demo"].sum()) if "Demo" in df.columns else 0
    
    # If Demo column missing, estimate from rate
    if total_demo == 0 and "Demo_Rate" in df.columns:
        total_demo = int((df["Leads"] * df["Demo_Rate"]).sum())
    
    demo_rate = total_demo / total_leads if total_leads > 0 else 0.0
    
    baseline = BaselineMetrics(
        current_leads=total_leads,
        current_demo=total_demo,
        current_demo_rate=demo_rate,
    )

    defaults_used = {
        "Cost_per_Lead_default": default_cost_per_lead,
        "Max_Scale_Percentage_default": default_max_scale,
    }

    model_assumptions = {
        "uses_sales_capacity_formula": True,
        "scaling_limits_enforced": True,
        "non_negative_allocations": True,
    }

    shadow_price_unit_definition = {
        "capacity": "Marginal additional demos per extra unit of sales capacity (lead)",
        "budget": "Marginal additional demos per extra budget unit",
    }
    
    if verbose:
        print(f"\n[BASELINE]")
        print(f"  Current Leads: {baseline.current_leads:,}")
        print(f"  Current Demo: {baseline.current_demo:,}")
        print(f"  Current Demo Rate: {baseline.current_demo_rate:.4f}")
    
    # Calculate capacity
    sales_capacity = calculate_sales_capacity(number_of_sales)
    capacity_remaining = max(0, sales_capacity - total_leads)
    
    if verbose:
        print(f"\n[CAPACITY]")
        print(f"  Sales Capacity: {sales_capacity:,}")
        print(f"  Capacity Remaining: {capacity_remaining:,}")
        if marketing_budget is None:
            print(f"  Marketing Budget: (no limit)")
        else:
            print(f"  Marketing Budget: {marketing_budget:,.0f}")
    
    # Handle edge cases
    if capacity_remaining <= 0:
        if verbose:
            print("\n[WARNING] No capacity remaining. Cannot optimize.")
        
        empty_allocations = []
        for _, row in df.iterrows():
            demo_rate = row["Demo_Rate"]
            cost_per_lead = row["Cost_per_Lead"]
            max_allowed = row["Leads"] * row["Max_Scale_Percentage"]
            demo_per_cost = (demo_rate / cost_per_lead) if cost_per_lead > 0 else 0.0

            empty_allocations.append(
                ChannelAllocation(
                    channel=row["Channel"],
                    current_leads=int(row["Leads"]),
                    additional_leads=0.0,
                    new_leads=row["Leads"],
                    demo_rate=demo_rate,
                    additional_demo=0.0,
                    cost_used=0.0,
                    max_allowed=max_allowed,
                    utilization_pct=0.0,
                    demo_per_cost=round(demo_per_cost, 6),
                    roi_index=0.0,
                    scaling_fully_used=False,
                )
            )
        
        return summarize_results(
            allocations=empty_allocations,
            total_additional_demo=0.0,
            total_cost=0.0,
            baseline=baseline,
            capacity_remaining=0.0,
            marketing_budget=marketing_budget,
            solver_status="No Capacity",
            shadow_price_capacity=None,
            shadow_price_budget=None,
            defaults_used=defaults_used,
            model_assumptions=model_assumptions,
            shadow_price_unit_definition=shadow_price_unit_definition,
        )
    
    if marketing_budget is not None and marketing_budget <= 0:
        if verbose:
            print("\n[WARNING] No budget available. Cannot optimize.")
        
        empty_allocations = []
        for _, row in df.iterrows():
            demo_rate = row["Demo_Rate"]
            cost_per_lead = row["Cost_per_Lead"]
            max_allowed = row["Leads"] * row["Max_Scale_Percentage"]
            demo_per_cost = (demo_rate / cost_per_lead) if cost_per_lead > 0 else 0.0

            empty_allocations.append(
                ChannelAllocation(
                    channel=row["Channel"],
                    current_leads=int(row["Leads"]),
                    additional_leads=0.0,
                    new_leads=row["Leads"],
                    demo_rate=demo_rate,
                    additional_demo=0.0,
                    cost_used=0.0,
                    max_allowed=max_allowed,
                    utilization_pct=0.0,
                    demo_per_cost=round(demo_per_cost, 6),
                    roi_index=0.0,
                    scaling_fully_used=False,
                )
            )
        
        return summarize_results(
            allocations=empty_allocations,
            total_additional_demo=0.0,
            total_cost=0.0,
            baseline=baseline,
            capacity_remaining=capacity_remaining,
            marketing_budget=0.0,
            solver_status="No Budget",
            shadow_price_capacity=None,
            shadow_price_budget=None,
            defaults_used=defaults_used,
            model_assumptions=model_assumptions,
            shadow_price_unit_definition=shadow_price_unit_definition,
        )
    
    # Build and solve LP model
    if verbose:
        print(f"\n[BUILDING LP MODEL]")
        print(f"  Channels: {len(df)}")
        print(f"  Decision Variables: {len(df)} (one per channel)")
    
    model, variables = build_lp_model(
        channel_df=df,
        capacity_remaining=capacity_remaining,
        marketing_budget=marketing_budget,
    )
    
    if verbose:
        print(f"\n[SOLVING]")
    
    status = solve_max_demo(model, verbose=verbose)
    
    if verbose:
        print(f"  Solver Status: {status}")

    # Shadow price approximation (dual values) for capacity and budget
    shadow_price_capacity: Optional[float] = None
    shadow_price_budget: Optional[float] = None
    try:
        if "Capacity_Constraint" in model.constraints:
            cap_pi = model.constraints["Capacity_Constraint"].pi
            if cap_pi is not None:
                shadow_price_capacity = float(cap_pi)
    except Exception:
        shadow_price_capacity = None
    try:
        if "Budget_Constraint" in model.constraints:
            bud_pi = model.constraints["Budget_Constraint"].pi
            if bud_pi is not None:
                shadow_price_budget = float(bud_pi)
    except Exception:
        shadow_price_budget = None
    
    # Extract solution
    allocations, total_additional_demo, total_cost = extract_solution(
        model=model,
        variables=variables,
        channel_df=df,
    )
    
    # Summarize results
    result = summarize_results(
        allocations=allocations,
        total_additional_demo=total_additional_demo,
        total_cost=total_cost,
        baseline=baseline,
        capacity_remaining=capacity_remaining,
        marketing_budget=marketing_budget,
        solver_status=status,
        shadow_price_capacity=shadow_price_capacity,
        shadow_price_budget=shadow_price_budget,
        defaults_used=defaults_used,
        model_assumptions=model_assumptions,
        shadow_price_unit_definition=shadow_price_unit_definition,
    )
    
    if verbose:
        print_optimization_result(result)
    
    return result


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def print_optimization_result(result: EnterpriseOptimizationResult) -> None:
    """Print formatted optimization result."""
    
    print("\n" + "=" * 60)
    print("  OPTIMIZATION RESULTS")
    print("=" * 60)
    
    print(f"\n[STATUS] {result.message}")
    print(f"  Solver Status: {result.optimization_result.solver_status}")
    
    print(f"\n[OPTIMIZATION SUMMARY]")
    print(f"  Additional Demo: +{result.optimization_result.total_additional_demo:.2f}")
    print(f"  New Total Demo: {result.optimization_result.new_total_demo:.2f}")
    print(f"  New Total Leads: {result.optimization_result.new_total_leads:.2f}")
    print(f"  New Demo Rate: {result.optimization_result.new_demo_rate:.4f}")
    
    rate_change = result.optimization_result.new_demo_rate - result.baseline.current_demo_rate
    print(f"  Rate Change: {'+' if rate_change >= 0 else ''}{rate_change:.4f}")
    
    print(f"\n[CONSTRAINT UTILIZATION]")
    print(f"  Capacity Used: {result.constraints_status.capacity_used:,.2f} / {result.constraints_status.capacity_limit:,.2f}")
    print(f"  Budget Used: {result.constraints_status.budget_used:,.2f} / {result.constraints_status.budget_limit:,.2f}")
    
    print(f"\n[ALLOCATION PLAN]")
    
    # Only show channels with allocation > 0
    active = result.allocation_plan[result.allocation_plan["Additional_Leads"] > 0]
    
    if len(active) > 0:
        print(active.to_string(index=False))
    else:
        print("  No allocation recommended.")
    
    print("\n" + "=" * 60)


def result_to_dict(result: EnterpriseOptimizationResult) -> dict:
    """
    Convert result to dictionary format.
    
    Args:
        result: EnterpriseOptimizationResult
        
    Returns:
        Dictionary with all result components
    """
    return {
        "baseline": {
            "current_leads": result.baseline.current_leads,
            "current_demo": result.baseline.current_demo,
            "current_demo_rate": result.baseline.current_demo_rate,
        },
        "optimization_result": {
            "total_additional_demo": result.optimization_result.total_additional_demo,
            "new_total_demo": result.optimization_result.new_total_demo,
            "new_total_leads": result.optimization_result.new_total_leads,
            "new_demo_rate": result.optimization_result.new_demo_rate,
            "solver_status": result.optimization_result.solver_status,
        },
        "allocation_plan": result.allocation_plan.to_dict(orient="records"),
        "constraints_status": {
            "capacity_used": result.constraints_status.capacity_used,
            "budget_used": result.constraints_status.budget_used,
            "capacity_remaining": result.constraints_status.capacity_remaining,
            "budget_remaining": result.constraints_status.budget_remaining,
            "capacity_limit": result.constraints_status.capacity_limit,
            "budget_limit": result.constraints_status.budget_limit,
            "capacity_utilization_pct": result.constraints_status.capacity_utilization_pct,
            "budget_utilization_pct": result.constraints_status.budget_utilization_pct,
            "demo_per_capacity": result.constraints_status.demo_per_capacity,
            "demo_per_budget": result.constraints_status.demo_per_budget,
        },
        "binding_constraint": result.binding_constraint,
        "shadow_price_capacity": result.shadow_price_capacity,
        "shadow_price_budget": result.shadow_price_budget,
        "executive_summary": result.executive_summary,
        "defaults_used": result.defaults_used,
        "model_assumptions": result.model_assumptions,
        "shadow_price_unit_definition": result.shadow_price_unit_definition,
        "feasible": result.feasible,
        "message": result.message,
    }


# =============================================================================
# CLI INTERFACE
# =============================================================================

def load_channel_data(path: Optional[str] = None) -> pd.DataFrame:
    """Load channel performance data from CSV."""
    if path is None:
        path = "data/processed/channel_performance_summary.csv"
    
    df = pd.read_csv(path)
    return df


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="PHASE 4: Enterprise Channel Optimization Engine"
    )
    parser.add_argument(
        "--sales", "-s",
        type=int,
        default=5,
        help="Number of sales representatives (default: 5)",
    )
    parser.add_argument(
        "--budget", "-b",
        type=float,
        default=100000.0,
        help="Marketing budget in THB (default: 100000)",
    )
    parser.add_argument(
        "--cost-per-lead", "-c",
        type=float,
        default=DEFAULT_COST_PER_LEAD,
        help=f"Default cost per lead (default: {DEFAULT_COST_PER_LEAD})",
    )
    parser.add_argument(
        "--max-scale", "-m",
        type=float,
        default=DEFAULT_MAX_SCALE_PERCENTAGE,
        help=f"Default max scale percentage (default: {DEFAULT_MAX_SCALE_PERCENTAGE})",
    )
    parser.add_argument(
        "--data", "-d",
        type=str,
        default="data/processed/channel_performance_summary.csv",
        help="Path to channel data CSV",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from: {args.data}")
    channel_df = load_channel_data(args.data)
    
    # Run optimization
    result = run_enterprise_optimization(
        channel_df=channel_df,
        number_of_sales=args.sales,
        marketing_budget=args.budget,
        default_cost_per_lead=args.cost_per_lead,
        default_max_scale=args.max_scale,
        verbose=True,  # Always verbose in CLI
    )
    
    return result


if __name__ == "__main__":
    main()
