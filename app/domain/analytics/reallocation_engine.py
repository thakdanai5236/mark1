"""
Reallocation Engine - Optimizes budget and resource allocation
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class AllocationRecommendation:
    """Recommendation for resource reallocation."""
    channel: str
    current_allocation: float
    recommended_allocation: float
    change_percentage: float
    expected_impact: float
    rationale: str


@dataclass
class ReallocationResult:
    """Result of reallocation optimization."""
    recommendations: List[AllocationRecommendation]
    total_budget: float
    expected_roi_improvement: float
    confidence_level: float


class ReallocationEngine:
    """
    Engine for optimizing marketing budget and resource allocation
    across channels based on performance data.
    """
    
    def __init__(self, min_allocation_pct: float = 0.05):
        """
        Initialize reallocation engine.
        
        Args:
            min_allocation_pct: Minimum allocation percentage per channel
        """
        self.min_allocation_pct = min_allocation_pct
    
    def calculate_channel_efficiency(
        self,
        data: pd.DataFrame,
        channel_col: str = "channel",
        cost_col: str = "cost",
        conversions_col: str = "conversions",
        revenue_col: str = "revenue"
    ) -> pd.DataFrame:
        """
        Calculate efficiency metrics for each channel.
        
        Args:
            data: Channel performance data
            channel_col: Column containing channel names
            cost_col: Column containing costs
            conversions_col: Column containing conversions
            revenue_col: Column containing revenue
            
        Returns:
            DataFrame with efficiency metrics per channel
        """
        efficiency = data.groupby(channel_col).agg({
            cost_col: "sum",
            conversions_col: "sum",
            revenue_col: "sum"
        }).reset_index()
        
        efficiency["cpa"] = efficiency[cost_col] / efficiency[conversions_col]
        efficiency["roi"] = (efficiency[revenue_col] - efficiency[cost_col]) / efficiency[cost_col]
        efficiency["conversion_rate"] = efficiency[conversions_col] / efficiency[cost_col] * 1000
        efficiency["revenue_per_cost"] = efficiency[revenue_col] / efficiency[cost_col]
        
        return efficiency
    
    def optimize_allocation(
        self,
        channel_efficiency: pd.DataFrame,
        total_budget: float,
        channel_col: str = "channel",
        efficiency_metric: str = "roi",
        constraints: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> ReallocationResult:
        """
        Optimize budget allocation across channels.
        
        Args:
            channel_efficiency: DataFrame with channel efficiency metrics
            total_budget: Total budget to allocate
            channel_col: Column containing channel names
            efficiency_metric: Metric to optimize for
            constraints: Dict of channel: (min_pct, max_pct) constraints
            
        Returns:
            ReallocationResult with recommendations
        """
        channels = channel_efficiency[channel_col].tolist()
        efficiencies = channel_efficiency[efficiency_metric].tolist()
        current_allocations = channel_efficiency["cost"].tolist()
        current_total = sum(current_allocations)
        
        # Normalize efficiencies to positive values
        min_eff = min(efficiencies)
        if min_eff < 0:
            adjusted_efficiencies = [e - min_eff + 0.1 for e in efficiencies]
        else:
            adjusted_efficiencies = efficiencies
        
        # Calculate efficiency-weighted allocation
        total_efficiency = sum(adjusted_efficiencies)
        raw_allocations = [
            (eff / total_efficiency) * total_budget
            for eff in adjusted_efficiencies
        ]
        
        # Apply constraints
        final_allocations = []
        for i, (channel, alloc) in enumerate(zip(channels, raw_allocations)):
            min_alloc = total_budget * self.min_allocation_pct
            max_alloc = total_budget * (1 - self.min_allocation_pct * (len(channels) - 1))
            
            if constraints and channel in constraints:
                min_pct, max_pct = constraints[channel]
                min_alloc = max(min_alloc, total_budget * min_pct)
                max_alloc = min(max_alloc, total_budget * max_pct)
            
            final_allocations.append(max(min_alloc, min(max_alloc, alloc)))
        
        # Normalize to ensure sum equals budget
        alloc_sum = sum(final_allocations)
        final_allocations = [a * total_budget / alloc_sum for a in final_allocations]
        
        # Create recommendations
        recommendations = []
        for i, channel in enumerate(channels):
            current_pct = current_allocations[i] / current_total if current_total > 0 else 0
            new_pct = final_allocations[i] / total_budget
            change_pct = (new_pct - current_pct) / current_pct * 100 if current_pct > 0 else 0
            
            if change_pct > 10:
                rationale = f"High {efficiency_metric} of {efficiencies[i]:.2f} justifies increased investment"
            elif change_pct < -10:
                rationale = f"Low {efficiency_metric} of {efficiencies[i]:.2f} suggests reducing investment"
            else:
                rationale = f"Current allocation is near optimal based on {efficiency_metric}"
            
            recommendations.append(AllocationRecommendation(
                channel=channel,
                current_allocation=current_allocations[i],
                recommended_allocation=final_allocations[i],
                change_percentage=change_pct,
                expected_impact=efficiencies[i] * final_allocations[i],
                rationale=rationale
            ))
        
        # Sort by change magnitude
        recommendations.sort(key=lambda x: abs(x.change_percentage), reverse=True)
        
        # Calculate expected improvement
        current_weighted_efficiency = sum(
            e * (a / current_total) for e, a in zip(efficiencies, current_allocations)
        ) if current_total > 0 else 0
        
        new_weighted_efficiency = sum(
            e * (a / total_budget) for e, a in zip(efficiencies, final_allocations)
        )
        
        improvement = (
            (new_weighted_efficiency - current_weighted_efficiency) / 
            current_weighted_efficiency * 100
        ) if current_weighted_efficiency > 0 else 0
        
        return ReallocationResult(
            recommendations=recommendations,
            total_budget=total_budget,
            expected_roi_improvement=improvement,
            confidence_level=0.75  # Base confidence
        )
    
    def simulate_reallocation(
        self,
        current_data: pd.DataFrame,
        recommendations: List[AllocationRecommendation],
        channel_col: str = "channel"
    ) -> Dict[str, float]:
        """
        Simulate the impact of reallocation recommendations.
        
        Args:
            current_data: Current performance data
            recommendations: Reallocation recommendations
            channel_col: Column containing channel names
            
        Returns:
            Dictionary with simulated outcomes
        """
        # This is a simplified simulation
        # In practice, would use more sophisticated models
        
        projected_metrics = {
            "total_conversions": 0,
            "total_revenue": 0,
            "total_cost": 0
        }
        
        for rec in recommendations:
            channel_data = current_data[current_data[channel_col] == rec.channel]
            if len(channel_data) > 0:
                # Scale metrics by allocation change
                scale_factor = rec.recommended_allocation / rec.current_allocation \
                    if rec.current_allocation > 0 else 1
                
                # Apply diminishing returns for large increases
                if scale_factor > 1.5:
                    scale_factor = 1 + (scale_factor - 1) * 0.7
                
                projected_metrics["total_conversions"] += \
                    channel_data["conversions"].sum() * scale_factor
                projected_metrics["total_revenue"] += \
                    channel_data["revenue"].sum() * scale_factor
                projected_metrics["total_cost"] += rec.recommended_allocation
        
        projected_metrics["roi"] = (
            projected_metrics["total_revenue"] - projected_metrics["total_cost"]
        ) / projected_metrics["total_cost"] if projected_metrics["total_cost"] > 0 else 0
        
        return projected_metrics
