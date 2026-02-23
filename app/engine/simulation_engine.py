"""
Simulation Engine - Runs what-if scenarios and forecasts
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


@dataclass
class ScenarioConfig:
    """Configuration for a simulation scenario."""
    name: str
    description: str
    parameters: Dict[str, Any]
    duration_days: int = 30


@dataclass 
class SimulationResult:
    """Result of a simulation run."""
    scenario_name: str
    metrics: Dict[str, float]
    timeline: Optional[pd.DataFrame] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    assumptions: Optional[List[str]] = None


class SimulationEngine:
    """
    Engine for running marketing simulations and what-if scenarios.
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize simulation engine.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def run_budget_scenario(
        self,
        baseline_data: pd.DataFrame,
        budget_change_pct: float,
        channel: Optional[str] = None,
        duration_days: int = 30
    ) -> SimulationResult:
        """
        Simulate impact of budget changes.
        
        Args:
            baseline_data: Historical performance data
            budget_change_pct: Percentage change in budget (-100 to +inf)
            channel: Specific channel or None for all
            duration_days: Simulation duration
            
        Returns:
            SimulationResult with projected metrics
        """
        # Calculate baseline metrics
        if channel:
            filtered_data = baseline_data[baseline_data["channel"] == channel]
        else:
            filtered_data = baseline_data
        
        baseline_daily = {
            "cost": filtered_data["cost"].sum() / len(filtered_data["date"].unique()),
            "conversions": filtered_data["conversions"].sum() / len(filtered_data["date"].unique()),
            "revenue": filtered_data["revenue"].sum() / len(filtered_data["date"].unique())
        }
        
        # Apply budget change with diminishing returns
        budget_multiplier = 1 + (budget_change_pct / 100)
        
        # Model diminishing returns
        if budget_multiplier > 1:
            effectiveness = 0.7 + 0.3 * (1 / budget_multiplier)
        else:
            effectiveness = 1.0  # Linear decrease for budget cuts
        
        projected = {
            "daily_cost": baseline_daily["cost"] * budget_multiplier,
            "daily_conversions": baseline_daily["conversions"] * budget_multiplier * effectiveness,
            "daily_revenue": baseline_daily["revenue"] * budget_multiplier * effectiveness
        }
        
        # Generate timeline with noise
        timeline_data = []
        for day in range(duration_days):
            noise = np.random.normal(1, 0.1)  # 10% daily variance
            timeline_data.append({
                "day": day + 1,
                "date": datetime.now() + timedelta(days=day),
                "cost": projected["daily_cost"] * noise,
                "conversions": projected["daily_conversions"] * noise,
                "revenue": projected["daily_revenue"] * noise
            })
        
        timeline = pd.DataFrame(timeline_data)
        
        # Calculate totals
        total_metrics = {
            "total_cost": projected["daily_cost"] * duration_days,
            "total_conversions": projected["daily_conversions"] * duration_days,
            "total_revenue": projected["daily_revenue"] * duration_days,
            "roi": (projected["daily_revenue"] - projected["daily_cost"]) / projected["daily_cost"],
            "cpa": projected["daily_cost"] / projected["daily_conversions"]
        }
        
        assumptions = [
            f"Budget change: {budget_change_pct:+.1f}%",
            f"Effectiveness factor: {effectiveness:.2f}",
            "Assumes stable market conditions",
            "Based on historical performance patterns"
        ]
        
        return SimulationResult(
            scenario_name=f"Budget {'Increase' if budget_change_pct > 0 else 'Decrease'} {abs(budget_change_pct):.0f}%",
            metrics=total_metrics,
            timeline=timeline,
            confidence_interval=(0.85, 1.15),  # ±15% confidence
            assumptions=assumptions
        )
    
    def run_channel_mix_scenario(
        self,
        baseline_data: pd.DataFrame,
        channel_allocations: Dict[str, float]
    ) -> SimulationResult:
        """
        Simulate impact of changing channel mix.
        
        Args:
            baseline_data: Historical performance data
            channel_allocations: Dict of channel: allocation_percentage
            
        Returns:
            SimulationResult with projected metrics
        """
        # Get channel efficiencies from baseline
        channel_metrics = baseline_data.groupby("channel").agg({
            "cost": "sum",
            "conversions": "sum",
            "revenue": "sum"
        }).reset_index()
        
        channel_metrics["roas"] = channel_metrics["revenue"] / channel_metrics["cost"]
        channel_metrics["cpa"] = channel_metrics["cost"] / channel_metrics["conversions"]
        
        total_budget = baseline_data["cost"].sum()
        
        projected_conversions = 0
        projected_revenue = 0
        
        for channel, allocation in channel_allocations.items():
            channel_data = channel_metrics[channel_metrics["channel"] == channel]
            if len(channel_data) > 0:
                channel_budget = total_budget * allocation
                roas = channel_data["roas"].values[0]
                cpa = channel_data["cpa"].values[0]
                
                projected_conversions += channel_budget / cpa
                projected_revenue += channel_budget * roas
        
        metrics = {
            "total_cost": total_budget,
            "total_conversions": projected_conversions,
            "total_revenue": projected_revenue,
            "overall_roas": projected_revenue / total_budget,
            "overall_cpa": total_budget / projected_conversions
        }
        
        return SimulationResult(
            scenario_name="Channel Mix Optimization",
            metrics=metrics,
            assumptions=[
                "Channel efficiencies remain constant",
                "No cross-channel effects considered"
            ]
        )
    
    def monte_carlo_forecast(
        self,
        baseline_data: pd.DataFrame,
        forecast_days: int = 30,
        n_simulations: int = 1000
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation for forecasting.
        
        Args:
            baseline_data: Historical data
            forecast_days: Number of days to forecast
            n_simulations: Number of simulation runs
            
        Returns:
            Forecast results with confidence intervals
        """
        # Calculate historical statistics
        daily_revenue = baseline_data.groupby("date")["revenue"].sum()
        mean_revenue = daily_revenue.mean()
        std_revenue = daily_revenue.std()
        
        # Run simulations
        simulated_totals = []
        for _ in range(n_simulations):
            daily_values = np.random.normal(mean_revenue, std_revenue, forecast_days)
            daily_values = np.maximum(daily_values, 0)  # No negative revenue
            simulated_totals.append(daily_values.sum())
        
        simulated_totals = np.array(simulated_totals)
        
        return {
            "mean_forecast": np.mean(simulated_totals),
            "median_forecast": np.median(simulated_totals),
            "std_forecast": np.std(simulated_totals),
            "ci_90": (np.percentile(simulated_totals, 5), np.percentile(simulated_totals, 95)),
            "ci_95": (np.percentile(simulated_totals, 2.5), np.percentile(simulated_totals, 97.5)),
            "worst_case": np.percentile(simulated_totals, 5),
            "best_case": np.percentile(simulated_totals, 95)
        }
    
    def compare_scenarios(
        self,
        scenarios: List[SimulationResult]
    ) -> pd.DataFrame:
        """
        Compare multiple simulation scenarios.
        
        Args:
            scenarios: List of simulation results
            
        Returns:
            DataFrame comparing scenarios
        """
        comparison_data = []
        for scenario in scenarios:
            row = {"scenario": scenario.scenario_name}
            row.update(scenario.metrics)
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
