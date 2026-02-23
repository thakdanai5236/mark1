"""
Tests for Simulation Engine
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from app.engine.simulation_engine import (
    SimulationEngine,
    ScenarioConfig,
    SimulationResult
)


@pytest.fixture
def baseline_data():
    """Create sample baseline data."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=90, freq="D")
    
    data = []
    for date in dates:
        for channel in ["Facebook", "Google", "LINE"]:
            data.append({
                "date": date,
                "channel": channel,
                "cost": np.random.uniform(1000, 5000),
                "conversions": np.random.randint(5, 30),
                "revenue": np.random.uniform(10000, 50000)
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def simulation_engine():
    """Create simulation engine instance."""
    return SimulationEngine(random_seed=42)


class TestSimulationEngine:
    """Tests for SimulationEngine class."""
    
    def test_run_budget_scenario_increase(self, simulation_engine, baseline_data):
        """Test budget increase scenario."""
        result = simulation_engine.run_budget_scenario(
            baseline_data=baseline_data,
            budget_change_pct=20,
            duration_days=30
        )
        
        assert isinstance(result, SimulationResult)
        assert "Budget Increase" in result.scenario_name
        assert result.metrics["total_cost"] > 0
    
    def test_run_budget_scenario_decrease(self, simulation_engine, baseline_data):
        """Test budget decrease scenario."""
        result = simulation_engine.run_budget_scenario(
            baseline_data=baseline_data,
            budget_change_pct=-20,
            duration_days=30
        )
        
        assert "Budget Decrease" in result.scenario_name
    
    def test_budget_scenario_channel_specific(self, simulation_engine, baseline_data):
        """Test budget scenario for specific channel."""
        result = simulation_engine.run_budget_scenario(
            baseline_data=baseline_data,
            budget_change_pct=50,
            channel="Facebook",
            duration_days=30
        )
        
        assert result.metrics is not None
    
    def test_budget_scenario_has_timeline(self, simulation_engine, baseline_data):
        """Test that budget scenario includes timeline."""
        result = simulation_engine.run_budget_scenario(
            baseline_data=baseline_data,
            budget_change_pct=10,
            duration_days=30
        )
        
        assert result.timeline is not None
        assert len(result.timeline) == 30
    
    def test_budget_scenario_has_assumptions(self, simulation_engine, baseline_data):
        """Test that scenario includes assumptions."""
        result = simulation_engine.run_budget_scenario(
            baseline_data=baseline_data,
            budget_change_pct=10,
            duration_days=30
        )
        
        assert result.assumptions is not None
        assert len(result.assumptions) > 0
    
    def test_run_channel_mix_scenario(self, simulation_engine, baseline_data):
        """Test channel mix scenario."""
        allocations = {
            "Facebook": 0.4,
            "Google": 0.4,
            "LINE": 0.2
        }
        
        result = simulation_engine.run_channel_mix_scenario(
            baseline_data=baseline_data,
            channel_allocations=allocations
        )
        
        assert result.scenario_name == "Channel Mix Optimization"
        assert result.metrics["total_cost"] > 0
    
    def test_monte_carlo_forecast(self, simulation_engine, baseline_data):
        """Test Monte Carlo forecasting."""
        forecast = simulation_engine.monte_carlo_forecast(
            baseline_data=baseline_data,
            forecast_days=30,
            n_simulations=100
        )
        
        assert "mean_forecast" in forecast
        assert "ci_90" in forecast
        assert "ci_95" in forecast
        assert forecast["mean_forecast"] > 0
    
    def test_monte_carlo_confidence_intervals(self, simulation_engine, baseline_data):
        """Test Monte Carlo confidence intervals are ordered."""
        forecast = simulation_engine.monte_carlo_forecast(
            baseline_data=baseline_data,
            forecast_days=30,
            n_simulations=1000
        )
        
        ci_90_low, ci_90_high = forecast["ci_90"]
        ci_95_low, ci_95_high = forecast["ci_95"]
        
        # 95% CI should be wider than 90% CI
        assert ci_95_low <= ci_90_low
        assert ci_95_high >= ci_90_high
    
    def test_compare_scenarios(self, simulation_engine, baseline_data):
        """Test scenario comparison."""
        scenarios = [
            simulation_engine.run_budget_scenario(baseline_data, 10, duration_days=30),
            simulation_engine.run_budget_scenario(baseline_data, -10, duration_days=30),
            simulation_engine.run_budget_scenario(baseline_data, 0, duration_days=30)
        ]
        
        comparison = simulation_engine.compare_scenarios(scenarios)
        
        assert len(comparison) == 3
        assert "scenario" in comparison.columns


class TestScenarioConfig:
    """Tests for ScenarioConfig dataclass."""
    
    def test_config_creation(self):
        """Test creating scenario config."""
        config = ScenarioConfig(
            name="Test Scenario",
            description="A test scenario",
            parameters={"budget_change": 0.2},
            duration_days=60
        )
        
        assert config.name == "Test Scenario"
        assert config.duration_days == 60


class TestSimulationResult:
    """Tests for SimulationResult dataclass."""
    
    def test_result_creation(self):
        """Test creating simulation result."""
        result = SimulationResult(
            scenario_name="Test",
            metrics={"revenue": 100000, "cost": 50000}
        )
        
        assert result.scenario_name == "Test"
        assert result.metrics["revenue"] == 100000
    
    def test_result_with_optional_fields(self):
        """Test result with optional fields."""
        result = SimulationResult(
            scenario_name="Test",
            metrics={"revenue": 100000},
            confidence_interval=(90000, 110000),
            assumptions=["Stable market"]
        )
        
        assert result.confidence_interval == (90000, 110000)
        assert len(result.assumptions) == 1


class TestDiminishingReturns:
    """Tests for diminishing returns modeling."""
    
    def test_large_budget_increase_has_diminishing_returns(self, simulation_engine, baseline_data):
        """Test that large budget increases show diminishing returns."""
        small_increase = simulation_engine.run_budget_scenario(
            baseline_data, budget_change_pct=20, duration_days=30
        )
        large_increase = simulation_engine.run_budget_scenario(
            baseline_data, budget_change_pct=100, duration_days=30
        )
        
        # Efficiency should be lower for large increase
        small_efficiency = small_increase.metrics["total_conversions"] / small_increase.metrics["total_cost"]
        large_efficiency = large_increase.metrics["total_conversions"] / large_increase.metrics["total_cost"]
        
        assert large_efficiency < small_efficiency


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
