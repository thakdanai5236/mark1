"""
Tests for Reallocation Engine
"""

import pytest
import pandas as pd
import numpy as np
from app.engine.reallocation_engine import (
    ReallocationEngine,
    AllocationRecommendation,
    ReallocationResult
)


@pytest.fixture
def channel_data():
    """Create sample channel performance data."""
    return pd.DataFrame({
        "channel": ["Facebook", "Google", "LINE", "Website"],
        "cost": [50000, 80000, 30000, 40000],
        "conversions": [100, 200, 50, 80],
        "revenue": [200000, 400000, 100000, 180000]
    })


@pytest.fixture
def reallocation_engine():
    """Create reallocation engine instance."""
    return ReallocationEngine(min_allocation_pct=0.05)


class TestReallocationEngine:
    """Tests for ReallocationEngine class."""
    
    def test_calculate_channel_efficiency(self, reallocation_engine, channel_data):
        """Test efficiency calculation."""
        efficiency = reallocation_engine.calculate_channel_efficiency(channel_data)
        
        assert "cpa" in efficiency.columns
        assert "roi" in efficiency.columns
        assert "roas" in efficiency.columns
        assert len(efficiency) == 4
    
    def test_cpa_calculation(self, reallocation_engine, channel_data):
        """Test CPA is calculated correctly."""
        efficiency = reallocation_engine.calculate_channel_efficiency(channel_data)
        
        # Facebook: 50000 / 100 = 500
        fb_row = efficiency[efficiency["channel"] == "Facebook"]
        assert abs(fb_row["cpa"].values[0] - 500) < 0.01
    
    def test_roi_calculation(self, reallocation_engine, channel_data):
        """Test ROI is calculated correctly."""
        efficiency = reallocation_engine.calculate_channel_efficiency(channel_data)
        
        # Facebook: (200000 - 50000) / 50000 = 3.0
        fb_row = efficiency[efficiency["channel"] == "Facebook"]
        assert abs(fb_row["roi"].values[0] - 3.0) < 0.01
    
    def test_optimize_allocation(self, reallocation_engine, channel_data):
        """Test optimization."""
        efficiency = reallocation_engine.calculate_channel_efficiency(channel_data)
        result = reallocation_engine.optimize_allocation(
            channel_efficiency=efficiency,
            total_budget=200000
        )
        
        assert isinstance(result, ReallocationResult)
        assert len(result.recommendations) == 4
        assert result.total_budget == 200000
    
    def test_allocation_sums_to_budget(self, reallocation_engine, channel_data):
        """Test allocations sum to total budget."""
        efficiency = reallocation_engine.calculate_channel_efficiency(channel_data)
        result = reallocation_engine.optimize_allocation(
            channel_efficiency=efficiency,
            total_budget=200000
        )
        
        total_allocated = sum(r.recommended_allocation for r in result.recommendations)
        assert abs(total_allocated - 200000) < 0.01
    
    def test_min_allocation_constraint(self, reallocation_engine, channel_data):
        """Test minimum allocation constraint."""
        efficiency = reallocation_engine.calculate_channel_efficiency(channel_data)
        result = reallocation_engine.optimize_allocation(
            channel_efficiency=efficiency,
            total_budget=200000
        )
        
        min_expected = 200000 * 0.05  # 5% min
        for rec in result.recommendations:
            assert rec.recommended_allocation >= min_expected * 0.99  # Allow small float error
    
    def test_custom_constraints(self, reallocation_engine, channel_data):
        """Test custom allocation constraints."""
        efficiency = reallocation_engine.calculate_channel_efficiency(channel_data)
        
        constraints = {
            "Facebook": (0.2, 0.3),  # 20-30%
            "Google": (0.3, 0.5)     # 30-50%
        }
        
        result = reallocation_engine.optimize_allocation(
            channel_efficiency=efficiency,
            total_budget=200000,
            constraints=constraints
        )
        
        # Check Facebook stays within constraints
        fb_rec = next(r for r in result.recommendations if r.channel == "Facebook")
        fb_pct = fb_rec.recommended_allocation / 200000
        assert 0.2 <= fb_pct <= 0.3
    
    def test_recommendation_has_rationale(self, reallocation_engine, channel_data):
        """Test recommendations include rationale."""
        efficiency = reallocation_engine.calculate_channel_efficiency(channel_data)
        result = reallocation_engine.optimize_allocation(
            channel_efficiency=efficiency,
            total_budget=200000
        )
        
        for rec in result.recommendations:
            assert rec.rationale is not None
            assert len(rec.rationale) > 0


class TestAllocationRecommendation:
    """Tests for AllocationRecommendation dataclass."""
    
    def test_recommendation_creation(self):
        """Test creating recommendation."""
        rec = AllocationRecommendation(
            channel="Facebook",
            current_allocation=50000,
            recommended_allocation=60000,
            change_percentage=20.0,
            expected_impact=180000,
            rationale="High ROI"
        )
        
        assert rec.channel == "Facebook"
        assert rec.change_percentage == 20.0


class TestReallocationResult:
    """Tests for ReallocationResult dataclass."""
    
    def test_result_creation(self):
        """Test creating result."""
        rec = AllocationRecommendation(
            channel="Test",
            current_allocation=1000,
            recommended_allocation=1200,
            change_percentage=20.0,
            expected_impact=5000,
            rationale="Test"
        )
        
        result = ReallocationResult(
            recommendations=[rec],
            total_budget=10000,
            expected_roi_improvement=15.0,
            confidence_level=0.8
        )
        
        assert len(result.recommendations) == 1
        assert result.expected_roi_improvement == 15.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
