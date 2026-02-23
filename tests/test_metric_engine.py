"""
Tests for Metric Engine
"""

import pytest
import pandas as pd
import numpy as np
from app.engine.metric_engine import MetricEngine, MetricResult


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        "channel": np.tile(["Facebook", "Google", "LINE"], 100),
        "leads": np.random.randint(100, 500, 300),
        "conversions": np.random.randint(10, 100, 300),
        "marketing_cost": np.random.uniform(5000, 20000, 300),
        "new_customers": np.random.randint(5, 50, 300),
        "revenue": np.random.uniform(20000, 100000, 300)
    })


@pytest.fixture
def metric_engine(sample_data):
    """Create metric engine with sample data."""
    return MetricEngine(data=sample_data)


class TestMetricEngine:
    """Tests for MetricEngine class."""
    
    def test_calculate_conversion_rate(self, metric_engine):
        """Test conversion rate calculation."""
        result = metric_engine.calculate_conversion_rate()
        
        assert isinstance(result, MetricResult)
        assert result.name == "Conversion Rate"
        assert result.unit == "%"
        assert 0 <= result.value <= 100
    
    def test_conversion_rate_grouped(self, metric_engine):
        """Test conversion rate with grouping."""
        result = metric_engine.calculate_conversion_rate(group_by="channel")
        
        assert result.breakdown is not None
        assert "Facebook" in result.breakdown
        assert "Google" in result.breakdown
        assert "LINE" in result.breakdown
    
    def test_calculate_cac(self, metric_engine):
        """Test CAC calculation."""
        result = metric_engine.calculate_cac()
        
        assert isinstance(result, MetricResult)
        assert result.name == "Customer Acquisition Cost"
        assert result.unit == "THB"
        assert result.value > 0
    
    def test_cac_grouped(self, metric_engine):
        """Test CAC with grouping."""
        result = metric_engine.calculate_cac(group_by="channel")
        
        assert result.breakdown is not None
        assert len(result.breakdown) == 3
    
    def test_calculate_roi(self, metric_engine):
        """Test ROI calculation."""
        result = metric_engine.calculate_roi()
        
        assert isinstance(result, MetricResult)
        assert result.name == "Marketing ROI"
        assert result.unit == "%"
    
    def test_roi_grouped(self, metric_engine):
        """Test ROI with grouping."""
        result = metric_engine.calculate_roi(group_by="channel")
        
        assert result.breakdown is not None
    
    def test_get_metrics_summary(self, metric_engine):
        """Test metrics summary."""
        summary = metric_engine.get_metrics_summary()
        
        assert "conversion_rate" in summary
        assert "cac" in summary
        assert "roi" in summary
    
    def test_set_data(self, sample_data):
        """Test setting data."""
        engine = MetricEngine()
        engine.set_data(sample_data)
        
        assert engine.data is not None
        assert len(engine.data) == len(sample_data)
    
    def test_empty_data(self):
        """Test with empty data."""
        engine = MetricEngine(data=pd.DataFrame())
        
        with pytest.raises(Exception):
            engine.calculate_conversion_rate()


class TestMetricResult:
    """Tests for MetricResult dataclass."""
    
    def test_metric_result_creation(self):
        """Test creating MetricResult."""
        result = MetricResult(
            name="Test Metric",
            value=42.5,
            unit="%"
        )
        
        assert result.name == "Test Metric"
        assert result.value == 42.5
        assert result.unit == "%"
    
    def test_metric_result_with_breakdown(self):
        """Test MetricResult with breakdown."""
        result = MetricResult(
            name="Test Metric",
            value=50.0,
            unit="%",
            breakdown={"A": 40.0, "B": 60.0}
        )
        
        assert result.breakdown is not None
        assert result.breakdown["A"] == 40.0
    
    def test_metric_result_with_trend(self):
        """Test MetricResult with trend."""
        result = MetricResult(
            name="Test Metric",
            value=50.0,
            unit="%",
            trend=10.5
        )
        
        assert result.trend == 10.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
