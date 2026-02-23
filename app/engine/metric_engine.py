"""
Metric Engine - Calculates marketing metrics and KPIs
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import pandas as pd


@dataclass
class MetricResult:
    """Container for metric calculation results."""
    name: str
    value: float
    unit: str
    period: Optional[str] = None
    breakdown: Optional[Dict[str, float]] = None
    trend: Optional[float] = None  # Percentage change


class MetricEngine:
    """Calculates marketing performance metrics."""
    
    def __init__(self, data: Optional[pd.DataFrame] = None):
        """
        Initialize metric engine.
        
        Args:
            data: Marketing data DataFrame
        """
        self.data = data
    
    def set_data(self, data: pd.DataFrame):
        """Set the data for metric calculations."""
        self.data = data
    
    def calculate_conversion_rate(
        self,
        leads_col: str = "leads",
        conversions_col: str = "conversions",
        group_by: Optional[str] = None
    ) -> MetricResult:
        """
        Calculate conversion rate.
        
        Args:
            leads_col: Column name for leads count
            conversions_col: Column name for conversions count
            group_by: Optional column to group by
            
        Returns:
            MetricResult with conversion rate
        """
        if group_by:
            grouped = self.data.groupby(group_by).agg({
                leads_col: "sum",
                conversions_col: "sum"
            })
            breakdown = (grouped[conversions_col] / grouped[leads_col] * 100).to_dict()
            total_rate = self.data[conversions_col].sum() / self.data[leads_col].sum() * 100
            
            return MetricResult(
                name="Conversion Rate",
                value=total_rate,
                unit="%",
                breakdown=breakdown
            )
        
        rate = self.data[conversions_col].sum() / self.data[leads_col].sum() * 100
        return MetricResult(name="Conversion Rate", value=rate, unit="%")
    
    def calculate_cac(
        self,
        cost_col: str = "marketing_cost",
        customers_col: str = "new_customers",
        group_by: Optional[str] = None
    ) -> MetricResult:
        """
        Calculate Customer Acquisition Cost (CAC).
        
        Args:
            cost_col: Column name for marketing cost
            customers_col: Column name for new customers
            group_by: Optional column to group by
            
        Returns:
            MetricResult with CAC
        """
        if group_by:
            grouped = self.data.groupby(group_by).agg({
                cost_col: "sum",
                customers_col: "sum"
            })
            breakdown = (grouped[cost_col] / grouped[customers_col]).to_dict()
            total_cac = self.data[cost_col].sum() / self.data[customers_col].sum()
            
            return MetricResult(
                name="Customer Acquisition Cost",
                value=total_cac,
                unit="THB",
                breakdown=breakdown
            )
        
        cac = self.data[cost_col].sum() / self.data[customers_col].sum()
        return MetricResult(name="Customer Acquisition Cost", value=cac, unit="THB")
    
    def calculate_roi(
        self,
        revenue_col: str = "revenue",
        cost_col: str = "marketing_cost",
        group_by: Optional[str] = None
    ) -> MetricResult:
        """
        Calculate Return on Investment (ROI).
        
        Args:
            revenue_col: Column name for revenue
            cost_col: Column name for marketing cost
            group_by: Optional column to group by
            
        Returns:
            MetricResult with ROI percentage
        """
        if group_by:
            grouped = self.data.groupby(group_by).agg({
                revenue_col: "sum",
                cost_col: "sum"
            })
            breakdown = ((grouped[revenue_col] - grouped[cost_col]) / grouped[cost_col] * 100).to_dict()
            total_revenue = self.data[revenue_col].sum()
            total_cost = self.data[cost_col].sum()
            total_roi = (total_revenue - total_cost) / total_cost * 100
            
            return MetricResult(
                name="Marketing ROI",
                value=total_roi,
                unit="%",
                breakdown=breakdown
            )
        
        revenue = self.data[revenue_col].sum()
        cost = self.data[cost_col].sum()
        roi = (revenue - cost) / cost * 100
        return MetricResult(name="Marketing ROI", value=roi, unit="%")
    
    def calculate_ltv(
        self,
        revenue_col: str = "customer_revenue",
        customer_id_col: str = "customer_id",
        avg_lifespan_months: float = 24
    ) -> MetricResult:
        """
        Calculate Customer Lifetime Value (LTV).
        
        Args:
            revenue_col: Column name for customer revenue
            customer_id_col: Column name for customer ID
            avg_lifespan_months: Average customer lifespan in months
            
        Returns:
            MetricResult with LTV
        """
        avg_monthly_revenue = (
            self.data.groupby(customer_id_col)[revenue_col].sum().mean()
        )
        ltv = avg_monthly_revenue * avg_lifespan_months
        
        return MetricResult(
            name="Customer Lifetime Value",
            value=ltv,
            unit="THB"
        )
    
    def get_metrics_summary(self) -> Dict[str, MetricResult]:
        """Get a summary of all key metrics."""
        return {
            "conversion_rate": self.calculate_conversion_rate(),
            "cac": self.calculate_cac(),
            "roi": self.calculate_roi()
        }
