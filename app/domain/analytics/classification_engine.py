"""
Classification Engine - Lead scoring and segmentation
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np


class LeadScore(Enum):
    """Lead scoring categories."""
    HOT = "Hot"
    WARM = "Warm"
    COLD = "Cold"
    DORMANT = "Dormant"


class CustomerSegment(Enum):
    """Customer segmentation categories."""
    HIGH_VALUE = "High Value"
    GROWING = "Growing"
    AT_RISK = "At Risk"
    CHURNED = "Churned"
    NEW = "New"


@dataclass
class ClassificationResult:
    """Result of a classification operation."""
    entity_id: str
    classification: str
    confidence: float
    factors: Dict[str, float]


class ClassificationEngine:
    """Engine for lead scoring and customer segmentation."""
    
    def __init__(self):
        """Initialize classification engine."""
        self.lead_scoring_weights = {
            "engagement_score": 0.3,
            "recency_score": 0.25,
            "frequency_score": 0.2,
            "monetary_score": 0.15,
            "fit_score": 0.1
        }
        
        self.segment_thresholds = {
            "high_value": {"min_ltv": 50000, "min_frequency": 5},
            "growing": {"min_growth_rate": 0.1, "min_orders": 2},
            "at_risk": {"days_since_purchase": 60, "declining_frequency": True}
        }
    
    def score_lead(
        self,
        engagement: float,
        recency_days: int,
        interaction_count: int,
        estimated_value: float,
        fit_score: float = 0.5
    ) -> ClassificationResult:
        """
        Score a single lead.
        
        Args:
            engagement: Engagement score (0-1)
            recency_days: Days since last interaction
            interaction_count: Total number of interactions
            estimated_value: Estimated potential value
            fit_score: How well lead fits ideal customer profile (0-1)
            
        Returns:
            ClassificationResult with lead score
        """
        # Calculate component scores
        recency_score = max(0, 1 - (recency_days / 90))  # 90-day decay
        frequency_score = min(1, interaction_count / 10)  # Cap at 10 interactions
        monetary_score = min(1, estimated_value / 100000)  # Cap at 100k
        
        factors = {
            "engagement": engagement,
            "recency": recency_score,
            "frequency": frequency_score,
            "monetary": monetary_score,
            "fit": fit_score
        }
        
        # Calculate weighted score
        total_score = (
            engagement * self.lead_scoring_weights["engagement_score"] +
            recency_score * self.lead_scoring_weights["recency_score"] +
            frequency_score * self.lead_scoring_weights["frequency_score"] +
            monetary_score * self.lead_scoring_weights["monetary_score"] +
            fit_score * self.lead_scoring_weights["fit_score"]
        )
        
        # Classify based on score
        if total_score >= 0.7:
            classification = LeadScore.HOT.value
        elif total_score >= 0.4:
            classification = LeadScore.WARM.value
        elif total_score >= 0.2:
            classification = LeadScore.COLD.value
        else:
            classification = LeadScore.DORMANT.value
        
        return ClassificationResult(
            entity_id="",  # To be filled by caller
            classification=classification,
            confidence=total_score,
            factors=factors
        )
    
    def score_leads_batch(
        self,
        leads_df: pd.DataFrame,
        engagement_col: str = "engagement_score",
        recency_col: str = "days_since_contact",
        frequency_col: str = "interaction_count",
        value_col: str = "estimated_value",
        id_col: str = "lead_id"
    ) -> pd.DataFrame:
        """
        Score multiple leads from a DataFrame.
        
        Args:
            leads_df: DataFrame with lead data
            engagement_col: Column name for engagement score
            recency_col: Column name for recency
            frequency_col: Column name for frequency
            value_col: Column name for estimated value
            id_col: Column name for lead ID
            
        Returns:
            DataFrame with lead scores and classifications
        """
        results = []
        
        for _, row in leads_df.iterrows():
            result = self.score_lead(
                engagement=row.get(engagement_col, 0.5),
                recency_days=row.get(recency_col, 30),
                interaction_count=row.get(frequency_col, 1),
                estimated_value=row.get(value_col, 10000)
            )
            results.append({
                id_col: row[id_col],
                "lead_score": result.confidence,
                "lead_classification": result.classification,
                **{f"factor_{k}": v for k, v in result.factors.items()}
            })
        
        return pd.DataFrame(results)
    
    def segment_customer(
        self,
        ltv: float,
        order_count: int,
        days_since_purchase: int,
        growth_rate: float
    ) -> ClassificationResult:
        """
        Segment a customer based on RFM-like metrics.
        
        Args:
            ltv: Customer lifetime value
            order_count: Total number of orders
            days_since_purchase: Days since last purchase
            growth_rate: Revenue growth rate
            
        Returns:
            ClassificationResult with segment
        """
        factors = {
            "ltv": ltv,
            "order_count": order_count,
            "recency": days_since_purchase,
            "growth_rate": growth_rate
        }
        
        # Determine segment
        if ltv >= self.segment_thresholds["high_value"]["min_ltv"] and \
           order_count >= self.segment_thresholds["high_value"]["min_frequency"]:
            segment = CustomerSegment.HIGH_VALUE.value
            confidence = 0.9
        elif days_since_purchase > 90:
            segment = CustomerSegment.CHURNED.value
            confidence = 0.85
        elif days_since_purchase > self.segment_thresholds["at_risk"]["days_since_purchase"]:
            segment = CustomerSegment.AT_RISK.value
            confidence = 0.75
        elif growth_rate >= self.segment_thresholds["growing"]["min_growth_rate"] and \
             order_count >= self.segment_thresholds["growing"]["min_orders"]:
            segment = CustomerSegment.GROWING.value
            confidence = 0.8
        else:
            segment = CustomerSegment.NEW.value
            confidence = 0.7
        
        return ClassificationResult(
            entity_id="",
            classification=segment,
            confidence=confidence,
            factors=factors
        )
