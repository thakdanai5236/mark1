"""
API Schemas - Pydantic models for request/response validation
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


# Analysis Schemas
class AnalysisRequest(BaseModel):
    """Request model for data analysis."""
    query: str = Field(..., description="Analysis query or question")
    data_source: Optional[str] = Field(None, description="Specific data source to analyze")
    date_range_start: Optional[datetime] = Field(None, description="Start of date range")
    date_range_end: Optional[datetime] = Field(None, description="End of date range")
    channels: Optional[List[str]] = Field(None, description="Channels to include")
    metrics: Optional[List[str]] = Field(None, description="Specific metrics to analyze")


class AnalysisResponse(BaseModel):
    """Response model for data analysis."""
    success: bool
    insights: List[str]
    recommendations: List[str]
    data: Optional[Dict[str, Any]] = None
    confidence: float
    sources: Optional[List[str]] = None


# Metric Schemas
class MetricRequest(BaseModel):
    """Request model for metric calculation."""
    metric_names: List[str] = Field(..., description="Names of metrics to calculate")
    date_range_start: Optional[datetime] = None
    date_range_end: Optional[datetime] = None
    group_by: Optional[str] = Field(None, description="Dimension to group by")
    filters: Optional[Dict[str, Any]] = Field(None, description="Filters to apply")


class MetricResponse(BaseModel):
    """Response model for metric calculation."""
    success: bool
    metrics: Dict[str, Any]
    period: Optional[str] = None
    comparison: Optional[Dict[str, float]] = None


# Reallocation Schemas
class AllocationConstraint(BaseModel):
    """Constraint for channel allocation."""
    channel: str
    min_percentage: float = Field(ge=0, le=1)
    max_percentage: float = Field(ge=0, le=1)


class ReallocationRequest(BaseModel):
    """Request model for budget reallocation."""
    total_budget: float = Field(..., gt=0)
    optimization_target: str = Field(default="roi", description="Metric to optimize")
    constraints: Optional[List[AllocationConstraint]] = None
    include_channels: Optional[List[str]] = None
    exclude_channels: Optional[List[str]] = None


class ChannelRecommendation(BaseModel):
    """Recommendation for a single channel."""
    channel: str
    current_allocation: float
    recommended_allocation: float
    change_percentage: float
    rationale: str


class ReallocationResponse(BaseModel):
    """Response model for budget reallocation."""
    success: bool
    recommendations: List[ChannelRecommendation]
    expected_improvement: float
    confidence_level: float = 0.75


# Simulation Schemas
class SimulationRequest(BaseModel):
    """Request model for simulation."""
    scenario_name: str
    scenario_type: str = Field(..., description="Type: budget_change, channel_mix, etc.")
    parameters: Dict[str, Any]
    duration_days: int = Field(default=30, ge=1, le=365)


class SimulationResponse(BaseModel):
    """Response model for simulation."""
    success: bool
    scenario_name: str
    results: Dict[str, Any]
    confidence_interval: Optional[tuple] = None
    assumptions: Optional[List[str]] = None


# Chat Schemas
class ChatRequest(BaseModel):
    """Request model for chat."""
    message: str = Field(..., min_length=1)
    conversation_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    """Response model for chat."""
    success: bool
    message: str
    sources: List[str]
    suggested_actions: Optional[List[str]] = None
    data: Optional[Dict[str, Any]] = None


# Lead Scoring Schemas
class LeadScoringRequest(BaseModel):
    """Request model for lead scoring."""
    lead_ids: Optional[List[str]] = None
    score_all: bool = False
    min_score_threshold: float = Field(default=0.0, ge=0, le=1)


class ScoredLead(BaseModel):
    """Scored lead result."""
    lead_id: str
    score: float
    classification: str
    factors: Dict[str, float]


class LeadScoringResponse(BaseModel):
    """Response model for lead scoring."""
    success: bool
    scored_leads: List[ScoredLead]
    summary: Dict[str, int]
