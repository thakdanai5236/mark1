"""
Reverse Engineering Engine - Analyzes successful outcomes to identify patterns
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class Pattern:
    """Represents a discovered pattern."""
    name: str
    description: str
    confidence: float
    supporting_evidence: List[str]
    actionable_insights: List[str]


@dataclass
class ReverseEngineeringResult:
    """Result of reverse engineering analysis."""
    target_metric: str
    patterns: List[Pattern]
    key_factors: Dict[str, float]
    recommendations: List[str]


class ReverseEngineeringEngine:
    """
    Analyzes successful outcomes to identify what made them successful.
    Useful for understanding why certain campaigns, channels, or segments perform well.
    """
    
    def __init__(self):
        """Initialize reverse engineering engine."""
        self.min_sample_size = 30
        self.significance_threshold = 0.05
    
    def analyze_success_factors(
        self,
        data: pd.DataFrame,
        outcome_col: str,
        success_threshold: float,
        feature_cols: List[str]
    ) -> ReverseEngineeringResult:
        """
        Analyze factors that contribute to successful outcomes.
        
        Args:
            data: DataFrame with campaign/lead data
            outcome_col: Column containing outcome metric
            success_threshold: Threshold for defining success
            feature_cols: Columns to analyze as potential factors
            
        Returns:
            ReverseEngineeringResult with patterns and insights
        """
        # Split into success and non-success groups
        successful = data[data[outcome_col] >= success_threshold]
        unsuccessful = data[data[outcome_col] < success_threshold]
        
        key_factors = {}
        patterns = []
        
        for col in feature_cols:
            # Compare distributions
            if data[col].dtype in ['float64', 'int64']:
                success_mean = successful[col].mean()
                all_mean = data[col].mean()
                
                if all_mean > 0:
                    factor_importance = (success_mean - all_mean) / all_mean
                    key_factors[col] = factor_importance
                    
                    if abs(factor_importance) > 0.2:
                        direction = "higher" if factor_importance > 0 else "lower"
                        patterns.append(Pattern(
                            name=f"{col}_pattern",
                            description=f"Successful outcomes have {direction} {col}",
                            confidence=min(0.95, 0.7 + abs(factor_importance)),
                            supporting_evidence=[
                                f"Success avg: {success_mean:.2f}",
                                f"Overall avg: {all_mean:.2f}"
                            ],
                            actionable_insights=[
                                f"Target {direction} {col} values for better outcomes"
                            ]
                        ))
            else:
                # Categorical analysis
                success_dist = successful[col].value_counts(normalize=True)
                all_dist = data[col].value_counts(normalize=True)
                
                for category in success_dist.index:
                    if category in all_dist.index:
                        lift = success_dist[category] / all_dist[category]
                        if lift > 1.3:
                            patterns.append(Pattern(
                                name=f"{col}_{category}_pattern",
                                description=f"{category} in {col} correlates with success",
                                confidence=min(0.9, 0.6 + (lift - 1) * 0.3),
                                supporting_evidence=[
                                    f"Success rate lift: {lift:.2f}x"
                                ],
                                actionable_insights=[
                                    f"Focus on {category} for {col}"
                                ]
                            ))
        
        # Generate recommendations
        recommendations = self._generate_recommendations(key_factors, patterns)
        
        return ReverseEngineeringResult(
            target_metric=outcome_col,
            patterns=patterns,
            key_factors=key_factors,
            recommendations=recommendations
        )
    
    def _generate_recommendations(
        self,
        factors: Dict[str, float],
        patterns: List[Pattern]
    ) -> List[str]:
        """Generate actionable recommendations from analysis."""
        recommendations = []
        
        # Sort factors by importance
        sorted_factors = sorted(
            factors.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        for factor, importance in sorted_factors[:5]:
            if importance > 0.2:
                recommendations.append(
                    f"Increase focus on {factor} - shows {importance*100:.1f}% positive correlation with success"
                )
            elif importance < -0.2:
                recommendations.append(
                    f"Reduce emphasis on {factor} - shows {abs(importance)*100:.1f}% negative correlation with success"
                )
        
        # Add pattern-based recommendations
        for pattern in patterns[:3]:
            recommendations.extend(pattern.actionable_insights)
        
        return recommendations
    
    def compare_segments(
        self,
        data: pd.DataFrame,
        segment_col: str,
        metric_cols: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare performance across different segments.
        
        Args:
            data: DataFrame with segment data
            segment_col: Column containing segment labels
            metric_cols: Columns to compare
            
        Returns:
            Dictionary of segment comparisons
        """
        results = {}
        
        for segment in data[segment_col].unique():
            segment_data = data[data[segment_col] == segment]
            results[segment] = {
                col: segment_data[col].mean()
                for col in metric_cols
            }
        
        return results
    
    def identify_winning_combinations(
        self,
        data: pd.DataFrame,
        outcome_col: str,
        combination_cols: List[str],
        min_samples: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Identify winning combinations of factors.
        
        Args:
            data: DataFrame with data
            outcome_col: Outcome metric column
            combination_cols: Columns to combine
            min_samples: Minimum samples for valid combination
            
        Returns:
            List of winning combinations with their performance
        """
        # Group by combinations
        grouped = data.groupby(combination_cols).agg({
            outcome_col: ['mean', 'count']
        }).reset_index()
        
        grouped.columns = combination_cols + ['avg_outcome', 'sample_count']
        
        # Filter by minimum samples
        valid = grouped[grouped['sample_count'] >= min_samples]
        
        # Sort by outcome
        top_combinations = valid.nlargest(10, 'avg_outcome')
        
        return top_combinations.to_dict('records')
