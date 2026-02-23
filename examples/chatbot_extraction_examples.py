"""
Example: Chatbot Response Extraction
====================================

This module demonstrates how to use the ChatbotResponseExtractor
for handling LLM responses in a chatbot context.
"""

from typing import Dict, Any
from app.llm.chatbot_extractor import ChatbotResponseExtractor, ChatbotResponse


def example_basic_response_extraction():
    """Example 1: Basic text response extraction."""
    
    extractor = ChatbotResponseExtractor()
    
    # Simulate LLM response
    llm_response = """
    Based on the data analysis, I found three key insights:
    
    1. Channel A has the highest demo rate at 45%
    2. Channel B shows consistent growth month-over-month
    3. Channel C has room for optimization in the demo-to-contact ratio
    """
    
    # Extract response
    chatbot_response = extractor.extract_chatbot_response(
        llm_response=llm_response,
        context_sources=["channel_analysis_report.csv"],
        confidence_threshold=0.5
    )
    
    print("Example 1: Basic Text Response")
    print(f"Message: {chatbot_response.message}")
    print(f"Type: {chatbot_response.response_type}")
    print(f"Confidence: {chatbot_response.confidence}")
    print(f"Sources: {chatbot_response.sources}")
    print()


def example_json_response_extraction():
    """Example 2: Structured JSON response extraction."""
    
    extractor = ChatbotResponseExtractor()
    
    # Simulate structured LLM response
    llm_response = """
    ```json
    {
        "message": "Analysis complete with key recommendations",
        "insights": [
            "Demo rate improved by 15% compared to last month",
            "Channel diversity increased from 3 to 5 active channels",
            "Cost per demo reduced by 22%"
        ],
        "recommendations": [
            "Scale investment in top-performing channels by 20%",
            "Run A/B test on new contact strategy for Channel D",
            "Pause underperforming campaigns in Channel C"
        ],
        "confidence_score": 0.92
    }
    ```
    """
    
    # Define expected schema
    schema = {
        "type": "object",
        "properties": {
            "message": {"type": "string"},
            "insights": {"type": "array"},
            "recommendations": {"type": "array"},
            "confidence_score": {"type": "number"}
        },
        "required": ["message", "insights"]
    }
    
    # Extract response
    chatbot_response = extractor.extract_chatbot_response(
        llm_response=llm_response,
        expected_schema=schema,
        context_sources=["marketing_master.csv", "channel_summary.csv"]
    )
    
    print("Example 2: JSON Response")
    print(f"Message: {chatbot_response.message}")
    print(f"Type: {chatbot_response.response_type}")
    print(f"Structured Data: {chatbot_response.data}")
    print(f"Sources: {chatbot_response.sources}")
    print()


def example_action_response_extraction():
    """Example 3: Action-based response extraction."""
    
    extractor = ChatbotResponseExtractor()
    
    # Simulate action-based response
    llm_response = """
    I'll analyze the channel performance data for you.
    
    ACTION: RUN_ANALYSIS
    
    This will calculate demo rates, identify growth channels, and 
    generate optimization recommendations based on the current data.
    """
    
    # Extract response
    chatbot_response = extractor.extract_chatbot_response(
        llm_response=llm_response,
        confidence_threshold=0.5
    )
    
    print("Example 3: Action Response")
    print(f"Message: {chatbot_response.message}")
    print(f"Type: {chatbot_response.response_type}")
    print(f"Action: {chatbot_response.action}")
    print()


def example_analysis_insights_extraction():
    """Example 4: Extract insights from analysis response."""
    
    extractor = ChatbotResponseExtractor()
    
    # Simulate analysis response
    analysis_response = """
    {
        "analysis_type": "channel",
        "total_leads": 5000,
        "total_demos": 1250,
        "overall_demo_rate": 0.25,
        "key_findings": [
            "Email channel outperforms average by 18%",
            "Social media shows highest growth trajectory at 35% month-over-month",
            "Direct sales underperforming with only 12% demo rate",
            "Referral channel has high-quality leads with 38% conversion"
        ],
        "top_channels": ["Email", "Referral", "Social Media"],
        "bottleneck_channels": ["Direct Sales", "Ads"],
        "recommendations": [
            "Increase email campaign frequency by 2x given strong performance",
            "Invest in social media expansion - shows best growth potential",
            "Revamp direct sales process or reduce allocation to this channel"
        ]
    }
    """
    
    # Extract insights
    chatbot_response = extractor.extract_insights_from_analysis(
        analysis_response,
        analysis_type="channel"
    )
    
    print("Example 4: Analysis Insights")
    print(f"Message:\n{chatbot_response.message}")
    print(f"Type: {chatbot_response.response_type}")
    print(f"Metadata: {chatbot_response.metadata}")
    print()


def example_error_handling():
    """Example 5: Error handling and fallback."""
    
    extractor = ChatbotResponseExtractor()
    
    # Empty response
    empty_response = extractor.extract_chatbot_response(
        llm_response="",
        fallback_text="I couldn't process that request. Please try again."
    )
    
    print("Example 5: Error Handling")
    print(f"Message: {empty_response.message}")
    print(f"Type: {empty_response.response_type}")
    print(f"Error: {empty_response.error}")
    print(f"Fallback: {empty_response.fallback}")
    print()


def example_batch_extraction():
    """Example 6: Batch response extraction."""
    
    extractor = ChatbotResponseExtractor()
    
    # Multiple responses
    responses = [
        "The demo rate has increased by 15% this month.",
        "```json\n{\"status\": \"success\", \"demos_scheduled\": 120}\n```",
        "I will optimize the channel allocation for you."
    ]
    
    sources_list = [
        ["report_march.csv"],
        ["live_data.json"],
        None
    ]
    
    # Extract all
    results = extractor.batch_extract_responses(responses, sources_list)
    
    print("Example 6: Batch Extraction")
    for i, response in enumerate(results, 1):
        print(f"Response {i}:")
        print(f"  Message: {response.message[:50]}...")
        print(f"  Type: {response.response_type}")
        print()


def example_response_to_dict():
    """Example 7: Convert response to dictionary for API."""
    
    extractor = ChatbotResponseExtractor()
    
    llm_response = "The analysis shows strong performance across all channels."
    
    chatbot_response = extractor.extract_chatbot_response(
        llm_response=llm_response,
        context_sources=["analysis_report.json"]
    )
    
    # Convert to dict for JSON serialization
    response_dict = chatbot_response.to_dict()
    
    print("Example 7: Response as Dictionary")
    import json
    print(json.dumps(response_dict, indent=2, ensure_ascii=False))
    print()


if __name__ == "__main__":
    print("=== Chatbot Response Extraction Examples ===\n")
    
    example_basic_response_extraction()
    example_json_response_extraction()
    example_action_response_extraction()
    example_analysis_insights_extraction()
    example_error_handling()
    example_batch_extraction()
    example_response_to_dict()
