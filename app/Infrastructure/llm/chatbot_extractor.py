"""
Chatbot Response Extractor - Extracts and structures LLM responses for chatbot
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import json
import re
from app.Infrastructure.llm.response_parser import ResponseParser, ParsedResponse


@dataclass
class ChatbotResponse:
    """Structured response for chatbot consumption."""
    
    # Core response
    message: str
    
    # Metadata
    confidence: float
    response_type: str  # 'text', 'json', 'action', 'error'
    
    # Structured data (if applicable)
    data: Optional[Dict[str, Any]] = None
    action: Optional[str] = None
    
    # Context/sources
    sources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Error handling
    error: Optional[str] = None
    fallback: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "message": self.message,
            "confidence": self.confidence,
            "type": self.response_type,
            "data": self.data,
            "action": self.action,
            "sources": self.sources,
            "metadata": self.metadata,
            "error": self.error,
            "fallback": self.fallback
        }


class ChatbotResponseExtractor:
    """Extracts and structures LLM responses for chatbot delivery."""
    
    def __init__(self):
        """Initialize extractor with parser."""
        self.parser = ResponseParser()
        self.error_messages = {
            "parse_error": "I encountered an issue parsing the response. Here's what I found:",
            "empty_response": "I received an empty response. Please try again.",
            "no_context": "I don't have sufficient context to answer this question.",
        }
    
    def extract_chatbot_response(
        self,
        llm_response: str,
        context_sources: Optional[List[str]] = None,
        expected_schema: Optional[Dict] = None,
        fallback_text: Optional[str] = None,
        confidence_threshold: float = 0.5
    ) -> ChatbotResponse:
        """
        Extract and structure LLM response for chatbot.
        
        Args:
            llm_response: Raw LLM response text
            context_sources: List of sources used for context
            expected_schema: Optional JSON schema for validation
            fallback_text: Text to use if parsing fails
            confidence_threshold: Minimum confidence score to accept
            
        Returns:
            ChatbotResponse ready for chat endpoint
        """
        if not llm_response or not llm_response.strip():
            return self._create_error_response(
                "empty_response",
                fallback_text or self.error_messages["empty_response"]
            )
        
        # Try to parse as JSON first
        json_parsed = self.parser.parse_json_response(
            llm_response,
            schema=expected_schema
        )
        
        if json_parsed.structured_data and json_parsed.confidence > confidence_threshold:
            return self._create_structured_response(
                json_parsed,
                context_sources
            )
        
        # If JSON parsing failed or confidence too low, extract action/intent
        action = self.parser.extract_action(llm_response)
        
        if action:
            return self._create_action_response(
                llm_response,
                action,
                context_sources
            )
        
        # Fall back to plain text response
        return self._create_text_response(
            llm_response,
            confidence=json_parsed.confidence if json_parsed.confidence > 0 else 0.7,
            context_sources=context_sources,
            fallback=True
        )
    
    def _create_text_response(
        self,
        text: str,
        confidence: float = 0.7,
        context_sources: Optional[List[str]] = None,
        fallback: bool = False
    ) -> ChatbotResponse:
        """Create a plain text response."""
        # Clean up text (remove markdown, code blocks, etc.)
        clean_text = self._clean_text(text)
        
        return ChatbotResponse(
            message=clean_text,
            confidence=confidence,
            response_type="text",
            sources=context_sources or [],
            fallback=fallback
        )
    
    def _create_structured_response(
        self,
        parsed: ParsedResponse,
        context_sources: Optional[List[str]] = None
    ) -> ChatbotResponse:
        """Create a structured (JSON) response."""
        # Extract main message if available
        message = ""
        if isinstance(parsed.structured_data, dict):
            message = parsed.structured_data.get(
                "message",
                parsed.structured_data.get(
                    "answer",
                    json.dumps(parsed.structured_data, ensure_ascii=False, indent=2)
                )
            )
        
        if not message:
            message = parsed.raw_content
        
        clean_text = self._clean_text(message)
        
        return ChatbotResponse(
            message=clean_text,
            confidence=parsed.confidence,
            response_type="json",
            data=parsed.structured_data,
            sources=context_sources or []
        )
    
    def _create_action_response(
        self,
        text: str,
        action: str,
        context_sources: Optional[List[str]] = None
    ) -> ChatbotResponse:
        """Create an action-based response."""
        clean_text = self._clean_text(text)
        
        return ChatbotResponse(
            message=clean_text,
            confidence=0.8,
            response_type="action",
            action=action,
            sources=context_sources or []
        )
    
    def _create_error_response(
        self,
        error_type: str,
        fallback_message: Optional[str] = None
    ) -> ChatbotResponse:
        """Create an error response."""
        message = fallback_message or self.error_messages.get(
            error_type,
            "An unexpected error occurred."
        )
        
        return ChatbotResponse(
            message=message,
            confidence=0.0,
            response_type="error",
            error=error_type,
            fallback=True
        )
    
    def _clean_text(self, text: str) -> str:
        """
        Clean LLM response text for chatbot display.
        
        - Remove code block markers
        - Remove markdown formatting
        - Clean up excessive whitespace
        """
        # Remove markdown code blocks
        text = re.sub(r'```[\w]*\n?(.*?)\n?```', r'\1', text, flags=re.DOTALL)
        
        # Remove markdown headers
        text = re.sub(r'#{1,6}\s+', '', text)
        
        # Remove markdown bold/italic
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        text = re.sub(r'__(.*?)__', r'\1', text)
        text = re.sub(r'_(.*?)_', r'\1', text)
        
        # Clean up whitespace
        text = re.sub(r'\n\n\n+', '\n\n', text)
        text = text.strip()
        
        return text
    
    def extract_insights_from_analysis(
        self,
        analysis_response: str,
        analysis_type: str = "channel"
    ) -> ChatbotResponse:
        """
        Extract key insights from analysis response.
        
        Args:
            analysis_response: Raw analysis from engine
            analysis_type: Type of analysis (channel, strategy, etc)
            
        Returns:
            ChatbotResponse with extracted insights
        """
        # Try to parse as JSON first
        parsed = self.parser.parse_json_response(analysis_response)
        
        if parsed.structured_data:
            insights = self._extract_key_insights(
                parsed.structured_data,
                analysis_type
            )
            message = self._format_insights(insights, analysis_type)
            
            return ChatbotResponse(
                message=message,
                confidence=parsed.confidence,
                response_type="json",
                data=parsed.structured_data,
                metadata={"insights": insights, "type": analysis_type}
            )
        
        # Fallback to extracting bullet points/summaries
        return self._create_text_response(
            analysis_response,
            confidence=0.6,
            fallback=True
        )
    
    def _extract_key_insights(
        self,
        data: Dict[str, Any],
        analysis_type: str
    ) -> List[str]:
        """Extract key insights from structured data."""
        insights = []
        
        # Look for common insight fields
        insight_keys = [
            "insights", "key_findings", "recommendations",
            "findings", "analysis", "summary"
        ]
        
        for key in insight_keys:
            if key in data:
                value = data[key]
                if isinstance(value, list):
                    insights.extend([str(i) for i in value[:3]])  # Top 3
                elif isinstance(value, str):
                    insights.append(value)
                elif isinstance(value, dict):
                    insights.append(json.dumps(value, ensure_ascii=False))
        
        # If no insights found, extract all string values
        if not insights:
            for v in data.values():
                if isinstance(v, str) and len(v) > 20:
                    insights.append(v)
                    if len(insights) >= 3:
                        break
        
        return insights[:5]  # Return top 5 insights
    
    def _format_insights(
        self,
        insights: List[str],
        analysis_type: str
    ) -> str:
        """Format insights for chatbot display."""
        if not insights:
            return "No specific insights available."
        
        header = f"Key {analysis_type.title()} Insights:\n\n"
        formatted = header + "\n".join(f"• {insight}" for insight in insights)
        
        return formatted
    
    def batch_extract_responses(
        self,
        responses: List[str],
        context_sources: Optional[List[List[str]]] = None
    ) -> List[ChatbotResponse]:
        """
        Extract multiple responses in batch.
        
        Args:
            responses: List of LLM responses
            context_sources: Optional list of source lists for each response
            
        Returns:
            List of ChatbotResponse objects
        """
        results = []
        for i, response in enumerate(responses):
            sources = context_sources[i] if context_sources else None
            extracted = self.extract_chatbot_response(
                response,
                context_sources=sources
            )
            results.append(extracted)
        
        return results
