"""
Response Parser - Parses and validates LLM responses
"""

from typing import Dict, List, Optional, Any, Type
from dataclasses import dataclass
import json
import re


@dataclass
class ParsedResponse:
    """Container for parsed LLM response."""
    raw_content: str
    structured_data: Optional[Dict[str, Any]] = None
    action: Optional[str] = None
    entities: Optional[List[Dict]] = None
    confidence: float = 1.0


class ResponseParser:
    """Parses and validates responses from LLM."""
    
    def __init__(self):
        """Initialize response parser."""
        self.json_pattern = re.compile(r'```json\s*(.*?)\s*```', re.DOTALL)
        self.code_pattern = re.compile(r'```(\w+)?\s*(.*?)\s*```', re.DOTALL)
    
    def parse_json_response(
        self,
        response: str,
        schema: Optional[Dict] = None
    ) -> ParsedResponse:
        """
        Parse JSON from LLM response.
        
        Args:
            response: Raw LLM response
            schema: Optional JSON schema for validation
            
        Returns:
            ParsedResponse with structured data
        """
        # Try to extract JSON from code blocks
        json_match = self.json_pattern.search(response)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON
            try:
                # Look for JSON object
                start = response.index('{')
                end = response.rindex('}') + 1
                json_str = response[start:end]
            except ValueError:
                return ParsedResponse(
                    raw_content=response,
                    confidence=0.5
                )
        
        try:
            data = json.loads(json_str)
            
            # Validate against schema if provided
            confidence = 1.0
            if schema:
                confidence = self._validate_against_schema(data, schema)
            
            return ParsedResponse(
                raw_content=response,
                structured_data=data,
                confidence=confidence
            )
        except json.JSONDecodeError as e:
            return ParsedResponse(
                raw_content=response,
                confidence=0.3
            )
    
    def _validate_against_schema(
        self,
        data: Dict,
        schema: Dict
    ) -> float:
        """
        Validate data against a simple schema.
        Returns confidence score based on validation.
        """
        required_fields = schema.get("required", [])
        properties = schema.get("properties", {})
        
        score = 1.0
        
        # Check required fields
        for field in required_fields:
            if field not in data:
                score -= 0.2
        
        # Check field types
        for field, value in data.items():
            if field in properties:
                expected_type = properties[field].get("type")
                if expected_type:
                    actual_type = type(value).__name__
                    type_mapping = {
                        "string": "str",
                        "number": ("int", "float"),
                        "integer": "int",
                        "boolean": "bool",
                        "array": "list",
                        "object": "dict"
                    }
                    expected = type_mapping.get(expected_type, expected_type)
                    if isinstance(expected, tuple):
                        if actual_type not in expected:
                            score -= 0.1
                    elif actual_type != expected:
                        score -= 0.1
        
        return max(0, score)
    
    def extract_action(self, response: str) -> Optional[str]:
        """
        Extract action/intent from response.
        
        Args:
            response: LLM response text
            
        Returns:
            Extracted action or None
        """
        action_patterns = [
            r'ACTION:\s*(\w+)',
            r'\[ACTION\]\s*(\w+)',
            r'I will\s+(\w+)',
            r'Let me\s+(\w+)'
        ]
        
        for pattern in action_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).lower()
        
        return None
    
    def extract_entities(
        self,
        response: str,
        entity_types: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Extract named entities from response.
        
        Args:
            response: LLM response text
            entity_types: Types of entities to extract
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        # Simple pattern-based extraction
        patterns = {
            "channel": r'(?:channel|platform)[:\s]+([A-Za-z\s]+)',
            "metric": r'(?:metric|KPI)[:\s]+([A-Za-z\s_]+)',
            "amount": r'(?:budget|cost|revenue)[:\s]+(\d+(?:,\d{3})*(?:\.\d{2})?)',
            "percentage": r'(\d+(?:\.\d+)?)\s*%',
            "date": r'(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4})'
        }
        
        for entity_type in entity_types:
            if entity_type in patterns:
                matches = re.findall(patterns[entity_type], response, re.IGNORECASE)
                for match in matches:
                    entities.append({
                        "type": entity_type,
                        "value": match.strip() if isinstance(match, str) else match
                    })
        
        return entities
    
    def parse_analysis_response(
        self,
        response: str
    ) -> Dict[str, Any]:
        """
        Parse an analysis response into structured sections.
        
        Args:
            response: LLM analysis response
            
        Returns:
            Dict with parsed sections
        """
        sections = {
            "summary": "",
            "findings": [],
            "recommendations": [],
            "metrics": {}
        }
        
        # Split by common section headers
        current_section = "summary"
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for section headers
            lower_line = line.lower()
            if 'finding' in lower_line or 'insight' in lower_line:
                current_section = "findings"
                continue
            elif 'recommend' in lower_line or 'suggestion' in lower_line:
                current_section = "recommendations"
                continue
            elif 'metric' in lower_line or 'kpi' in lower_line:
                current_section = "metrics"
                continue
            elif 'summary' in lower_line or 'overview' in lower_line:
                current_section = "summary"
                continue
            
            # Add content to current section
            if current_section == "summary":
                sections["summary"] += line + " "
            elif current_section in ["findings", "recommendations"]:
                if line.startswith(('-', '•', '*', '1', '2', '3', '4', '5')):
                    line = re.sub(r'^[-•*\d.]+\s*', '', line)
                if line:
                    sections[current_section].append(line)
        
        sections["summary"] = sections["summary"].strip()
        
        return sections
