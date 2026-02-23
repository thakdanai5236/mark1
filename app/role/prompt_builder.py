"""
Prompt Builder - Constructs prompts with persona and context
"""

from typing import Optional, List, Dict, Any
from app.role.persona import Persona


class PromptBuilder:
    """Builds prompts combining persona, context, and user queries."""
    
    def __init__(self, persona: Persona):
        self.persona = persona
    
    def build_prompt(
        self,
        user_query: str,
        context: Optional[str] = None,
        additional_instructions: Optional[List[str]] = None,
        output_format: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Build a complete prompt with system and user messages.
        
        Args:
            user_query: The user's question or request
            context: Retrieved context from RAG
            additional_instructions: Extra instructions to include
            output_format: Desired output format specification
            
        Returns:
            Dictionary with 'system' and 'user' message content
        """
        system_prompt = self.persona.to_system_prompt()
        
        if additional_instructions:
            instructions_str = "\n".join(f"- {i}" for i in additional_instructions)
            system_prompt += f"\n\nAdditional Instructions:\n{instructions_str}"
        
        if output_format:
            system_prompt += f"\n\nOutput Format: {output_format}"
        
        user_message = user_query
        if context:
            user_message = f"""Context Information:
{context}

---

User Query: {user_query}"""
        
        return {
            "system": system_prompt,
            "user": user_message
        }
    
    def build_analysis_prompt(
        self,
        data_summary: str,
        analysis_type: str,
        specific_questions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Build a prompt for data analysis tasks."""
        
        query = f"""Please perform a {analysis_type} analysis on the following data:

{data_summary}
"""
        
        if specific_questions:
            questions_str = "\n".join(f"- {q}" for q in specific_questions)
            query += f"\n\nSpecifically address:\n{questions_str}"
        
        return self.build_prompt(
            user_query=query,
            output_format="Structured analysis with key findings and recommendations"
        )
