"""
LLM Client - Handles communication with language models
"""

from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
import json


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    def generate_with_functions(
        self,
        messages: List[Dict[str, str]],
        functions: List[Dict],
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Generate a response with function calling."""
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI API client implementation."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4",
        default_temperature: float = 0.7
    ):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key
            model: Model to use
            default_temperature: Default temperature for generation
        """
        self.api_key = api_key
        self.model = model
        self.default_temperature = default_temperature
        self._client = None
    
    @property
    def client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: int = 2000
    ) -> str:
        """
        Generate a response from OpenAI.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Generation temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            Generated text response
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or self.default_temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    
    def generate_with_functions(
        self,
        messages: List[Dict[str, str]],
        functions: List[Dict],
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generate a response with function calling.
        
        Args:
            messages: List of message dicts
            functions: List of function definitions
            temperature: Generation temperature
            
        Returns:
            Dict with 'content' and/or 'function_call'
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            functions=functions,
            function_call="auto",
            temperature=temperature or self.default_temperature
        )
        
        message = response.choices[0].message
        result = {"content": message.content}
        
        if message.function_call:
            result["function_call"] = {
                "name": message.function_call.name,
                "arguments": json.loads(message.function_call.arguments)
            }
        
        return result
    
    def generate_structured(
        self,
        messages: List[Dict[str, str]],
        response_schema: Dict,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generate a structured JSON response.
        
        Args:
            messages: List of message dicts
            response_schema: JSON schema for response
            temperature: Generation temperature
            
        Returns:
            Parsed JSON response
        """
        # Add schema instruction to system message
        schema_instruction = f"\n\nRespond with valid JSON matching this schema:\n{json.dumps(response_schema, indent=2)}"
        
        enhanced_messages = messages.copy()
        if enhanced_messages and enhanced_messages[0]["role"] == "system":
            enhanced_messages[0]["content"] += schema_instruction
        else:
            enhanced_messages.insert(0, {
                "role": "system",
                "content": schema_instruction
            })
        
        response = self.generate(enhanced_messages, temperature)
        
        # Parse JSON from response
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            raise ValueError(f"Could not parse JSON from response: {response}")


class OllamaClient(BaseLLMClient):
    """Ollama local LLM client implementation."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama2",
        default_temperature: float = 0.7
    ):
        """
        Initialize Ollama client.
        
        Args:
            base_url: Ollama server base URL (default: localhost:11434)
            model: Model name to use
            default_temperature: Default temperature for generation
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.default_temperature = default_temperature
        self._client = None
    
    @property
    def client(self):
        """Lazy initialization of Ollama client."""
        if self._client is None:
            import requests
            self._client = requests
        return self._client
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: int = 2000
    ) -> str:
        """
        Generate a response from Ollama.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Generation temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            Generated text response
        """
        import requests
        
        # Convert messages to prompt for Ollama
        prompt = self._format_prompt(messages)
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "temperature": temperature or self.default_temperature,
                    "stream": False
                },
                timeout=300
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "").strip()
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama request failed: {e}")
    
    def generate_with_functions(
        self,
        messages: List[Dict[str, str]],
        functions: List[Dict],
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generate a response (Ollama doesn't support true function calling).
        
        Falls back to basic generation with function descriptions in prompt.
        
        Args:
            messages: List of message dicts
            functions: List of function definitions
            temperature: Generation temperature
            
        Returns:
            Dict with 'content'
        """
        # Add function descriptions to prompt
        function_str = "\n\nAvailable functions:\n"
        for func in functions:
            function_str += f"- {func.get('name')}: {func.get('description')}\n"
        
        messages = messages.copy()
        if messages and messages[0]["role"] == "system":
            messages[0]["content"] += function_str
        else:
            messages.insert(0, {"role": "system", "content": function_str})
        
        content = self.generate(messages, temperature)
        return {"content": content}
    
    def generate_structured(
        self,
        messages: List[Dict[str, str]],
        response_schema: Dict,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generate a structured JSON response.
        
        Args:
            messages: List of message dicts
            response_schema: JSON schema for response
            temperature: Generation temperature
            
        Returns:
            Parsed JSON response
        """
        # Add schema instruction to prompt
        schema_instruction = f"\n\nRespond with valid JSON matching this schema:\n{json.dumps(response_schema, indent=2)}\n\nStart your response with {{ and end with }}"
        
        enhanced_messages = messages.copy()
        if enhanced_messages and enhanced_messages[0]["role"] == "system":
            enhanced_messages[0]["content"] += schema_instruction
        else:
            enhanced_messages.insert(0, {
                "role": "system",
                "content": schema_instruction
            })
        
        response = self.generate(enhanced_messages, temperature)
        
        # Parse JSON from response
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            raise ValueError(f"Could not parse JSON from response: {response}")
    
    def _format_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Format messages into a prompt for Ollama."""
        prompt = ""
        
        for msg in messages:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            
            if role == "SYSTEM":
                prompt += f"System: {content}\n\n"
            elif role == "USER":
                prompt += f"User: {content}\n\n"
            elif role == "ASSISTANT":
                prompt += f"Assistant: {content}\n\n"
        
        prompt += "Assistant:"
        return prompt
    
    def health_check(self) -> bool:
        """Check if Ollama server is running."""
        import requests
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False


class LLMClientFactory:
    """Factory for creating LLM clients."""
    
    @staticmethod
    def create(
        provider: str = "ollama",
        **kwargs
    ) -> BaseLLMClient:
        """
        Create an LLM client.
        
        Args:
            provider: LLM provider name ('openai' or 'ollama')
            **kwargs: Provider-specific arguments
            
        Returns:
            LLM client instance
        """
        if provider == "openai":
            return OpenAIClient(**kwargs)
        elif provider == "ollama":
            return OllamaClient(**kwargs)
        else:
            raise ValueError(f"Unknown provider: {provider}")
