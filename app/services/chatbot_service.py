"""
Chatbot Service - End-to-end chat service with RAG integration
"""

from typing import Optional, Dict, Any, List
import asyncio
from app.Infrastructure.llm.chatbot_extractor import ChatbotResponseExtractor, ChatbotResponse
from app.Infrastructure.llm.client import OpenAIClient, OllamaClient, LLMClientFactory
from app.role.prompt_builder import PromptBuilder
from app.role.persona import Persona
from app.Infrastructure.rag.retriever import Retriever
from app.Infrastructure.rag.context_builder import ContextBuilder
from app.config import settings


class ChatbotService:
    """Service for handling chatbot conversations with RAG and LLM."""
    
    def __init__(
        self,
        llm_client: Optional[OpenAIClient] = None,
        retriever: Optional[Retriever] = None,
        use_rag: bool = True
    ):
        """
        Initialize chatbot service.
        
        Args:
            llm_client: OpenAI client (lazy-initialized if not provided)
            retriever: RAG retriever (lazy-initialized if not provided)
            use_rag: Whether to use RAG for context
        """
        self.llm_client = llm_client
        self.retriever = retriever
        self.use_rag = use_rag
        
        # Initialize extractors and builders
        self.response_extractor = ChatbotResponseExtractor()
        self.context_builder = ContextBuilder(
            max_tokens=4000,
            include_sources=True
        )
        
        # Default persona for marketing analytics
        self.default_persona = Persona(
            name="Mark Analytics Agent",
            role="Marketing Analytics Expert",
            description="An AI-powered agent specializing in marketing analytics, demo rate optimization, and channel performance analysis.",
            expertise=[
                "Channel performance analysis",
                "Demo rate optimization",
                "Budget allocation strategies",
                "Lead scoring and segmentation",
                "Conversion funnel analysis"
            ],
            communication_style="Professional, data-driven, and insights-focused",
            goals=[
                "Optimize demo rates across marketing channels",
                "Provide actionable recommendations for channel allocation",
                "Identify growth opportunities and bottlenecks"
            ]
        )
        self.prompt_builder = PromptBuilder(self.default_persona)
        
        # Conversation history for context
        self.conversation_history: Dict[str, List[Dict[str, str]]] = {}
    
    def _ensure_llm_client(self):
        """Lazy initialize LLM client based on configuration."""
        if self.llm_client is None:
            provider = settings.LLM_PROVIDER.lower()
            
            if provider == "ollama":
                # Use Ollama (runs locally)
                self.llm_client = OllamaClient(
                    base_url=settings.OLLAMA_BASE_URL,
                    model=settings.OLLAMA_MODEL,
                    default_temperature=settings.LLM_TEMPERATURE
                )
                
                # Check if Ollama is running
                if not self.llm_client.health_check():
                    raise RuntimeError(
                        f"Ollama server is not running at {settings.OLLAMA_BASE_URL}. "
                        f"Make sure Ollama is installed and running: ollama serve"
                    )
            
            elif provider == "openai":
                # Use OpenAI (requires API key)
                if not settings.OPENAI_API_KEY:
                    raise ValueError(
                        "OPENAI_API_KEY not configured. "
                        "Set LLM_PROVIDER=ollama or provide OPENAI_API_KEY"
                    )
                
                self.llm_client = OpenAIClient(
                    api_key=settings.OPENAI_API_KEY,
                    model=settings.LLM_MODEL,
                    default_temperature=settings.LLM_TEMPERATURE
                )
            
            else:
                raise ValueError(
                    f"Unknown LLM_PROVIDER: {provider}. "
                    f"Use 'ollama' or 'openai'"
                )
    
    def _ensure_retriever(self):
        """Lazy initialize retriever."""
        if self.retriever is None:
            if not self.use_rag:
                return
            from app.Infrastructure.rag.retriever import Retriever
            self.retriever = Retriever(
                vector_store_path=settings.VECTOR_STORE_PATH,
                embedding_model=settings.EMBEDDING_MODEL
            )
    
    async def chat(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        retrieve_context: bool = True,
        max_context_docs: int = 3
    ) -> ChatbotResponse:
        """
        Process a chat message and return response.
        
        Args:
            message: User message
            conversation_id: Optional conversation ID for history
            context: Optional additional context
            retrieve_context: Whether to use RAG for context
            max_context_docs: Maximum documents to retrieve
            
        Returns:
            ChatbotResponse with message and metadata
        """
        self._ensure_llm_client()
        
        # Retrieve context from RAG if enabled
        rag_context = ""
        sources = []
        
        if self.use_rag and retrieve_context:
            self._ensure_retriever()
            rag_context, sources = await self._retrieve_context(
                message,
                max_docs=max_context_docs
            )
        
        # Build prompt
        prompt_data = self.prompt_builder.build_prompt(
            user_query=message,
            context=rag_context,
            additional_instructions=[
                "Be concise and actionable",
                "Reference specific metrics and data when available",
                "Provide recommendations based on the analysis"
            ]
        )
        
        # Maintain conversation history
        if conversation_id:
            if conversation_id not in self.conversation_history:
                self.conversation_history[conversation_id] = []
            
            self.conversation_history[conversation_id].append({
                "role": "user",
                "content": message
            })
        
        # Get LLM response
        try:
            llm_response = await self._get_llm_response(
                prompt_data,
                conversation_id
            )
            
            # Extract structured response
            chatbot_response = self.response_extractor.extract_chatbot_response(
                llm_response,
                context_sources=sources,
                fallback_text="I've processed your request. Here's what I found..."
            )
            
        except Exception as e:
            return self.response_extractor._create_error_response(
                "parse_error",
                f"I encountered an error: {str(e)[:100]}. Please try again."
            )
        
        # Update conversation history with assistant response
        if conversation_id:
            self.conversation_history[conversation_id].append({
                "role": "assistant",
                "content": chatbot_response.message
            })
        
        return chatbot_response
    
    async def _retrieve_context(
        self,
        query: str,
        max_docs: int = 3
    ) -> tuple:
        """
        Retrieve context from RAG system.
        
        Args:
            query: User query for retrieval
            max_docs: Maximum documents to retrieve
            
        Returns:
            Tuple of (context_string, sources_list)
        """
        try:
            # Retrieve documents
            documents = await self.retriever.retrieve(
                query,
                k=max_docs,
                score_threshold=0.0
            )
            
            if not documents:
                return "", []
            
            # Build context string
            context = self.context_builder.build_context(
                documents,
                min_score=0.0
            )
            
            # Extract sources
            sources = list(set([doc.source for doc in documents]))
            
            return context, sources
            
        except Exception as e:
            print(f"Warning: RAG retrieval failed: {e}")
            return "", []
    
    async def _get_llm_response(
        self,
        prompt_data: Dict[str, str],
        conversation_id: Optional[str] = None
    ) -> str:
        """
        Get response from LLM.
        
        Args:
            prompt_data: Dict with 'system' and 'user' prompts
            conversation_id: Optional conversation ID for history
            
        Returns:
            LLM response text
        """
        messages = []
        
        # Add conversation history if available
        if conversation_id and conversation_id in self.conversation_history:
            for msg in self.conversation_history[conversation_id]:
                messages.append(msg)
        else:
            # Add current message only
            messages.append({
                "role": "user",
                "content": prompt_data["user"]
            })
        
        # Call LLM (running sync function in executor for async context)
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.llm_client.generate(
                messages=[
                    {"role": "system", "content": prompt_data["system"]}
                ] + messages
            )
        )
        
        return response
    
    async def analyze(
        self,
        analysis_type: str,
        data_summary: str,
        specific_questions: Optional[List[str]] = None
    ) -> ChatbotResponse:
        """
        Perform targeted analysis on data.
        
        Args:
            analysis_type: Type of analysis (channel, strategy, etc)
            data_summary: Summary of data to analyze
            specific_questions: Optional specific questions to address
            
        Returns:
            ChatbotResponse with analysis insights
        """
        self._ensure_llm_client()
        
        # Build analysis prompt
        prompt_data = self.prompt_builder.build_analysis_prompt(
            data_summary=data_summary,
            analysis_type=analysis_type,
            specific_questions=specific_questions
        )
        
        # Get LLM response
        try:
            loop = asyncio.get_event_loop()
            llm_response = await loop.run_in_executor(
                None,
                lambda: self.llm_client.generate(
                    messages=[
                        {"role": "system", "content": prompt_data["system"]},
                        {"role": "user", "content": prompt_data["user"]}
                    ]
                )
            )
            
            # Extract insights from analysis
            response = self.response_extractor.extract_insights_from_analysis(
                llm_response,
                analysis_type=analysis_type
            )
            
            return response
            
        except Exception as e:
            return self.response_extractor._create_error_response(
                "parse_error",
                f"Analysis failed: {str(e)[:100]}"
            )
    
    def clear_conversation(self, conversation_id: str):
        """Clear conversation history."""
        if conversation_id in self.conversation_history:
            del self.conversation_history[conversation_id]
    
    def get_conversation_history(
        self,
        conversation_id: str,
        last_n: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """Get conversation history."""
        if conversation_id not in self.conversation_history:
            return []
        
        history = self.conversation_history[conversation_id]
        if last_n:
            return history[-last_n:]
        return history


# Global singleton instance (lazy-loaded)
_chatbot_service: Optional[ChatbotService] = None


def get_chatbot_service() -> ChatbotService:
    """Get or create global chatbot service."""
    global _chatbot_service
    if _chatbot_service is None:
        _chatbot_service = ChatbotService(use_rag=True)
    return _chatbot_service
