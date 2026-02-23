# Chatbot Response Extraction System

## Overview

The Chatbot Response Extraction system is a comprehensive solution for processing LLM responses and preparing them for chatbot delivery. It combines:

- **Response Parsing** - Extracts structured data from unstructured LLM outputs
- **Context Management** - Preserves RAG source information
- **Error Handling** - Graceful fallbacks for parsing failures
- **Type Detection** - Identifies response types (text, JSON, action-based, errors)
- **Confidence Scoring** - Indicates reliability of parsed responses

## Architecture

```
LLM Response
    ↓
ChatbotResponseExtractor
    ├─ ResponseParser (parse JSON, extract actions)
    ├─ Text Cleaner (remove markdown, code blocks)
    └─ Structured Data Extractor (JSON, insights, entities)
    ↓
ChatbotResponse (structured output)
    ├─ message (clean text for UI)
    ├─ confidence (0-1 score)
    ├─ type (text/json/action/error)
    ├─ data (structured JSON data if available)
    ├─ sources (RAG sources used)
    └─ metadata (additional context)
    ↓
API Response / Chat Endpoint
```

## Core Components

### 1. ChatbotResponseExtractor

**File:** `app/llm/chatbot_extractor.py`

Main class for extracting responses. Key methods:

```python
extractor = ChatbotResponseExtractor()

# Extract single response
response = extractor.extract_chatbot_response(
    llm_response="Raw LLM output",
    context_sources=["source1.csv", "source2.json"],
    expected_schema={"type": "object", "properties": {...}},
    confidence_threshold=0.5
)

# Extract insights from analysis
response = extractor.extract_insights_from_analysis(
    analysis_response=json_string,
    analysis_type="channel"
)

# Batch extract multiple responses
responses = extractor.batch_extract_responses(
    responses=[resp1, resp2, ...],
    context_sources=[[sources1], [sources2], ...]
)
```

### 2. ChatbotResponse

**Data Structure:** Structured container for chatbot delivery

```python
@dataclass
class ChatbotResponse:
    message: str                                    # Clean text for display
    confidence: float                               # 0-1 confidence score
    response_type: str                             # text/json/action/error
    data: Optional[Dict[str, Any]]                 # Structured data if JSON
    action: Optional[str]                          # Action name if action-based
    sources: List[str]                             # RAG source references
    metadata: Dict[str, Any]                       # Additional context
    error: Optional[str]                           # Error type if applicable
    fallback: bool                                 # Whether fallback was used
```

### 3. ChatbotService

**File:** `app/services/chatbot_service.py`

End-to-end chat service with RAG integration:

```python
service = ChatbotService(use_rag=True)

# Process chat message
response = await service.chat(
    message="What are the best performing channels?",
    conversation_id="conv_123",
    retrieve_context=True,
    max_context_docs=3
)

# Perform targeted analysis
response = await service.analyze(
    analysis_type="channel",
    data_summary="...",
    specific_questions=["Q1", "Q2"]
)

# Get conversation history
history = service.get_conversation_history("conv_123")

# Clear conversation
service.clear_conversation("conv_123")
```

## Response Types

### Text Responses
Plain text responses cleaned of markdown:

```python
ChatbotResponse(
    message="The demo rate improved by 15%...",
    response_type="text",
    confidence=0.7
)
```

### JSON Responses
Structured data extracted from JSON blocks:

```python
ChatbotResponse(
    message="Analysis complete",
    response_type="json",
    data={"insights": [...], "recommendations": [...]},
    confidence=0.95
)
```

### Action Responses
Responses indicating specific actions:

```python
ChatbotResponse(
    message="I will optimize channel allocation...",
    response_type="action",
    action="optimize_allocation",
    confidence=0.8
)
```

### Error Responses
Graceful error handling with fallbacks:

```python
ChatbotResponse(
    message="I couldn't process that. Please try again.",
    response_type="error",
    error="parse_error",
    fallback=True,
    confidence=0.0
)
```

## API Integration

### Chat Endpoint

**Route:** `POST /chat`

**Request:**
```json
{
    "message": "What channels should I focus on?",
    "conversation_id": "conv_123",
    "context": {}
}
```

**Response:**
```json
{
    "success": true,
    "message": "Based on the analysis...",
    "sources": ["channel_analysis.csv"],
    "data": {"insights": [...]},
    "suggested_actions": ["optimize_allocation"]
}
```

## Usage Examples

### Example 1: Basic Response Extraction

```python
from app.llm.chatbot_extractor import ChatbotResponseExtractor

extractor = ChatbotResponseExtractor()

response = extractor.extract_chatbot_response(
    llm_response="The data shows strong growth trends.",
    context_sources=["growth_report.csv"]
)

print(response.message)  # Clean text
print(response.confidence)  # 0.7
print(response.sources)  # ["growth_report.csv"]
```

### Example 2: With Expected Schema

```python
schema = {
    "type": "object",
    "properties": {
        "message": {"type": "string"},
        "insights": {"type": "array"},
        "recommendations": {"type": "array"}
    },
    "required": ["message"]
}

response = extractor.extract_chatbot_response(
    llm_response=llm_output,
    expected_schema=schema,
    confidence_threshold=0.8
)

# Will only accept JSON responses with >0.8 confidence
if response.type == "json" and response.confidence > 0.8:
    use_structured_data(response.data)
```

### Example 3: Analysis Insights

```python
response = extractor.extract_insights_from_analysis(
    analysis_response=analysis_json,
    analysis_type="channel"
)

# Automatic extraction of insights, formatted for display
print(response.message)
# Output:
# Key Channel Insights:
# • Channel A shows 15% improvement
# • Channel B has highest conversion rate
# ...
```

### Example 4: Full Chat Flow with Service

```python
from app.services.chatbot_service import ChatbotService

service = ChatbotService(use_rag=True)

# Chat with conversation history
response = await service.chat(
    message="What should I do next?",
    conversation_id="user_123"
)

# Response includes RAG context
print(f"Message: {response.message}")
print(f"Sources: {response.sources}")  # From RAG retrieval
print(f"Confidence: {response.confidence}")
```

## Confidence Scoring

Confidence scores indicate reliability:

- **0.95-1.0**: High confidence, structured JSON with valid schema
- **0.8-0.94**: Good confidence, clear action or well-formed JSON
- **0.5-0.79**: Medium confidence, plain text or partial parsing
- **0.0-0.49**: Low confidence, fallback text or error handling

Use confidence scores to:
- Route to human review if < 0.5
- Use structured data only if > 0.8
- Show certainty indicators in UI

## Error Handling

The system gracefully handles errors:

```python
response = extractor.extract_chatbot_response(
    llm_response="",  # Empty response
    fallback_text="I couldn't process that. Please try again."
)

# Result:
# response.type = "error"
# response.message = "I couldn't process that. Please try again."
# response.fallback = True
# response.error = "empty_response"
```

## Configuration

**Environment Variables** (in `.env`):

```
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4
LLM_TEMPERATURE=0.7
VECTOR_STORE_PATH=data/vector_store
EMBEDDING_MODEL=text-embedding-ada-002
```

**ChatbotService Configuration:**

```python
service = ChatbotService(
    llm_client=custom_client,      # Optional custom LLM client
    retriever=custom_retriever,    # Optional custom RAG retriever
    use_rag=True                   # Enable/disable RAG
)
```

## Performance Considerations

- **JSON Parsing**: Attempts direct JSON extraction before schema validation
- **Action Detection**: Uses regex patterns for fast intent detection
- **Context Limits**: Respects token limits (default 4000 tokens)
- **Batch Processing**: Efficient batch extraction for multiple responses
- **Conversation History**: Maintained in memory (configurable storage)

## Testing

Run examples:

```bash
cd Mark Rag
python examples/chatbot_extraction_examples.py
```

## Design Principles

1. **Deterministic Parsing** - No randomness in response extraction
2. **Graceful Degradation** - Always produces valid response, even on error
3. **Confidence-Based** - Scores indicate reliability for downstream use
4. **RAG-Aware** - Preserves source information from context
5. **Extensible** - Easy to add custom parsers and response types

## Future Enhancements

- [ ] Support for multi-turn analysis workflows
- [ ] Caching of frequently-asked insights
- [ ] Custom response templates per analysis type
- [ ] Export conversation history to structured formats
- [ ] Integration with conversation analytics
