# Ollama Integration Guide

## Overview

Mark1 now supports **Ollama** for running local LLMs, eliminating the need for OpenAI API keys and cloud dependencies. Ollama is the default LLM provider.

## What is Ollama?

[Ollama](https://ollama.ai) is a lightweight framework for running large language models locally on your machine.

**Advantages:**
- ✅ Free and open-source
- ✅ Runs entirely offline (no API calls)
- ✅ Privacy-preserving (data stays local)
- ✅ Fast inference on modern hardware
- ✅ Supports multiple model families

## Installation

### Step 1: Install Ollama

Download from [ollama.ai](https://ollama.ai) for your OS:

- **macOS**: [ollama-darwin.zip](https://ollama.ai/download)
- **Linux**: `curl https://ollama.ai/install.sh | sh`
- **Windows**: Download installer from [ollama.ai](https://ollama.ai)

### Step 2: Start Ollama Server

```bash
ollama serve
```

This starts the Ollama server on `http://localhost:11434` (default).

### Step 3: Pull a Model

In another terminal, pull a language model:

```bash
# Fast, lightweight model (good for testing)
ollama pull mistral

# Larger, more capable model
ollama pull llama2

# Alternative models
ollama pull neural-chat
ollama pull orca-mini
ollama pull dolphin-mixtral
```

List installed models:
```bash
ollama list
```

## Configuration

### Default Setup (Ollama)

The application uses Ollama by default. Ensure your `.env` file has:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2
```

### Switch to OpenAI

If you prefer OpenAI:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4
```

## Usage

### Start the Application

1. **Start Ollama server** (in one terminal):
   ```bash
   ollama serve
   ```

2. **Start Mark1 API** (in another terminal):
   ```bash
   python app/main.py
   ```

### Via API

```bash
# Chat endpoint
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the best performing channels?",
    "conversation_id": "user_123"
  }'
```

### Via Python

```python
from app.services.chatbot_service import ChatbotService
import asyncio

async def main():
    service = ChatbotService(use_rag=True)
    
    response = await service.chat(
        message="What should I optimize?",
        conversation_id="conv_1"
    )
    
    print(response.message)
    print(f"Confidence: {response.confidence}")
    print(f"Sources: {response.sources}")

asyncio.run(main())
```

## Recommended Models

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| **mistral** | 7B | ⚡⚡⚡ | ⭐⭐⭐⭐ | Balanced, fast |
| **llama2** | 7B/13B/70B | ⚡⚡ | ⭐⭐⭐⭐⭐ | General purpose |
| **neural-chat** | 7B | ⚡⚡⚡ | ⭐⭐⭐⭐ | Chat optimized |
| **orca-mini** | 3B | ⚡⚡⚡⚡ | ⭐⭐⭐ | Quick testing |
| **dolphin-mixtral** | 47B | ⚡ | ⭐⭐⭐⭐⭐ | Premium quality |

### For Marketing Analytics

**Recommended:** `mistral` or `neural-chat`
- Fast response times
- Good instruction following
- Suitable for structured analysis

## Hardware Requirements

| Model | VRAM Required | Notes |
|-------|---------------|-------|
| mistral (7B) | 8GB | Minimum recommended |
| llama2 (7B) | 8GB | Works well |
| llama2 (13B) | 16GB | Better quality |
| orca-mini (3B) | 4GB | Minimal viable |

## Performance Tips

1. **Use smaller models for faster responses:**
   ```bash
   ollama pull mistral  # Faster than llama2
   ```

2. **Increase GPU memory if available:**
   Models run faster on GPU (if your hardware supports it)

3. **Tune temperature for consistency:**
   - Lower (0.3-0.5): More deterministic, factual
   - Higher (0.7-0.9): More creative, varied

4. **Batch requests** for multiple analyses:
   ```python
   responses = extractor.batch_extract_responses(responses_list)
   ```

## Troubleshooting

### Issue: "Connection refused on localhost:11434"

**Solution:** Start Ollama server:
```bash
ollama serve
```

### Issue: Model not found

**Solution:** Pull the model:
```bash
ollama pull mistral
```

### Issue: Out of Memory (OOM)

**Solution:** Use smaller model:
```bash
# Switch from llama2 to mistral
ollama pull mistral
# Update .env: OLLAMA_MODEL=mistral
```

### Issue: Slow responses

**Solution:** 
- Use faster model (mistral, orca-mini)
- Check available GPU memory
- Run health check:
  ```python
  from app.llm.client import OllamaClient
  client = OllamaClient()
  print(client.health_check())  # Should print True
  ```

## Model Management

### List installed models
```bash
ollama list
```

### Remove a model
```bash
ollama rm llama2
```

### Run specific model directly
```bash
ollama run mistral
# Type your prompt and press Enter
```

## Advanced Configuration

### Custom Ollama Server

Using Ollama on a different machine:

```env
OLLAMA_BASE_URL=http://192.168.1.100:11434
OLLAMA_MODEL=mistral
```

### Model-specific Settings

Create a custom Modelfile:

```modelfile
FROM mistral
PARAMETER temperature 0.7
PARAMETER num_predict 2000
```

Load custom model:
```bash
ollama create my-custom-model -f Modelfile
ollama run my-custom-model
```

## Comparing with OpenAI

| Aspect | Ollama | OpenAI |
|--------|--------|--------|
| **Cost** | Free ✅ | $0.03-$0.06/1K tokens |
| **Privacy** | Local ✅ | Cloud ⚠️ |
| **Speed** | Fast (GPU) ✅ | Very fast (fast servers) |
| **Quality** | Good | Excellent |
| **Setup** | Simple ✅ | Requires API key |
| **Internet** | Optional | Required |

## Switching Providers

To easily switch between Ollama and OpenAI:

```python
from app.services.chatbot_service import ChatbotService
from app.llm.client import OllamaClient, OpenAIClient

# Use Ollama
llm = OllamaClient(model="mistral")
service = ChatbotService(llm_client=llm)

# Or use OpenAI
llm = OpenAIClient(api_key="sk-...", model="gpt-4")
service = ChatbotService(llm_client=llm)
```

## Production Deployment

For production use with Ollama:

1. **Run Ollama in background:**
   ```bash
   nohup ollama serve > ollama.log 2>&1 &
   ```

2. **Use process manager (e.g., systemd):**
   ```bash
   sudo systemctl enable ollama
   sudo systemctl start ollama
   ```

3. **Monitor health:**
   ```python
   from app.llm.client import OllamaClient
   
   client = OllamaClient()
   assert client.health_check(), "Ollama not running!"
   ```

4. **Configure firewall** (if on different machine):
   ```bash
   # Allow port 11434
   ufw allow 11434
   ```

## Resources

- **Ollama Documentation**: https://github.com/ollama/ollama
- **Model Registry**: https://ollama.ai/library
- **GitHub Issues**: https://github.com/ollama/ollama/issues
- **Community Discord**: Ollama official Discord community

## Examples

### Example 1: Basic Chat with Ollama

```python
from app.services.chatbot_service import ChatbotService
import asyncio

async def main():
    # Ollama is used by default (no API key needed)
    service = ChatbotService(use_rag=False)
    
    response = await service.chat(
        message="Analyze the demo rate trends"
    )
    
    print(response.message)

asyncio.run(main())
```

### Example 2: With RAG Context

```python
async def main():
    service = ChatbotService(use_rag=True)  # Uses Ollama
    
    response = await service.chat(
        message="What's driving demo rate increases?",
        retrieve_context=True,
        max_context_docs=5
    )
    
    print(f"Response: {response.message}")
    print(f"Sources: {response.sources}")
    print(f"Confidence: {response.confidence:.2f}")

asyncio.run(main())
```

### Example 3: Batch Analysis

```python
from app.llm.chatbot_extractor import ChatbotResponseExtractor

extractor = ChatbotResponseExtractor()

responses = [
    "Channel A: 45% demo rate",
    "Channel B: emerging with 30% rate",
    "Channel C: declining trend"
]

results = extractor.batch_extract_responses(responses)

for i, result in enumerate(results):
    print(f"Response {i+1}: {result.message}")
```

## Support

Having issues? Check:

1. Is Ollama running? `ollama serve`
2. Is the model pulled? `ollama list`
3. Is the server accessible? `curl http://localhost:11434/api/tags`
4. Check logs: `cat ollama.log` (if using nohup)
