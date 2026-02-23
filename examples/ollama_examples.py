"""
Example: Using Mark1 Chatbot with Ollama
========================================

This example demonstrates how to use the chatbot service with Ollama.
Make sure Ollama is running: ollama serve
"""

import asyncio
import json
from app.services.chatbot_service import ChatbotService
from app.llm.client import OllamaClient


async def example_basic_chat():
    """Example 1: Basic chat with Ollama."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Chat with Ollama")
    print("=" * 60)
    
    service = ChatbotService(use_rag=False)
    
    try:
        response = await service.chat(
            message="What are the key metrics for marketing performance?",
            conversation_id="demo_1"
        )
        
        print(f"\nUser: What are the key metrics for marketing performance?")
        print(f"Assistant: {response.message}")
        print(f"Confidence: {response.confidence:.2f}")
        print(f"Response Type: {response.response_type}")
        
    except RuntimeError as e:
        print(f"Error: {e}")
        print("\nMake sure Ollama is running: ollama serve")


async def example_conversation():
    """Example 2: Multi-turn conversation with Ollama."""
    print("\n" + "=" * 60)
    print("Example 2: Multi-turn Conversation")
    print("=" * 60)
    
    service = ChatbotService(use_rag=False)
    conversation_id = "demo_2"
    
    try:
        messages = [
            "What are the main marketing channels?",
            "How can we improve demo rates?",
            "What's the ROI of channel A?"
        ]
        
        for i, message in enumerate(messages, 1):
            print(f"\n[Turn {i}]")
            print(f"User: {message}")
            
            response = await service.chat(
                message=message,
                conversation_id=conversation_id
            )
            
            print(f"Assistant: {response.message[:200]}...")
            
            # Small delay between messages
            await asyncio.sleep(1)
        
        # Show conversation history
        history = service.get_conversation_history(conversation_id)
        print(f"\nConversation history ({len(history)} messages):")
        for msg in history[:2]:
            print(f"  - {msg['role']}: {msg['content'][:50]}...")
        
    except RuntimeError as e:
        print(f"Error: {e}")


async def example_ollama_health_check():
    """Example 3: Check Ollama server health."""
    print("\n" + "=" * 60)
    print("Example 3: Ollama Health Check")
    print("=" * 60)
    
    try:
        client = OllamaClient(
            base_url="http://localhost:11434",
            model="mistral"
        )
        
        is_healthy = client.health_check()
        
        if is_healthy:
            print("\n✅ Ollama server is running and healthy")
            print(f"Base URL: {client.base_url}")
            print(f"Model: {client.model}")
        else:
            print("\n❌ Ollama server is not responding")
            print("\nTo start Ollama, run:")
            print("  ollama serve")
            print("\nTo pull a model, run:")
            print("  ollama pull mistral")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")


async def example_model_comparison():
    """Example 4: Compare different Ollama models."""
    print("\n" + "=" * 60)
    print("Example 4: Model Comparison")
    print("=" * 60)
    
    models = ["mistral", "neural-chat"]
    question = "Summarize the benefits of marketing analytics"
    
    for model in models:
        print(f"\nTesting model: {model}")
        print("-" * 40)
        
        try:
            client = OllamaClient(
                base_url="http://localhost:11434",
                model=model
            )
            
            if not client.health_check():
                print(f"⚠️  Model '{model}' not found. Pull it with:")
                print(f"   ollama pull {model}")
                continue
            
            response = client.generate(
                messages=[{"role": "user", "content": question}],
                temperature=0.5,
                max_tokens=150
            )
            
            print(f"Response: {response[:150]}...")
            
        except Exception as e:
            print(f"Error with {model}: {e}")
        
        await asyncio.sleep(1)


async def example_structured_analysis():
    """Example 5: Structured analysis request."""
    print("\n" + "=" * 60)
    print("Example 5: Structured Analysis")
    print("=" * 60)
    
    service = ChatbotService(use_rag=False)
    
    try:
        response = await service.analyze(
            analysis_type="channel",
            data_summary="""
            Channel Performance Data:
            - Email: 5000 leads, 1500 demos (30% rate)
            - Social: 3000 leads, 900 demos (30% rate)
            - Direct: 2000 leads, 200 demos (10% rate)
            """,
            specific_questions=[
                "Which channel is most efficient?",
                "What's driving performance differences?",
                "Where should we invest next?"
            ]
        )
        
        print(f"\nAnalysis Response:")
        print(response.message)
        print(f"\nConfidence: {response.confidence:.2f}")
        
    except RuntimeError as e:
        print(f"Error: {e}")


async def example_with_custom_client():
    """Example 6: Using custom Ollama client."""
    print("\n" + "=" * 60)
    print("Example 6: Custom Ollama Configuration")
    print("=" * 60)
    
    # Create custom Ollama client
    ollama_client = OllamaClient(
        base_url="http://localhost:11434",
        model="mistral",  # Change to your preferred model
        default_temperature=0.5  # Lower temp for more deterministic responses
    )
    
    # Create service with custom client
    service = ChatbotService(llm_client=ollama_client, use_rag=False)
    
    try:
        response = await service.chat(
            message="What's a good demo rate benchmark?",
            conversation_id="custom_1"
        )
        
        print(f"Response: {response.message}")
        print(f"Model used: {ollama_client.model}")
        print(f"Temperature: {ollama_client.default_temperature}")
        
    except RuntimeError as e:
        print(f"Error: {e}")


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Mark1 Chatbot - Ollama Examples")
    print("=" * 60)
    print("\nMake sure Ollama is running before starting!")
    print("Terminal 1: ollama serve")
    print("Terminal 2: python examples/ollama_examples.py")
    
    # Example 0: Health check first
    await example_ollama_health_check()
    
    # If health check passes, run other examples
    try:
        await example_basic_chat()
        await example_conversation()
        await example_structured_analysis()
        await example_with_custom_client()
        await example_model_comparison()
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("\nTroubleshooting:")
        print("1. Check if Ollama is running: ollama serve")
        print("2. Check if model is pulled: ollama pull mistral")
        print("3. Check firewall: port 11434 should be accessible")


if __name__ == "__main__":
    asyncio.run(main())
