import os
from dotenv import load_dotenv
from agents import OpenAIChatCompletionsModel
from openai import AsyncOpenAI

def create_model(provider="openai", model_name=None):
    """
    Create model object based on provider and model name.
    
    Args:
        provider: Either "openai" (default) or "openrouter"
        model_name: Model identifier
    
    Returns:
        Model object or model name string
    """
    if provider == "openrouter":

        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required for OpenRouter")
        
        if not model_name:
            model_name = os.getenv("OPENROUTER_MODEL_NAME", "anthropic/claude-3.5-sonnet")
            
        model_obj = OpenAIChatCompletionsModel(
            model=model_name,
            openai_client=AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_key,
            ),
        )
        print(f"✅ Using OpenRouter model: {model_name}")
        return model_obj
    else:
        # For OpenAI, just return the model name string
        if not model_name:
            model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-5")
        print(f"✅ Using OpenAI model: {model_name}")
        return model_name