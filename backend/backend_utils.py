import os
import re
import json
from typing import List, Dict, Any
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


def extract_hypotheses(buffer: str) -> List[Dict[str, Any]]:
    """
    Extract all complete hypothesis JSON blocks from text buffer.
    
    Args:
        buffer: Text buffer that may contain ```json...``` blocks
        
    Returns:
        List of validated hypothesis dictionaries
    """
    # Regex to capture content between ```json and ```
    pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(pattern, buffer, re.DOTALL)
    
    hypotheses = []
    for match in matches:
        try:
            # Parse JSON
            hypothesis = json.loads(match.strip())
            
            # Validate required fields
            required_fields = ['claim', 'dataset', 'metric', 'baseline', 
                             'success_threshold', 'budget']
            
            if all(field in hypothesis for field in required_fields):
                # Add metadata for tracking
                hypothesis['_extracted'] = True
                hypothesis['_id'] = len(hypotheses) + 1
                hypotheses.append(hypothesis)
            else:
                # Log validation failure (optional)
                missing = [f for f in required_fields if f not in hypothesis]
                if missing:
                    print(f"Hypothesis missing fields: {missing}")
                    
        except json.JSONDecodeError as e:
            # Silently skip malformed JSON (or optionally log)
            print(f"Failed to parse hypothesis JSON: {e}")
            continue
        except Exception as e:
            # Catch any other unexpected errors
            print(f"Error processing hypothesis: {e}")
            continue
    
    return hypotheses