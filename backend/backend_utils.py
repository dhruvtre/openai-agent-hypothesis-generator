import os
import re
import json
from datetime import datetime
from pathlib import Path
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
    Handles both single hypothesis objects and arrays of hypotheses.
    
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
            parsed = json.loads(match.strip())
            
            # Handle both single hypothesis and array of hypotheses
            if isinstance(parsed, list):
                # It's an array of hypotheses
                items = parsed
            elif isinstance(parsed, dict):
                # It's a single hypothesis
                items = [parsed]
            else:
                continue
            
            # Validate required fields for each hypothesis
            required_fields = ['claim', 'dataset', 'metric', 'baseline', 
                             'success_threshold', 'budget', 'reasoning', 'citations']
            
            for item in items:
                if all(field in item for field in required_fields):
                    # Add metadata for tracking
                    item['_extracted'] = True
                    item['_id'] = len(hypotheses) + 1
                    hypotheses.append(item)
                else:
                    # Log validation failure (optional)
                    missing = [f for f in required_fields if f not in item]
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


def save_session(domain: str, num_hypotheses: int, research_idea: str, 
                provider: str, model_name: str, raw_output: str, 
                extracted_hypotheses: list, extraction_stats: dict,
                tool_interactions: list = None) -> str:
    """
    Save complete session data to JSON file.
    
    Args:
        domain: Research domain
        num_hypotheses: Number of hypotheses requested
        research_idea: Original research idea text
        provider: Model provider (openai/openrouter)
        model_name: Model name used
        raw_output: Complete raw text output from agent
        extracted_hypotheses: List of extracted hypothesis dictionaries
        extraction_stats: Stats from extraction process
        tool_interactions: List of complete tool interaction data (optional)
        
    Returns:
        Path to saved session file
    """
    # Create sessions directory if it doesn't exist
    sessions_dir = Path("sessions")
    sessions_dir.mkdir(exist_ok=True)
    
    # Generate session ID from timestamp and domain
    timestamp = datetime.now()
    domain_slug = domain.replace(" ", "-").replace("/", "-")[:30]
    session_id = f"{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}_{domain_slug}"
    
    # Build session data
    session_data = {
        "session_id": session_id,
        "timestamp": timestamp.isoformat(),
        "metadata": {
            "domain": domain,
            "num_requested": num_hypotheses,
            "model_provider": provider,
            "model_name": model_name
        },
        "research_idea": research_idea,
        "raw_output": raw_output,
        "extracted_hypotheses": extracted_hypotheses,
        "extraction_stats": extraction_stats,
        "tool_interactions": tool_interactions or []  # NEW: Complete tool interaction data
    }
    
    # Save to file
    session_file = sessions_dir / f"{session_id}.json"
    with open(session_file, 'w', encoding='utf-8') as f:
        json.dump(session_data, f, indent=2, ensure_ascii=False)
    
    return str(session_file)