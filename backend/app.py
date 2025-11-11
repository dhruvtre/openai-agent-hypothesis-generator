"""
FastAPI wrapper that provides OpenAI-compatible API for the Hypothesis Generator Agent.
This allows the agent to work with OpenAI's ChatKit.js frontend.
"""

import os
import json
import time
import asyncio
from typing import Optional, AsyncGenerator
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from openai_agent import OpenAIAgentWrapper
from tools import literature_agent
from context import hypothesis_generator_instructions, ResearchContext
from backend_utils import create_model, save_session

load_dotenv()

app = FastAPI()

# Configure CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI-compatible request/response models
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "hypothesis-generator"
    messages: list[Message]
    stream: Optional[bool] = True
    # Custom fields for your agent
    metadata: Optional[dict] = Field(default_factory=lambda: {
        "domain": "AI for Drug Discovery",
        "num_hypotheses": 3
    })

# Initialize agent
def create_agent():
    provider = os.getenv("MODEL_PROVIDER", "openai")
    
    if provider == "openai":
        model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4")
    else:
        model_name = os.getenv("OPENROUTER_MODEL_NAME")
    
    hypothesis_model = create_model(provider, model_name)
    
    return OpenAIAgentWrapper(
        name="Hypothesis Generator Agent",
        instructions=hypothesis_generator_instructions,
        tools=[literature_agent.as_tool(
            tool_name="literature_search",
            tool_description="Search for academic and scholarly information"
        )],
        model=hypothesis_model
    )

agent = create_agent()

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.
    Translates between OpenAI's format and your agent's format.
    """
    
    # Extract the user's message
    user_message = ""
    for msg in request.messages:
        if msg.role == "user":
            user_message = msg.content
            break
    
    if not user_message:
        raise HTTPException(status_code=400, detail="No user message found")
    
    # Extract metadata for context
    domain = request.metadata.get("domain", "AI for Drug Discovery")
    num_hypotheses = request.metadata.get("num_hypotheses", 3)
    
    context = ResearchContext(
        problem_space_title=domain,
        number_of_hypothesis=num_hypotheses
    )
    
    # Format the prompt
    prompt = f"Please generate hypotheses for the following research idea: {user_message}"
    
    if request.stream:
        return StreamingResponse(
            generate_stream(prompt, context),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            }
        )
    else:
        # Non-streaming response
        response = await agent.run(prompt, context=context)
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }

async def generate_stream(prompt: str, context: ResearchContext) -> AsyncGenerator[str, None]:
    """
    Generate OpenAI-compatible SSE stream from agent output.
    """
    
    # Initialize session tracking variables
    raw_output = ""
    extracted_hypotheses = []
    extraction_stats = {}
    tool_interactions = []
    
    # Send initial stream chunk
    chunk_id = f"chatcmpl-{int(time.time())}"
    
    # Initial chunk
    initial_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "hypothesis-generator",
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "content": ""},
            "finish_reason": None
        }]
    }
    yield f"data: {json.dumps(initial_chunk)}\n\n"
    
    # Stream content from agent with hypothesis extraction
    async for event in agent.run_stream_with_extraction(prompt=prompt, context=context):
        if event["type"] == "text":
            # Collect raw output for session saving
            raw_output += event["data"]
            
            # Stream text chunks in OpenAI format
            chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "hypothesis-generator",
                "choices": [{
                    "index": 0,
                    "delta": {"content": event["data"]},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            
        elif event["type"] == "tool_call":
            # Optional: Send tool call info as a system message
            chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "hypothesis-generator",
                "choices": [{
                    "index": 0,
                    "delta": {"content": f"\n[{event['data']}]\n"},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        
        elif event["type"] == "hypothesis_found":
            # Collect extracted hypothesis for session saving
            extracted_hypotheses.append(event["data"])
            
            # Send hypothesis extraction event to frontend
            chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "hypothesis-generator",
                "choices": [{
                    "index": 0,
                    "delta": {"content": f"\n[HYPOTHESIS {event['progress']} EXTRACTED]\n{event['summary']}\n"},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            
        elif event["type"] == "extraction_complete":
            # Collect extraction stats and tool interactions for session saving
            extraction_stats = {
                "total_extracted": event.get('total_hypotheses', 0),
                "expected": event.get('expected', None),
                "extraction_time": time.time(),
                "message": event.get('message', '')
            }
            # Collect tool interactions for enhanced logging
            tool_interactions = event.get('tool_interactions', [])
            
            # Send extraction summary to frontend
            chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk", 
                "created": int(time.time()),
                "model": "hypothesis-generator",
                "choices": [{
                    "index": 0,
                    "delta": {"content": f"\n=== EXTRACTION COMPLETE ===\n{event['message']}\n"},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
    
    # Send final chunk
    final_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "hypothesis-generator",
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    
    # Send [DONE] marker
    yield "data: [DONE]\n\n"
    
    # Save session data after streaming completes
    try:
        # Extract metadata from context
        domain = context.problem_space_title
        num_hypotheses = context.number_of_hypothesis
        
        # Get provider/model info from environment
        provider = os.getenv("MODEL_PROVIDER", "openai")
        if provider == "openai":
            model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4")
        else:
            model_name = os.getenv("OPENROUTER_MODEL_NAME", "unknown")
        
        # Use the prompt as research_idea (best we can extract from API request)
        research_idea = prompt.replace("Please generate hypotheses for the following research idea: ", "")
        
        session_file = save_session(
            domain=domain,
            num_hypotheses=num_hypotheses,
            research_idea=research_idea,
            provider=provider,
            model_name=model_name,
            raw_output=raw_output,
            extracted_hypotheses=extracted_hypotheses,
            extraction_stats=extraction_stats,
            tool_interactions=tool_interactions  # NEW: Pass tool interaction data
        )
        
        # Log successful save (optional - won't reach frontend)
        print(f"API session saved: {session_file}")
        print(f"Extracted {len(extracted_hypotheses)}/{num_hypotheses} hypotheses")
        print(f"Tool interactions: {len(tool_interactions)}")
        
    except Exception as e:
        # Log error but don't break the API response
        print(f"Warning: Failed to save API session: {e}")
        print(f"Session data: {len(raw_output)} chars, {len(extracted_hypotheses)} hypotheses, {len(tool_interactions)} tool calls")

if __name__ == "__main__":
    import uvicorn
    print("Starting Hypothesis Generator API on http://localhost:8000")
    print("OpenAI-compatible endpoint: POST http://localhost:8000/v1/chat/completions")
    uvicorn.run(app, host="0.0.0.0", port=8000)