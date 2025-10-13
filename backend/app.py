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
from backend_utils import create_model

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
    
    # Stream content from agent
    async for event in agent.run_stream(prompt=prompt, context=context):
        if event["type"] == "text":
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

if __name__ == "__main__":
    import uvicorn
    print("Starting Hypothesis Generator API on http://localhost:8000")
    print("OpenAI-compatible endpoint: POST http://localhost:8000/v1/chat/completions")
    uvicorn.run(app, host="0.0.0.0", port=8000)