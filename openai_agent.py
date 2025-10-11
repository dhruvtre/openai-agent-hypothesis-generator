"""
Hypothesis Generator Agent using the OpenAI Agents SDK.

A specialized agent for generating research hypotheses with literature search capabilities,
supporting both OpenAI and OpenRouter model providers with real-time streaming output.
"""

import os
import asyncio
from dotenv import load_dotenv
from agents import Agent, Runner, OpenAIChatCompletionsModel
from tools import literature_agent
from context import hypothesis_generator_instructions, ResearchContext
from backend_utils import create_model

load_dotenv()


class OpenAIAgentWrapper:
    """Wrapper class for OpenAI Agent operations."""
    
    def __init__(self, name="Assistant", instructions="You are a helpful assistant.", model=None, tools=None):
        """
        Initialize the OpenAI Agent.
        
        Args:
            name: Name of the agent
            instructions: System instructions for the agent
        """
        agent_kwargs = {
            "name": name,
            "instructions": instructions,
        }
        if model is not None:
            agent_kwargs["model"] = model
        if tools is not None:
            agent_kwargs["tools"] = tools
            
        self.agent = Agent(**agent_kwargs)
    
    async def run(self, prompt: str, context=None):
        """
        Run the agent with a given prompt.
        
        Args:
            prompt: The user prompt to process
            
        Returns:
            The agent's response
        """
        result = await Runner.run(self.agent, prompt, context=context)
        return result.final_output
    
    async def run_stream(self, prompt: str, context=None):
        """
        Run the agent with streaming output.
        
        Args:
            prompt: The user prompt to process
            context: Optional context for the agent
            
        Yields:
            Stream events from the agent
        """
        from openai.types.responses import ResponseTextDeltaEvent
        from agents import ItemHelpers
        
        result = Runner.run_streamed(self.agent, prompt, context=context)
        
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                # Stream text deltas as they come
                yield {"type": "text", "data": event.data.delta}
            elif event.type == "run_item_stream_event":
                if event.item.type == "tool_call_item":
                    yield {"type": "tool_call", "data": f"Calling tool: {event.item.raw_item.name}"}
                elif event.item.type == "tool_call_output_item":
                    yield {"type": "tool_output", "data": event.item.output}
                elif event.item.type == "message_output_item":
                    text = ItemHelpers.text_message_output(event.item)
                    if text:
                        yield {"type": "message", "data": text}


async def main():
    """Main function to demonstrate OpenAI agent usage."""
    
    # Configure model provider
    provider = os.getenv("MODEL_PROVIDER", "openai")
    print(f"Model provider: {provider}\n")

    print("\n=== Streaming Hypothesis Generation ===")
    hypothesis_model = create_model(provider, "gpt-5" if provider == "openai" else "anthropic/claude-sonnet-4.5")
    
    hypotheses_generator_agent = OpenAIAgentWrapper(name="Hypotheses Generator Agent 1",
                                                    instructions=hypothesis_generator_instructions,
                                                    tools=[literature_agent.as_tool(
                                                        tool_name="literature_search",
                                                        tool_description="Search for academic and scholarly information"
                                                    )],
                                                    model=hypothesis_model)
    context = ResearchContext(
        problem_space_title="AI for Scientific Discovery",
        number_of_hypothesis=2
    )
    
    # Use streaming instead of waiting for complete response
    print("Generating hypotheses (streaming):\n")
    async for event in hypotheses_generator_agent.run_stream("Generate hypotheses for me.", context=context):
        if event["type"] == "text":
            # Print text as it streams in, character by character
            print(event["data"], end="", flush=True)
        elif event["type"] == "tool_call":
            # Show when a tool is being called
            print(f"\n>>> {event['data']}\n")
        elif event["type"] == "tool_output":
            # Show tool results (truncated for readability)
            output_preview = str(event['data'])[:100] + "..." if len(str(event['data'])) > 100 else str(event['data'])
            print(f">>> Tool result: {output_preview}\n")
    


if __name__ == "__main__":
    asyncio.run(main())