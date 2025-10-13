"""
Hypothesis Generator Agent using the OpenAI Agents SDK.

A specialized agent for generating research hypotheses with literature search capabilities,
supporting both OpenAI and OpenRouter model providers with real-time streaming output.
"""

import os
import json
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
                    # Extract tool name
                    tool_name = event.item.raw_item.name
                    tool_info = f"Calling tool: {tool_name}"
                    
                    # Try to extract arguments and show meaningful information
                    if hasattr(event.item.raw_item, 'arguments') and event.item.raw_item.arguments:
                        try:
                            args = json.loads(event.item.raw_item.arguments)
                            # For literature search, show the query parameter
                            if tool_name == "literature_search" and 'query' in args:
                                tool_info += f" - Searching for: '{args['query']}'"
                            elif args:
                                # For other tools, show key parameters (truncate if too long)
                                params = []
                                for key, value in list(args.items())[:2]:  # Show max 2 params
                                    if isinstance(value, str) and len(value) > 50:
                                        value = value[:47] + "..."
                                    params.append(f"{key}: {value}")
                                if params:
                                    tool_info += f" - Parameters: {', '.join(params)}"
                        except (json.JSONDecodeError, AttributeError):
                            pass
                    
                    yield {"type": "tool_call", "data": tool_info}
                elif event.item.type == "tool_call_output_item":
                    yield {"type": "tool_output", "data": event.item.output}
                elif event.item.type == "message_output_item":
                    text = ItemHelpers.text_message_output(event.item)
                    if text:
                        yield {"type": "message", "data": text}


def get_user_input():
    """Get user input for hypothesis generation parameters."""
    print("=== Hypothesis Generator Setup ===")
    domain = input("Research domain (e.g., 'AI for Drug Discovery'): ")
    num_hypotheses = int(input("Number of hypotheses to generate: "))
    research_idea = input("Describe your research idea: ")
    return domain, num_hypotheses, research_idea


async def main():
    """Main function to demonstrate OpenAI agent usage."""
    
    # Get user input
    domain, num_hypotheses, research_idea = get_user_input()
    
    # Configure model provider
    provider = os.getenv("MODEL_PROVIDER", "openai")
    print(f"\nModel provider: {provider}")

    print("\n=== Streaming Hypothesis Generation ===")
    
    # Get model name from environment variables
    if provider == "openai":
        model_name = os.getenv("OPENAI_MODEL_NAME")
    else:
        model_name = os.getenv("OPENROUTER_MODEL_NAME")
    
    hypothesis_model = create_model(provider, model_name)
    
    hypotheses_generator_agent = OpenAIAgentWrapper(name="Hypotheses Generator Agent 1",
                                                    instructions=hypothesis_generator_instructions,
                                                    tools=[literature_agent.as_tool(
                                                        tool_name="literature_search",
                                                        tool_description="Search for academic and scholarly information"
                                                    )],
                                                    model=hypothesis_model)
    context = ResearchContext(
        problem_space_title=domain,
        number_of_hypothesis=num_hypotheses
    )
    
    # Use streaming instead of waiting for complete response
    print(f"Generating {num_hypotheses} hypotheses for {domain}...\n")
    async for event in hypotheses_generator_agent.run_stream(prompt=f"Please generate hypotheses for the following research idea : {research_idea}", context=context):
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