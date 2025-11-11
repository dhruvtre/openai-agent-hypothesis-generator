"""
Hypothesis Generator Agent using the OpenAI Agents SDK.

A specialized agent for generating research hypotheses with literature search capabilities,
supporting both OpenAI and OpenRouter model providers with real-time streaming output.
"""

import os
import json
import asyncio
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from agents import Agent, Runner, OpenAIChatCompletionsModel
from tools import literature_agent
from context import hypothesis_generator_instructions, ResearchContext
from backend_utils import create_model, save_session

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
                    print(f"🔧 DEBUG: Tool call detected - {event.item.raw_item.name}")
                    # Extract complete tool call information
                    tool_call_data = {
                        "tool_name": event.item.raw_item.name,
                        "timestamp": datetime.now().isoformat(),
                        "input_args": None
                    }
                    
                    # Parse full arguments (not truncated)
                    if hasattr(event.item.raw_item, 'arguments') and event.item.raw_item.arguments:
                        try:
                            tool_call_data["input_args"] = json.loads(event.item.raw_item.arguments)
                        except json.JSONDecodeError:
                            tool_call_data["input_args"] = {"raw_arguments": event.item.raw_item.arguments}
                    
                    # Create display string for console (keep existing UX)
                    tool_info = f"Calling tool: {tool_call_data['tool_name']}"
                    if tool_call_data["input_args"] and isinstance(tool_call_data["input_args"], dict):
                        if tool_call_data['tool_name'] == "literature_search" and 'query' in tool_call_data["input_args"]:
                            tool_info += f" - Searching for: '{tool_call_data['input_args']['query']}'"
                        else:
                            # Show key parameters (truncate for display only)
                            params = []
                            for key, value in list(tool_call_data["input_args"].items())[:2]:
                                if isinstance(value, str) and len(value) > 50:
                                    display_value = value[:47] + "..."
                                else:
                                    display_value = value
                                params.append(f"{key}: {display_value}")
                            if params:
                                tool_info += f" - Parameters: {', '.join(params)}"
                    
                    yield {
                        "type": "tool_call", 
                        "data": tool_info,
                        "tool_interaction": tool_call_data  # NEW: Complete tool data
                    }
                elif event.item.type == "tool_call_output_item":
                    print(f"🔧 DEBUG: Tool output detected")
                    # Extract complete tool output information
                    tool_output_data = {
                        "timestamp": datetime.now().isoformat(),
                        "output": event.item.output,
                        "output_length": len(str(event.item.output)) if event.item.output else 0
                    }
                    
                    yield {
                        "type": "tool_output", 
                        "data": event.item.output,
                        "tool_interaction": tool_output_data  # NEW: Complete output data
                    }
                elif event.item.type == "message_output_item":
                    text = ItemHelpers.text_message_output(event.item)
                    if text:
                        yield {"type": "message", "data": text}

    async def run_stream_with_extraction(self, prompt: str, context=None):
        """
        Enhanced streaming with real-time hypothesis extraction and tool interaction logging.
        
        Delegates to existing run_stream() and adds hypothesis detection events.
        Maintains all original functionality while adding extraction capabilities.
        
        Args:
            prompt: The user prompt to process
            context: Optional context (contains expected hypothesis count)
            
        Yields:
            - All original stream events from run_stream()
            - "hypothesis_found" events when complete JSON blocks detected
            - "extraction_complete" summary event at the end
            - Enhanced events include tool_interaction data for logging
        """
        from backend_utils import extract_hypotheses
        
        # Extract expected count from context for progress tracking
        expected_count = getattr(context, 'number_of_hypothesis', None) if context else None
        
        # Initialize tracking variables
        text_buffer = ""
        extracted_count = 0
        pending_tool_calls = []  # Store tool calls in order
        completed_interactions = []  # Store completed tool interactions

        # Delegate to existing run_stream and enhance with extraction
        async for event in self.run_stream(prompt, context):
            # Process tool interactions for logging (sequential pairing)
            if event.get("type") == "tool_call" and "tool_interaction" in event:
                call_data = event["tool_interaction"]
                print(f"🔧 DEBUG: Storing tool call #{len(pending_tool_calls) + 1} - {call_data['tool_name']}")
                pending_tool_calls.append(call_data)
                
            elif event.get("type") == "tool_output" and "tool_interaction" in event:
                output_data = event["tool_interaction"]
                print(f"🔧 DEBUG: Processing tool output, {len(pending_tool_calls)} pending calls")
                
                # Match with oldest pending tool call (FIFO order)
                if pending_tool_calls:
                    call_data = pending_tool_calls.pop(0)  # Remove first (oldest) call
                    interaction = {
                        "tool_name": call_data["tool_name"],
                        "timestamp_start": call_data["timestamp"],
                        "timestamp_end": output_data["timestamp"],
                        "input_args": call_data["input_args"],
                        "output": output_data["output"],
                        "output_length": output_data["output_length"]
                    }
                    
                    # Calculate duration if both timestamps are valid
                    try:
                        start_time = datetime.fromisoformat(call_data["timestamp"])
                        end_time = datetime.fromisoformat(output_data["timestamp"])
                        interaction["duration_ms"] = int((end_time - start_time).total_seconds() * 1000)
                    except:
                        interaction["duration_ms"] = None
                    
                    completed_interactions.append(interaction)
                    print(f"🔧 DEBUG: Created interaction #{len(completed_interactions)} - {interaction['tool_name']}, duration: {interaction.get('duration_ms')}ms")
            
            # Always yield original events first (preserves existing behavior)
            yield event

            # Only process text events for hypothesis extraction
            if event.get("type") != "text":
                continue

            # Accumulate text in buffer
            text_buffer += event["data"]
            
            # Run extraction on the growing buffer
            hypotheses = extract_hypotheses(text_buffer)

            # Skip if no new hypotheses found
            if len(hypotheses) <= extracted_count:
                continue

            # Emit events for newly found hypotheses
            for index in range(extracted_count, len(hypotheses)):
                hypothesis = hypotheses[index]
                extracted_count += 1
                
                # Build progress indicator
                progress = f"{extracted_count}/{expected_count}" if expected_count else str(extracted_count)
                
                yield {
                    "type": "hypothesis_found",
                    "data": hypothesis,
                    "id": extracted_count,
                    "progress": progress,
                    "summary": (
                        f"Hypothesis {extracted_count}: "
                        f"{hypothesis.get('claim', '')[:80]}..."
                    ),
                }

        # Emit final summary with tool interactions
        print(f"🔧 DEBUG: Final - {len(completed_interactions)} tool interactions to include in extraction_complete")
        if extracted_count > 0 or completed_interactions:
            yield {
                "type": "extraction_complete",
                "total_hypotheses": extracted_count,
                "expected": expected_count,
                "tool_interactions": completed_interactions,  # NEW: Complete tool data
                "message": (
                    f"Successfully extracted {extracted_count} hypothesis"
                    f"{'es' if extracted_count != 1 else ''}"
                    + (f" (expected {expected_count})" if expected_count else "")
                    + f" | Tool calls: {len(completed_interactions)}"
                ),
            }


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
    
    # Initialize session tracking variables
    raw_output = ""
    extracted_hypotheses = []
    extraction_stats = {}
    tool_interactions = []
    
    # Use streaming instead of waiting for complete response
    print(f"Generating {num_hypotheses} hypotheses for {domain}...\n")
    async for event in hypotheses_generator_agent.run_stream_with_extraction(prompt=f"Please generate hypotheses for the following research idea : {research_idea}", context=context):
        if event["type"] == "text":
            # Print text as it streams in, character by character
            print(event["data"], end="", flush=True)
            # Collect raw output
            raw_output += event["data"]
        elif event["type"] == "tool_call":
            # Show when a tool is being called
            print(f"\n>>> {event['data']}\n")
            # Track tool calls in raw output
            raw_output += f"\n>>> {event['data']}\n"
        elif event["type"] == "tool_output":
            # Show tool results (truncated for readability)
            output_preview = str(event['data'])[:100] + "..." if len(str(event['data'])) > 100 else str(event['data'])
            print(f">>> Tool result: {output_preview}\n")
            # Track tool results in raw output
            raw_output += f">>> Tool result: {output_preview}\n"
        elif event["type"] == "hypothesis_found":
            # Show real-time hypothesis detection
            print(f"\n{'='*60}")
            print(f"[{event['progress']} HYPOTHESIS EXTRACTED]")
            print(f"Summary: {event['summary']}")
            print(f"\nFull hypothesis data:")
            hypothesis = event['data']
            for key, value in hypothesis.items():
                if key.startswith('_'):  # Skip metadata fields for cleaner display
                    continue
                print(f"  • {key}: {value}")
            print(f"{'='*60}\n")
            # Collect extracted hypothesis
            extracted_hypotheses.append(hypothesis)
        elif event["type"] == "extraction_complete":
            # Final summary of extraction
            print(f"\n{'='*60}")
            print("=== EXTRACTION COMPLETE ===")
            print(event['message'])
            print(f"Total extracted: {event.get('total_hypotheses', 0)}")
            print(f"Expected: {event.get('expected', 'N/A')}")
            print(f"{'='*60}\n")
            # Collect extraction stats and tool interactions
            extraction_stats = {
                "total_extracted": event.get('total_hypotheses', 0),
                "expected": event.get('expected', None),
                "extraction_time": datetime.now().isoformat(),
                "message": event.get('message', '')
            }
            # Collect tool interactions for enhanced logging
            tool_interactions = event.get('tool_interactions', [])
    
    # Save session after streaming completes
    try:
        session_file = save_session(
            domain=domain,
            num_hypotheses=num_hypotheses, 
            research_idea=research_idea,
            provider=provider,
            model_name=model_name or "unknown",
            raw_output=raw_output,
            extracted_hypotheses=extracted_hypotheses,
            extraction_stats=extraction_stats,
            tool_interactions=tool_interactions  # NEW: Pass tool interaction data
        )
        print(f"\n{'='*60}")
        print("=== SESSION SAVED ===")
        print(f"Session saved to: {session_file}")
        print(f"Hypotheses extracted: {len(extracted_hypotheses)}")
        print(f"Tool interactions: {len(tool_interactions)}")
        print(f"Raw output length: {len(raw_output)} characters")
        print(f"{'='*60}\n")
    except Exception as e:
        print(f"\n⚠️  Warning: Failed to save session: {e}")
        print("Session data was generated but not persisted.")
    


if __name__ == "__main__":
    asyncio.run(main())