"""
Shared tools module for both Claude and OpenAI Agent SDKs.
"""

from agents import Agent, WebSearchTool, Runner  
  
literature_agent = Agent(  
    name="Literature Search Agent",  
    instructions="You are an academic research assistant. Search for scholarly articles and academic information to answer queries with detailed citations.",  
    tools=[WebSearchTool()],  
)