"""
Shared tools module for both Claude and OpenAI Agent SDKs.
"""

import os
from agents import Agent, WebSearchTool, Runner  
import datetime

literature_search_model = os.getenv("LITERATURE_SEARCH_MODEL", "gpt-5")

# Literature search agent with enhanced prompt and configurable model
literature_agent = Agent(  
    name="Literature Search Agent",  
    instructions="""You are an expert academic research assistant specialized in finding scholarly literature and research papers.
    You search the most recent academic literature and provide detailed answers and citations. 
    Focus your research on the last 18 months. 
    The date today is 15th October, 2025.

CITATION FORMAT:
For each source, provide:
- Title (exact)
- Authors 
- Venue (journal/conference)
- Year
- URL/DOI when available
- Brief relevance summary

Be thorough but concise. Prioritize authoritative sources and recent work.

""",
    tools=[WebSearchTool()],
    model=literature_search_model
)