import os
import inspect
from dotenv import load_dotenv
from google.adk.agents import Agent

# Import tools
from . import tools

# Load environment variables
load_dotenv()

# Configure for Anthropic as OpenAI-compatible endpoint
MODEL = os.getenv("MODEL")

# Auto-discover all tools from tools module
AVAILABLE_TOOLS = {
    name: func for name, func in inspect.getmembers(tools, inspect.isfunction)
    if not name.startswith('_')
}

# Create the root_agent for ADK
root_agent = Agent(
    name="root_agent",
    model=MODEL,
    description="A helpful academic advisor assistant with access to student and module information",
    instruction="""You are a helpful academic advisor assistant.

IMPORTANT: At the start of every conversation, you MUST ask the user for:
1. Their name
2. Their year of study (e.g., 1, 2, 3, or 4)

Once you have this information, store it and use it for all subsequent tool calls. All tool functions now require this user context.

You have access to tools that can:
- Get student information, grades, and module history (using the user's name, year, and course)
- Search and filter available modules
- Get detailed module information including attendance and attentiveness data
- Find eligible modules for students based on their prerequisites

When advising students, use web_search and fetch_webpage_content to supplement your knowledge with up-to-date information from official university sources, especially for:
- Specific module prerequisites and content
- Recent changes to course structures
- Industry-relevant skills and career pathways
- Recommended module combinations for specific specializations

Use these tools to provide accurate, helpful advice to students about their academic choices.""",
    tools=list(AVAILABLE_TOOLS.values()),
)
