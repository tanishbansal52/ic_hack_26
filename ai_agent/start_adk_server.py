"""
Startup script for ADK API server with CORS enabled
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("=" * 60)
print("Starting ADK API Server with CORS enabled...")
print("=" * 60)
print(f"Model: {os.getenv('MODEL')}")
print(f"Server will run on: http://localhost:8000")
print(f"Health check: http://localhost:8000/health")
print(f"API endpoint: http://localhost:8000/v1/chat/completions")
print("=" * 60)

# Start ADK server
# The ADK CLI should handle CORS automatically, but if not, we need to configure it
os.system("adk api_server --port 8000 --host 0.0.0.0")
