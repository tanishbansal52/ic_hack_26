# 🚀 Starting the Integrated App

## Prerequisites
- Python environment with ADK installed: `pip install google-adk`
- Node.js and npm installed
- Environment variables configured in `ai_agent/.env`

## Start All Services (3 Terminals Required)

### Terminal 1: ADK API Server (AI Agent) ⚡
```bash
cd ai_agent
python start_adk_server.py
# OR manually:
# adk api_server --port 8000 --host 0.0.0.0
```
**Expected output:**
```
Starting ADK API Server...
Server running on http://localhost:8000
```

**Test it's working:**
```bash
curl http://localhost:8000/health
# Should return: {"status":"ok"}
```

### Terminal 2: Module ROI Backend (Flask) 📊
```bash
cd back-end
python webapp.py
```
**Runs on:** http://localhost:5000

**Test it's working:**
```bash
curl http://localhost:5000/
# Should return module data
```

### Terminal 3: React Frontend (Vite) 🎨
```bash
cd my-app
npm run dev
```
**Runs on:** http://localhost:5173

## Testing the Full Integration

1. **Open browser:** http://localhost:5173
2. **Fill in your info:**
   - Name: Your name
   - Year: 2
   - Course: Computer Science

3. **Click "🤖 AI Advisor"** to open the chat

4. **Test questions:**
   - "What modules should I take?"
   - "Tell me about CS301"
   - "What are the prerequisites for Machine Learning?"
   - "Which modules have good ROI?"

## Architecture

```
┌─────────────────────┐
│   React Frontend    │
│  (localhost:5173)   │
└──────────┬──────────┘
           │
           ├─────────────────────┐
           │                     │
           v                     v
┌─────────────────────┐  ┌──────────────────┐
│   ADK API Server    │  │  Flask Backend   │
│  (localhost:8000)   │  │ (localhost:5000) │
│                     │  │                  │
│  • AI Agent         │  │  • Module ROI    │
│  • Tool Execution   │  │  • Database      │
└─────────────────────┘  └──────────────────┘
```

## API Endpoints

### ADK Agent API (Port 8000)
- **POST** `/v1/chat/completions` - Chat with AI agent
  ```json
  {
    "model": "root_agent",
    "messages": [{"role": "user", "content": "Hello"}]
  }
  ```

### Module ROI API (Port 5000)
- **GET** `/` - Health check
- **GET** `/module/<code>` - Get module ROI data

## Troubleshooting

### ❌ "404 Not Found" from ADK
**Problem:** ADK server not running or wrong endpoint

**Solutions:**
1. Check ADK is running: `curl http://localhost:8000/health`
2. Verify ADK installed: `pip show google-adk`
3. Check console logs in Terminal 1 for errors
4. Try restarting: Kill process and run `adk api_server` again

### ❌ "Connection Refused"
**Problem:** Server not started or wrong port

**Solutions:**
1. Check all 3 terminals are running
2. Verify ports not in use: `lsof -i :8000,5000,5173`
3. Restart servers one by one

### ❌ "CORS Error" in Browser
**Problem:** Frontend can't access backend

**Solutions:**
1. Frontend uses proxy (should work automatically)
2. Check Vite config has proxy setup
3. Restart React dev server: `npm run dev`

### ❌ Tools Not Working
**Problem:** Agent can't access student/module data

**Solutions:**
1. Check `ai_agent/my_agent/tools.py` exists
2. Verify agent imports tools correctly
3. Look at ADK server logs for tool execution errors

### ❌ Environment Variables
**Problem:** Missing API keys

**Solutions:**
1. Check `ai_agent/.env` exists
2. Verify `ANTHROPIC_API_KEY` is set (or appropriate API key)
3. Check `MODEL` is set to valid model name

## Quick Health Check Script

```bash
#!/bin/bash
echo "Checking ADK Server..."
curl -s http://localhost:8000/health && echo " ✅" || echo " ❌"

echo "Checking Flask Backend..."
curl -s http://localhost:5000/ && echo " ✅" || echo " ❌"

echo "Checking React Frontend..."
curl -s http://localhost:5173/ && echo " ✅" || echo " ❌"
```

Save as `check_servers.sh`, make executable: `chmod +x check_servers.sh`

## Manual Testing

Test ADK server directly:
```bash
# Test health endpoint
curl http://localhost:8000/health

# Test chat endpoint
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "root_agent",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```
