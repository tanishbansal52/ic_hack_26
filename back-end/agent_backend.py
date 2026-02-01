from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

# Add ai_agent to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ai_agent'))

from my_agent.agent import root_agent

app = Flask(__name__)
CORS(app)

# Store conversation state per session
conversation_sessions = {}

@app.route('/api/agent/chat', methods=['POST'])
def chat():
    """
    Handle chat messages to the AI agent.
    Expects: { "message": "user message", "session_id": "unique_id", "user_context": { "name": "...", "year": 2, "course": "..." } }
    Returns: { "response": "agent response", "session_id": "unique_id" }
    """
    data = request.json
    message = data.get('message', '')
    session_id = data.get('session_id', 'default')
    user_context = data.get('user_context', {})
    
    if not message:
        return jsonify({"error": "Message is required"}), 400
    
    try:
        # Initialize session if needed
        if session_id not in conversation_sessions:
            conversation_sessions[session_id] = {
                'history': [],
                'user_context': user_context
            }
        
        # Update user context if provided
        if user_context:
            conversation_sessions[session_id]['user_context'] = user_context
        
        # Build context-aware message
        context_msg = message
        if user_context.get('name') or user_context.get('year') or user_context.get('course'):
            context_info = []
            if user_context.get('name'):
                context_info.append(f"Name: {user_context['name']}")
            if user_context.get('year'):
                context_info.append(f"Year: {user_context['year']}")
            if user_context.get('course'):
                context_info.append(f"Course: {user_context['course']}")
            context_msg = f"[User Context: {', '.join(context_info)}]\n\n{message}"
        
        # Get agent response
        response = root_agent.run(context_msg)
        
        # Store in history
        conversation_sessions[session_id]['history'].append({
            'user': message,
            'agent': response
        })
        
        return jsonify({
            "response": response,
            "session_id": session_id
        })
    
    except Exception as e:
        print(f"Error in agent chat: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/agent/clear', methods=['POST'])
def clear_session():
    """Clear a conversation session"""
    data = request.json
    session_id = data.get('session_id', 'default')
    
    if session_id in conversation_sessions:
        del conversation_sessions[session_id]
    
    return jsonify({"message": "Session cleared"})

@app.route('/api/agent/history', methods=['GET'])
def get_history():
    """Get conversation history for a session"""
    session_id = request.args.get('session_id', 'default')
    
    history = conversation_sessions.get(session_id, {}).get('history', [])
    return jsonify({"history": history})

@app.route('/api/agent/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "sessions": len(conversation_sessions)})

if __name__ == '__main__':
    print("Starting AI Agent Backend on port 5001...")
    print("Make sure the AI agent is properly configured with API keys!")
    app.run(debug=True, port=5001, host='0.0.0.0')
