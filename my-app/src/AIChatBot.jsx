import React, { useState, useEffect, useRef } from 'react';

const AIChatBot = ({ userContext }) => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [sessionId] = useState(() => `session_${Date.now()}`);
  const [userId] = useState(() => `user_${Date.now()}`);
  const [sessionCreated, setSessionCreated] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Create session first, then show greeting
    const initializeChat = async () => {
      try {
        const response = await fetch(`/apps/my_agent/users/${userId}/sessions/${sessionId}`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({})
        });
        
        if (response.ok || response.status === 409) {
          // 409 means session already exists, which is fine
          setSessionCreated(true);
          console.log('ADK session ready:', sessionId);
        } else {
          console.error('Failed to create session:', await response.text());
        }
      } catch (error) {
        console.error('Error creating session:', error);
      }
      
      // Show initial greeting
      setMessages([{
        role: 'agent',
        content: 'Hi! I\'m your academic advisor assistant. I can help you with module recommendations, prerequisites, and academic planning. What would you like to know?',
        timestamp: new Date().toISOString()
      }]);
    };
    
    initializeChat();
  }, [userId, sessionId]);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = {
      role: 'user',
      content: input,
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      // Build context-aware message
      let contextMessage = input;
      if (userContext.name || userContext.year || userContext.course) {
        const contextParts = [];
        if (userContext.name) contextParts.push(`Name: ${userContext.name}`);
        if (userContext.year) contextParts.push(`Year: ${userContext.year}`);
        if (userContext.course) contextParts.push(`Course: ${userContext.course}`);
        contextMessage = `[User Context: ${contextParts.join(', ')}]\n\n${input}`;
      }

      // Call ADK API server using correct /run endpoint via Vite proxy
      const response = await fetch('/run', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          appName: 'my_agent',
          userId: userId,
          sessionId: sessionId,
          newMessage: {
            role: 'user',
            parts: [{ text: contextMessage }]
          }
        })
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('ADK API Response:', response.status, errorText);
        throw new Error(`ADK API error: ${response.status}`);
      }

      const data = await response.json();
      console.log('ADK API Response:', data);
      
      // Handle ADK response format - data is an array of events
      let agentResponse = '';
      if (Array.isArray(data)) {
        // Find the last model response with text
        for (let i = data.length - 1; i >= 0; i--) {
          const event = data[i];
          if (event.content?.role === 'model' && event.content?.parts) {
            for (const part of event.content.parts) {
              if (part.text) {
                agentResponse = part.text;
                break;
              }
            }
            if (agentResponse) break;
          }
        }
      }
      
      if (!agentResponse) {
        throw new Error('No text response found in ADK response');
      }

      const agentMessage = {
        role: 'agent',
        content: agentResponse,
        timestamp: new Date().toISOString()
      };

      setMessages(prev => [...prev, agentMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = {
        role: 'agent',
        content: `Sorry, I encountered an error: ${error.message}\n\n**Troubleshooting:**\n1. Make sure ADK API server is running: \`cd ai_agent && adk api_server\`\n2. Check it's running on http://localhost:8000\n3. Try: \`curl http://localhost:8000/list-apps\``,
        timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => {
    setMessages([{
      role: 'agent',
      content: 'Chat cleared. How can I help you today?',
      timestamp: new Date().toISOString()
    }]);
  };

  return (
    <div style={{
      width: '100%',
      maxWidth: '900px',
      margin: '0 auto',
      border: '1px solid #ddd',
      borderRadius: '10px',
      overflow: 'hidden',
      boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
      display: 'flex',
      flexDirection: 'column',
      height: '650px',
      background: 'white'
    }}>
      {/* Header */}
      <div style={{
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        color: 'white',
        padding: '20px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <div>
          <h3 style={{ margin: 0, fontSize: '20px', fontWeight: 'bold' }}>🎓 Academic Advisor AI</h3>
          <p style={{ margin: '5px 0 0 0', fontSize: '13px', opacity: 0.9 }}>
            Get personalized module recommendations and academic advice
          </p>
        </div>
        <button
          onClick={clearChat}
          style={{
            background: 'rgba(255,255,255,0.2)',
            border: 'none',
            color: 'white',
            padding: '10px 18px',
            borderRadius: '6px',
            cursor: 'pointer',
            fontSize: '13px',
            fontWeight: '600',
            transition: 'all 0.2s'
          }}
          onMouseEnter={(e) => e.target.style.background = 'rgba(255,255,255,0.3)'}
          onMouseLeave={(e) => e.target.style.background = 'rgba(255,255,255,0.2)'}
        >
          🗑️ Clear Chat
        </button>
      </div>

      {/* Messages */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        padding: '20px',
        background: '#f8f9fa'
      }}>
        {messages.map((msg, idx) => (
          <div
            key={idx}
            style={{
              display: 'flex',
              justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start',
              marginBottom: '16px',
              animation: 'fadeIn 0.3s ease-in'
            }}
          >
            <div
              style={{
                maxWidth: '75%',
                padding: '14px 18px',
                borderRadius: msg.role === 'user' ? '18px 18px 4px 18px' : '18px 18px 18px 4px',
                background: msg.role === 'user'
                  ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
                  : 'white',
                color: msg.role === 'user' ? 'white' : '#333',
                boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                wordWrap: 'break-word',
                whiteSpace: 'pre-wrap'
              }}
            >
              <div style={{ fontSize: '15px', lineHeight: '1.6' }}>
                {msg.content}
              </div>
              <div style={{
                fontSize: '11px',
                marginTop: '6px',
                opacity: 0.7,
                textAlign: msg.role === 'user' ? 'right' : 'left'
              }}>
                {new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </div>
            </div>
          </div>
        ))}
        {loading && (
          <div style={{ display: 'flex', justifyContent: 'flex-start', marginBottom: '16px' }}>
            <div style={{
              padding: '14px 18px',
              borderRadius: '18px 18px 18px 4px',
              background: 'white',
              boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
            }}>
              <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
                <div className="dot-flashing"></div>
                <span style={{ fontSize: '13px', color: '#666', marginLeft: '8px' }}>AI is thinking...</span>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div style={{
        padding: '18px',
        background: 'white',
        borderTop: '1px solid #e0e0e0'
      }}>
        {userContext.name && (
          <div style={{
            fontSize: '12px',
            color: '#666',
            marginBottom: '10px',
            padding: '8px 12px',
            background: '#f0f0f0',
            borderRadius: '6px',
            display: 'inline-block'
          }}>
            👤 {userContext.name} • Year {userContext.year} • {userContext.course}
          </div>
        )}
        <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-end' }}>
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask about modules, prerequisites, recommendations..."
            style={{
              flex: 1,
              padding: '14px',
              border: '1px solid #ddd',
              borderRadius: '10px',
              resize: 'none',
              fontFamily: 'inherit',
              fontSize: '15px',
              minHeight: '52px',
              maxHeight: '120px',
              outline: 'none',
              transition: 'border-color 0.2s'
            }}
            onFocus={(e) => e.target.style.borderColor = '#667eea'}
            onBlur={(e) => e.target.style.borderColor = '#ddd'}
            disabled={loading}
          />
          <button
            onClick={sendMessage}
            disabled={loading || !input.trim()}
            style={{
              padding: '14px 28px',
              background: loading || !input.trim()
                ? '#ccc'
                : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              color: 'white',
              border: 'none',
              borderRadius: '10px',
              cursor: loading || !input.trim() ? 'not-allowed' : 'pointer',
              fontSize: '15px',
              fontWeight: 'bold',
              transition: 'all 0.2s',
              minWidth: '90px'
            }}
          >
            {loading ? '...' : '📤 Send'}
          </button>
        </div>
      </div>

      <style>{`
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        .dot-flashing {
          position: relative;
          width: 10px;
          height: 10px;
          border-radius: 5px;
          background-color: #667eea;
          color: #667eea;
          animation: dotFlashing 1s infinite linear alternate;
          animation-delay: .5s;
        }

        .dot-flashing::before, .dot-flashing::after {
          content: '';
          display: inline-block;
          position: absolute;
          top: 0;
        }

        .dot-flashing::before {
          left: -15px;
          width: 10px;
          height: 10px;
          border-radius: 5px;
          background-color: #667eea;
          color: #667eea;
          animation: dotFlashing 1s infinite alternate;
          animation-delay: 0s;
        }

        .dot-flashing::after {
          left: 15px;
          width: 10px;
          height: 10px;
          border-radius: 5px;
          background-color: #667eea;
          color: #667eea;
          animation: dotFlashing 1s infinite alternate;
          animation-delay: 1s;
        }

        @keyframes dotFlashing {
          0% {
            background-color: #667eea;
          }
          50%, 100% {
            background-color: #d4d9f7;
          }
        }
      `}</style>
    </div>
  );
};

export default AIChatBot;
