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
          
          // Send initial context message to the agent
          if (userContext.name || userContext.year || userContext.course) {
            await sendContextToAgent();
          }
        } else {
          console.error('Failed to create session:', await response.text());
        }
      } catch (error) {
        console.error('Error creating session:', error);
      }
      
      // Show initial greeting with personalization
      setMessages([{
        role: 'agent',
        content: `Hi${userContext.name ? ' ' + userContext.name : ''}! I'm your academic advisor assistant. I can see you're in Year ${userContext.year}${userContext.course ? ' studying ' + userContext.course : ''}. I can help you with module recommendations, prerequisites, and academic planning. What would you like to know?`,
        timestamp: new Date().toISOString()
      }]);
    };
    
    // Function to send context to agent without showing in chat
    const sendContextToAgent = async () => {
      const contextParts = [];
      if (userContext.name) contextParts.push(`Name: ${userContext.name}`);
      if (userContext.year) contextParts.push(`Year: ${userContext.year}`);
      if (userContext.course) contextParts.push(`Course: ${userContext.course}`);
      const contextMessage = `[User Context: ${contextParts.join(', ')}]\n\nPlease acknowledge this context silently and use it for future recommendations.`;

      try {
        await fetch('/run', {
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
      } catch (error) {
        console.error('Error sending context:', error);
      }
    };
    
    initializeChat();
  }, [userId, sessionId, userContext]);

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
      maxWidth: '1000px',
      margin: '0 auto',
      borderRadius: '28px',
      overflow: 'hidden',
      boxShadow: 
        '0 8px 32px rgba(31, 38, 135, 0.2), ' +
        'inset 0 1px 0 rgba(255, 255, 255, 0.5), ' +
        'inset 0 -1px 0 rgba(255, 255, 255, 0.2), ' +
        '0 20px 60px rgba(0, 0, 0, 0.15)',
      display: 'flex',
      flexDirection: 'column',
      height: '700px',
      background: 'rgba(255, 255, 255, 0.2)',
      backdropFilter: 'blur(40px) saturate(180%)',
      WebkitBackdropFilter: 'blur(40px) saturate(180%)',
      border: '1.5px solid rgba(255, 255, 255, 0.4)'
    }}>
      {/* Header */}
      <div style={{
        background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.95) 0%, rgba(139, 92, 246, 0.95) 100%)',
        color: 'white',
        padding: '24px 28px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        boxShadow: 
          '0 4px 12px rgba(99, 102, 241, 0.3), ' +
          'inset 0 1px 0 rgba(255, 255, 255, 0.3)',
        borderBottom: '1px solid rgba(255, 255, 255, 0.2)'
      }}>
        <div>
          <h3 style={{ 
            margin: 0, 
            fontSize: '24px', 
            fontWeight: '800',
            display: 'flex',
            alignItems: 'center',
            gap: '10px'
          }}>
            Academic Advisor
          </h3>
          <p style={{ 
            margin: '8px 0 0 0', 
            fontSize: '14px', 
            opacity: 0.95,
            fontWeight: '500'
          }}>
            Get module recommendations based on your attendance and performance
          </p>
        </div>
        <button
          onClick={clearChat}
          style={{
            background: 'rgba(255, 255, 255, 0.2)',
            backdropFilter: 'blur(10px)',
            WebkitBackdropFilter: 'blur(10px)',
            border: '2px solid rgba(255, 255, 255, 0.4)',
            color: 'white',
            padding: '12px 22px',
            borderRadius: '12px',
            cursor: 'pointer',
            fontSize: '14px',
            fontWeight: '700',
            transition: 'all 0.3s',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            boxShadow: 
              '0 4px 12px rgba(0, 0, 0, 0.15), ' +
              'inset 0 1px 0 rgba(255, 255, 255, 0.4)'
          }}
          onMouseEnter={(e) => {
            e.target.style.background = 'rgba(255, 255, 255, 0.3)';
            e.target.style.transform = 'translateY(-2px)';
            e.target.style.boxShadow = 
              '0 6px 16px rgba(0, 0, 0, 0.2), ' +
              'inset 0 1px 0 rgba(255, 255, 255, 0.5)';
          }}
          onMouseLeave={(e) => {
            e.target.style.background = 'rgba(255, 255, 255, 0.2)';
            e.target.style.transform = 'translateY(0)';
            e.target.style.boxShadow = 
              '0 4px 12px rgba(0, 0, 0, 0.15), ' +
              'inset 0 1px 0 rgba(255, 255, 255, 0.4)';
          }}
        >
          <span>Clear Chat</span>
        </button>
      </div>

      {/* Messages */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        padding: '28px',
        background: 'rgba(255, 255, 255, 0.05)'
      }}>
        {messages.map((msg, idx) => (
          <div
            key={idx}
            style={{
              display: 'flex',
              justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start',
              marginBottom: '20px',
              animation: 'fadeInUp 0.4s ease-out'
            }}
          >
            {msg.role === 'agent' && (
              <div style={{
                width: '36px',
                height: '36px',
                borderRadius: '50%',
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                marginRight: '12px',
                flexShrink: 0,
                boxShadow: '0 2px 8px rgba(102, 126, 234, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.3)'
              }}>
              </div>
            )}
            <div
              style={{
                maxWidth: '70%',
                padding: '16px 20px',
                borderRadius: msg.role === 'user' ? '20px 20px 4px 20px' : '20px 20px 20px 4px',
                background: msg.role === 'user'
                  ? 'linear-gradient(135deg, rgba(99, 102, 241, 0.9) 0%, rgba(139, 92, 246, 0.9) 100%)'
                  : 'rgba(255, 255, 255, 0.4)',
                backdropFilter: 'blur(20px) saturate(180%)',
                WebkitBackdropFilter: 'blur(20px) saturate(180%)',
                color: msg.role === 'user' ? 'white' : '#2d3748',
                boxShadow: msg.role === 'user' 
                  ? '0 4px 16px rgba(99, 102, 241, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.4)'
                  : '0 4px 16px rgba(31, 38, 135, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.6)',
                wordWrap: 'break-word',
                whiteSpace: 'pre-wrap',
                border: msg.role === 'user' 
                  ? '1px solid rgba(255, 255, 255, 0.3)' 
                  : '1.5px solid rgba(255, 255, 255, 0.4)'
              }}
            >
              <div style={{ fontSize: '15px', lineHeight: '1.7', fontWeight: msg.role === 'agent' ? '500' : '400' }}>
                {msg.content}
              </div>
              <div style={{
                fontSize: '11px',
                marginTop: '8px',
                opacity: msg.role === 'user' ? 0.8 : 0.6,
                textAlign: msg.role === 'user' ? 'right' : 'left',
                fontWeight: '600'
              }}>
                {new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </div>
            </div>
            {msg.role === 'user' && (
              <div style={{
                width: '36px',
                height: '36px',
                borderRadius: '50%',
                background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                marginLeft: '12px',
                flexShrink: 0,
                boxShadow: '0 2px 8px rgba(240, 147, 251, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.3)'
              }}>
              </div>
            )}
          </div>
        ))}
        {loading && (
          <div style={{ display: 'flex', justifyContent: 'flex-start', marginBottom: '20px', alignItems: 'center' }}>
            <div style={{
              width: '36px',
              height: '36px',
              borderRadius: '50%',
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              marginRight: '12px',
              flexShrink: 0,
              boxShadow: '0 2px 8px rgba(102, 126, 234, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.3)'
            }}>
            </div>
            <div style={{
              padding: '16px 20px',
              borderRadius: '20px 20px 20px 4px',
              background: 'rgba(255, 255, 255, 0.4)',
              backdropFilter: 'blur(20px) saturate(180%)',
              WebkitBackdropFilter: 'blur(20px) saturate(180%)',
              boxShadow: 
                '0 4px 16px rgba(31, 38, 135, 0.15), ' +
                'inset 0 1px 0 rgba(255, 255, 255, 0.6)',
              border: '1.5px solid rgba(255, 255, 255, 0.4)'
            }}>
              <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                <div className="dot-flashing"></div>
                <span style={{ fontSize: '14px', color: '#718096', marginLeft: '12px', fontWeight: '600' }}>AI is thinking...</span>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div style={{
        padding: '24px 28px',
        background: 'rgba(255, 255, 255, 0.15)',
        backdropFilter: 'blur(20px)',
        WebkitBackdropFilter: 'blur(20px)',
        borderTop: '1.5px solid rgba(255, 255, 255, 0.3)',
        boxShadow: 
          '0 -4px 12px rgba(31, 38, 135, 0.1), ' +
          'inset 0 1px 0 rgba(255, 255, 255, 0.4)'
      }}>
        {userContext.name && (
          <div style={{
            fontSize: '13px',
            color: 'rgba(45, 55, 72, 0.9)',
            marginBottom: '14px',
            padding: '10px 16px',
            background: 'rgba(255, 255, 255, 0.4)',
            backdropFilter: 'blur(10px)',
            WebkitBackdropFilter: 'blur(10px)',
            borderRadius: '10px',
            display: 'inline-flex',
            alignItems: 'center',
            gap: '8px',
            fontWeight: '600',
            border: '1.5px solid rgba(255, 255, 255, 0.5)',
            boxShadow: 
              '0 2px 8px rgba(31, 38, 135, 0.1), ' +
              'inset 0 1px 0 rgba(255, 255, 255, 0.6)'
          }}>
            <span>{userContext.name} • Year {userContext.year} • {userContext.course}</span>
          </div>
        )}
        <div style={{ display: 'flex', gap: '14px', alignItems: 'flex-end' }}>
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask about modules, prerequisites, recommendations..."
            style={{
              flex: 1,
              padding: '16px 18px',
              border: '2px solid rgba(255, 255, 255, 0.4)',
              borderRadius: '16px',
              resize: 'none',
              fontFamily: 'inherit',
              fontSize: '15px',
              minHeight: '56px',
              maxHeight: '140px',
              outline: 'none',
              transition: 'all 0.3s',
              background: 'rgba(255, 255, 255, 0.4)',
              backdropFilter: 'blur(10px)',
              WebkitBackdropFilter: 'blur(10px)',
              color: '#2d3748',
              boxShadow: 
                '0 2px 8px rgba(31, 38, 135, 0.1), ' +
                'inset 0 1px 0 rgba(255, 255, 255, 0.5)'
            }}
            onFocus={(e) => {
              e.target.style.borderColor = 'rgba(139, 92, 246, 0.8)';
              e.target.style.background = 'rgba(255, 255, 255, 0.5)';
              e.target.style.boxShadow = 
                '0 0 0 4px rgba(139, 92, 246, 0.15), ' +
                '0 4px 12px rgba(139, 92, 246, 0.2), ' +
                'inset 0 1px 0 rgba(255, 255, 255, 0.6)';
            }}
            onBlur={(e) => {
              e.target.style.borderColor = 'rgba(255, 255, 255, 0.4)';
              e.target.style.background = 'rgba(255, 255, 255, 0.4)';
              e.target.style.boxShadow = 
                '0 2px 8px rgba(31, 38, 135, 0.1), ' +
                'inset 0 1px 0 rgba(255, 255, 255, 0.5)';
            }}
            disabled={loading}
          />
          <button
            onClick={sendMessage}
            disabled={loading || !input.trim()}
            style={{
              padding: '16px 32px',
              background: loading || !input.trim()
                ? 'rgba(203, 213, 224, 0.4)'
                : 'linear-gradient(135deg, rgba(99, 102, 241, 0.95) 0%, rgba(139, 92, 246, 0.95) 100%)',
              backdropFilter: 'blur(10px)',
              WebkitBackdropFilter: 'blur(10px)',
              color: 'white',
              border: loading || !input.trim() 
                ? '2px solid rgba(203, 213, 224, 0.3)'
                : '2px solid rgba(255, 255, 255, 0.4)',
              borderRadius: '16px',
              cursor: loading || !input.trim() ? 'not-allowed' : 'pointer',
              fontSize: '15px',
              fontWeight: '700',
              transition: 'all 0.3s',
              minWidth: '110px',
              boxShadow: loading || !input.trim() 
                ? 'none' 
                : '0 4px 16px rgba(99, 102, 241, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.4)',
              display: 'flex',
              alignItems: 'center',
              gap: '8px'
            }}
            onMouseEnter={(e) => {
              if (!loading && input.trim()) {
                e.target.style.transform = 'translateY(-2px)';
                e.target.style.boxShadow = 
                  '0 6px 20px rgba(99, 102, 241, 0.5), ' +
                  'inset 0 1px 0 rgba(255, 255, 255, 0.5)';
              }
            }}
            onMouseLeave={(e) => {
              e.target.style.transform = 'translateY(0)';
              e.target.style.boxShadow = loading || !input.trim() 
                ? 'none' 
                : '0 4px 16px rgba(99, 102, 241, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.4)';
            }}
          >
            {loading ? (
              <>
                <div className="spinner-small"></div>
                <span>Sending</span>
              </>
            ) : (
              <>
                <span>Send</span>
              </>
            )}
          </button>
        </div>
      </div>

      <style>{`
        @keyframes fadeInUp {
          from {
            opacity: 0;
            transform: translateY(20px);
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

        .spinner-small {
          width: 16px;
          height: 16px;
          border: 3px solid rgba(255, 255, 255, 0.3);
          border-top-color: white;
          border-radius: 50%;
          animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};

export default AIChatBot;
