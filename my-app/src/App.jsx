import React, { useState } from 'react';
import ModuleCard from './ModuleCard';
import AIChatBot from './AIChatBot';

// Hardcoded list from your DB for now
const modules = [
  { id: 1, code: 'DOC601', name: 'Machine Learning' },
  { id: 2, code: 'DOC602', name: 'Computer Graphics' },
  { id: 3, code: 'DOC404', name: 'Professional Skills' }
];

function App() {
  const [showChat, setShowChat] = useState(false);
  const [userContext, setUserContext] = useState({
    name: '',
    year: 2,
    course: 'Computer Science'
  });

  return (
    <div style={{ padding: '40px', fontFamily: 'Arial, sans-serif', minHeight: '100vh', background: '#f5f7fa' }}>
      {/* Header */}
      <div style={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center',
        marginBottom: '30px',
        flexWrap: 'wrap',
        gap: '20px'
      }}>
        <div>
          <h1 style={{ margin: '0 0 10px 0', fontSize: '32px', color: '#333' }}>Module Selection Helper</h1>
          <p style={{ margin: 0, fontSize: '16px', color: '#666' }}>
            {showChat ? 'Chat with your AI academic advisor' : 'Select a module to see if attendance actually matters.'}
          </p>
        </div>
        
        <button
          onClick={() => setShowChat(!showChat)}
          style={{
            padding: '14px 28px',
            background: showChat 
              ? 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)'
              : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            color: 'white',
            border: 'none',
            borderRadius: '10px',
            cursor: 'pointer',
            fontSize: '16px',
            fontWeight: 'bold',
            boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
            transition: 'all 0.3s',
            display: 'flex',
            alignItems: 'center',
            gap: '8px'
          }}
          onMouseEnter={(e) => e.target.style.transform = 'translateY(-2px)'}
          onMouseLeave={(e) => e.target.style.transform = 'translateY(0)'}
        >
          {showChat ? '📊 Show Modules' : '🤖 AI Advisor'}
        </button>
      </div>

      {/* User Context Form */}
      {!showChat && (
        <div style={{
          background: 'white',
          padding: '24px',
          borderRadius: '12px',
          marginBottom: '30px',
          border: '1px solid #e0e0e0',
          boxShadow: '0 2px 8px rgba(0,0,0,0.05)'
        }}>
          <h3 style={{ margin: '0 0 16px 0', fontSize: '18px', color: '#333' }}>📋 Your Information</h3>
          <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap' }}>
            <div style={{ flex: '1 1 200px' }}>
              <label style={{ display: 'block', marginBottom: '8px', fontSize: '14px', fontWeight: '600', color: '#555' }}>
                Name:
              </label>
              <input
                type="text"
                value={userContext.name}
                onChange={(e) => setUserContext({...userContext, name: e.target.value})}
                placeholder="Your name"
                style={{
                  width: '100%',
                  padding: '10px 14px',
                  border: '1px solid #ddd',
                  borderRadius: '8px',
                  fontSize: '14px',
                  outline: 'none',
                  transition: 'border-color 0.2s'
                }}
                onFocus={(e) => e.target.style.borderColor = '#667eea'}
                onBlur={(e) => e.target.style.borderColor = '#ddd'}
              />
            </div>
            <div style={{ flex: '1 1 150px' }}>
              <label style={{ display: 'block', marginBottom: '8px', fontSize: '14px', fontWeight: '600', color: '#555' }}>
                Year:
              </label>
              <select
                value={userContext.year}
                onChange={(e) => setUserContext({...userContext, year: parseInt(e.target.value)})}
                style={{
                  width: '100%',
                  padding: '10px 14px',
                  border: '1px solid #ddd',
                  borderRadius: '8px',
                  fontSize: '14px',
                  outline: 'none',
                  background: 'white',
                  cursor: 'pointer'
                }}
              >
                <option value={1}>1st Year</option>
                <option value={2}>2nd Year</option>
                <option value={3}>3rd Year</option>
                <option value={4}>4th Year</option>
              </select>
            </div>
            <div style={{ flex: '1 1 200px' }}>
              <label style={{ display: 'block', marginBottom: '8px', fontSize: '14px', fontWeight: '600', color: '#555' }}>
                Course:
              </label>
              <input
                type="text"
                value={userContext.course}
                onChange={(e) => setUserContext({...userContext, course: e.target.value})}
                placeholder="e.g., Computer Science"
                style={{
                  width: '100%',
                  padding: '10px 14px',
                  border: '1px solid #ddd',
                  borderRadius: '8px',
                  fontSize: '14px',
                  outline: 'none',
                  transition: 'border-color 0.2s'
                }}
                onFocus={(e) => e.target.style.borderColor = '#667eea'}
                onBlur={(e) => e.target.style.borderColor = '#ddd'}
              />
            </div>
          </div>
        </div>
      )}

      {/* Conditional Rendering */}
      {showChat ? (
        <AIChatBot userContext={userContext} />
      ) : (
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(auto-fit, minmax(450px, 1fr))', 
          gap: '20px' 
        }}>
          {modules.map(mod => (
            <ModuleCard key={mod.id} module={mod} />
          ))}
        </div>
      )}
    </div>
  );
}

export default App;