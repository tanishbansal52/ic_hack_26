import React, { useState } from 'react';
import ModuleCard from './ModuleCard';
import AIChatBot from './AIChatBot';
import './App.css';

// Hardcoded list from your DB for now

function App() {
  const [showChat, setShowChat] = useState(false);
  const [userContext, setUserContext] = useState({
    name: '',
    year: 2,
    course: 'Computer Science'
  });

  const [modules, setModules] = useState({});

  React.useEffect(() => {
    fetch('/src/assets/module_effectiveness.json').then(response => response.json()).then(data => {
      console.log(data)
      setModules(data);
    })

  }, [])

  return (
    <div className="app-container">
      {/* Animated Background */}
      <div className="animated-background">
        <div className="gradient-orb orb-1"></div>
        <div className="gradient-orb orb-2"></div>
        <div className="gradient-orb orb-3"></div>
      </div>

      <div className="content-wrapper">
        {/* Header */}
        <div className="header-container">
          <div className="header-text">
            <h1 className="main-title">
              Module Performance Tracker
            </h1>
            <p className="subtitle">
              {showChat ? 'Get personalized advice on your modules' : 'See how attendance impacts your grades'}
            </p>
          </div>
          
          <button
            onClick={() => setShowChat(!showChat)}
            className={`toggle-button ${showChat ? 'chat-mode' : 'module-mode'}`}
          >
            <span>{showChat ? 'Show Modules' : 'AI Advisor'}</span>
          </button>
        </div>

      {/* User Context Form */}
      <div className="user-context-card">
        <h3 className="context-title">
          Your Information
        </h3>
        <div className="context-form">
          <div className="form-field">
            <label className="field-label">Name</label>
            <input
              type="text"
              value={userContext.name}
              onChange={(e) => setUserContext({...userContext, name: e.target.value})}
              placeholder="Enter your name"
              className="field-input"
            />
          </div>
          <div className="form-field">
            <label className="field-label">Year</label>
            <select
              value={userContext.year}
              onChange={(e) => setUserContext({...userContext, year: parseInt(e.target.value)})}
              className="field-select"
            >
              <option value={1}>1st Year</option>
              <option value={2}>2nd Year</option>
              <option value={3}>3rd Year</option>
              <option value={4}>4th Year</option>
            </select>
          </div>
          <div className="form-field">
            <label className="field-label">Course</label>
            <input
              type="text"
              value={userContext.course}
              onChange={(e) => setUserContext({...userContext, course: e.target.value})}
              placeholder="e.g., Computer Science"
              className="field-input"
            />
          </div>
        </div>
      </div>

      {/* Conditional Rendering */}
      <div className="main-content">
        {showChat ? (
          <div className="chat-container-wrapper">
            <AIChatBot userContext={userContext} />
          </div>
        ) : (
          <div className="modules-grid">
            {Object.entries(modules).map(([key, mod], index) => (
              <div 
                key={key} 
                className="module-card-wrapper"
                style={{ animationDelay: `${index * 0.1}s` }}
              >
                <ModuleCard module={mod} moduleName={key} />
              </div>
            ))}
          </div>
        )}
      </div>
      </div>
    </div>
  );
}

export default App;