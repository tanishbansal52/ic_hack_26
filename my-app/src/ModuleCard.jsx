import React, { useState } from 'react';
import ROIScatterPlot from './ROIScatterPlot';
import './ModuleCard.css';

const ModuleCard = ({ module }) => {
  const [showROI, setShowROI] = useState(false);
  const [roiData, setRoiData] = useState(null);
  const [loading, setLoading] = useState(false);

  // This simulates fetching data from your Python backend
  const handleCalculateROI = async () => {
    if (showROI) {
      setShowROI(false); // Toggle off if already open
      return;
    }

    setLoading(true);
    
    // TODO: Replace this with actual fetch('/api/modules/' + module.id + '/roi')
    // Simulating API delay and response based on your seed data
    setTimeout(() => {
        // Hardcoded response based on your "Machine Learning" seed data
        const mockResponse = {
            r_squared: 0.95,
            points: [
                { student_id: 101, attentiveness: 91, grade: 92.5 }, // Good student
                { student_id: 102, attentiveness: 47, grade: 65.0 }, // Average
                { student_id: 103, attentiveness: 15, grade: 38.0 }, // Slacker
            ]
        };
        setRoiData(mockResponse);
        setShowROI(true);
        setLoading(false);
    }, 800);
  };

  return (
    <div className="module-card">
      <div className="module-header">
        <div className="module-icon">📚</div>
        <div className="module-info">
          <div className="module-code">{module.code}</div>
          <h2 className="module-name">{module.name}</h2>
          {module.description && (
            <p className="module-description">{module.description}</p>
          )}
        </div>
      </div>
      
      <button 
        onClick={handleCalculateROI}
        className={`roi-button ${showROI ? 'active' : ''} ${loading ? 'loading' : ''}`}
        disabled={loading}
      >
        {loading ? (
          <>
            <span className="spinner"></span>
            <span>Calculating...</span>
          </>
        ) : showROI ? (
          <>
            <span>👁️</span>
            <span>Hide ROI Analysis</span>
          </>
        ) : (
          <>
            <span>📈</span>
            <span>Analyze Attendance ROI</span>
          </>
        )}
      </button>

      {showROI && roiData && (
        <div className="roi-content">
          <ROIScatterPlot data={roiData.points} rSquared={roiData.r_squared} />
        </div>
      )}
    </div>
  );
};

export default ModuleCard;