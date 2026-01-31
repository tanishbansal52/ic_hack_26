import React, { useState } from 'react';
import ROIScatterPlot from './ROIScatterPlot';

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
    <div style={{ border: '1px solid #ddd', borderRadius: '10px', padding: '20px', margin: '10px', boxShadow: '0 2px 5px rgba(0,0,0,0.1)' }}>
      <h2>{module.code}: {module.name}</h2>
      
      <button 
        onClick={handleCalculateROI}
        style={{
            padding: '10px 20px', 
            background: showROI ? '#d9534f' : '#0275d8', 
            color: 'white', 
            border: 'none', 
            borderRadius: '5px', 
            cursor: 'pointer'
        }}
      >
        {loading ? 'Calculating...' : showROI ? 'Hide ROI' : 'Calculate ROI'}
      </button>

      {showROI && roiData && (
        <ROIScatterPlot data={roiData.points} rSquared={roiData.r_squared} />
      )}
    </div>
  );
};

export default ModuleCard;