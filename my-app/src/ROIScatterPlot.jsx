import React from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Label } from 'recharts';

const ROIScatterPlot = ({ data, rSquared }) => {

  const getAdvice = (r2) => {
    if (r2 >= 0.7) {
      return {
        text: "🚨 CRITICAL: High attendance is strongly linked to good grades. Don't skip!",
        color: "#d9534f", 
        bg: "#fde8e8"
      };
    } else if (r2 >= 0.4) {
      return {
        text: "⚠️ MODERATE: Attendance helps, but self-study is also effective.",
        color: "#f0ad4e", 
        bg: "#fcf8e3"
      };
    } else {
      return {
        text: "✅ RELAXED: Low correlation. You can likely pass by studying the slides.",
        color: "#5cb85c", 
        bg: "#dff0d8"
      };
    }
  };

  const advice = getAdvice(rSquared);

  return (
    <div style={{ marginTop: '20px', background: '#fff', padding: '15px', borderRadius: '8px', border: '1px solid #eee' }}>
      
      <h3 style={{ textAlign: 'center', color: '#333', marginBottom: '5px' }}>
        Attendance ROI (R² = {rSquared})
      </h3>

      <div style={{ 
        backgroundColor: advice.bg, 
        color: advice.color, 
        padding: '10px', 
        borderRadius: '5px', 
        textAlign: 'center',
        marginBottom: '15px',
        fontWeight: 'bold',
        border: `1px solid ${advice.color}`
      }}>
        {advice.text}
      </div>

      <div style={{ height: '300px', width: '100%' }}>
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart 
            margin={{ top: 20, right: 20, bottom: 50, left: 0 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            
            <XAxis type="number" dataKey="attentiveness" name="Avg Attentiveness" unit="%" domain={[0, 100]}>
              <Label value="Attendance / Attentiveness (%)" offset={-30} position="insideBottom" />
            </XAxis>

            <YAxis type="number" dataKey="grade" name="Final Grade" unit="%" domain={[0, 100]}>
              <Label value="Final Grade (%)" angle={-90} position="insideLeft" offset={10} style={{ textAnchor: 'middle' }} />
            </YAxis>

            <Tooltip cursor={{ strokeDasharray: '3 3' }} />
            <Scatter name="Students" data={data} fill="#8884d8" />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default ROIScatterPlot;