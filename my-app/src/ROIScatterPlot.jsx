import React from 'react';
// 1. IMPORT 'Label' here
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Label } from 'recharts';

const ROIScatterPlot = ({ data, rSquared }) => {
  return (
    <div style={{ height: '350px', marginTop: '20px', background: '#fff', padding: '15px', borderRadius: '8px', border: '1px solid #eee' }}>
      <h3 style={{ textAlign: 'center', color: '#333', marginBottom: '10px' }}>
        Attendance ROI (R² = {rSquared})
      </h3>
      <ResponsiveContainer width="100%" height="90%">
        <ScatterChart 
          // 2. INCREASE MARGINS so labels fit
          margin={{ top: 20, right: 30, bottom: 40, left: 40 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          
          {/* X AXIS */}
          <XAxis type="number" dataKey="attentiveness" name="Avg Attentiveness" unit="%" domain={[0, 100]}>
            {/* 3. ADD LABEL */}
            <Label value="Attendance / Attentiveness (%)" offset={-20} position="insideBottom" />
          </XAxis>

          {/* Y AXIS */}
          <YAxis type="number" dataKey="grade" name="Final Grade" unit="%" domain={[0, 100]}>
            {/* 3. ADD LABEL */}
            <Label value="Final Grade (%)" angle={-90} position="insideLeft" offset={10} style={{ textAnchor: 'middle' }} />
          </YAxis>

          <Tooltip cursor={{ strokeDasharray: '3 3' }} />
          <Scatter name="Students" data={data} fill="#8884d8" />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
};

export default ROIScatterPlot;