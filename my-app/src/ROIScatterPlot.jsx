import React from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Label, ReferenceLine } from 'recharts';

const ROIScatterPlot = ({ data, rSquared }) => {

  const getAdvice = (r2) => {
    if (r2 >= 0.7) {
      return {
        text: "High attendance is strongly linked to good grades. Don't skip!",
        title: "CRITICAL",
        color: "#e53e3e",
        bg: "linear-gradient(135deg, #feb2b2 0%, #fc8181 100%)"
      };
    } else if (r2 >= 0.4) {
      return {
        text: "Attendance helps, but self-study is also effective.",
        title: "MODERATE",
        color: "#dd6b20",
        bg: "linear-gradient(135deg, #fbd38d 0%, #f6ad55 100%)"
      };
    } else {
      return {
        text: "Low correlation. You can likely pass by studying the slides.",
        title: "RELAXED",
        color: "#38a169",
        bg: "linear-gradient(135deg, #9ae6b4 0%, #68d391 100%)"
      };
    }
  };

  const advice = getAdvice(rSquared);

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      return (
        <div style={{
          background: 'rgba(255, 255, 255, 0.4)',
          backdropFilter: 'blur(20px) saturate(180%)',
          WebkitBackdropFilter: 'blur(20px) saturate(180%)',
          padding: '14px 18px',
          borderRadius: '14px',
          boxShadow: 
            '0 8px 24px rgba(31, 38, 135, 0.2), ' +
            'inset 0 1px 0 rgba(255, 255, 255, 0.6)',
          border: '1.5px solid rgba(255, 255, 255, 0.5)'
        }}>
          <p style={{ margin: '0 0 6px 0', fontWeight: 'bold', color: '#2d3748' }}>
            Student #{payload[0].payload.student_id}
          </p>
          <p style={{ margin: '4px 0', fontSize: '14px', color: '#4a5568' }}>
            <strong>Attendance:</strong> {payload[0].value}%
          </p>
          <p style={{ margin: '4px 0', fontSize: '14px', color: '#4a5568' }}>
            <strong>Grade:</strong> {payload[1].value}%
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div style={{ 
      marginTop: '20px', 
      background: 'rgba(255, 255, 255, 0.3)',
      backdropFilter: 'blur(30px) saturate(180%)',
      WebkitBackdropFilter: 'blur(30px) saturate(180%)',
      padding: '28px', 
      borderRadius: '20px',
      border: '1.5px solid rgba(255, 255, 255, 0.5)',
      boxShadow: 
        '0 8px 24px rgba(31, 38, 135, 0.15), ' +
        'inset 0 1px 0 rgba(255, 255, 255, 0.6), ' +
        'inset 0 -1px 0 rgba(255, 255, 255, 0.2)'
    }}>
      
      <h3 style={{ 
        textAlign: 'center', 
        color: '#2d3748', 
        marginBottom: '8px',
        fontSize: '22px',
        fontWeight: '700',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap: '10px'
      }}>
        Attendance ROI Analysis
      </h3>

      <p style={{
        textAlign: 'center',
        color: '#718096',
        fontSize: '14px',
        marginBottom: '20px'
      }}>
        R² = {rSquared.toFixed(2)} • {advice.title}
      </p>

      <div style={{ 
        background: advice.bg,
        backdropFilter: 'blur(10px)',
        WebkitBackdropFilter: 'blur(10px)',
        padding: '18px 24px', 
        borderRadius: '16px', 
        textAlign: 'center',
        marginBottom: '24px',
        fontWeight: '600',
        fontSize: '15px',
        color: 'white',
        boxShadow: 
          '0 8px 20px rgba(0, 0, 0, 0.15), ' +
          'inset 0 1px 0 rgba(255, 255, 255, 0.4)',
        border: '1px solid rgba(255, 255, 255, 0.3)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap: '12px'
      }}>
        <span>{advice.text}</span>
      </div>

      <div style={{ height: '350px', width: '100%' }}>
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart 
            margin={{ top: 20, right: 20, bottom: 60, left: 20 }}
          >
            <defs>
              <linearGradient id="scatterGradient" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stopColor="#667eea" />
                <stop offset="100%" stopColor="#764ba2" />
              </linearGradient>
            </defs>
            
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            
            <XAxis 
              type="number" 
              dataKey="attentiveness" 
              name="Avg Attentiveness" 
              unit="%" 
              domain={[0, 100]}
              tick={{ fill: '#4a5568', fontSize: 12, fontWeight: 600 }}
              stroke="#cbd5e0"
            >
              <Label 
                value="Attendance (%)" 
                offset={-40} 
                position="insideBottom" 
                style={{ fill: '#2d3748', fontWeight: 'bold', fontSize: 14 }}
              />
            </XAxis>

            <YAxis 
              type="number" 
              dataKey="grade" 
              name="Final Grade" 
              unit="%" 
              domain={[0, 100]}
              tick={{ fill: '#4a5568', fontSize: 12, fontWeight: 600 }}
              stroke="#cbd5e0"
            >
              <Label 
                value="Final Grade (%)" 
                angle={-90} 
                position="insideLeft" 
                offset={0}
                style={{ textAnchor: 'middle', fill: '#2d3748', fontWeight: 'bold', fontSize: 14 }} 
              />
            </YAxis>

            <Tooltip content={<CustomTooltip />} />
            
            <ReferenceLine 
              y={70} 
              stroke="#48bb78" 
              strokeDasharray="5 5" 
              label={{ value: 'Pass Line (70%)', position: 'right', fill: '#48bb78', fontWeight: 'bold' }} 
            />
            
            <Scatter 
              name="Students" 
              data={data} 
              fill="url(#scatterGradient)"
              r={8}
              strokeWidth={2}
              stroke="white"
            />
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      <div style={{
        marginTop: '20px',
        padding: '18px',
        background: 'rgba(99, 102, 241, 0.15)',
        backdropFilter: 'blur(10px)',
        WebkitBackdropFilter: 'blur(10px)',
        borderRadius: '12px',
        fontSize: '13px',
        color: '#2d3748',
        lineHeight: '1.7',
        border: '1px solid rgba(99, 102, 241, 0.2)',
        boxShadow: 
          '0 4px 12px rgba(99, 102, 241, 0.1), ' +
          'inset 0 1px 0 rgba(255, 255, 255, 0.5)'
      }}>
        <strong>How to read this:</strong> Each dot represents a student. The higher the R² value (closer to 1.0), 
        the stronger the correlation between attendance and grades. A high R² means attendance really matters for this module!
      </div>
    </div>
  );
};

export default ROIScatterPlot;