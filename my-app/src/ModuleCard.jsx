import React from 'react';
import './ModuleCard.css';

const ModuleCard = ({ moduleName, module }) => {
  const moduleData = module;

  // Get category color based on effectiveness
  const getCategoryColor = (category) => {
    switch (category) {
      case 'Very Effective':
        return '#10b981';
      case 'Effective':
        return '#22c55e';
      case 'Slightly Effective':
        return '#eab308';
      case 'Not Effective':
        return '#ef4444';
      default:
        return '#6b7280';
    }
  };

  const categoryColor = getCategoryColor(moduleData.category);

  return (
    <div className="module-card">
      <div className="module-header">
        <div className="module-icon">
          <div style={{
            width: '48px',
            height: '48px',
            borderRadius: '12px',
            background: `linear-gradient(135deg, ${categoryColor} 0%, ${categoryColor}aa 100%)`,
            boxShadow: `0 4px 12px ${categoryColor}66, inset 0 1px 0 rgba(255, 255, 255, 0.3)`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'white',
            fontWeight: 'bold',
            fontSize: '18px'
          }}>
            {Math.round(moduleData.effectiveness_score)}
          </div>
        </div>
        <div className="module-info">
          <h2 className="module-name">{moduleName}</h2>
          <span 
            className="module-category" 
            style={{ 
              color: categoryColor, 
              fontWeight: '600',
              fontSize: '14px'
            }}
          >
            {moduleData.category} {moduleData.is_significant ? '✓' : ''}
          </span>
        </div>
      </div>

      <div className="module-stats">
        <div className="stats-section">
          <h3 className="stats-title">Effectiveness Score</h3>
          <div className="effectiveness-bar">
            <div 
              className="effectiveness-fill" 
              style={{ 
                width: `${moduleData.effectiveness_score}%`,
                backgroundColor: categoryColor
              }}
            ></div>
          </div>
          <span className="effectiveness-value">{moduleData.effectiveness_score.toFixed(2)} / 100</span>
        </div>

        <div className="stats-grid">
          <div className="stat-item">
            <span className="stat-label">Attendance Mean</span>
            <span className="stat-value">{moduleData.attendance_mean.toFixed(2)}%</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Attendance Std</span>
            <span className="stat-value">±{moduleData.attendance_std.toFixed(2)}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Grade Mean</span>
            <span className="stat-value">{moduleData.grade_mean.toFixed(2)}%</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Grade Std</span>
            <span className="stat-value">±{moduleData.grade_std.toFixed(2)}</span>
          </div>
        </div>

        <div className="stats-section">
          <h3 className="stats-title">Correlation Analysis</h3>
          <div className="stats-grid">
            <div className="stat-item">
              <span className="stat-label">Pearson r</span>
              <span className="stat-value">{moduleData.pearson_r.toFixed(4)}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Pearson p-value</span>
              <span className="stat-value">{moduleData.pearson_p.toFixed(6)}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Spearman r</span>
              <span className="stat-value">{moduleData.spearman_r.toFixed(4)}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">R²</span>
              <span className="stat-value">{moduleData.r_squared.toFixed(4)}</span>
            </div>
          </div>
        </div>

        <div className="stats-section">
          <h3 className="stats-title">Regression</h3>
          <div className="stats-grid">
            <div className="stat-item">
              <span className="stat-label">Slope</span>
              <span className="stat-value">{moduleData.slope.toFixed(4)}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Normalized Slope</span>
              <span className="stat-value">{moduleData.normalized_slope.toFixed(4)}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Intercept</span>
              <span className="stat-value">{moduleData.intercept.toFixed(4)}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Sample Size</span>
              <span className="stat-value">{moduleData.sample_size}</span>
            </div>
          </div>
        </div>

        <div className="stats-section">
          <h3 className="stats-title">Score Breakdown</h3>
          <div className="breakdown-grid">
            <div className="breakdown-item">
              <span className="breakdown-label">Correlation Strength</span>
              <div className="breakdown-bar">
                <div 
                  className="breakdown-fill" 
                  style={{ width: `${(moduleData.score_breakdown.correlation_strength / 40) * 100}%` }}
                ></div>
              </div>
              <span className="breakdown-value">{moduleData.score_breakdown.correlation_strength.toFixed(2)} / 40</span>
            </div>
            <div className="breakdown-item">
              <span className="breakdown-label">Variance Explained</span>
              <div className="breakdown-bar">
                <div 
                  className="breakdown-fill" 
                  style={{ width: `${(moduleData.score_breakdown.variance_explained / 30) * 100}%` }}
                ></div>
              </div>
              <span className="breakdown-value">{moduleData.score_breakdown.variance_explained.toFixed(2)} / 30</span>
            </div>
            <div className="breakdown-item">
              <span className="breakdown-label">Statistical Significance</span>
              <div className="breakdown-bar">
                <div 
                  className="breakdown-fill" 
                  style={{ width: `${(moduleData.score_breakdown.statistical_significance / 20) * 100}%` }}
                ></div>
              </div>
              <span className="breakdown-value">{moduleData.score_breakdown.statistical_significance.toFixed(2)} / 20</span>
            </div>
            <div className="breakdown-item">
              <span className="breakdown-label">Effect Size</span>
              <div className="breakdown-bar">
                <div 
                  className="breakdown-fill" 
                  style={{ width: `${(moduleData.score_breakdown.effect_size / 10) * 100}%` }}
                ></div>
              </div>
              <span className="breakdown-value">{moduleData.score_breakdown.effect_size.toFixed(2)} / 10</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModuleCard;