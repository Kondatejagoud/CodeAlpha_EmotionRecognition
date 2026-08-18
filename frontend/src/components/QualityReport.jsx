import React from 'react';

export default function QualityReport({ quality }) {
  if (!quality) return null;

  const { status, reasons, duration, sample_rate, rms, clipping_ratio, silence_ratio, snr_est } = quality;

  const getStatusColor = (status) => {
    switch (status) {
      case 'GOOD': return 'status-good';
      case 'WARNING': return 'status-warning';
      case 'UNSUITABLE': return 'status-unsuitable';
      default: return '';
    }
  };

  return (
    <div className="report-card">
      <div className="report-header">
        <h4>Audio Quality Analysis</h4>
        <span className={`status-badge ${getStatusColor(status)}`}>{status}</span>
      </div>

      {reasons && reasons.length > 0 && (
        <div className="reasons-list">
          {reasons.map((reason, idx) => (
            <div key={idx} className={`reason-item ${status === 'UNSUITABLE' ? 'reason-err' : 'reason-warn'}`}>
              • {reason}
            </div>
          ))}
        </div>
      )}

      <div className="metrics-grid">
        <div className="metric-box">
          <span className="metric-label">Duration</span>
          <span className="metric-value">{duration}s</span>
        </div>
        
        <div className="metric-box">
          <span className="metric-label">Sample Rate</span>
          <span className="metric-value">{sample_rate} Hz</span>
        </div>
        
        <div className="metric-box">
          <span className="metric-label">Loudness (RMS)</span>
          <span className="metric-value">{rms}</span>
        </div>
        
        <div className="metric-box">
          <span className="metric-label">SNR (Estimated)</span>
          <span className="metric-value">{snr_est} dB</span>
        </div>
        
        <div className="metric-box">
          <span className="metric-label">Clipping Ratio</span>
          <span className="metric-value">{(clipping_ratio * 100).toFixed(1)}%</span>
        </div>

        <div className="metric-box">
          <span className="metric-label">Silence Ratio</span>
          <span className="metric-value">{(silence_ratio * 100).toFixed(0)}%</span>
        </div>
      </div>
    </div>
  );
}
