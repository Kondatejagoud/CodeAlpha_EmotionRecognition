import React from 'react';

export default function PredictionHistory({ history, onSelectRecord }) {
  const getReliabilityBadge = (rel) => {
    switch (rel) {
      case 'HIGH': return 'rel-high';
      case 'MODERATE': return 'rel-mod';
      case 'UNCERTAIN': return 'rel-unc';
      default: return '';
    }
  };

  const getQualityBadge = (status) => {
    switch (status) {
      case 'GOOD': return 'q-good';
      case 'WARNING': return 'q-warn';
      case 'UNSUITABLE': return 'q-uns';
      default: return '';
    }
  };

  return (
    <div className="history-card">
      <div className="history-header">
        <h3>Prediction History</h3>
        <span className="history-count">{history.length} records</span>
      </div>

      <div className="history-list">
        {history.length === 0 ? (
          <p className="no-history-text">No predictions logged yet. Analyze an audio file or microphone recording to save history.</p>
        ) : (
          history.map((record) => (
            <div 
              key={record.id} 
              className="history-item"
              onClick={() => onSelectRecord && onSelectRecord(record)}
            >
              <div className="history-item-top">
                <span className="history-emotion">{record.prediction}</span>
                <span className={`badge ${getReliabilityBadge(record.reliability)}`}>
                  {record.reliability === "UNCERTAIN" ? "UNCERTAIN" : `${(record.probability * 100).toFixed(0)}%`}
                </span>
              </div>
              
              <div className="history-item-bottom">
                <span className="history-time">{record.timestamp}</span>
                <span className={`badge ${getQualityBadge(record.audio_quality.status)}`}>
                  AQA: {record.audio_quality.status}
                </span>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
