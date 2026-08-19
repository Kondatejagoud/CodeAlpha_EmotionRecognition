import React, { useState, useEffect } from 'react';
import AudioRecorder from './AudioRecorder';
import AudioUpload from './AudioUpload';
import QualityReport from './QualityReport';
import AttentionPlot from './AttentionPlot';
import GradCamPlot from './GradCamPlot';
import PredictionHistory from './PredictionHistory';

const API_BASE = "http://127.0.0.1:8000/api";

export default function Dashboard() {
  const [activePrediction, setActivePrediction] = useState(null);
  const [historyList, setHistoryList] = useState([]);
  const [modelInfo, setModelInfo] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  useEffect(() => {
    fetchHistory();
    fetchModelInfo();
  }, []);

  const fetchHistory = async () => {
    try {
      const response = await fetch(`${API_BASE}/history?limit=10`);
      if (response.ok) {
        const data = await response.json();
        setHistoryList(data);
      }
    } catch (err) {
      console.error("Error fetching prediction history:", err);
    }
  };

  const handleClearHistory = async () => {
    try {
      const response = await fetch(`${API_BASE}/history`, {
        method: "DELETE"
      });
      if (response.ok) {
        setHistoryList([]);
        setActivePrediction(null);
      } else {
        alert("Failed to clear prediction history.");
      }
    } catch (err) {
      console.error("Error clearing prediction history:", err);
      alert(`Clear history failed: ${err.message}`);
    }
  };

  const fetchModelInfo = async () => {
    try {
      const response = await fetch(`${API_BASE}/model-info`);
      if (response.ok) {
        const data = await response.json();
        setModelInfo(data);
      }
    } catch (err) {
      console.error("Error fetching model info:", err);
    }
  };

  const handleAudioAnalysis = async (fileOrBlob) => {
    setIsAnalyzing(true);
    setActivePrediction(null);
    
    const formData = new FormData();
    // Use filename 'recording.wav' if passing blob from recorder
    if (fileOrBlob instanceof Blob) {
      formData.append("file", fileOrBlob, "recording.wav");
    } else {
      formData.append("file", fileOrBlob);
    }

    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Server error analyzing audio.");
      }

      const data = await response.json();
      setActivePrediction(data);
      fetchHistory(); // refresh local database list
    } catch (err) {
      console.error("Analysis failed:", err);
      alert(`Inference failed: ${err.message}`);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getReliabilityColor = (rel) => {
    switch (rel) {
      case 'HIGH': return 'rel-high';
      case 'MODERATE': return 'rel-mod';
      case 'UNCERTAIN': return 'rel-unc';
      default: return '';
    }
  };

  return (
    <div className="dashboard-grid">
      <header className="app-header">
        <div className="header-title">
          <h1>EmotionSense AI</h1>
          <p>Explainable Speech Emotion Recognition System</p>
        </div>
        
        {modelInfo && (
          <div className="model-info-badge">
            <span className="info-title">Model: {modelInfo.model_name || "CNN-BiLSTM-Attention"}</span>
            <span className="info-meta">Test Acc: {modelInfo.metrics?.test_accuracy ? `${(modelInfo.metrics.test_accuracy * 100).toFixed(1)}%` : "Loading..."}</span>
          </div>
        )}
      </header>

      <main className="dashboard-body">
        {/* Left column - Controls */}
        <section className="controls-column">
          <AudioRecorder onRecordingComplete={handleAudioAnalysis} isAnalyzing={isAnalyzing} />
          <AudioUpload onUploadComplete={handleAudioAnalysis} isAnalyzing={isAnalyzing} />
          
          {modelInfo && (
            <div className="model-card-metadata">
              <h4>Active Model Metadata</h4>
              <div className="meta-row">
                <span className="meta-label">Architecture</span>
                <span className="meta-val">{modelInfo.architecture || "cnn-bilstm-attention"}</span>
              </div>
              <div className="meta-row">
                <span className="meta-label">Features</span>
                <span className="meta-val">{modelInfo.feature_representation || "mel"}</span>
              </div>
              <div className="meta-row">
                <span className="meta-label">Train Actors</span>
                <span className="meta-val">{modelInfo.training_actors || "1-18"}</span>
              </div>
              <div className="meta-row">
                <span className="meta-label">Val Actors</span>
                <span className="meta-val">{modelInfo.validation_actors || "19-20"}</span>
              </div>
              <div className="meta-row">
                <span className="meta-label">Test Actors</span>
                <span className="meta-val">{modelInfo.test_actors || "21-24"}</span>
              </div>
            </div>
          )}
        </section>

        {/* Center column - Analysis results */}
        <section className="analysis-column">
          {isAnalyzing && (
            <div className="analysis-loading">
              <div className="spinner"></div>
              <p>Analyzing speech audio features & computing predictions...</p>
            </div>
          )}

          {!isAnalyzing && !activePrediction && (
            <div className="analysis-placeholder">
              <div className="placeholder-icon">🎙️</div>
              <h3>Ready for Input</h3>
              <p>Upload a vocal statement or record your voice live using the controls on the left. The system will inspect audio quality and run predictions.</p>
            </div>
          )}

          {!isAnalyzing && activePrediction && (
            <div className="analysis-results">
              {/* Primary prediction board */}
              <div className="results-header-board">
                <div className="board-main-emotion">
                  <span className="board-label">Predicted Emotion</span>
                  <span className="board-value">{activePrediction.prediction}</span>
                </div>
                
                <div className="board-reliability">
                  <span className="board-label">Reliability</span>
                  <span className={`reliability-tag ${getReliabilityColor(activePrediction.reliability)}`}>
                    {activePrediction.reliability}
                  </span>
                </div>
              </div>

              {/* Grid with metrics & top alternatives */}
              <div className="details-subgrid">
                {/* Top alternative predictions */}
                <div className="alternatives-card">
                  <h4>Top Predictions</h4>
                  <div className="alt-list">
                    {activePrediction.top_predictions.map((p, idx) => (
                      <div key={idx} className="alt-item">
                        <span className="alt-emotion">{p.emotion}</span>
                        <div className="alt-bar-wrapper">
                          <div 
                            className="alt-bar" 
                            style={{ width: `${p.probability * 100}%` }}
                          />
                        </div>
                        <span className="alt-percentage">{(p.probability * 100).toFixed(0)}%</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Quality Metrics */}
                <QualityReport quality={activePrediction.audio_quality} />
              </div>

              {/* Explainability visualization displays */}
              {activePrediction.audio_quality.status !== "UNSUITABLE" && (
                <div className="explainability-section">
                  <h3>Model Explainability Interpretations</h3>
                  
                  {activePrediction.explainability.attention && activePrediction.explainability.attention.length > 0 && (
                    <AttentionPlot attention={activePrediction.explainability.attention} />
                  )}
                  
                  {activePrediction.explainability.grad_cam && activePrediction.explainability.grad_cam.length > 0 && (
                    <GradCamPlot gradCam={activePrediction.explainability.grad_cam} />
                  )}
                </div>
              )}
            </div>
          )}
        </section>

        {/* Right column - SQL logs */}
        <section className="history-column">
          <PredictionHistory 
            history={historyList} 
            onSelectRecord={(record) => {
              // Map prediction history record back to details view
              setActivePrediction({
                prediction: record.prediction,
                probability: record.probability,
                reliability: record.reliability,
                top_predictions: record.top_predictions,
                audio_quality: record.audio_quality,
                explainability: {
                  grad_cam: [], // SQLite history doesn't store heavy Grad-CAM matrices
                  attention: []
                },
                model_metadata: {
                  model_name: record.model_used,
                  architecture: "Saved Record",
                  training_date: record.timestamp
                }
              });
            }} 
            onClearHistory={handleClearHistory}
          />
        </section>
      </main>
      <footer className="app-footer" style={{ padding: "1.5rem", borderTop: "1px solid #eee", marginTop: "2rem", textAlign: "center", fontSize: "0.85rem", color: "#666" }}>
        <p className="limitation-notice">
          <strong>Limitation Notice:</strong> This system predicts acoustic patterns associated with emotion categories in the training dataset. It does not determine a person's actual psychological or mental state.
        </p>
      </footer>
    </div>
  );
}
