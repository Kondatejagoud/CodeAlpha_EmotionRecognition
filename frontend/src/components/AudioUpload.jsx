import React, { useState } from 'react';

export default function AudioUpload({ onUploadComplete, isAnalyzing }) {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith("audio/") || file.name.endsWith(".wav") || file.name.endsWith(".mp3")) {
        setSelectedFile(file);
      } else {
        alert("Please upload a valid audio file (.wav, .mp3).");
      }
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const handleUpload = () => {
    if (selectedFile) {
      onUploadComplete(selectedFile);
    }
  };

  const formatBytes = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="audio-card">
      <h3>Audio File Upload</h3>
      
      <div 
        className={`upload-zone ${dragActive ? "drag-active" : ""} ${selectedFile ? "has-file" : ""}`}
        onDragEnter={handleDrag}
        onDragOver={handleDrag}
        onDragLeave={handleDrag}
        onDrop={handleDrop}
      >
        <input 
          type="file" 
          id="file-upload-input"
          accept="audio/*,.wav,.mp3"
          onChange={handleChange}
          style={{ display: 'none' }}
        />
        
        {!selectedFile ? (
          <label htmlFor="file-upload-input" className="upload-label">
            <div className="upload-icon">📁</div>
            <p>Drag & drop speech audio file, or <span>browse files</span></p>
            <p className="file-formats">Supports WAV, MP3</p>
          </label>
        ) : (
          <div className="file-info-container">
            <div className="file-icon">🎵</div>
            <div className="file-details">
              <p className="file-name">{selectedFile.name}</p>
              <p className="file-size">{formatBytes(selectedFile.size)}</p>
            </div>
            
            <div className="upload-actions">
              <button 
                className="btn btn-secondary btn-sm"
                onClick={() => setSelectedFile(null)}
                disabled={isAnalyzing}
              >
                Clear
              </button>
              
              <button 
                className="btn btn-success btn-sm"
                onClick={handleUpload}
                disabled={isAnalyzing}
              >
                {isAnalyzing ? "Analyzing..." : "Analyze"}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
