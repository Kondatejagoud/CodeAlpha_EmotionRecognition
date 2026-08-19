import React, { useRef, useEffect } from 'react';

export default function GradCamPlot({ gradCam }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current || !gradCam || gradCam.length === 0) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // Grad-CAM matrix shape details: rows = n_mels (usually 128), cols = time_steps (usually 130)
    const rows = gradCam.length;
    const cols = gradCam[0].length;

    // Create an offscreen canvas to render the raw heatmap pixels first
    const offscreen = document.createElement('canvas');
    offscreen.width = cols;
    offscreen.height = rows;
    const offscreenCtx = offscreen.getContext('2d');
    
    const imgData = offscreenCtx.createImageData(cols, rows);

    // Helper: custom plasma/spectrogram colormap
    const getColor = (v) => {
      // Input v is in [0, 1]
      // Colors: Slate-900 (0.0) -> Purple (0.35) -> Hot Pink (0.7) -> Orange (0.9) -> Yellow/White (1.0)
      let r, g, b;
      
      if (v < 0.35) {
        // Interpolate Slate-900 (15, 23, 42) to Purple (99, 102, 241)
        const t = v / 0.35;
        r = 15 + t * (99 - 15);
        g = 23 + t * (102 - 23);
        b = 42 + t * (241 - 42);
      } else if (v < 0.7) {
        // Interpolate Purple (99, 102, 241) to Hot Pink (236, 72, 153)
        const t = (v - 0.35) / 0.35;
        r = 99 + t * (236 - 99);
        g = 102 + t * (72 - 102);
        b = 241 + t * (153 - 241);
      } else if (v < 0.9) {
        // Interpolate Hot Pink (236, 72, 153) to Orange (249, 115, 22)
        const t = (v - 0.7) / 0.2;
        r = 236 + t * (249 - 236);
        g = 72 + t * (115 - 72);
        b = 153 + t * (22 - 153);
      } else {
        // Interpolate Orange (249, 115, 22) to Bright Yellow (253, 224, 71)
        const t = (v - 0.9) / 0.1;
        r = 249 + t * (253 - 249);
        g = 115 + t * (224 - 115);
        b = 22 + t * (71 - 22);
      }
      return [Math.round(r), Math.round(g), Math.round(b)];
    };

    // Fill offscreen buffer (flip vertically since mel arrays start low freq to high)
    for (let y = 0; y < rows; y++) {
      const targetRow = rows - y - 1; // flip vertically
      for (let x = 0; x < cols; x++) {
        const val = gradCam[targetRow][x];
        const [r, g, b] = getColor(val);
        
        const idx = (y * cols + x) * 4;
        imgData.data[idx] = r;
        imgData.data[idx+1] = g;
        imgData.data[idx+2] = b;
        imgData.data[idx+3] = 255; // Alpha full opacity
      }
    }

    offscreenCtx.putImageData(imgData, 0, 0);

    // Draw stretched spectrogram on main canvas
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, width, height);

    // We leave 40px left padding for Y axis and 20px bottom padding for X axis
    const graphWidth = width - 50;
    const graphHeight = height - 30;
    const startX = 40;
    const startY = 10;

    // Enable hardware interpolation smoothing
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(offscreen, startX, startY, graphWidth, graphHeight);

    // 2. Draw Y Axis labels (Frequency pitch range)
    ctx.fillStyle = '#94a3b8';
    ctx.font = '9px sans-serif';
    ctx.fillText("High Pitch", 5, startY + 10);
    ctx.fillText("Low Pitch", 5, startY + graphHeight - 5);

    // 3. Draw X Axis labels (Time duration seconds)
    ctx.strokeStyle = '#475569';
    ctx.lineWidth = 0.5;
    ctx.fillStyle = '#94a3b8';
    ctx.beginPath();
    ctx.moveTo(startX, startY + graphHeight);
    ctx.lineTo(startX + graphWidth, startY + graphHeight);
    ctx.stroke();

    for (let i = 0; i <= 3; i++) {
      const labelX = startX + (graphWidth * i) / 3;
      ctx.fillText(`${i}.0s`, labelX - 10, startY + graphHeight + 15);
    }

  }, [gradCam]);

  return (
    <div className="xai-plot-card">
      <div className="xai-header">
        <h4>Mel Spectrogram Grad-CAM Activation</h4>
        <span className="xai-desc">Visualizes time-frequency hotspots that triggered model classification</span>
      </div>
      
      <div className="canvas-wrapper">
        <canvas 
          ref={canvasRef} 
          width={600} 
          height={160} 
          className="xai-canvas"
        />
      </div>
      
      <div className="xai-legend">
        <span className="legend-item"><span className="legend-gradient"></span> Low Importance → High Importance</span>
      </div>
    </div>
  );
}
