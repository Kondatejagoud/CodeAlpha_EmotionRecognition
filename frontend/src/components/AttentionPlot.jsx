import React, { useRef, useEffect } from 'react';

export default function AttentionPlot({ attention }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current || !attention || attention.length === 0) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // Clear canvas
    ctx.fillStyle = '#0f172a'; // dark theme slate-900
    ctx.fillRect(0, 0, width, height);

    const length = attention.length;
    const step = width / length;

    // Normalize attention weights for visual contrast
    const maxVal = Math.max(...attention);
    const normalizedAttn = maxVal > 0 ? attention.map(w => w / maxVal) : attention;

    // 1. Draw Timeline Grid
    ctx.strokeStyle = '#334155'; // grid lines
    ctx.lineWidth = 0.5;
    ctx.font = '10px sans-serif';
    ctx.fillStyle = '#64748b';
    
    // Draw 3 grids representing 1s, 2s, 3s
    for (let i = 1; i <= 3; i++) {
      const x = (width * i) / 3;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
      ctx.fillText(`${i}.0s`, x - 25, height - 5);
    }

    // 2. Draw Attention Highlight Shading (Background glow)
    for (let i = 0; i < length; i++) {
      const w = normalizedAttn[i];
      const x = i * step;
      
      // Shade regions where attention is high
      if (w > 0.15) {
        ctx.fillStyle = `rgba(16, 185, 129, ${w * 0.35})`; // emerald highlight with alpha
        ctx.fillRect(x, 0, step + 1, height);
      }
    }

    // 3. Draw Simulated Symmetrical Voice Waveform Envelope
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    
    // Using a pseudo-random wave envelope modulated by attention/sine wave
    // to give it a realistic speech look
    const wavePoints = [];
    for (let i = 0; i < length; i++) {
      const attn = normalizedAttn[i];
      const x = i * step;
      
      // Modulating a base envelope shaped like word vocalizations
      const baseEnvelope = 0.15 + 0.5 * Math.sin((i / length) * Math.PI) * (0.8 + 0.2 * Math.sin(i * 0.4));
      // Boost envelope slightly where attention is high to simulate speech activity
      const amp = baseEnvelope * (0.3 + 0.7 * attn) * (height / 2.3);
      
      wavePoints.push({ x, amp });
    }

    // Draw Top Half of Symmetrical Envelope
    ctx.strokeStyle = '#475569'; // wave gray-600
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    for (let i = 0; i < wavePoints.length; i++) {
      ctx.lineTo(wavePoints[i].x, height / 2 - wavePoints[i].amp);
    }
    ctx.lineTo(width, height / 2);
    ctx.stroke();

    // Draw Bottom Half of Symmetrical Envelope
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    for (let i = 0; i < wavePoints.length; i++) {
      ctx.lineTo(wavePoints[i].x, height / 2 + wavePoints[i].amp);
    }
    ctx.lineTo(width, height / 2);
    ctx.stroke();

    // 4. Draw Attention Weights Line (Vibrant overlay)
    ctx.beginPath();
    ctx.strokeStyle = '#10b981'; // Emerald-500
    ctx.lineWidth = 2.5;
    
    for (let i = 0; i < length; i++) {
      const attn = normalizedAttn[i];
      const x = i * step;
      // Draw line peaking at the top for higher attention
      const y = height - (attn * (height - 25)) - 15;
      
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();

    // Draw glowing circles at highest attention points
    for (let i = 0; i < length; i++) {
      const attn = normalizedAttn[i];
      if (attn > 0.85) {
        const x = i * step;
        const y = height - (attn * (height - 25)) - 15;
        
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, 2 * Math.PI);
        ctx.fillStyle = '#f43f5e'; // Red indicator
        ctx.fill();
        
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 9px sans-serif';
        ctx.fillText("Peak Focus", x - 25, y - 8);
      }
    }

  }, [attention]);

  return (
    <div className="xai-plot-card">
      <div className="xai-header">
        <h4>Temporal Attention Mapping</h4>
        <span className="xai-desc">Highlights which syllables/timestamps driven model's prediction</span>
      </div>
      
      <div className="canvas-wrapper">
        <canvas 
          ref={canvasRef} 
          width={600} 
          height={140} 
          className="xai-canvas"
        />
      </div>
      <div className="xai-legend">
        <span className="legend-item"><span className="legend-dot wave-dot"></span> Speech Audio Envelope</span>
        <span className="legend-item"><span className="legend-dot attn-dot"></span> Attention Weight ($\alpha_t$)</span>
        <span className="legend-item"><span className="legend-dot focus-dot"></span> Focus Regions</span>
      </div>
    </div>
  );
}
