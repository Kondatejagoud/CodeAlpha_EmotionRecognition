import React, { useState, useRef, useEffect } from 'react';

export default function AudioRecorder({ onRecordingComplete, isAnalyzing }) {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [audioUrl, setAudioUrl] = useState(null);
  const [audioBlob, setAudioBlob] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const timerRef = useRef(null);
  const audioPlayerRef = useRef(null);
  const canvasRef = useRef(null);
  const animationFrameRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const streamRef = useRef(null);

  useEffect(() => {
    return () => {
      stopTimer();
      cancelAnimationFrame(animationFrameRef.current);
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
  }, []);

  const startTimer = () => {
    setRecordingTime(0);
    timerRef.current = setInterval(() => {
      setRecordingTime((prev) => prev + 1);
    }, 1000);
  };

  const stopTimer = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const startRecording = async () => {
    audioChunksRef.current = [];
    setAudioUrl(null);
    setAudioBlob(null);
    
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        const url = URL.createObjectURL(blob);
        setAudioBlob(blob);
        setAudioUrl(url);
        
        // Clean up mic stream tracks
        stream.getTracks().forEach(track => track.stop());
      };

      // Set up Web Audio API visualizer
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      audioContextRef.current = audioContext;
      
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
      analyserRef.current = analyser;

      mediaRecorder.start();
      setIsRecording(true);
      startTimer();
      drawVisualizer();
    } catch (err) {
      console.error("Error accessing microphone:", err);
      alert("Microphone access denied. Please grant permission in your browser.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      stopTimer();
      cancelAnimationFrame(animationFrameRef.current);
    }
  };

  const drawVisualizer = () => {
    if (!canvasRef.current || !analyserRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const analyser = analyserRef.current;
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const draw = () => {
      animationFrameRef.current = requestAnimationFrame(draw);
      analyser.getByteFrequencyData(dataArray);

      ctx.fillStyle = '#0f172a'; // dark theme bg
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      const barWidth = (canvas.width / bufferLength) * 2.5;
      let barHeight;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        barHeight = dataArray[i] / 1.5;

        // Custom gradient style: indigo to emerald
        const grad = ctx.createLinearGradient(0, canvas.height, 0, 0);
        grad.addColorStop(0, '#6366f1'); // Indigo
        grad.addColorStop(1, '#10b981'); // Emerald
        
        ctx.fillStyle = grad;
        ctx.fillRect(x, canvas.height - barHeight, barWidth - 1, barHeight);

        x += barWidth;
      }
    };
    draw();
  };

  const togglePlayback = () => {
    const audio = audioPlayerRef.current;
    if (!audio) return;

    if (isPlaying) {
      audio.pause();
      setIsPlaying(false);
    } else {
      audio.play();
      setIsPlaying(true);
    }
  };

  const handleAnalyze = () => {
    if (audioBlob) {
      onRecordingComplete(audioBlob);
    }
  };

  return (
    <div className="audio-card">
      <h3>Microphone Recording</h3>
      
      <div className="recorder-container">
        <canvas 
          ref={canvasRef} 
          width={300} 
          height={80} 
          className="visualizer-canvas"
        />
        
        <div className="timer-text">
          {isRecording ? formatTime(recordingTime) : audioUrl ? "Recording captured" : "00:00"}
        </div>

        <div className="recorder-buttons">
          {!isRecording ? (
            <button 
              className="btn btn-primary btn-record"
              onClick={startRecording}
              disabled={isAnalyzing}
            >
              <span className="dot record-dot"></span> Record
            </button>
          ) : (
            <button 
              className="btn btn-danger btn-stop"
              onClick={stopRecording}
            >
              <span className="square stop-square"></span> Stop
            </button>
          )}

          {audioUrl && !isRecording && (
            <>
              <button 
                className="btn btn-secondary"
                onClick={togglePlayback}
              >
                {isPlaying ? "Pause" : "Play"}
              </button>
              
              <button 
                className="btn btn-success"
                onClick={handleAnalyze}
                disabled={isAnalyzing}
              >
                {isAnalyzing ? "Analyzing..." : "Analyze"}
              </button>
            </>
          )}
        </div>

        {audioUrl && (
          <audio 
            ref={audioPlayerRef} 
            src={audioUrl} 
            onEnded={() => setIsPlaying(false)}
            style={{ display: 'none' }}
          />
        )}
      </div>
    </div>
  );
}
