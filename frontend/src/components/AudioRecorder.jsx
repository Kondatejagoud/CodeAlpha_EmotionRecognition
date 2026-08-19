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

  const processorRef = useRef(null);
  const leftchannel = useRef([]);
  const recordingLength = useRef(0);
  const sampleRateRef = useRef(44100);

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
    setAudioUrl(null);
    setAudioBlob(null);
    leftchannel.current = [];
    recordingLength.current = 0;
    
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      
      // Set up Web Audio API
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      audioContextRef.current = audioContext;
      sampleRateRef.current = audioContext.sampleRate;
      
      const source = audioContext.createMediaStreamSource(stream);
      
      // Analyser for visualizer
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
      analyserRef.current = analyser;

      // Script processor node (bufferSize=4096, 1 input channel, 1 output channel)
      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;
      
      processor.onaudioprocess = (e) => {
        const left = e.inputBuffer.getChannelData(0);
        leftchannel.current.push(new Float32Array(left));
        recordingLength.current += left.length;
      };
      
      source.connect(processor);
      processor.connect(audioContext.destination);

      setIsRecording(true);
      startTimer();
      drawVisualizer();
    } catch (err) {
      console.error("Error accessing microphone:", err);
      alert("Microphone access denied. Please grant permission in your browser.");
    }
  };

  const stopRecording = () => {
    if (isRecording) {
      setIsRecording(false);
      stopTimer();
      cancelAnimationFrame(animationFrameRef.current);
      
      // Disconnect nodes
      if (processorRef.current) {
        processorRef.current.disconnect();
      }
      if (audioContextRef.current) {
        audioContextRef.current.close().catch(err => console.error(err));
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      
      // Flatten buffers and encode to standard 16-bit PCM WAV
      const samples = mergeBuffers(leftchannel.current, recordingLength.current);
      const wavBlob = encodeWAV(samples, sampleRateRef.current);
      const url = URL.createObjectURL(wavBlob);
      
      setAudioBlob(wavBlob);
      setAudioUrl(url);
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

function mergeBuffers(channelBuffer, recordingLength) {
  const result = new Float32Array(recordingLength);
  let offset = 0;
  for (let i = 0; i < channelBuffer.length; i++) {
    const buffer = channelBuffer[i];
    result.set(buffer, offset);
    offset += buffer.length;
  }
  return result;
}

function encodeWAV(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  /* RIFF identifier */
  writeString(view, 0, 'RIFF');
  /* file length */
  view.setUint32(4, 36 + samples.length * 2, true);
  /* RIFF type */
  writeString(view, 8, 'WAVE');
  /* format chunk identifier */
  writeString(view, 12, 'fmt ');
  /* format chunk length */
  view.setUint32(16, 16, true);
  /* sample format (raw) */
  view.setUint16(20, 1, true);
  /* channel count (mono) */
  view.setUint16(22, 1, true);
  /* sample rate */
  view.setUint32(24, sampleRate, true);
  /* byte rate (sample rate * block align) */
  view.setUint32(28, sampleRate * 2, true);
  /* block align (channel count * bytes per sample) */
  view.setUint16(32, 2, true);
  /* bits per sample */
  view.setUint16(34, 16, true);
  /* data chunk identifier */
  writeString(view, 36, 'data');
  /* data chunk length */
  view.setUint32(40, samples.length * 2, true);

  floatTo16BitPCM(view, 44, samples);

  return new Blob([view], { type: 'audio/wav' });
}

function floatTo16BitPCM(output, offset, input) {
  for (let i = 0; i < input.length; i++, offset += 2) {
    let s = Math.max(-1, Math.min(1, input[i]));
    output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
}

function writeString(view, offset, string) {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}
