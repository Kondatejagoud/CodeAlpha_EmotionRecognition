import numpy as np
import librosa
from typing import Dict, Any, Tuple

class AudioQualityAnalyzer:
    """
    Analyzes audio properties before running inference to prevent garbage predictions.
    Determines if audio quality is GOOD, WARNING, or UNSUITABLE.
    """
    def __init__(
        self,
        min_duration: float = 0.5,
        warn_duration: float = 1.0,
        unsuitable_sr: int = 8000,
        warn_sr: int = 16000,
        min_rms: float = 0.002,
        warn_rms: float = 0.008,
        unsuitable_clipping: float = 0.20,
        warn_clipping: float = 0.05,
        unsuitable_silence_ratio: float = 0.85,
        warn_silence_ratio: float = 0.50,
        unsuitable_snr: float = 3.0,
        warn_snr: float = 12.0
    ):
        self.min_duration = min_duration
        self.warn_duration = warn_duration
        self.unsuitable_sr = unsuitable_sr
        self.warn_sr = warn_sr
        self.min_rms = min_rms
        self.warn_rms = warn_rms
        self.unsuitable_clipping = unsuitable_clipping
        self.warn_clipping = warn_clipping
        self.unsuitable_silence_ratio = unsuitable_silence_ratio
        self.warn_silence_ratio = warn_silence_ratio
        self.unsuitable_snr = unsuitable_snr
        self.warn_snr = warn_snr

    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        Loads the audio file using its original sample rate (sr=None) to analyze
        native characteristics.
        """
        try:
            y, sr = librosa.load(file_path, sr=None, mono=True)
        except Exception as e:
            return {
                "status": "UNSUITABLE",
                "error": f"Failed to load audio file: {str(e)}",
                "duration": 0.0,
                "sample_rate": 0,
                "rms": 0.0,
                "clipping_ratio": 0.0,
                "silence_ratio": 1.0,
                "snr_est": 0.0
            }

        duration = librosa.get_duration(y=y, sr=sr)
        
        # Calculate RMS energy of audio frames
        frame_length = min(2048, len(y))
        hop_length = min(512, len(y) // 4)
        if frame_length < 16: # Edge case: extremely short files
            return {
                "status": "UNSUITABLE",
                "error": "Audio clip too short to extract frames.",
                "duration": duration,
                "sample_rate": sr,
                "rms": 0.0,
                "clipping_ratio": 0.0,
                "silence_ratio": 1.0,
                "snr_est": 0.0
            }

        rms_frames = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        mean_rms = float(np.mean(rms_frames))
        max_rms = float(np.max(rms_frames))

        # Clipping ratio (samples very close to full dynamic scale)
        clipping_samples = np.sum(np.abs(y) >= 0.99)
        clipping_ratio = float(clipping_samples / len(y))

        # Silence ratio (frames with RMS energy below standard silent threshold)
        # Using a default db threshold relative to peak energy
        ref_db = 20 * np.log10(max(max_rms, 1e-6))
        db_frames = 20 * np.log10(rms_frames + 1e-6)
        silent_frames = np.sum(ref_db - db_frames > 25.0) # frames 25dB quieter than peak
        silence_ratio = float(silent_frames / len(rms_frames))

        # SNR Estimation
        # Sort RMS frames to find loud parts vs quiet background noise floor
        sorted_rms = np.sort(rms_frames)
        n_frames = len(sorted_rms)
        
        # Check if the signal is steady (e.g. pure tone generator in testing)
        p10 = np.percentile(sorted_rms, 10)
        p90 = np.percentile(sorted_rms, 90)
        rms_diff = float(p90 - p10)
        
        if rms_diff < 0.01 and mean_rms > 0.01:
            snr_est = 50.0
        else:
            # Estimate signal power from upper 30% loudest frames
            signal_rms = np.mean(sorted_rms[int(n_frames * 0.7):]) if n_frames >= 3 else mean_rms
            # Estimate noise power from lowest 15% quietest frames
            noise_rms = np.mean(sorted_rms[:max(1, int(n_frames * 0.15))]) if n_frames >= 3 else 1e-5
            
            # SNR in dB
            snr_est = float(20 * np.log10(max(signal_rms, 1e-6) / (max(noise_rms, 1e-6) + 1e-8)))
            if snr_est < 0:
                snr_est = 0.0

        # Run Threshold Evaluation
        status, reasons = self.evaluate_status(
            duration, sr, mean_rms, max_rms, clipping_ratio, silence_ratio, snr_est
        )

        return {
            "status": status,
            "reasons": reasons,
            "duration": round(duration, 2),
            "sample_rate": sr,
            "rms": round(mean_rms, 4),
            "clipping_ratio": round(clipping_ratio, 4),
            "silence_ratio": round(silence_ratio, 2),
            "snr_est": round(snr_est, 1)
        }

    def evaluate_status(
        self,
        duration: float,
        sr: int,
        mean_rms: float,
        max_rms: float,
        clipping_ratio: float,
        silence_ratio: float,
        snr_est: float
    ) -> Tuple[str, list]:
        """
        Applies rules to determine status.
        """
        reasons = []
        is_unsuitable = False
        is_warning = False

        # 1. Duration check
        if duration < self.min_duration:
            is_unsuitable = True
            reasons.append(f"Audio duration is too short ({duration:.2f}s, min is {self.min_duration}s).")
        elif duration < self.warn_duration:
            is_warning = True
            reasons.append(f"Short audio clip ({duration:.2f}s), predictions may be less stable.")

        # 2. Sample rate check
        if sr < self.unsuitable_sr:
            is_unsuitable = True
            reasons.append(f"Sample rate is too low ({sr}Hz, minimum supported is {self.unsuitable_sr}Hz).")
        elif sr < self.warn_sr:
            is_warning = True
            reasons.append(f"Sub-optimal sample rate ({sr}Hz), resampling will be applied.")

        # 3. RMS check (volume/silence)
        if max_rms < self.min_rms:
            is_unsuitable = True
            reasons.append("Audio is virtually silent (RMS level below threshold).")
        elif mean_rms < self.warn_rms:
            is_warning = True
            reasons.append("Low average recording volume. Try speaking closer to the microphone.")

        # 4. Clipping check
        if clipping_ratio > self.unsuitable_clipping:
            is_unsuitable = True
            reasons.append(f"Severe audio clipping ({clipping_ratio*100:.1f}% of samples). The signal is heavily distorted.")
        elif clipping_ratio > self.warn_clipping:
            is_warning = True
            reasons.append(f"Moderate audio clipping ({clipping_ratio*100:.1f}%). Audio levels may be too high.")

        # 5. Silence Ratio check
        if silence_ratio > self.unsuitable_silence_ratio:
            is_unsuitable = True
            reasons.append(f"Too much silence ({silence_ratio*100:.0f}% of clip).")
        elif silence_ratio > self.warn_silence_ratio:
            is_warning = True
            reasons.append(f"High ratio of silence ({silence_ratio*100:.0f}%).")

        # 6. SNR check
        if snr_est < self.unsuitable_snr:
            is_unsuitable = True
            reasons.append(f"Signal-to-noise ratio is too low ({snr_est:.1f} dB). Too much background noise.")
        elif snr_est < self.warn_snr:
            is_warning = True
            reasons.append(f"Low signal-to-noise ratio ({snr_est:.1f} dB). Noise may affect prediction accuracy.")

        if is_unsuitable:
            return "UNSUITABLE", reasons
        elif is_warning:
            return "WARNING", reasons
        else:
            return "GOOD", ["Audio quality looks good!"]
