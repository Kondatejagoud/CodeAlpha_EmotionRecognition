import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, Optional

class AudioProcessor:
    """
    Handles audio loading, preprocessing (resampling, mono conversion, silence trimming,
    duration normalization), and training-only data augmentation.
    """
    def __init__(
        self,
        target_sr: int = 22050,
        target_duration: float = 3.0,
        trim_db: float = 30.0,
        normalize_volume: bool = True
    ):
        self.target_sr = target_sr
        self.target_duration = target_duration
        self.target_samples = int(target_sr * target_duration)
        self.trim_db = trim_db
        self.normalize_volume = normalize_volume

    def load_and_preprocess(self, file_path: str, augment: bool = False, aug_config: Optional[dict] = None) -> Tuple[np.ndarray, int]:
        """
        Loads the audio file, converts to mono, resamples, trims silence,
        normalizes volume, pads/truncates, and optionally applies augmentations.
        """
        # 1. Load audio (resample to target sample rate, force mono)
        y, sr = librosa.load(file_path, sr=self.target_sr, mono=True)

        # 2. Trim silence
        y_trimmed, _ = librosa.effects.trim(y, top_db=self.trim_db)
        if len(y_trimmed) == 0:
            # If the entire file is trimmed as silence, keep original
            y_trimmed = y

        # 3. Apply augmentation if requested (ONLY for training)
        if augment and aug_config:
            y_trimmed = self.apply_augmentations(y_trimmed, sr, aug_config)

        # 4. Normalize duration (pad or truncate)
        y_normalized = self.normalize_duration(y_trimmed)

        # 5. Peak normalization
        if self.normalize_volume:
            max_val = np.max(np.abs(y_normalized))
            if max_val > 0:
                y_normalized = y_normalized / max_val * 0.95 # scale to 0.95 to avoid clipping

        return y_normalized, self.target_sr

    def normalize_duration(self, y: np.ndarray) -> np.ndarray:
        """
        Pads audio with zeros if it is shorter than target_samples,
        or truncates if it is longer.
        """
        if len(y) < self.target_samples:
            # Pad with zeros (center padding)
            pad_left = (self.target_samples - len(y)) // 2
            pad_right = self.target_samples - len(y) - pad_left
            return np.pad(y, (pad_left, pad_right), mode='constant')
        elif len(y) > self.target_samples:
            # Truncate (center cropping)
            start = (len(y) - self.target_samples) // 2
            return y[start:start + self.target_samples]
        return y

    def apply_augmentations(self, y: np.ndarray, sr: int, config: dict) -> np.ndarray:
        """
        Applies audio data augmentations: noise injection, pitch shifting,
        time stretching, and random volume scaling.
        """
        y_aug = y.copy()

        # 1. Time stretching
        if config.get("time_stretch", False):
            # random rate between min and max (e.g. 0.8 to 1.25)
            min_rate = config.get("stretch_min", 0.8)
            max_rate = config.get("stretch_max", 1.2)
            rate = np.random.uniform(min_rate, max_rate)
            # librosa time_stretch can error if audio is too short or has NaN
            try:
                y_aug = librosa.effects.time_stretch(y_aug, rate=rate)
            except Exception as e:
                # Fallback to original if stretching fails
                pass

        # 2. Pitch shifting
        if config.get("pitch_shift", False):
            min_steps = config.get("pitch_min", -2.0)
            max_steps = config.get("pitch_max", 2.0)
            steps = np.random.uniform(min_steps, max_steps)
            try:
                y_aug = librosa.effects.pitch_shift(y_aug, sr=sr, n_steps=steps)
            except Exception as e:
                pass

        # 3. Noise injection
        if config.get("noise_injection", False):
            noise_factor = config.get("noise_factor", 0.005)
            # Alternatively, add white noise scaled relative to RMS
            noise = np.random.randn(len(y_aug))
            y_aug = y_aug + noise_factor * noise

        # 4. Volume perturbation (gain adjustment)
        if config.get("volume_perturbation", False):
            min_gain = config.get("gain_min", 0.8)
            max_gain = config.get("gain_max", 1.2)
            gain = np.random.uniform(min_gain, max_gain)
            y_aug = y_aug * gain

        return y_aug
