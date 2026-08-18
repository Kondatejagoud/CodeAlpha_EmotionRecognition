import numpy as np
import librosa
import matplotlib.pyplot as plt
import io
from typing import Dict, Tuple

class FeatureExtractor:
    """
    Extracts acoustic features from audio signals.
    Supports 1D flattened features (for baseline SVM) and 2D arrays (for CNN / RNN models).
    """
    def __init__(
        self,
        sr: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        n_mfcc: int = 13,
        n_chroma: int = 12
    ):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.n_chroma = n_chroma

    def extract_all_features(self, y: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extracts all requested raw feature arrays from the audio signal.
        """
        # Mel Spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels
        )
        mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # MFCC, Delta, and Delta-Delta
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length)
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)

        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_chroma=self.n_chroma)

        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=self.n_fft, hop_length=self.hop_length)

        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)

        # RMS Energy
        rms = librosa.feature.rms(y=y, frame_length=self.n_fft, hop_length=self.hop_length)

        return {
            "mel_db": mel_db,
            "mfcc": mfcc,
            "delta_mfcc": delta_mfcc,
            "delta2_mfcc": delta2_mfcc,
            "chroma": chroma,
            "zcr": zcr,
            "spectral_centroid": spectral_centroid,
            "spectral_bandwidth": spectral_bandwidth,
            "spectral_rolloff": spectral_rolloff,
            "rms": rms
        }

    def get_flattened_features(self, y: np.ndarray) -> np.ndarray:
        """
        Computes statistical summaries (mean, std, max, min) of acoustic features.
        Used for baseline classical ML models like SVM.
        """
        features_dict = self.extract_all_features(y)
        feature_vectors = []

        # Loop through each feature, calculate statistics over time (axis=1)
        for key, feat in features_dict.items():
            # For 2D features like Mel Spectrogram, MFCC, Delta MFCC, Delta-delta, Chroma
            # we summarize over time (axis=-1)
            mean = np.mean(feat, axis=-1)
            std = np.std(feat, axis=-1)
            max_val = np.max(feat, axis=-1)
            min_val = np.min(feat, axis=-1)

            feature_vectors.extend([mean, std, max_val, min_val])

        # Concatenate all statistical arrays into one flat 1D vector
        return np.concatenate([np.atleast_1d(x) for x in feature_vectors])

    def get_mel_spectrogram_2d(self, y: np.ndarray) -> np.ndarray:
        """
        Extracts Mel Spectrogram and returns it as a 2D array formatted for PyTorch CNNs:
        Shape: (1, n_mels, time_steps)
        """
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels
        )
        mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        # Add channel dimension
        return np.expand_dims(mel_db, axis=0) # Shape: (1, n_mels, time_steps)

    def get_mel_and_mfcc_2d(self, y: np.ndarray) -> np.ndarray:
        """
        Extracts Mel Spectrogram and MFCCs and stacks them.
        Shape: (2, n_mels, time_steps)
        """
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels
        )
        mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length)
        
        # Interpolate MFCCs up to match n_mels shape or vice versa for combining
        # Easier alternative: pad MFCCs with zeros to match n_mels dimensions or resize
        # Let's just resize mfcc array or pad it to n_mels.
        # But actually, we can just interpolate or pad mfccs:
        padded_mfcc = np.zeros_like(mel_db)
        padded_mfcc[:self.n_mfcc, :] = mfcc
        
        return np.stack([mel_db, padded_mfcc], axis=0) # Shape: (2, n_mels, time_steps)

    def generate_spectrogram_image(self, mel_db: np.ndarray) -> bytes:
        """
        Helper to generate an in-memory PNG bytes file of the spectrogram.
        """
        fig, ax = plt.subplots(figsize=(6, 3))
        # Display spectrogram
        img = librosa.display.specshow(
            mel_db, sr=self.sr, hop_length=self.hop_length, x_axis='time', y_axis='mel', ax=ax, cmap='viridis'
        )
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_title("Mel Spectrogram")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

    def generate_mfcc_image(self, mfcc: np.ndarray) -> bytes:
        """
        Helper to generate an in-memory PNG bytes file of the MFCCs.
        """
        fig, ax = plt.subplots(figsize=(6, 3))
        img = librosa.display.specshow(
            mfcc, sr=self.sr, hop_length=self.hop_length, x_axis='time', ax=ax, cmap='coolwarm'
        )
        fig.colorbar(img, ax=ax)
        ax.set_title("MFCC")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()
