import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import joblib
import librosa
from typing import Dict, Any, Tuple

from ml.preprocessing.audio_processor import AudioProcessor
from ml.features.feature_extractor import FeatureExtractor
from backend.app.services.audio_analyzer import AudioQualityAnalyzer
from backend.app.services.explainability import extract_explanations
from ml.models.cnn_lstm_att import SpeechCNNLSTMAttention
from ml.models.cnn_lstm import SpeechCNNLSTM
from ml.models.cnn import SpeechCNN

class EmotionPredictor:
    """
    Main inference interface. Integrates AudioQualityAnalyzer, AudioProcessor,
    FeatureExtractor, trained Neural Network models, and explainability features.
    """
    def __init__(
        self,
        model_name: str = "cnn-bilstm-attention",
        feature_type: str = "mel",
        model_dir: str = "ml/models/saved"
    ):
        self.model_name = model_name
        self.feature_type = feature_type
        self.model_dir = model_dir
        
        self.classes = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
        
        self.processor = AudioProcessor(target_sr=22050, target_duration=3.0)
        self.extractor = FeatureExtractor(sr=22050)
        self.analyzer = AudioQualityAnalyzer()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.metadata = {}
        
        # Load training-only Mel normalization stats
        self.norm_mean = None
        self.norm_std = None
        norm_path = "models/mel_normalization.json"
        if not os.path.exists(norm_path):
            norm_path = os.path.join(self.model_dir, "mel_normalization.json")
        if os.path.exists(norm_path):
            try:
                with open(norm_path, 'r') as f:
                    norm_data = json.load(f)
                self.norm_mean = np.array(norm_data["mean"], dtype=np.float32)
                self.norm_std = np.array(norm_data["std"], dtype=np.float32)
                print(f"EmotionPredictor loaded Mel normalization parameters from {norm_path}")
            except Exception as e:
                print(f"Warning: Failed to load Mel normalization: {e}")

        # Load calibration parameters
        self.temperature = 1.0
        self.confidence_threshold = 0.40
        self.margin_threshold = 0.15
        
        cal_path = "models/calibration_config.json"
        if not os.path.exists(cal_path):
            cal_path = os.path.join(self.model_dir, "calibration_config.json")
        if os.path.exists(cal_path):
            try:
                with open(cal_path, 'r') as f:
                    cal_data = json.load(f)
                self.temperature = cal_data.get("temperature", 1.0)
                self.confidence_threshold = cal_data.get("confidence_threshold", 0.40)
                self.margin_threshold = cal_data.get("margin_threshold", 0.15)
                print(f"EmotionPredictor loaded calibration config from {cal_path}")
            except Exception as e:
                print(f"Warning: Failed to load calibration config: {e}")

        # Load model and metadata
        self._load_predictor()

    def _load_predictor(self):
        """
        Loads model state dict and associated version metadata.
        """
        # Check with feature suffix first, then without as fallback
        model_path = os.path.join(self.model_dir, f"{self.model_name}_{self.feature_type}_best.pt")
        if not os.path.exists(model_path):
            model_path = os.path.join(self.model_dir, f"{self.model_name}_best.pt")
            
        meta_path = os.path.join(self.model_dir, f"{self.model_name}_{self.feature_type}_metadata.json")
        if not os.path.exists(meta_path):
            meta_path = os.path.join(self.model_dir, f"{self.model_name}_metadata.json")
        
        # Load metadata if exists
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "model_name": self.model_name,
                "architecture": "cnn-bilstm-attention",
                "dataset": "RAVDESS Speech",
                "feature_representation": self.feature_type,
                "training_date": "N/A",
                "metrics": {"test_accuracy": 0.0}
            }

        # Initialize appropriate model graph
        # Try to find model file, otherwise we will initialize a blank model for evaluation/testing
        if self.model_name == "cnn-bilstm-attention":
            self.model = SpeechCNNLSTMAttention(num_classes=8, n_mels=128)
        elif self.model_name == "cnn-bilstm":
            self.model = SpeechCNNLSTM(num_classes=8, n_mels=128)
        elif self.model_name == "cnn":
            self.model = SpeechCNN(num_classes=8, n_mels=128)
        else:
            raise ValueError(f"Unknown architecture model: {self.model_name}")

        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Loaded trained model weights from {model_path}")
            except Exception as e:
                print(f"Warning: Failed to load trained weights: {e}. Model is uninitialized.")
        else:
            print(f"Warning: No trained weights found at {model_path}. Running with random initialization.")
            
        self.model.to(self.device)
        self.model.eval()

    def predict(self, file_path: str) -> Dict[str, Any]:
        """
        Runs the complete inference pipeline on an audio file, using multi-window
        segmentation and temperature probability calibration.
        """
        # 1. Run Audio Quality Analyzer
        quality = self.analyzer.analyze_file(file_path)
        
        if quality["status"] == "UNSUITABLE":
            return {
                "prediction": "UNCERTAIN",
                "probability": 0.0,
                "reliability": "UNCERTAIN",
                "top_predictions": [],
                "audio_quality": quality,
                "explainability": {
                    "grad_cam": [],
                    "attention": []
                },
                "model_metadata": self.metadata,
                "final_prediction": "UNCERTAIN",
                "aggregated_probabilities": {},
                "window_predictions": []
            }

        # 2. Load raw audio (resampled to target rate)
        y_raw, sr = librosa.load(file_path, sr=self.processor.target_sr, mono=True)
        # Trim silence
        y_trimmed, _ = librosa.effects.trim(y_raw, top_db=self.processor.trim_db)
        if len(y_trimmed) == 0:
            y_trimmed = y_raw

        w_samples = self.processor.target_samples
        hop_samples = int(w_samples * 0.5) # 50% overlap

        # Slice into windows of exactly w_samples
        y_windows = []
        if len(y_trimmed) <= w_samples:
            pad_left = (w_samples - len(y_trimmed)) // 2
            pad_right = w_samples - len(y_trimmed) - pad_left
            y_windows.append(np.pad(y_trimmed, (pad_left, pad_right), mode='constant'))
        else:
            for start in range(0, len(y_trimmed) - w_samples + 1, hop_samples):
                y_windows.append(y_trimmed[start:start + w_samples])
            if (len(y_trimmed) - w_samples) % hop_samples != 0:
                y_windows.append(y_trimmed[-w_samples:])

        # Make predictions for each window
        all_probs = []
        window_predictions = []
        max_conf = -1.0
        best_input_tensor = None
        best_top1_idx = 0

        # Mean and standard deviation scaling vectors
        mean_reshaped = self.norm_mean.reshape(1, -1, 1) if self.norm_mean is not None else None
        std_reshaped = self.norm_std.reshape(1, -1, 1) if self.norm_std is not None else None

        for w_idx, window in enumerate(y_windows):
            # Volume peak normalize the window segment
            max_val = np.max(np.abs(window))
            if max_val > 0:
                window = window / max_val * 0.95

            # Extract features
            if self.feature_type == "mel":
                feat = self.extractor.get_mel_spectrogram_2d(window)
                if mean_reshaped is not None and std_reshaped is not None:
                    feat = (feat - mean_reshaped) / std_reshaped
            elif self.feature_type == "mel_mfcc":
                feat = self.extractor.get_mel_and_mfcc_2d(window)
                if self.norm_mean is not None and self.norm_std is not None:
                    feat[0] = (feat[0] - self.norm_mean.reshape(-1, 1)) / self.norm_std.reshape(-1, 1)
            else:
                feat = self.extractor.get_mel_spectrogram_2d(window)

            input_tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(input_tensor)
                if isinstance(outputs, tuple):
                    logits, _ = outputs
                else:
                    logits = outputs

                # Calibrate probabilities using temperature scaling
                calibrated_logits = logits / self.temperature
                probs = F.softmax(calibrated_logits, dim=1).squeeze(0).cpu().numpy()

            top_idx = np.argmax(probs)
            conf = float(probs[top_idx])

            if conf > max_conf:
                max_conf = conf
                best_input_tensor = input_tensor
                best_top1_idx = top_idx

            all_probs.append(probs)
            window_predictions.append({
                "window_index": w_idx,
                "prediction": self.classes[top_idx].capitalize(),
                "confidence": conf,
                "probabilities": {self.classes[i].capitalize(): float(probs[i]) for i in range(len(self.classes))}
            })

        # Aggregate probabilities across windows using RMS energy weights
        rms_weights = []
        for window in y_windows:
            rms = np.sqrt(np.mean(window ** 2))
            rms_weights.append(max(rms, 1e-4))

        total_weight = sum(rms_weights)
        aggregated_probs = np.zeros_like(all_probs[0])
        for w_idx, probs in enumerate(all_probs):
            aggregated_probs += (rms_weights[w_idx] / total_weight) * probs

        # Sort aggregated probabilities
        sorted_indices = np.argsort(aggregated_probs)[::-1]
        top_predictions = [
            {"emotion": self.classes[idx].capitalize(), "probability": float(aggregated_probs[idx])}
            for idx in sorted_indices
        ]

        top1_idx = sorted_indices[0]
        top1_prob = float(aggregated_probs[top1_idx])
        top2_prob = float(aggregated_probs[sorted_indices[1]])
        final_prediction = self.classes[top1_idx].capitalize()

        # Run explainability on the dominant window (with highest confidence)
        grad_cam_heatmap, attn_weights = extract_explanations(self.model, best_input_tensor, best_top1_idx)
        grad_cam_list = grad_cam_heatmap.tolist() if grad_cam_heatmap is not None else []
        attn_list = attn_weights.tolist() if attn_weights is not None else []

        # Determine uncertainty and reliability using calibrated validation thresholds
        is_uncertain = (top1_prob < self.confidence_threshold) or ((top1_prob - top2_prob) < self.margin_threshold)

        if is_uncertain:
            prediction = "UNCERTAIN"
            reliability = "UNCERTAIN"
        else:
            prediction = final_prediction
            if top1_prob >= 0.70:
                reliability = "HIGH"
            else:
                reliability = "MODERATE"

        return {
            "prediction": prediction,
            "probability": round(top1_prob, 3),
            "reliability": reliability,
            "top_predictions": top_predictions[:3],
            "audio_quality": quality,
            "explainability": {
                "grad_cam": grad_cam_list,
                "attention": attn_list
            },
            "model_metadata": self.metadata,
            "final_prediction": final_prediction,
            "aggregated_probabilities": {self.classes[i].capitalize(): float(aggregated_probs[i]) for i in range(len(self.classes))},
            "window_predictions": window_predictions
        }
