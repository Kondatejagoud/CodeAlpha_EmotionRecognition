import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import joblib
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
        Analyzes quality, extracts features, runs inference, validates uncertainty,
        and computes Grad-CAM/attention values.
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
                "model_metadata": self.metadata
            }

        # 2. Load & preprocess audio
        y_wave, sr = self.processor.load_and_preprocess(file_path, augment=False)
        
        # 3. Extract 2D features
        if self.feature_type == "mel":
            feat = self.extractor.get_mel_spectrogram_2d(y_wave)
        elif self.feature_type == "mel_mfcc":
            feat = self.extractor.get_mel_and_mfcc_2d(y_wave)
        else:
            # Fallback to standard mel
            feat = self.extractor.get_mel_spectrogram_2d(y_wave)
            
        # Convert to tensor and send to device
        input_tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(self.device) # shape: (1, channels, n_mels, time_steps)
        
        # 4. Model Forward Pass
        outputs = self.model(input_tensor)
        if isinstance(outputs, tuple):
            logits, _ = outputs
        else:
            logits = outputs
            
        probs = F.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
        
        # Sort predictions
        sorted_indices = np.argsort(probs)[::-1]
        top_predictions = [
            {"emotion": self.classes[idx].capitalize(), "probability": float(probs[idx])}
            for idx in sorted_indices
        ]
        
        top1_idx = sorted_indices[0]
        top1_prob = float(probs[top1_idx])
        top2_prob = float(probs[sorted_indices[1]])
        raw_prediction = self.classes[top1_idx].capitalize()

        # 5. Local Explainability extraction
        # Grad-CAM and Attention
        grad_cam_heatmap, attn_weights = extract_explanations(self.model, input_tensor, top1_idx)
        
        # Convert arrays to serializable forms
        grad_cam_list = grad_cam_heatmap.tolist() if grad_cam_heatmap is not None else []
        attn_list = attn_weights.tolist() if attn_weights is not None else []

        # 6. Uncertainty Check & Reliability determination
        # Criteria:
        # - Probability < 0.40 OR
        # - Margin between 1st and 2nd top predictions is < 0.15
        is_uncertain = (top1_prob < 0.40) or ((top1_prob - top2_prob) < 0.15)
        
        if is_uncertain:
            prediction = "UNCERTAIN"
            reliability = "UNCERTAIN"
        else:
            prediction = raw_prediction
            if top1_prob >= 0.70:
                reliability = "HIGH"
            else:
                reliability = "MODERATE"

        return {
            "prediction": prediction,
            "probability": round(top1_prob, 3),
            "reliability": reliability,
            "top_predictions": top_predictions[:3], # Top-3 predictions
            "audio_quality": quality,
            "explainability": {
                "grad_cam": grad_cam_list,
                "attention": attn_list
            },
            "model_metadata": self.metadata
        }
