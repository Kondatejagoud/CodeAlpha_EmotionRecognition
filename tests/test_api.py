import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import io

# We patch predictor loading inside endpoints *before* importing endpoints/main
with patch("ml.inference.predictor.EmotionPredictor") as mock_pred_class:
    # Setup mock instance
    mock_instance = MagicMock()
    mock_instance.model_name = "cnn-bilstm-attention"
    mock_instance.feature_type = "mel"
    mock_instance.metadata = {"model_name": "cnn-bilstm-attention", "test_accuracy": 0.85}
    mock_instance.predict.return_value = {
        "prediction": "Happy",
        "probability": 0.820,
        "reliability": "HIGH",
        "top_predictions": [
            {"emotion": "Happy", "probability": 0.820},
            {"emotion": "Calm", "probability": 0.120},
            {"emotion": "Neutral", "probability": 0.060}
        ],
        "audio_quality": {
            "status": "GOOD",
            "duration": 3.0,
            "sample_rate": 22050,
            "rms": 0.05,
            "clipping_ratio": 0.0,
            "silence_ratio": 0.1,
            "snr_est": 32.0
        },
        "explainability": {
            "grad_cam": [[0.1, 0.2], [0.3, 0.4]],
            "attention": [0.05, 0.95]
        }
    }
    mock_pred_class.return_value = mock_instance

    from backend.app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "EmotionSense AI API"}

def test_model_info():
    response = client.get("/api/model-info")
    assert response.status_code == 200
    assert response.json()["model_name"] == "cnn-bilstm-attention"

def test_predict_endpoint():
    # Generate mock wave file in memory
    audio_data = io.BytesIO(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x22\x56\x00\x00\x44\xac\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00")
    files = {"file": ("test.wav", audio_data, "audio/wav")}
    
    response = client.post("/api/predict", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "Happy"
    assert data["reliability"] == "HIGH"
    assert "audio_quality" in data
    assert "prediction_id" in data

def test_history_endpoint():
    response = client.get("/api/history?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    # Check that the prediction we just posted is in history
    if len(data) > 0:
        assert data[0]["prediction"] == "Happy"
        assert "audio_quality" in data[0]
