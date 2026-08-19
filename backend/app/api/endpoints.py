import os
import shutil
import tempfile
from fastapi import APIRouter, File, UploadFile, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, Any, List

from backend.app.core.database import get_db
from backend.app.models.schema import PredictionHistory
from ml.inference.predictor import EmotionPredictor
from backend.app.core.config import settings

router = APIRouter()

# Initialize the inference engine
# Defaults to CNN-BiLSTM-Attention with Mel features
try:
    predictor = EmotionPredictor(model_name="cnn-bilstm-attention", feature_type="mel")
except Exception as e:
    print(f"Error loading emotion predictor: {e}. Inference will run with dummy configurations.")
    predictor = None

@router.get("/health")
def health_check() -> Dict[str, str]:
    """
    API Health status.
    """
    return {"status": "healthy", "service": "EmotionSense AI API"}

@router.get("/model-info")
def model_info() -> Dict[str, Any]:
    """
    Returns metadata about the active ML model loaded in the server.
    """
    if not predictor:
        return {"error": "Model predictor not loaded."}
    return predictor.metadata

@router.post("/predict")
def predict_emotion(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Upload an audio file (.wav, .mp3, etc.) to evaluate speaker emotional state.
    Runs pre-inference audio quality check.
    """
    if not predictor:
        raise HTTPException(status_code=500, detail="Inference engine is not initialized.")

    # 1. Save uploaded file to a temporary location
    suffix = os.path.splitext(file.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name

    try:
        # 2. Run prediction
        result = predictor.predict(temp_path)
        
        # 3. Log results to database (except when audio is completely UNSUITABLE or error occurs)
        quality = result["audio_quality"]
        top_preds = result["top_predictions"]
        
        # We fill values for database
        top1_emo = top_preds[0]["emotion"] if len(top_preds) > 0 else "N/A"
        top1_pr = top_preds[0]["probability"] if len(top_preds) > 0 else 0.0
        top2_emo = top_preds[1]["emotion"] if len(top_preds) > 1 else "N/A"
        top2_pr = top_preds[1]["probability"] if len(top_preds) > 1 else 0.0
        top3_emo = top_preds[2]["emotion"] if len(top_preds) > 2 else "N/A"
        top3_pr = top_preds[2]["probability"] if len(top_preds) > 2 else 0.0

        history_record = PredictionHistory(
            filename=file.filename,
            prediction=result["prediction"],
            probability=result["probability"],
            reliability=result["reliability"],
            top1_emotion=top1_emo,
            top1_prob=top1_pr,
            top2_emotion=top2_emo,
            top2_prob=top2_pr,
            top3_emotion=top3_emo,
            top3_prob=top3_pr,
            duration=quality["duration"],
            sample_rate=quality["sample_rate"],
            rms=quality["rms"],
            clipping_ratio=quality["clipping_ratio"],
            silence_ratio=quality["silence_ratio"],
            snr_est=quality["snr_est"],
            audio_quality_status=quality["status"],
            model_used=f"{predictor.model_name} ({predictor.feature_type})"
        )
        db.add(history_record)
        db.commit()
        db.refresh(history_record)
        
        # Include record ID in result response
        result["prediction_id"] = history_record.id
        
        return result
    except Exception as e:
        print(f"Error predicting emotion for upload: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@router.post("/predict/file")
def predict_file_alias(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Alias route for /predict
    """
    return predict_emotion(file, db)

@router.post("/predict/audio")
def predict_audio_alias(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Alias route for /predict
    """
    return predict_emotion(file, db)

@router.get("/history")
def get_prediction_history(
    limit: int = 10,
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """
    Fetches recent prediction records.
    """
    records = db.query(PredictionHistory).order_by(PredictionHistory.timestamp.desc()).limit(limit).all()
    history = []
    
    for r in records:
        history.append({
            "id": r.id,
            "timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "filename": r.filename,
            "prediction": r.prediction,
            "probability": r.probability,
            "reliability": r.reliability,
            "top_predictions": [
                {"emotion": r.top1_emotion, "probability": r.top1_prob},
                {"emotion": r.top2_emotion, "probability": r.top2_prob},
                {"emotion": r.top3_emotion, "probability": r.top3_prob}
            ],
            "audio_quality": {
                "status": r.audio_quality_status,
                "duration": r.duration,
                "sample_rate": r.sample_rate,
                "rms": r.rms,
                "clipping_ratio": r.clipping_ratio,
                "silence_ratio": r.silence_ratio,
                "snr_est": r.snr_est
            },
            "model_used": r.model_used
        })
    return history

@router.delete("/history")
def clear_prediction_history(db: Session = Depends(get_db)) -> Dict[str, str]:
    """
    Deletes all records from the prediction history database.
    """
    try:
        db.query(PredictionHistory).delete()
        db.commit()
        return {"status": "success", "message": "Prediction history cleared successfully."}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to clear history: {str(e)}")
