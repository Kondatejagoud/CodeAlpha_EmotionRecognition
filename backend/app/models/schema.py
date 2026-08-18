from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from backend.app.core.database import Base

class PredictionHistory(Base):
    """
    SQLAlchemy model storing ser prediction details and audio quality analysis logs.
    """
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    filename = Column(String, index=True)
    prediction = Column(String)
    probability = Column(Float)
    reliability = Column(String)
    
    # Top-3 predictions
    top1_emotion = Column(String)
    top1_prob = Column(Float)
    top2_emotion = Column(String)
    top2_prob = Column(Float)
    top3_emotion = Column(String)
    top3_prob = Column(Float)
    
    # Audio quality metrics
    duration = Column(Float)
    sample_rate = Column(Integer)
    rms = Column(Float)
    clipping_ratio = Column(Float)
    silence_ratio = Column(Float)
    snr_est = Column(Float)
    audio_quality_status = Column(String)
    
    # Model Metadata
    model_used = Column(String)
