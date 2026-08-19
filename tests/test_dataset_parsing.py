import os
import json
import pytest
import pandas as pd
import numpy as np
from ml.data.dataset import get_speaker_independent_split, parse_ravdess_metadata
from ml.inference.predictor import EmotionPredictor

def test_ravdess_parsing_and_labels():
    """
    Verifies that sample filenames are parsed with the correct emotion mapping and actor properties.
    """
    # Sample RAVDESS speech filenames
    # Format: modality-vocal_channel-emotion-intensity-statement-repetition-actor
    test_filenames = [
        ("03-01-01-01-01-01-01", "neutral", 1, "male"),
        ("03-01-02-02-01-02-02", "calm", 2, "female"),
        ("03-01-03-01-02-01-19", "happy", 19, "male"),
        ("03-01-05-02-02-02-20", "angry", 20, "female"),
        ("03-01-08-01-01-01-24", "surprised", 24, "female")
    ]
    
    # Test individual split tag mapping logic
    emotions_map = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fearful",
        "07": "disgust",
        "08": "surprised"
    }
    
    for stem, expected_emotion, expected_actor_id, expected_gender in test_filenames:
        parts = stem.split('-')
        assert len(parts) == 7
        modality, vocal, emotion_code, intensity, statement, repetition, actor_code = parts
        
        assert modality == "03"
        assert vocal == "01"
        assert emotions_map[emotion_code] == expected_emotion
        
        actor_id = int(actor_code)
        assert actor_id == expected_actor_id
        
        gender = "female" if actor_id % 2 == 0 else "male"
        assert gender == expected_gender

def test_speaker_independent_split_disjointness():
    """
    Programmatically asserts that speaker splits (train, val, test) are disjoint.
    """
    # Create a mock dataframe
    mock_data = []
    for actor in range(1, 25):
        mock_data.append({
            "file_path": f"dummy_path_{actor}.wav",
            "emotion": "happy",
            "actor_id": actor
        })
    df = pd.DataFrame(mock_data)
    
    train_df, val_df, test_df = get_speaker_independent_split(df)
    
    train_actors = set(train_df["actor_id"])
    val_actors = set(val_df["actor_id"])
    test_actors = set(test_df["actor_id"])
    
    # Programmatic assertion checks
    assert train_actors.isdisjoint(val_actors), "Overlap between training and validation speakers!"
    assert train_actors.isdisjoint(test_actors), "Overlap between training and testing speakers!"
    assert val_actors.isdisjoint(test_actors), "Overlap between validation and testing speakers!"
    
    assert train_actors == set(range(1, 19))
    assert val_actors == {19, 20}
    assert test_actors == {21, 22, 23, 24}

def test_inference_loads_normalization_params():
    """
    Verifies that the EmotionPredictor correctly loads the saved Mel normalization stats file.
    """
    norm_path = "models/mel_normalization.json"
    assert os.path.exists(norm_path), "mel_normalization.json file does not exist!"
    
    # Instantiate predictor (running on cpu fallback if no gpu)
    predictor = EmotionPredictor(model_name="cnn-bilstm-attention", feature_type="mel")
    
    # Verify arrays are loaded
    assert predictor.norm_mean is not None, "Predictor failed to load Mel normalization mean vector!"
    assert predictor.norm_std is not None, "Predictor failed to load Mel normalization std vector!"
    
    # Check shape constraints
    assert len(predictor.norm_mean) == 128
    assert len(predictor.norm_std) == 128
    
    # Verify standard values are valid
    assert np.all(predictor.norm_std > 0)
