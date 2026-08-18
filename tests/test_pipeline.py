import os
import tempfile
import numpy as np
import pytest
import soundfile as sf
from ml.preprocessing.audio_processor import AudioProcessor
from ml.features.feature_extractor import FeatureExtractor
from backend.app.services.audio_analyzer import AudioQualityAnalyzer

@pytest.fixture
def dummy_audio_files():
    """
    Generates temporary WAV files of different audio conditions for testing:
    1. Clean 1kHz sine wave
    2. Silent audio
    3. Clipped sine wave
    """
    temp_dir = tempfile.mkdtemp()
    
    sr = 22050
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # 1. Clean Sine Wave
    y_clean = 0.5 * np.sin(2 * np.pi * 440 * t) # 440Hz tone
    clean_path = os.path.join(temp_dir, "clean.wav")
    sf.write(clean_path, y_clean, sr)
    
    # 2. Silent audio
    y_silent = np.zeros(int(sr * duration))
    silent_path = os.path.join(temp_dir, "silent.wav")
    sf.write(silent_path, y_silent, sr)
    
    # 3. Clipped audio (amplitude > 1.0, written and clipped)
    y_clipped = np.sin(2 * np.pi * 440 * t) * 5.0 # Large amplitude to cause clipping
    y_clipped = np.clip(y_clipped, -1.0, 1.0)
    clipped_path = os.path.join(temp_dir, "clipped.wav")
    sf.write(clipped_path, y_clipped, sr)

    # 4. Short audio
    y_short = 0.5 * np.sin(2 * np.pi * 440 * t[:int(sr * 0.2)]) # 0.2 seconds
    short_path = os.path.join(temp_dir, "short.wav")
    sf.write(short_path, y_short, sr)

    yield {
        "clean": clean_path,
        "silent": silent_path,
        "clipped": clipped_path,
        "short": short_path,
        "sr": sr
    }
    
    # Cleanup temp directory
    for f in [clean_path, silent_path, clipped_path, short_path]:
        try:
            os.remove(f)
        except OSError:
            pass
    try:
        os.rmdir(temp_dir)
    except OSError:
        pass

def test_audio_processor(dummy_audio_files):
    processor = AudioProcessor(target_sr=22050, target_duration=3.0)
    
    # Test loading clean audio
    y, sr = processor.load_and_preprocess(dummy_audio_files["clean"])
    assert sr == 22050
    assert len(y) == 22050 * 3.0
    
    # Test loading short audio
    y_short, sr_short = processor.load_and_preprocess(dummy_audio_files["short"])
    assert sr_short == 22050
    assert len(y_short) == 22050 * 3.0 # Should be padded to 3.0s

    # Test augmentations do not error
    aug_config = {
        "time_stretch": True,
        "stretch_min": 0.9,
        "stretch_max": 1.1,
        "pitch_shift": True,
        "pitch_min": -1.0,
        "pitch_max": 1.0,
        "noise_injection": True,
        "noise_factor": 0.002,
        "volume_perturbation": True,
        "gain_min": 0.9,
        "gain_max": 1.1
    }
    y_aug, _ = processor.load_and_preprocess(dummy_audio_files["clean"], augment=True, aug_config=aug_config)
    assert len(y_aug) == 22050 * 3.0

def test_feature_extractor(dummy_audio_files):
    processor = AudioProcessor(target_sr=22050, target_duration=3.0)
    extractor = FeatureExtractor(sr=22050)
    
    y, _ = processor.load_and_preprocess(dummy_audio_files["clean"])
    
    # Test 2D Mel spectrogram
    mel_2d = extractor.get_mel_spectrogram_2d(y)
    assert mel_2d.ndim == 3 # Channels, n_mels, time_steps
    assert mel_2d.shape[0] == 1 # 1 channel
    assert mel_2d.shape[1] == extractor.n_mels
    
    # Test flattened feature vectors
    flat_feat = extractor.get_flattened_features(y)
    assert flat_feat.ndim == 1
    assert len(flat_feat) > 0

def test_audio_quality_analyzer(dummy_audio_files):
    analyzer = AudioQualityAnalyzer()
    
    # Test clean WAV (should be GOOD)
    res_clean = analyzer.analyze_file(dummy_audio_files["clean"])
    assert res_clean["status"] == "GOOD"
    assert res_clean["clipping_ratio"] < 0.01
    
    # Test silent WAV (should be UNSUITABLE)
    res_silent = analyzer.analyze_file(dummy_audio_files["silent"])
    assert res_silent["status"] == "UNSUITABLE"
    assert "silent" in "".join(res_silent["reasons"]).lower()
    
    # Test clipped WAV (should be WARNING or UNSUITABLE depending on clipping percentage)
    res_clipped = analyzer.analyze_file(dummy_audio_files["clipped"])
    assert res_clipped["status"] in ["WARNING", "UNSUITABLE"]
    assert res_clipped["clipping_ratio"] > 0.10
    
    # Test short WAV (should be UNSUITABLE)
    res_short = analyzer.analyze_file(dummy_audio_files["short"])
    assert res_short["status"] == "UNSUITABLE"
    assert "short" in "".join(res_short["reasons"]).lower()
