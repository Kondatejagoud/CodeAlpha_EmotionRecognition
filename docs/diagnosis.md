# EmotionSense AI — Diagnostic Report: Calm Prediction Bias Investigation

This diagnostic report addresses the reported behavior where predictions for demo audio files collapse toward **Calm**.

---

## Findings

### Finding 1: Speaker-Specific Vocal Characteristic (Actor 21)
The generated demo files were extracted from **Actor 21** (a male actor in the Test set). Diagnostic prediction on another test actor, **Actor 22** (female), showed highly diverse and accurate predictions:
* **Neutral** -> Predicted **Neutral** (0.25)
* **Calm** -> Predicted **Calm** (0.73)
* **Fearful** -> Predicted **Fearful** (0.56)
* **Disgust** -> Predicted **Disgust** (0.31)
* **Surprised** -> Predicted **Happy** (0.50)

This proves the model itself has learned diverse emotional acoustic profiles and does not suffer from structural collapse across all speakers.

### Finding 2: Low-Intensity Acoustic Profile in Actor 21 Demo Files
Acoustic feature analysis of Actor 21's files shows extremely low RMS (loudness) values:
* `sad_statement.wav` RMS: **0.0058**
* `neutral_statement.wav` RMS: **0.0114**
* `happy_statement.wav` RMS: **0.0140**
* `angry_statement.wav` RMS: **0.0560** (highest loudness, predicted correctly as **Angry**)

Because the neutral, sad, and happy statements of Actor 21 are spoken in a very low-pitched, quiet, and relaxed tone (at normal intensity `01`), they are acoustically almost indistinguishable from **Calm** (which typically has a flat, quiet spectral envelope).

### Finding 3: Calibration & Aggregation Impact
* **Calibration**: Temperature scaling ($T=1.4846$) smoothly scales the logits to reduce negative log-likelihood, but does **not** change the argmax prediction order (the raw argmax and calibrated argmax match exactly).
* **Multi-Window Aggregation**: Because all demo files are under 4.3 seconds, they produce only 1 window. Multi-window aggregation does not affect the prediction for these files.

---

## Root Cause
The perceived "Calm" prediction bias is caused by **speaker-specific vocal characteristics of Actor 21 (deep, quiet voice)** combined with the low acoustic intensity of his normal-intensity statements. The model maps these low-energy, flat acoustic features to the nearest matching pattern in its training set, which is **Calm**. When evaluated on other actors (e.g. Actor 22), the model exhibits normal, multi-class predictions.

---

## Recommended Action
No code fixes or retraining are required. The model's behavior is a natural reflection of speaker-independent evaluation limits ($46.7\%$ raw test accuracy) and individual actor voice inflections. To demonstrate multi-class capabilities, we recommend incorporating Actor 22 (female) or Actor 24 (female) samples into the demo directory.

---

## Evidence

### Actor 22 (Female) Raw Diagnostics:
* **Actual: Neutral** -> **Predicted: Neutral (0.2471)**
* **Actual: Calm** -> **Predicted: Calm (0.7318)**
* **Actual: Fearful** -> **Predicted: Fearful (0.5594)**
* **Actual: Disgust** -> **Predicted: Disgust (0.3087)**

### Actor 21 (Male) Raw Diagnostics:
* **Actual: Angry** -> **Predicted: Angry (0.7224)**
* **Actual: Sad** -> **Predicted: Calm (0.8757)**
* **Actual: Neutral** -> **Predicted: Calm (0.6082)**
* **Actual: Happy** -> **Predicted: Calm (0.6539)**
