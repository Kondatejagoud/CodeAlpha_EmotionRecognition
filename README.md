# EmotionSense AI

## Overview
EmotionSense AI is a research-grade, explainable Speech Emotion Recognition (SER) system that predicts emotional categories from live microphone inputs or uploaded audio files. Developed as part of the **CodeAlpha Machine Learning Internship**, it addresses the core issues of standard emotion recognition models—specifically speaker data leakage and the black-box nature of deep learning predictions—by introducing speaker-independent evaluation, a pre-inference audio quality safety gate, and local explainability visuals.

## Problem Statement
Standard Speech Emotion Recognition systems often suffer from over-optimistic performance metrics due to "speaker leakage" (where audio samples from the same speaker appear in both the training and test sets). When deployed in the real world on unseen speakers, these models crash in accuracy. Furthermore, deep neural networks are black boxes, offering no explanation for *why* an emotion was predicted, which makes them unsuitable for transparent applications.

## Features
1. **Audio Quality Analysis (AQA)**: Filters and inspects incoming audio for sample rate, duration, loudness, clipping, and signal-to-noise ratio before predictions run.
2. **Speaker-Independent Splits**: Isolates speakers completely across datasets to ensure genuine out-of-sample performance reporting.
3. **Multi-Window sliding-window inference**: Supports variable-length live recordings by sliding a 3.0s window with 50% overlap and aggregating predictions using speech-energy (RMS) weights.
4. **Soft Temporal Self-Attention**: Focuses on specific syllables and emphasis regions over time, rendering a timeline of attention weights.
5. **Spectrogram Grad-CAM heatmaps**: Backpropagates activation scores to visual frequency hotspots.
6. **Calibrated Safety Gates**: Applies post-training temperature scaling to calibrate probabilities, rejecting predictions that fall below reliability thresholds.

## Architecture
```
                  +----------------------------------------------+
                  |               Web Frontend                   |
                  |  - Record (Mic) / Upload File                |
                  |  - Visualizers (Wavesurfer, Spectrogram)     |
                  |  - Local/Global Explainability Dashboard      |
                  +-----------------------+----------------------+
                                          |
                                          | POST /predict
                                          v
                  +----------------------------------------------+
                  |               FastAPI Backend                |
                  |  - API Routing & History Persistence         |
                  |  - Prediction DB Logging (SQLite)            |
                  +-----------------------+----------------------+
                                          |
                                          v
                  +----------------------------------------------+
                  |            Audio Quality Analyzer            |
                  |  - Inspects clipping, silence ratio, RMS, SNR|
                  |  - Decides status: GOOD, WARNING, UNSUITABLE |
                  +-----------------------+----------------------+
                                          | (If GOOD/WARNING)
                                          v
                  +----------------------------------------------+
                  |          Preprocessing & Feature Ext.        |
                  |  - Trimming, resampling, mono conversion     |
                  |  - Mel Spectrogram / MFCC generation         |
                  +-----------------------+----------------------+
                                          |
                                          v
                  +----------------------------------------------+
                  |                 Inference Pipeline           |
                  |  - Level D Model: CNN -> BiLSTM -> Attention |
                  |  - Temperature scaling calibration (T=1.48)  |
                  |  - Grad-CAM on Mel Spectrogram features       |
                  |  - Temporal Attention Weights Extractor       |
                  |  - Uncertainty safety gates validation        |
                  +----------------------------------------------+
```

## Dataset
The system uses the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset, containing 1,440 emotional utterances from 24 actors expressing 8 emotions (neutral, calm, happy, sad, angry, fearful, disgust, surprised).

## Speaker-Independent Evaluation
To report honest metrics, the dataset is divided by actors to ensure complete speaker isolation:
* **Training Actors**: 1–18 (1,080 files)
* **Validation Actors**: 19–20 (120 files)
* **Testing Actors**: 21–24 (240 files)

## Preprocessing
* Resampling to exactly **22050 Hz** mono.
* Silence trimming at **30 dB** threshold.
* Segment padding and cropping to a unified **3.0s** duration.
* Channel Z-score scaling using parameters computed strictly over training actors: [`mel_normalization.json`](file:///c:/Users/goudt/OneDrive/Desktop/intern/Emotion_Recognition/models/mel_normalization.json).

## Model Architecture
* **SpeechCNNLSTMAttention**:
  1. **2D CNN Feature Extractor**: 4 convolutional layers (with BatchNorm, MaxPool, and Dropout) extracting time-frequency local maps.
  2. **Bidirectional LSTM Temporal Layer**: Learns bidirectional temporal sequencing.
  3. **Soft Self-Attention Layer**: Learns weighted syllable-level importances over time.

## Training
The training pipeline features:
* **Optimizer**: `AdamW` with weight decay.
* **Scheduling**: `ReduceLROnPlateau` reducing learning rates on validation plateaus.
* **Class Weighting**: Dynamically balanced loss weights to handle neutral sample imbalances.
* **Checkpointing**: Automatic early stopping and saving based on minimum validation loss.

## Evaluation
A single test run on the untouched test set (Actors 21–24) using the best pre-trained model checkpoint achieves:
* **Overall Raw Test Accuracy**: **46.7%**
* **Macro-F1 Score**: **40.6%**
* **Calibrated Safety-Gated Accuracy**: **64.5%** (at **51.7%** coverage).

## Explainability
* **Temporal Attention Mapping**: Overlays frame-level self-attention weights onto the audio timeline, identifying the exact milliseconds representing the highest emotional intensity.
* **Spectrogram Grad-CAM**: Highlights which frequency bins (Hz) in the Mel Spectrogram drove the convolutional layers to classify the emotion.

## Audio Quality Analysis
Before predictions are run, the input audio is analyzed across:
* **Duration**: Discards files that are too short.
* **Sample Rate**: Verifies sampling rates.
* **RMS Energy**: Measures speech signal loudness.
* **Clipping Ratio**: Determines if the signal suffers from gain distortion.
* **Silence Ratio**: Identifies if the clip is mostly silence.
* **SNR Estimate**: Computes background noise ratio.
* **AQA Status**: Evaluates to `GOOD`, `WARNING` (if slight issues exist), or `UNSUITABLE` (disallows prediction).

## Uncertainty / Reliability
A post-training Temperature parameter ($T = 1.4846$) is applied to calibrate validation probabilities, lowering Expected Calibration Error (ECE) from **14.34%** to **7.52%**. 
Predictions are marked **UNCERTAIN** if:
1. Calibrated confidence probability is $< 0.40$.
2. Margin between top-two predictions is $< 0.15$.

## Technology Stack
* **Backend**: FastAPI, SQLAlchemy (SQLite database), PyTorch, Librosa.
* **Frontend**: React, Vite, WaveSurfer.js, Chart.js.

## Project Structure
```
Emotion_Recognition/
├── backend/
│   ├── app/
│   │   ├── api/endpoints.py         # Prediction REST endpoints
│   │   ├── core/database.py         # SQLite connection setup
│   │   ├── services/audio_analyzer.py # AQA service
│   │   └── main.py                  # API server init
│   └── requirements.txt
├── frontend/
│   ├── src/components/Dashboard.jsx # UI Grid controller
│   └── package.json
├── ml/
│   ├── data/dataset.py               # Dataset splits downloader
│   ├── models/cnn_lstm_att.py       # PyTorch Model graph
│   └── training/train.py             # Training loop
├── demo/                             # Sample audio files
├── models/                           # Configurations and нормализаторы
└── README.md
```

## Installation
Clone the repository and install the backend packages:
```bash
pip install -r requirements.txt
```

## Running the Backend
Launch the FastAPI development server:
```bash
python backend/app/main.py
```
API docs are available at `http://127.0.0.1:8000/docs`.

## Running the Frontend
1. Change directory to frontend and install npm components:
   ```bash
   cd frontend
   npm install
   ```
2. Start the Vite React app:
   ```bash
   npm run dev
   ```
   Open `http://localhost:5173` in your web browser.

## Example Prediction
Using a file from the `demo/` folder, uploading `demo/angry_statement.wav` results in:
```json
{
  "prediction": "Angry",
  "probability": 0.812,
  "reliability": "HIGH",
  "audio_quality": { "status": "GOOD", "rms": 0.054, "snr_est": 25.4 }
}
```

## Limitations
* **Speaker Inflections**: Studio-recorded datasets can have exaggerated speech inflections compared to natural colloquial discussions.
* **Accents & Accent Shift**: Performance may vary for distinct accents outside of the training set.

## Ethical Considerations
Vocal properties vary across individuals, cultures, and age groups. Classification is strictly an acoustic matching pattern and must not be used as clinical psychological evaluations.

## Future Improvements
* Multi-lingual model training.
* Noise-resilient Mel filterbanks.

## Internship Task
This project was developed for the **CodeAlpha Machine Learning Internship** as part of the final Speech Emotion Recognition systems engineering submission.
