# EmotionSense AI — Explainable Speech Emotion Recognition System

EmotionSense AI is a research-grade, explainable Speech Emotion Recognition (SER) system that predicts emotional states from microphone recording inputs or uploaded audio files. 

This project goes beyond typical tutorial setups (which suffer from speaker data leakage and black-box predictions) by implementing a multi-level modeling pipeline (up to **CNN-BiLSTM-Attention**), a pre-inference **Audio Quality Analyzer**, and local/global explainability indicators (including **Grad-CAM spectrogram hotspots** and **temporal attention timelines**).

---

## Key Highlights & Features
1. **Audio Quality Analyzer (AQA)**: Validates duration, clipping ratio, silence ratio, RMS, and SNR estimates before feeding inputs to models, ensuring predictions are based on clean speech data.
2. **Speaker-Independent splits**: Avoids speaker data leakage by keeping training (actors 1-18), validation (actors 19-20), and testing (actors 21-24) speakers completely isolated.
3. **Soft Self-Attention Mechanism**: Directs the network to focus on specific syllables or emphasis regions over time, exporting visual frame-level attention weights ($\alpha_t$) onto a timeline.
4. **Grad-CAM Spectrogram Heatmaps**: Backpropagates activation scores to the last CNN convolutional layer to display exactly which frequency bands drove prediction.
5. **Prediction Reliability Flagging**: Incorporates uncertainty handling (refuses to present a confident label if probability is low ($< 0.40$) or margin is tight ($< 0.15$)).

---

## System Architecture

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
                  |  - Grad-CAM on Mel Spectrogram features       |
                  |  - Temporal Attention Weights Extractor       |
                  |  - Uncertainty handling & reliability checks  |
                  +----------------------------------------------+
```

---

## Project Directory Structure

```
Emotion_Recognition/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── endpoints.py         # Prediction & History REST routes
│   │   ├── core/
│   │   │   ├── config.py            # env loader
│   │   │   └── database.py          # SQLAlchemy Session setup
│   │   ├── models/
│   │   │   └── schema.py            # Prediction log table model
│   │   ├── services/
│   │   │   ├── audio_analyzer.py    # Pre-inference Audio Quality Analyzer
│   │   │   ├── explainability.py    # Grad-CAM and attention calculations
│   │   │   └── inference.py
│   │   └── main.py                  # Server app init
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── AudioRecorder.jsx    # Microphone recording + Canvas visualizer
│   │   │   ├── AudioUpload.jsx      # Drag & Drop file upload
│   │   │   ├── QualityReport.jsx    # Displays AQA metrics
│   │   │   ├── AttentionPlot.jsx    # Timeline attention weight overlay
│   │   │   ├── GradCamPlot.jsx      # Heatmap overlay on spectrogram
│   │   │   ├── PredictionHistory.jsx # Log lists
│   │   │   └── Dashboard.jsx        # Grid layout orchestrator
│   │   ├── App.jsx
│   │   ├── index.css                # Custom glassmorphic CSS styling
│   │   └── main.jsx
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
│
├── ml/
│   ├── data/
│   │   └── dataset.py               # RAVDESS zip downloader & parser
│   ├── preprocessing/
│   │   └── audio_processor.py       # Audio duration padding & trim
│   ├── features/
│   │   └── feature_extractor.py     # Mel spect & MFCC feature mapping
│   ├── models/
│   │   ├── baselines.py             # Level A: SVM baseline
│   │   ├── cnn.py                   # Level B: 2D Speech CNN
│   │   ├── cnn_lstm.py              # Level C: CNN-BiLSTM
│   │   └── cnn_lstm_att.py          # Level D: CNN-BiLSTM-Attention (Best)
│   ├── training/
│   │   └── train.py                 # Multi-level ablation run script
│   └── evaluation/
│       └── evaluate.py              # Speaker-independent test evaluation script
│
├── docs/
│   └── model_card.md                # Research-style model documentation
├── experiments/                     # CSV charts, training logs, results
└── tests/                           # Pytest unit testing suite
```

---

## Installation & Setup

### Prerequisites
- Python 3.8+
- Node.js (with npm)

### Backend setup
1. In the repository root, install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Setup environment variables by editing `.env` (a template exists as `.env.example`).
3. Run the API backend server:
   ```bash
   python backend/app/main.py
   ```
   The interactive Swagger documentation will be available at `http://127.0.0.1:8000/docs`.

### Frontend Setup
1. Open a new terminal in the `frontend` folder:
   ```bash
   cd frontend
   npm install
   ```
2. Launch the React development server:
   ```bash
   npm run dev
   ```
3. Open `http://localhost:5173` in your browser.

---

## ML Pipeline Workflows

### Programmatic Download & Splits
To run training, execute:
```bash
python ml/training/train.py
```
This script will:
1. Programmatically download the RAVDESS Speech dataset (~248MB) from Zenodo.
2. Group files speaker-independently:
   - **Training Set**: Actors 1–18
   - **Validation Set**: Actors 19–20
   - **Testing Set**: Actors 21–24
3. Apply training-only audio augmentations (stretch, pitch shift, volume scaling, white noise injection).
4. Run a multi-model **Ablation Study** comparing feature configurations. Logs results to `experiments/model_comparison.csv`.

### Speaker-Independent Evaluation
To evaluate the final model and produce reports (including confusion matrices, class difficulty, and gender/actor breakdowns):
```bash
python ml/evaluation/evaluate.py
```
Results will save to `experiments/evaluation_results.json` and a plot to `experiments/confusion_matrix.png`.

---

## Testing
Run the modular test suite to verify code stability:
```bash
python -m pytest tests/
```

---

## Limitations & Ethical Considerations
- **No Diagnostics**: This system maps raw vocal acoustic patterns to emotional classes. It **does not** determine a speaker's actual psychological state, truthfulness, or diagnose mental health conditions.
- **Actor Inflection Bias**: The training set features studio actors. Real-world natural speaking styles can have different vocal patterns.
- **Accents**: The system is trained on North American English speakers and may exhibit performance shifts for other language speakers or distinct accents.
