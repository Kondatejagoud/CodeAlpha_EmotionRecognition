# Screen-Recorded Demo Video Script — EmotionSense AI

This guide contains the exact step-by-step recording sequence, captions, mouse movements, and configurations required to create a professional 2-minute video for your LinkedIn submission.

---

## Video Specifications
* **Target Duration**: ~2 minutes and 20 seconds (140 seconds total).
* **Audio**: Clean, subtle, low-volume instrumental background music (ambient/tech synth style). No voice-over.
* **Resolution**: 1080p (1920x1080) in full-screen window mode.
* **Recording Prep**: 
  1. Clear prediction history by clicking the **Clear** button on the history panel.
  2. Open the browser console and close it to verify no red console error statements are showing.
  3. Close any unrelated background programs or OS notification trays.
  4. Ensure your browser address bar is hidden or uses a clean local URL (`http://localhost:5173`).

---

## Step-by-Step Recording Sequence & Captions

| Scene # | Scene Title | Duration | UI Target Element | Action / Mouse Movement | On-Screen Caption Text |
| :---: | :--- | :---: | :--- | :--- | :--- |
| **1** | Title | 8s | Header Area / Full Dashboard | Keep cursor stationary in empty space. Let dashboard display in a clean glassmorphic state. | **EmotionSense AI**<br>Explainable Speech Emotion Recognition<br>CNN + BiLSTM + Attention |
| **2** | Dashboard Walkthrough | 12s | Full Dashboard | Slowly hover mouse from left controls column (Microphone/Upload) to center prediction panels and right History sidebar. | **End-to-end Speech Emotion Recognition System**<br>Speaker-independent evaluation<br>Train: Actors 1–18 \| Val: 19–20 \| Test: 21–24 |
| **3** | Audio Input | 15s | Audio File Upload | Hover to drag-and-drop file upload. Drag the `demo/angry_statement.wav` file (or click and select it). Click **Analyze**. | **Audio preprocessing → Mel Spectrogram → Neural Network inference** |
| **4** | Prediction | 15s | Primary Prediction Board | Highlight the predicted emotion **Angry** (53%), alternative class predictions, and the safety-gated reliability status. | **Prediction with probability distribution and input-quality analysis** |
| **5** | Audio Quality Analysis | 10s | Audio Quality Card | Highlight duration ($4.14\text{s}$), sample rate ($48\text{k}\text{Hz}$), RMS loudness ($0.026$), clipping ($0\%$), and silence ratio. | **Audio Quality Analysis helps identify recordings that may affect prediction reliability.** |
| **6** | Attention Explainability | 15s | Temporal Attention Plot | Slowly scroll down to "Model Explainability Interpretations" and highlight the soft self-attention peaks over time. | **Temporal Attention Mapping**<br>Highlights temporal regions receiving greater attention from the model. |
| **7** | Grad-CAM Explainability | 15s | Mel Spectrogram Grad-CAM | Hover over the Grad-CAM heatmap overlay showing time-frequency activation hotspots. | **Mel Spectrogram Grad-CAM**<br>Highlights time-frequency regions contributing to the CNN representation. |
| **8** | Model Metadata | 12s | Left Sidebar Metadata | Scroll back to the top left and point cursor to the active architecture and split badges. | **Speaker-independent evaluation reduces speaker leakage.** |
| **9** | Performance | 10s | Header badge | Highlight the Test Acc badge ($46.7\%$). | **Final speaker-independent test accuracy: 46.7%**<br>Safety-gated accuracy: 64.5% at 51.7% coverage. |
| **10** | SQLite History | 10s | Right Prediction History | Hover over the SQLite database history showing saved records. Click a record to show it loads correctly. | **Prediction history stored locally using SQLite** |
| **11** | Technology Stack | 10s | README / Tech list | Briefly display files list or overlay text. | **Python • PyTorch • Librosa • Scikit-learn**<br>FastAPI • React • SQLite<br>Explainable AI |
| **12** | Limitation Notice | 8s | Footer disclaimer | Highlight the disclaimer text at the bottom footer. | **Important Limitation**<br>This system predicts acoustic patterns associated with emotion categories in the training data. It does not determine a person's actual psychological or mental state. |
| **13** | Outro | 5s | Clean Screen | Transition to logo or black screen. | **EmotionSense AI**<br>Built for the CodeAlpha Machine Learning Internship<br>GitHub Repository: (Check caption links) |

---

## Final Upload Checklist (LinkedIn Submission)

- [ ] **FastAPI Backend is running** (`python backend/app/main.py` is active on port 8000).
- [ ] **React Frontend is running** (`npm run dev` active on port 5173).
- [ ] **Database cleared** before starting the capture.
- [ ] **WAV demo files** ready at hand inside your local demo folder.
- [ ] **Instrumental audio** added at low gain (e.g. -20dB) so it does not overpower the visuals.
- [ ] **Disclaimer visible** in both the video footer and the LinkedIn post description.
- [ ] **Internship mention**: CodeAlpha Machine Learning Internship included in your LinkedIn text block.
