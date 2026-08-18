# Model Card - Speech Emotion Recognition Models (Level A - D)

This Model Card details the four architectures evaluated for **EmotionSense AI**, trained on the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS).

## Intended Use
- **Primary Use Case**: Predicting raw acoustic markers of emotion in human voice signals from audio files or microphone captures.
- **Out of Scope**: Clinical psychological diagnosis, lie detection, or surveillance tracking. Predictions represent structural vocal acoustic behaviors, which may differ from a speaker's actual psychological state.

## Dataset Details
- **Source**: RAVDESS Speech Dataset (24 actors, 12 male, 12 female vocalizing statements in neutral, calm, happy, sad, angry, fearful, disgust, and surprised states).
- **Speaker-Independent Split**:
  - **Training Set**: Actors 1–18 (1080 files, including augmentations).
  - **Validation Set**: Actors 19–20 (120 files, clean).
  - **Test Set**: Actors 21–24 (240 files, clean).
- **Data Hygiene**: Augmentations (noise, shift, speed, gain) are applied **only to the training split** to ensure clean validation and testing.

## Training Procedure & Hyperparameters
- **Acoustic Front-end**: 128-band Mel Spectrogram (22050Hz, n_fft=2048, hop_length=512).
- **Optimizer**: AdamW (Learning Rate = 1e-3, Weight Decay = 1e-4).
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=3).
- **Framework**: PyTorch.

---

## Model Architectures & Ablation Matrix

### Level A: Support Vector Machine (SVM)
- **Features**: Aggregated statistical summaries (mean, std, max, min) of MFCCs (or MFCC + Delta + Delta-Delta).
- **Pros**: Fast to train, highly reproducible, resistant to overfitting on small data.
- **Cons**: Completely discards temporal ordering of vocal inflections.

### Level B: 2D Speech CNN
- **Features**: 2D Mel Spectrogram inputs.
- **Architecture**: 4 Conv2D blocks with Batch Normalization, Max Pooling, and Dropout, feeding a dense classifier.
- **Pros**: Captures local time-frequency features (formant tracks, energy bands).
- **Cons**: Flat representation in dense layers ignores long-term sequence dynamics.

### Level C: CNN-BiLSTM
- **Features**: Frame-level CNN spatial features mapped to temporal sequence steps.
- **Architecture**: CNN feature maps pooled along frequency, fed into a Bidirectional LSTM (BiLSTM), followed by average sequence pooling.
- **Pros**: Models sequence directionality and temporal progression of speech.

### Level D: CNN-BiLSTM-Attention (Best Model)
- **Features**: CNN-BiLSTM sequence hidden states pooled via a Soft Self-Attention Layer.
- **Architecture**: Same as Level C, but average sequence pooling is replaced by:
  \[
  u_t = \tanh(W_a h_t + b_a)^T v_a
  \]
  \[
  \alpha_t = \frac{\exp(u_t)}{\sum \exp(u_i)}
  \]
- **Pros**: Most expressive architecture; produces temporal attention weights ($\alpha_t$) that identify key vocal frames triggering predictions.

---

## Technical Limitations
- **Actor Bias**: Since RAVDESS consists of theatrical actors expressing emotions in a noiseless studio, models might overfit to actor-style dramatic inflections.
- **Language / Cultural Limitations**: Native language in the dataset is North American English. The model might perform differently on non-English speakers or speakers with varying regional accents due to differences in pitch contours.
- **Environment Limitations**: Background noise, echo, and poor microphone responses (e.g. low SNR) will degrade performance. The pre-inference Audio Quality Analyzer is designed to reject such inputs.

## Ethical Considerations
- **TRUE STATE MISALIGNMENT WARNING**: Predicting vocal acoustic properties is not equivalent to reading a person's true psychological state or diagnosing mental health. True emotion expression is complex, culturally specific, and context-dependent.
- **No Diagnostics**: This model should not be used as a diagnostics tool for psychiatry or legal truth-verification.
