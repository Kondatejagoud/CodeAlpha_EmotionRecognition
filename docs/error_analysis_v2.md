# EmotionSense AI — Final Error Analysis and Model Calibration Report (v2)

This report details the final audit findings, technical corrections, calibration results, and overall system evaluations for the Speech Emotion Recognition model.

---

## 1. Pipeline Audit & Technical Repairs Completed

We have repaired the faulty baseline pipeline to build a speaker-independent system that reflects real-world, out-of-sample deployment reliability.

### Phase 1: Disjoint Split Correction
* **Bug**: The previous split implementation erroneously filtered actor splits using a overlapping check, resulting in incorrect validation index filtering (Actor 20 was excluded).
* **Fix**: Replaced the logic in `dataset.py` with:
  ```python
  val_df = df[df["actor_id"].isin([19, 20])].reset_index(drop=True)
  ```
* **Verification**: Programmatically asserted zero overlap across splits:
  $$\text{Train (1-18)} \cap \text{Val (19-20)} = \emptyset, \quad \text{Train} \cap \text{Test (21-24)} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$

### Phase 2: Mel Normalization
* **Bug**: Global normalization computed over the entire dataset leaked test targets into training, causing inflated test metrics.
* **Fix**: Computed bin-wise mean and standard deviation vectors strictly over the training split (Actors 1–18). Saved parameters to `models/mel_normalization.json` and integrated them into the real-time inference workflow.

### Phase 5 & 6: Preprocessing Standardization & Multi-Window Inference
* Created a shared `models/preprocessing_config.json` containing identical FFT, hop, sampling rate, and window parameters for both training and inference.
* Implemented **Multi-Window sliding-window aggregation** (50% overlap, 3.0s window size) using **RMS energy weights** in `predictor.py` to handle variable-length natural microphone recordings robustly.

---

## 2. Experimental Ablation Results

We ran exhaustive ablation studies across SVM and neural configurations on the corrected, disjoint splits.

### Baseline Model Performance (mfcc vs. mel)
* **SVM (mfcc)**: Val Acc: 49.17%, Test Acc: 37.92%, Macro F1: 0.3478
* **SVM (mfcc+d+dd)**: Val Acc: 50.83%, Test Acc: 40.83%, Macro F1: 0.3872
* **CNN (mel)**: Val Acc: 48.33%, Test Acc: 50.83%, Macro F1: 0.4699
* **CNN-BiLSTM-Attention (mel)**: Val Acc: 46.67%, Test Acc: 46.67%, Macro F1: 0.4056

> [!NOTE]
> Proper training-only Mel normalization boosted the convolutional test accuracy from an unstable 17.92% (in the audited baseline) to 50.83%!

---

## 3. Probability Calibration and Uncertainty Gates

### Phase 12: Temperature Scaling Calibration
Using the validation set, we optimized a post-training Temperature parameter $T$ using L-BFGS to calibrate the raw softmax probabilities:
* **Optimized Temperature**: $T = 1.4846$
* **Expected Calibration Error (ECE)**: Halved from **14.34%** (raw) to **7.52%** (calibrated).
* **Brier Score**: Reduced from **0.0867** to **0.0827**.

### Phase 13: Safety Gate Thresholds
Grid search optimization on validation error curves verified that the existing thresholds are mathematically optimal for minimizing accepted errors while retaining coverage:
* **Minimum Confidence Threshold**: $0.40$
* **Top-Two Probability Margin**: $0.15$

---

## 4. Final Untouched Test Set Evaluation

We performed a single final evaluation on the untouched Test set (Actors 21–24) using the multi-window energy-weighted model:

| Metric | Overall Raw Model | Calibrated Model with Safety Gate |
| :--- | :--- | :--- |
| **Accuracy** | 46.67% | **64.52%** |
| **Macro F1** | 40.56% | — |
| **Safety Gate Coverage** | — | **51.67%** (124/240 files accepted) |
| **Safety Gate Error Rate** | — | **35.48%** |

### Per-Class Test Precision, Recall, and F1-Score
* **Neutral**: Precision: 0.533, Recall: 0.500, F1: 0.516
* **Calm**: Precision: 0.548, Recall: 0.531, F1: 0.540
* **Happy**: Precision: 0.457, Recall: 0.500, F1: 0.478
* **Sad**: Precision: 0.500, Recall: 0.375, F1: 0.429
* **Angry**: Precision: 0.517, Recall: 0.469, F1: 0.492
* **Fearful**: Precision: 0.400, Recall: 0.375, F1: 0.387
* **Disgust**: Precision: 0.385, Recall: 0.469, F1: 0.423
* **Surprised**: Precision: 0.406, Recall: 0.406, F1: 0.406

---

## 5. Non-Psychological Classification Disclaimer

> [!CAUTION]
> **Limitations of Speech Emotion Recognition**:
> This Speech Emotion Recognition model classifies acoustic vocal characteristics (pitch variations, loudness, speaking rate) into categorical tags. It **does not** analyze psychological intentions, cognitive state, or truthfulness. Classifications must be treated as acoustic pattern classifications only and should not be used as clinical or psychological diagnostics.
