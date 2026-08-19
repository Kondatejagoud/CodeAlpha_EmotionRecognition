import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import label_binarize

import sys
# Add project root directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ml.data.dataset import download_ravdess, parse_ravdess_metadata, get_speaker_independent_split
from ml.preprocessing.audio_processor import AudioProcessor
from ml.features.feature_extractor import FeatureExtractor
from ml.training.train import SpeechDataset
from ml.inference.predictor import EmotionPredictor

# Hardcoded classes
CLASSES = ["Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprised"]

def run_evaluation(model_name: str = "cnn-bilstm-attention", feature_type: str = "mel"):
    """
    Evaluates the model on the test split (unseen actors 21-24).
    Produces detailed metrics, speaker analysis, error reports, and plots.
    """
    # 1. Initialize predictor
    print(f"Initializing predictor for model: {model_name}...")
    predictor = EmotionPredictor(model_name=model_name, feature_type=feature_type)
    
    # 2. Get dataset metadata
    data_path = download_ravdess()
    metadata_df = parse_ravdess_metadata(data_path)
    
    if metadata_df.empty:
        print("Dataset is empty. Cannot run evaluation.")
        return
        
    _, _, test_df = get_speaker_independent_split(metadata_df)
    
    # 3. Predict on all test set examples
    y_true = []
    y_pred = []
    y_probs = []
    filenames = []
    genders = []
    actor_ids = []
    
    print(f"Running prediction on {len(test_df)} test samples...")
    for idx, row in test_df.iterrows():
        file_path = row['file_path']
        true_emotion = row['emotion'].capitalize()
        
        # Predict
        res = predictor.predict(file_path)
        
        # Handle "UNCERTAIN" fallback
        pred_emotion = res["prediction"]
        if pred_emotion == "UNCERTAIN":
            # If the model is uncertain, get the top raw prediction
            if len(res["top_predictions"]) > 0:
                pred_emotion = res["top_predictions"][0]["emotion"]
            else:
                pred_emotion = "Neutral" # Fallback
                
        # Probs mapping
        prob_dict = {item["emotion"]: item["probability"] for item in res["top_predictions"]}
        prob_arr = [prob_dict.get(cls, 0.0) for cls in CLASSES]
        
        y_true.append(CLASSES.index(true_emotion))
        y_pred.append(CLASSES.index(pred_emotion))
        y_probs.append(prob_arr)
        
        filenames.append(row['filename'])
        genders.append(row['gender'])
        actor_ids.append(row['actor_id'])
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)
    
    # 4. Compute global metrics
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    _, _, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    
    # ROC-AUC calculation if multi-class binarization works
    try:
        y_true_bin = label_binarize(y_true, classes=list(range(8)))
        roc_auc = roc_auc_score(y_true_bin, y_probs, multi_class='ovr', average='macro')
    except Exception:
        roc_auc = 0.0
        
    print(f"\nGlobal Test Performance Metrics:")
    print(f"  Accuracy:          {acc:.4f}")
    print(f"  Macro Precision:   {precision:.4f}")
    print(f"  Macro Recall:      {recall:.4f}")
    print(f"  Macro F1-Score:    {f1:.4f}")
    print(f"  Weighted F1-Score: {weighted_f1:.4f}")
    print(f"  Macro ROC-AUC:     {roc_auc:.4f}")
    
    # 5. Per-class performance
    class_p, class_r, class_f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    per_class_metrics = {}
    for i, cls in enumerate(CLASSES):
        per_class_metrics[cls] = {
            "precision": round(float(class_p[i]), 4),
            "recall": round(float(class_r[i]), 4),
            "f1_score": round(float(class_f1[i]), 4)
        }
        
    # 6. Speaker subgroup evaluation (Gender/Actor breakdown)
    gender_results = {}
    for g in ["male", "female"]:
        idx_g = [i for i, gender in enumerate(genders) if gender == g]
        if idx_g:
            g_acc = accuracy_score(y_true[idx_g], y_pred[idx_g])
            gender_results[g] = round(float(g_acc), 4)
            
    actor_results = {}
    for actor in set(actor_ids):
        idx_act = [i for i, act in enumerate(actor_ids) if act == actor]
        if idx_act:
            act_acc = accuracy_score(y_true[idx_act], y_pred[idx_act])
            actor_results[f"Actor_{actor}"] = round(float(act_acc), 4)

    # 7. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot & save Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=CLASSES, yticklabels=CLASSES,
           title="Confusion Matrix (Test Set)",
           ylabel="True Emotion",
           xlabel="Predicted Emotion")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Annotate matrix blocks
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2. else "black")
    plt.tight_layout()
    os.makedirs("experiments", exist_ok=True)
    cm_path = "experiments/confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    
    # 8. Error analysis details
    # Easiest vs Hardest
    sorted_classes_by_f1 = sorted(CLASSES, key=lambda c: per_class_metrics[c]["f1_score"])
    difficulty_ranking = [
        {"emotion": cls, "f1_score": per_class_metrics[cls]["f1_score"]}
        for cls in sorted_classes_by_f1
    ]
    
    # Common confusion pairs
    confusions = []
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if i != j and cm[i, j] > 0:
                confusions.append({
                    "true": CLASSES[i],
                    "predicted": CLASSES[j],
                    "count": int(cm[i, j])
                })
    # Sort confusions by frequency
    confusions = sorted(confusions, key=lambda x: x["count"], reverse=True)

    # Gather a sample of misclassified files
    misclassified_examples = []
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            misclassified_examples.append({
                "filename": filenames[i],
                "gender": genders[i],
                "actor_id": int(actor_ids[i]),
                "true_emotion": CLASSES[y_true[i]],
                "predicted_emotion": CLASSES[y_pred[i]]
            })
            if len(misclassified_examples) >= 10: # Limit sample
                break
                
    # 9. Save all evaluation output to JSON
    eval_output = {
        "model_name": model_name,
        "feature_representation": feature_type,
        "evaluation_timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "global_metrics": {
            "accuracy": round(float(acc), 4),
            "macro_precision": round(float(precision), 4),
            "macro_recall": round(float(recall), 4),
            "macro_f1": round(float(f1), 4),
            "weighted_f1": round(float(weighted_f1), 4),
            "macro_roc_auc": round(float(roc_auc), 4)
        },
        "per_class_metrics": per_class_metrics,
        "speaker_gender_breakdown": gender_results,
        "individual_actor_accuracy": actor_results,
        "common_confusions": confusions[:5],
        "class_difficulty_ranking": difficulty_ranking,
        "misclassified_samples": misclassified_examples
    }
    
    eval_json_path = "experiments/evaluation_results.json"
    with open(eval_json_path, 'w') as f:
        json.dump(eval_output, f, indent=4)
        
    print(f"Evaluation report saved to {eval_json_path}")
    print(f"Confusion matrix plot saved to {cm_path}")
    
    return eval_output

if __name__ == "__main__":
    try:
        run_evaluation()
    except Exception as e:
        print(f"Evaluation failed: {e}")
