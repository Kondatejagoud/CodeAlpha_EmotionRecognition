import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Add project root directory to python path
sys.path.append("c:/Users/goudt/OneDrive/Desktop/intern/Emotion_Recognition")

from ml.data.dataset import get_speaker_independent_split, parse_ravdess_metadata, download_ravdess
from ml.preprocessing.audio_processor import AudioProcessor
from ml.features.feature_extractor import FeatureExtractor
from ml.models.cnn_lstm_att import SpeechCNNLSTMAttention

# Set random seeds for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class ExpDataset(Dataset):
    def __init__(self, df, processor, extractor, mean=None, std=None):
        self.df = df
        self.processor = processor
        self.extractor = extractor
        self.mean = mean
        self.std = std
        self.classes = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.features = []
        self.labels = []
        
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            file_path = row['file_path']
            label = self.class_to_idx[row['emotion']]
            
            y, _ = self.processor.load_and_preprocess(file_path, augment=False)
            feat = self.extractor.get_mel_spectrogram_2d(y) # Shape: (1, n_mels, time_steps)
            
            self.features.append(feat)
            self.labels.append(label)
            
        # If mean and std are not provided, calculate them (for training only)
        if self.mean is None or self.std is None:
            stacked = np.stack(self.features, axis=0) # (N, 1, n_mels, time)
            # average over N and time
            self.mean = np.mean(stacked, axis=(0, 3)).squeeze() # shape (n_mels,)
            self.std = np.std(stacked, axis=(0, 3)).squeeze()
            self.std = np.clip(self.std, a_min=1e-7, a_max=None)
            
        # Standardize features
        mean_reshaped = self.mean.reshape(1, -1, 1)
        std_reshaped = self.std.reshape(1, -1, 1)
        self.norm_features = []
        for feat in self.features:
            norm_feat = (feat - mean_reshaped) / std_reshaped
            self.norm_features.append(torch.tensor(norm_feat, dtype=torch.float32))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.norm_features[idx], torch.tensor(self.labels[idx], dtype=torch.long)

def train_eval_model(train_loader, val_loader, epochs=8, device='cpu'):
    model = SpeechCNNLSTMAttention(num_classes=8, n_mels=128)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    best_val_f1 = 0.0
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
        # Eval
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(y.cpu().numpy())
                
        val_acc = accuracy_score(val_labels, val_preds)
        val_p, val_r, val_f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='macro', zero_division=0)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            
    return best_val_acc, best_val_f1, model

def eval_multi_window(df, model, mean, std, window_size=3.0, overlap=0.5, device='cpu'):
    # Evaluation with Strategy D: Sliding Window Inference
    model.eval()
    processor_full = AudioProcessor(target_sr=22050, target_duration=10.0, trim_db=30.0) # Load full length
    extractor = FeatureExtractor(sr=22050)
    
    classes = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    val_preds = []
    val_labels = []
    
    sr = 22050
    w_samples = int(window_size * sr)
    hop_samples = int(w_samples * (1 - overlap))
    
    mean_reshaped = mean.reshape(1, -1, 1)
    std_reshaped = std.reshape(1, -1, 1)
    
    for idx in range(len(df)):
        row = df.iloc[idx]
        file_path = row['file_path']
        label = class_to_idx[row['emotion']]
        
        # Load raw full audio
        y, _ = processor_full.load_and_preprocess(file_path, augment=False)
        # Remove normalization padding to get raw trimmed signal
        
        # If signal is shorter than window, pad it
        if len(y) <= w_samples:
            pad_len = w_samples - len(y)
            y_windowed = [np.pad(y, (0, pad_len), mode='constant')]
        else:
            # Slice into windows
            y_windowed = []
            for start in range(0, len(y) - w_samples + 1, hop_samples):
                y_windowed.append(y[start:start + w_samples])
            # If last part is not covered, add one final window
            if (len(y) - w_samples) % hop_samples != 0:
                y_windowed.append(y[-w_samples:])
                
        # Run all windows through model and average probabilities
        logits_list = []
        for chunk in y_windowed:
            feat = extractor.get_mel_spectrogram_2d(chunk)
            norm_feat = (feat - mean_reshaped) / std_reshaped
            feat_tensor = torch.tensor(norm_feat, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits, _ = model(feat_tensor)
                probs = torch.softmax(logits, dim=1)
                logits_list.append(probs.cpu().numpy()[0])
                
        # Mean aggregation
        avg_probs = np.mean(logits_list, axis=0)
        pred = np.argmax(avg_probs)
        val_preds.append(pred)
        val_labels.append(label)
        
    acc = accuracy_score(val_labels, val_preds)
    _, _, f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='macro', zero_division=0)
    return acc, f1

def run_experiment():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running experiments on device: {device}")
    
    path = download_ravdess()
    df = parse_ravdess_metadata(path)
    train_df, val_df, _ = get_speaker_independent_split(df)
    
    results = {}
    
    # ----------------------------------------------------
    # Experiment A: 3.0s Center Crop (trim_db = 30.0)
    # ----------------------------------------------------
    print("\n==================================================")
    print("Strategy A: 3-second Center Crop (trim_db=30)")
    print("==================================================")
    proc_a = AudioProcessor(target_sr=22050, target_duration=3.0, trim_db=30.0)
    ext_a = FeatureExtractor(sr=22050)
    
    train_ds_a = ExpDataset(train_df, proc_a, ext_a)
    val_ds_a = ExpDataset(val_df, proc_a, ext_a, mean=train_ds_a.mean, std=train_ds_a.std)
    
    train_ldr_a = DataLoader(train_ds_a, batch_size=32, shuffle=True)
    val_ldr_a = DataLoader(val_ds_a, batch_size=32, shuffle=False)
    
    acc_a, f1_a, model_a = train_eval_model(train_ldr_a, val_ldr_a, epochs=3, device=device)
    results["Strategy A (3.0s crop)"] = {"Val Accuracy": acc_a, "Val Macro F1": f1_a}
    print(f"Strategy A Results: Val Acc: {acc_a:.4f}, Val Macro F1: {f1_a:.4f}")
    
    # Save Strategy A model weights momentarily for Strategy D
    torch.save(model_a.state_dict(), "scratch_model_a.pt")
    
    # ----------------------------------------------------
    # Experiment B: 5.3s Padding/Cropping (trim_db = 30.0)
    # ----------------------------------------------------
    print("\n==================================================")
    print("Strategy B: 5.3-second Center Crop/Pad (trim_db=30)")
    print("==================================================")
    proc_b = AudioProcessor(target_sr=22050, target_duration=5.3, trim_db=30.0)
    ext_b = FeatureExtractor(sr=22050)
    
    train_ds_b = ExpDataset(train_df, proc_b, ext_b)
    val_ds_b = ExpDataset(val_df, proc_b, ext_b, mean=train_ds_b.mean, std=train_ds_b.std)
    
    train_ldr_b = DataLoader(train_ds_b, batch_size=32, shuffle=True)
    val_ldr_b = DataLoader(val_ds_b, batch_size=32, shuffle=False)
    
    acc_b, f1_b, _ = train_eval_model(train_ldr_b, val_ldr_b, epochs=3, device=device)
    results["Strategy B (5.3s crop/pad)"] = {"Val Accuracy": acc_b, "Val Macro F1": f1_b}
    print(f"Strategy B Results: Val Acc: {acc_b:.4f}, Val Macro F1: {f1_b:.4f}")
    
    # ----------------------------------------------------
    # Experiment C: 4.0s Intelligent Padding (trim_db = 40.0)
    # ----------------------------------------------------
    print("\n==================================================")
    print("Strategy C: 4.0s Crop/Pad (trim_db=40)")
    print("==================================================")
    proc_c = AudioProcessor(target_sr=22050, target_duration=4.0, trim_db=40.0)
    ext_c = FeatureExtractor(sr=22050)
    
    train_ds_c = ExpDataset(train_df, proc_c, ext_c)
    val_ds_c = ExpDataset(val_df, proc_c, ext_c, mean=train_ds_c.mean, std=train_ds_c.std)
    
    train_ldr_c = DataLoader(train_ds_c, batch_size=32, shuffle=True)
    val_ldr_c = DataLoader(val_ds_c, batch_size=32, shuffle=False)
    
    acc_c, f1_c, _ = train_eval_model(train_ldr_c, val_ldr_c, epochs=3, device=device)
    results["Strategy C (4.0s crop/pad, trim_db=40)"] = {"Val Accuracy": acc_c, "Val Macro F1": f1_c}
    print(f"Strategy C Results: Val Acc: {acc_c:.4f}, Val Macro F1: {f1_c:.4f}")
    
    # ----------------------------------------------------
    # Experiment D: Multi-window Inference
    # ----------------------------------------------------
    print("\n==================================================")
    print("Strategy D: Multi-Window Inference (window=3s, overlap=0.5)")
    print("==================================================")
    acc_d, f1_d = eval_multi_window(val_df, model_a, train_ds_a.mean, train_ds_a.std, window_size=3.0, overlap=0.5, device=device)
    results["Strategy D (3.0s Multi-window inference)"] = {"Val Accuracy": acc_d, "Val Macro F1": f1_d}
    print(f"Strategy D Results: Val Acc: {acc_d:.4f}, Val Macro F1: {f1_d:.4f}")
    
    # Save the results table
    os.makedirs("experiments", exist_ok=True)
    with open("experiments/duration_experiment_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nAll experiments completed!")
    print(json.dumps(results, indent=4))
    
    # Cleanup temp model weights
    if os.path.exists("scratch_model_a.pt"):
        os.remove("scratch_model_a.pt")

if __name__ == "__main__":
    run_experiment()
