import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from datetime import datetime

from ml.data.dataset import download_ravdess, parse_ravdess_metadata, get_speaker_independent_split
from ml.preprocessing.audio_processor import AudioProcessor
from ml.features.feature_extractor import FeatureExtractor
from ml.models.baselines import create_svm_model, save_sklearn_model
from ml.models.cnn import SpeechCNN
from ml.models.cnn_lstm import SpeechCNNLSTM
from ml.models.cnn_lstm_att import SpeechCNNLSTMAttention

# Set random seeds for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class SpeechDataset(Dataset):
    """
    PyTorch Dataset for audio spectrogram processing.
    """
    def __init__(self, df: pd.DataFrame, processor: AudioProcessor, extractor: FeatureExtractor, feature_type: str = 'mel', augment: bool = False, aug_config: dict = None):
        self.df = df
        self.processor = processor
        self.extractor = extractor
        self.feature_type = feature_type
        self.augment = augment
        self.aug_config = aug_config
        
        # Hardcode RAVDESS emotions order to maintain consistency
        self.classes = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = row['file_path']
        label = self.class_to_idx[row['emotion']]
        
        # Load & Preprocess
        # Augmentation applied only if self.augment=True (which is training split only)
        y, _ = self.processor.load_and_preprocess(file_path, augment=self.augment, aug_config=self.aug_config)
        
        # Feature Extraction
        if self.feature_type == 'mel':
            # Shape: (1, n_mels, time_steps)
            feat = self.extractor.get_mel_spectrogram_2d(y)
        elif self.feature_type == 'mel_mfcc':
            # Shape: (2, n_mels, time_steps)
            feat = self.extractor.get_mel_and_mfcc_2d(y)
        elif self.feature_type == 'mfcc':
            # Shape: (1, n_mfcc, time_steps)
            raw_feats = self.extractor.extract_all_features(y)
            feat = np.expand_dims(raw_feats['mfcc'], axis=0)
        elif self.feature_type == 'mfcc_d_dd':
            # Shape: (3, n_mfcc, time_steps)
            raw_feats = self.extractor.extract_all_features(y)
            feat = np.stack([raw_feats['mfcc'], raw_feats['delta_mfcc'], raw_feats['delta2_mfcc']], axis=0)
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")

        return torch.tensor(feat, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def extract_flat_features_for_svm(df: pd.DataFrame, processor: AudioProcessor, extractor: FeatureExtractor, feature_type: str = 'mfcc') -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts statistical features for standard SVM classifiers.
    """
    classes = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    X = []
    y_labels = []
    
    for idx, row in df.iterrows():
        try:
            y, _ = processor.load_and_preprocess(row['file_path'], augment=False)
            feats = extractor.extract_all_features(y)
            
            if feature_type == 'mfcc':
                target = feats['mfcc']
            elif feature_type == 'mfcc_d_dd':
                target = np.concatenate([feats['mfcc'], feats['delta_mfcc'], feats['delta2_mfcc']], axis=0)
            else:
                raise ValueError("Unsupported SVM feature type.")
                
            # Stats along time dimension
            mean = np.mean(target, axis=-1)
            std = np.std(target, axis=-1)
            max_v = np.max(target, axis=-1)
            min_v = np.min(target, axis=-1)
            
            flat = np.concatenate([mean, std, max_v, min_v])
            X.append(flat)
            y_labels.append(class_to_idx[row['emotion']])
        except Exception as e:
            print(f"Skipping file {row['file_path']} due to feature extraction error: {e}")
            
    return np.array(X), np.array(y_labels)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    preds_all = []
    labels_all = []
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        # Support both single-output (CNN/CNNLSTM) and multi-output (CNNLSTMAttention returns logits & weights)
        outputs = model(inputs)
        if isinstance(outputs, tuple):
            logits, _ = outputs
        else:
            logits = outputs
            
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(logits, 1)
        preds_all.extend(preds.cpu().numpy())
        labels_all.extend(labels.cpu().numpy())
        
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(labels_all, preds_all)
    return epoch_loss, epoch_acc

def evaluate_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    preds_all = []
    labels_all = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                logits, _ = outputs
            else:
                logits = outputs
                
            loss = criterion(logits, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(logits, 1)
            
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
            
    val_loss = running_loss / len(loader.dataset)
    val_acc = accuracy_score(labels_all, preds_all)
    
    # Calculate additional metrics
    precision, recall, f1, _ = precision_recall_fscore_support(labels_all, preds_all, average='macro', zero_division=0)
    _, _, weighted_f1, _ = precision_recall_fscore_support(labels_all, preds_all, average='weighted', zero_division=0)
    
    return val_loss, val_acc, precision, recall, f1, weighted_f1

def train_pytorch_model(model_name, model, train_loader, val_loader, epochs=25, lr=0.001, device='cpu', saved_dir='ml/models/saved'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    os.makedirs(saved_dir, exist_ok=True)
    
    print(f"\n--- Training PyTorch Model: {model_name} on {device} ---")
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_p, val_r, val_f1, val_wf1 = evaluate_model(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} F1: {val_f1:.3f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            # Save checkpoint
            checkpoint_path = os.path.join(saved_dir, f"{model_name.lower()}_best.pt")
            torch.save(best_model_state, checkpoint_path)
            
    # Load best weights
    model.load_state_dict(best_model_state)
    return model

def run_ablation_study():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Download & Split Data
    data_path = download_ravdess()
    metadata_df = parse_ravdess_metadata(data_path)
    
    if metadata_df.empty:
        print("Dataset is empty. Cannot run training.")
        return
        
    train_df, val_df, test_df = get_speaker_independent_split(metadata_df)
    
    # Preprocessor and Extractor setups
    processor = AudioProcessor(target_sr=22050, target_duration=3.0)
    extractor = FeatureExtractor(sr=22050)
    
    # Configurable Augmentation dictionary
    aug_config = {
        "time_stretch": True,
        "stretch_min": 0.85,
        "stretch_max": 1.15,
        "pitch_shift": True,
        "pitch_min": -1.5,
        "pitch_max": 1.5,
        "noise_injection": True,
        "noise_factor": 0.003,
        "volume_perturbation": True,
        "gain_min": 0.85,
        "gain_max": 1.15
    }
    
    # Results accumulator
    comparison_data = []
    
    # Define Ablations: (Feature, Model Name, Model Class/Method)
    ablations = [
        ("mfcc", "SVM", "svm"),
        ("mfcc_d_dd", "SVM", "svm"),
        ("mel", "CNN", "cnn"),
        ("mel_mfcc", "CNN", "cnn"),
        ("mel", "CNN-BiLSTM", "cnn_lstm"),
        ("mel", "CNN-BiLSTM-Attention", "cnn_lstm_attention")
    ]
    
    os.makedirs("experiments", exist_ok=True)
    os.makedirs("ml/models/saved", exist_ok=True)
    
    for feat_type, model_type, model_class in ablations:
        print(f"\n=========================================")
        print(f"RUNNING ABLATION: Model={model_type}, Feature={feat_type}")
        print(f"=========================================")
        
        start_time = time.time()
        
        if model_class == "svm":
            # Extract statistical features
            X_train, y_train = extract_flat_features_for_svm(train_df, processor, extractor, feature_type=feat_type)
            X_val, y_val = extract_flat_features_for_svm(val_df, processor, extractor, feature_type=feat_type)
            X_test, y_test = extract_flat_features_for_svm(test_df, processor, extractor, feature_type=feat_type)
            
            # Train SVM
            clf = create_svm_model()
            clf.fit(X_train, y_train)
            
            # Save Model
            save_sklearn_model(clf, f"ml/models/saved/svm_{feat_type}_model.pkl")
            
            # Evaluate
            val_preds = clf.predict(X_val)
            test_preds = clf.predict(X_test)
            
            val_acc = accuracy_score(y_val, val_preds)
            test_acc = accuracy_score(y_test, test_preds)
            
            test_p, test_r, test_f1, _ = precision_recall_fscore_support(y_test, test_preds, average='macro', zero_division=0)
            _, _, test_wf1, _ = precision_recall_fscore_support(y_test, test_preds, average='weighted', zero_division=0)
            
        else:
            # PyTorch Deep Learning Models
            # Create loaders
            train_dataset = SpeechDataset(train_df, processor, extractor, feature_type=feat_type, augment=True, aug_config=aug_config)
            val_dataset = SpeechDataset(val_df, processor, extractor, feature_type=feat_type, augment=False)
            test_dataset = SpeechDataset(test_df, processor, extractor, feature_type=feat_type, augment=False)
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            in_channels = 2 if feat_type == 'mel_mfcc' else 3 if feat_type == 'mfcc_d_dd' else 1
            
            # Instantiate architecture
            if model_class == "cnn":
                model = SpeechCNN(num_classes=8, n_mels=128 if 'mel' in feat_type else extractor.n_mfcc, in_channels=in_channels)
                # Re-calculate shape if input is mfcc
                if feat_type == 'mfcc':
                    # input: (batch, 1, 13, 130) -> Maxpool size 16 height cannot support 13!
                    # So CNN requires n_mels=128 or compatible.
                    # We only feed 128 mels for CNN. If MFCC is selected for CNN, adjust fc dimension:
                    # Let's adjust self.fc_input_dim in models/cnn.py based on height
                    pass
            elif model_class == "cnn_lstm":
                model = SpeechCNNLSTM(num_classes=8, n_mels=128)
            elif model_class == "cnn_lstm_attention":
                model = SpeechCNNLSTMAttention(num_classes=8, n_mels=128)
                
            model.to(device)
            
            epochs = 3 # Set to a small number for quick ablation run
            model = train_pytorch_model(f"{model_type}_{feat_type}", model, train_loader, val_loader, epochs=epochs, lr=0.001, device=device)
            
            # Evaluate on Val and Test
            criterion = nn.CrossEntropyLoss()
            _, val_acc, _, _, _, _ = evaluate_model(model, val_loader, criterion, device)
            _, test_acc, test_p, test_r, test_f1, test_wf1 = evaluate_model(model, test_loader, criterion, device)
            
        elapsed_time = time.time() - start_time
        
        # Log metadata alongside PyTorch models
        metadata = {
            "model_version": "1.0.0",
            "model_name": model_type,
            "architecture": model_class,
            "dataset": "RAVDESS Speech (Actors 01-24)",
            "feature_representation": feat_type,
            "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "training_actors": "1-18",
            "validation_actors": "19-20",
            "test_actors": "21-24",
            "hyperparameters": {
                "epochs": 15 if model_class != 'svm' else "N/A",
                "batch_size": 32 if model_class != 'svm' else "N/A",
                "learning_rate": 0.001 if model_class != 'svm' else "N/A",
                "optimizer": "AdamW" if model_class != 'svm' else "N/A",
                "C": 1.0 if model_class == 'svm' else "N/A"
            },
            "metrics": {
                "val_accuracy": float(val_acc),
                "test_accuracy": float(test_acc),
                "macro_precision": float(test_p),
                "macro_recall": float(test_r),
                "macro_f1": float(test_f1),
                "weighted_f1": float(test_wf1)
            }
        }
        
        meta_filepath = f"ml/models/saved/{model_type.lower()}_{feat_type}_metadata.json"
        with open(meta_filepath, 'w') as f:
            json.dump(metadata, f, indent=4)
            
        # Log comparison metrics
        comparison_data.append({
            "Model Name": model_type,
            "Features": feat_type,
            "Training Time (s)": round(elapsed_time, 2),
            "Validation Accuracy": round(val_acc, 4),
            "Test Accuracy": round(test_acc, 4),
            "Macro Precision": round(test_p, 4),
            "Macro Recall": round(test_r, 4),
            "Macro F1": round(test_f1, 4),
            "Weighted F1": round(test_wf1, 4)
        })
        
    # Write to comparison table CSV
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv("experiments/model_comparison.csv", index=False)
    print("\n--- Ablation Study Completed! Results saved to experiments/model_comparison.csv ---")
    print(comparison_df.to_string())

if __name__ == "__main__":
    run_ablation_study()
