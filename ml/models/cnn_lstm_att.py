import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class SoftAttention(nn.Module):
    """
    Applies a soft self-attention mechanism over sequence frames.
    """
    def __init__(self, hidden_dim: int, attn_dim: int = 128):
        super(SoftAttention, self).__init__()
        self.W = nn.Linear(hidden_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # h shape: (batch_size, time_steps, hidden_dim)
        
        # Linear projection to attention space
        # u shape: (batch_size, time_steps, attn_dim)
        u = torch.tanh(self.W(h))
        
        # Calculate scores
        # scores shape: (batch_size, time_steps, 1)
        scores = self.v(u)
        
        # Softmax over time_steps
        # weights shape: (batch_size, time_steps, 1)
        weights = F.softmax(scores, dim=1)
        
        # Weighted sum of states
        # context shape: (batch_size, hidden_dim)
        context = torch.sum(weights * h, dim=1)
        
        return context, weights.squeeze(-1) # (batch, hidden_dim), (batch, time_steps)


class SpeechCNNLSTMAttention(nn.Module):
    """
    CNN + BiLSTM + Attention model for Speech Emotion Recognition.
    Level D model.
    Extracts time-frequency representations using CNN, models temporal patterns with BiLSTM,
    and pools predictions using soft self-attention to highlight influential speech segments.
    """
    def __init__(self, num_classes: int = 8, n_mels: int = 128, lstm_hidden: int = 128, attn_dim: int = 128):
        super(SpeechCNNLSTMAttention, self).__init__()
        
        # CNN Feature Extractor
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        # MaxPool2d only on height (frequency) to preserve time dimensions
        self.pool = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        # After 4 max-pooling operations on height:
        # Height: 128 -> 64 -> 32 -> 16 -> 8
        self.cnn_out_channels = 128
        self.cnn_out_height = n_mels // 16 # 8
        self.lstm_in_dim = self.cnn_out_channels * self.cnn_out_height # 1024
        
        # BiLSTM Layer
        self.lstm = nn.LSTM(
            input_size=self.lstm_in_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention Layer
        self.attention = SoftAttention(hidden_dim=lstm_hidden * 2, attn_dim=attn_dim)
        
        # Classifier
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input shape: (batch, 1, height, width) -> e.g. (batch, 1, 128, 130)
        
        # Pass through CNN
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        
        # Permute to sequence shape: (batch, width, channels * height)
        batch_size, channels, height, width = x.size()
        x = x.permute(0, 3, 1, 2).contiguous() # (batch, width, channels, height)
        x = x.view(batch_size, width, channels * height) # (batch, time_steps, 1024)
        
        x = self.dropout(x)
        
        # Pass to BiLSTM
        # lstm_out shape: (batch, time_steps, hidden_size * 2)
        lstm_out, _ = self.lstm(x)
        
        # Pass to Attention layer
        # attn_out shape: (batch, hidden_size * 2), weights shape: (batch, time_steps)
        attn_out, attn_weights = self.attention(lstm_out)
        
        # Classifier
        logits = self.fc(self.dropout(attn_out))
        
        return logits, attn_weights
