import torch
import torch.nn as nn

class SpeechCNNLSTM(nn.Module):
    """
    CNN + BiLSTM model for Speech Emotion Recognition.
    Level C model.
    CNN extracts localized time-frequency representations, which are treated as a
    temporal sequence and fed into a BiLSTM layer. Classification uses average pooling over time.
    """
    def __init__(self, num_classes: int = 8, n_mels: int = 128, lstm_hidden: int = 128):
        super(SpeechCNNLSTM, self).__init__()
        
        # Conv blocks designed to reduce frequency (height) while preserving time (width)
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
        self.lstm_in_dim = self.cnn_out_channels * self.cnn_out_height # 128 * 8 = 1024
        
        # BiLSTM Layer
        self.lstm = nn.LSTM(
            input_size=self.lstm_in_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Classifier
        self.fc = nn.Linear(lstm_hidden * 2, num_classes) # Bidirectional doubles hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch, 1, height, width) -> e.g. (batch, 1, 128, 130)
        
        # Pass through CNN
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        
        # Permute to sequence shape: (batch, width, channels * height)
        # Current shape: (batch, channels=128, height=8, width=time_steps)
        batch_size, channels, height, width = x.size()
        x = x.permute(0, 3, 1, 2).contiguous() # (batch, width, channels, height)
        x = x.view(batch_size, width, channels * height) # (batch, time_steps, 1024)
        
        x = self.dropout(x)
        
        # Pass to BiLSTM
        # lstm_out shape: (batch, time_steps, hidden_size * 2)
        lstm_out, _ = self.lstm(x)
        
        # Mean pooling across the time dimension
        out = torch.mean(lstm_out, dim=1) # (batch, hidden_size * 2)
        
        # Fully connected
        out = self.fc(self.dropout(out))
        return out
