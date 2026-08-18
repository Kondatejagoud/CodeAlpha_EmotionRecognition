import torch
import torch.nn as nn
import torch.nn.functional as F

class SpeechCNN(nn.Module):
    """
    2D Convolutional Neural Network for Speech Emotion Recognition.
    Acts on 2D Mel Spectrogram inputs of shape (batch_size, 1, n_mels, time_steps).
    Level B model.
    """
    def __init__(self, num_classes: int = 8, n_mels: int = 128, in_channels: int = 1):
        super(SpeechCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        
        # We need to compute the shape after 4 max-pooling operations (128 -> 64 -> 32 -> 16 -> 8)
        # For n_mels=128, height becomes 8.
        # For target duration of 3s, time_steps is 130. 130 -> 65 -> 32 -> 16 -> 8.
        # Height: 128 / 16 = 8
        # Width: 130 / 16 = 8
        self.fc_input_dim = 128 * 8 * 8
        
        # Classifier
        self.fc1 = nn.Linear(self.fc_input_dim, 256)
        self.fc_bn = nn.BatchNorm1d(256)
        self.out = nn.Linear(256, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # conv blocks
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        x = self.dropout(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Dense layers
        x = F.relu(self.fc_bn(self.fc1(x)))
        x = self.dropout(x)
        x = self.out(x)
        return x
        
    def get_last_conv_layer(self) -> nn.Module:
        """
        Returns the last convolutional layer. Used for Grad-CAM.
        """
        return self.conv4
