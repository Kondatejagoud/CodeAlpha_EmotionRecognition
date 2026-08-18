import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional

class GradCAM:
    """
    Computes Grad-CAM map on the final convolutional layer of a PyTorch speech model.
    Grad-CAM highlights time-frequency regions in the Mel Spectrogram that drove prediction.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register PyTorch Hooks
        self.forward_hook = self.target_layer.register_forward_hook(self._save_activation)
        self.backward_hook = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove_hooks(self):
        """
        Removes hooks to prevent memory leaks or issues during repeated inference.
        """
        self.forward_hook.remove()
        self.backward_hook.remove()

    def generate_heatmap(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.gradients = None
        self.activations = None
        
        # Forward pass
        self.model.eval()
        self.model.zero_grad()
        
        # Input tensor needs gradient enabled
        input_tensor.requires_grad = True
        
        outputs = self.model(input_tensor)
        if isinstance(outputs, tuple):
            logits, _ = outputs
        else:
            logits = outputs
            
        score = logits[0, class_idx]
        
        # Backward pass
        score.backward()
        
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks failed to capture gradients/activations. Ensure target_layer is correctly in model graph.")
            
        # Extract gradients and activations
        gradients = self.gradients.cpu().data.numpy()[0] # Shape: (C, H, W)
        activations = self.activations.cpu().data.numpy()[0] # Shape: (C, H, W)
        
        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2)) # Shape: (C,)
        
        # Weighted combination of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32) # Shape: (H, W)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # Apply ReLU (we only care about features that positively influence prediction score)
        cam = np.maximum(cam, 0)
        
        # Normalize between 0 and 1
        max_val = np.max(cam)
        if max_val > 0:
            cam = cam / max_val
            
        return cam

def extract_explanations(
    model: nn.Module,
    input_tensor: torch.Tensor,
    predicted_class: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extracts Grad-CAM heatmap and/or temporal attention weights from the speech model.
    """
    grad_cam_heatmap = None
    attention_weights = None
    
    # Check if model has conv4 (target for Grad-CAM)
    target_conv = None
    if hasattr(model, 'conv4'):
        target_conv = model.conv4
    elif hasattr(model, 'module') and hasattr(model.module, 'conv4'):
        target_conv = model.module.conv4
        
    if target_conv is not None:
        try:
            # Generate Grad-CAM heatmap
            gcam = GradCAM(model, target_conv)
            grad_cam_heatmap = gcam.generate_heatmap(input_tensor, predicted_class)
            gcam.remove_hooks()
        except Exception as e:
            print(f"Grad-CAM extraction failed: {e}")
            
    # Check if forward pass yields attention weights (CNN-BiLSTM-Attention model)
    model.eval()
    with torch.no_grad():
        try:
            outputs = model(input_tensor)
            if isinstance(outputs, tuple) and len(outputs) == 2:
                # CNN-BiLSTM-Attention yields (logits, attn_weights)
                _, weights = outputs
                attention_weights = weights.cpu().squeeze(0).numpy() # (time_steps,)
        except Exception as e:
            print(f"Attention extraction failed: {e}")
            
    return grad_cam_heatmap, attention_weights
