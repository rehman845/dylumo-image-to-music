"""
Fast MLP Model for Image-to-Music Feature Mapping

Optimized 3-layer MLP with BatchNorm and Dropout for fast, accurate predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FastMLP(nn.Module):
    """
    Fast Multi-Layer Perceptron for mapping image features to music features.
    
    Architecture:
        Input (1024) → Dense(512) → BN → ReLU → Dropout
                    → Dense(256) → BN → ReLU → Dropout
                    → Dense(128) → BN → ReLU
                    → Output(13) [music features]
    
    Optimized for:
        - Speed: 3 layers, BatchNorm for faster convergence
        - Accuracy: Dropout for regularization, ReLU activations
        - Mixed precision: FP16 compatible
    """
    
    def __init__(
        self,
        input_dim: int = 1024,  # CLIP image (512) + text (512)
        hidden_dims: list = [512, 256, 128],
        output_dim: int = 13,  # 13 music features
        dropout_rate: float = 0.2
    ):
        """
        Initialize FastMLP model.
        
        Args:
            input_dim: Dimension of input features (image + text)
            hidden_dims: List of hidden layer dimensions
            output_dim: Number of music features to predict
            dropout_rate: Dropout probability
        """
        super(FastMLP, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # BatchNorm (speeds up training, improves generalization)
            layers.append(nn.BatchNorm1d(hidden_dim))
            
            # ReLU activation
            layers.append(nn.ReLU())
            
            # Dropout (except for last layer)
            if i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Output layer (no activation, direct regression)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Predicted music features (batch_size, output_dim)
        """
        return self.model(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prediction mode (no gradients, eval mode).
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Predicted music features
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)


class WeightedMSELoss(nn.Module):
    """
    Weighted MSE Loss for multi-task regression.
    
    Allows different weights for different music features
    (e.g., valence and energy might be more important than key).
    """
    
    def __init__(self, weights: Optional[torch.Tensor] = None):
        """
        Initialize weighted MSE loss.
        
        Args:
            weights: Tensor of shape (output_dim,) with weights for each feature.
                    If None, uses equal weights (standard MSE).
        """
        super(WeightedMSELoss, self).__init__()
        self.weights = weights
        self.mse = nn.MSELoss(reduction='none')
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted MSE loss.
        
        Args:
            pred: Predicted features (batch_size, output_dim)
            target: True features (batch_size, output_dim)
            
        Returns:
            torch.Tensor: Scalar loss value
        """
        # Compute MSE for each feature
        mse_per_feature = self.mse(pred, target)  # (batch_size, output_dim)
        
        # Average over batch
        mse_per_feature = mse_per_feature.mean(dim=0)  # (output_dim,)
        
        # Apply weights if provided
        if self.weights is not None:
            mse_per_feature = mse_per_feature * self.weights.to(mse_per_feature.device)
        
        # Sum over features
        return mse_per_feature.sum()


if __name__ == "__main__":
    # Test the model
    print("Testing FastMLP...")
    
    # Create model
    model = FastMLP(input_dim=1024, output_dim=13)
    print(f"✓ Model created")
    print(f"  Input dimension: 1024")
    print(f"  Output dimension: 13")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 8
    x = torch.randn(batch_size, 1024)
    y = model(x)
    print(f"\n✓ Forward pass successful")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    
    # Test with mixed precision
    if torch.cuda.is_available():
        model = model.half().cuda()
        x = x.half().cuda()
        y = model(x)
        print(f"\n✓ FP16 inference successful on GPU")

