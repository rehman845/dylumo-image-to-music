"""
FastMLP Model for Image-to-Music Feature Mapping

Architecture: 1024 -> 512 -> 256 -> 128 -> 13
Maps CLIP image features (1024-dim) to Spotify audio features (13-dim)
"""

import torch
import torch.nn as nn


class FastMLP(nn.Module):
    """
    Fast Multi-Layer Perceptron for mapping image features to music features.
    
    Architecture:
        Input (1024) -> Hidden (512) -> Hidden (256) -> Hidden (128) -> Output (13)
    
    Uses:
        - BatchNorm for stable training
        - ReLU activation
        - Dropout for regularization
    """
    
    def __init__(
        self,
        input_dim: int = 1024,
        output_dim: int = 13,
        hidden_dims: list = None,
        dropout: float = 0.2
    ):
        """
        Initialize FastMLP.
        
        Args:
            input_dim: Dimension of input features (CLIP = 1024)
            output_dim: Dimension of output features (Spotify audio features = 13)
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer (no activation - we want raw predictions for regression)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        return self.network(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict music features from image features.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Predicted music features of shape (batch_size, output_dim)
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_dims': self.hidden_dims
        }, path)
        print(f"[OK] Model saved to {path}")
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'FastMLP':
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            input_dim=checkpoint['input_dim'],
            output_dim=checkpoint['output_dim'],
            hidden_dims=checkpoint['hidden_dims']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        print(f"[OK] Model loaded from {path}")
        return model


# Feature names for Spotify audio features (13 dimensions)
SPOTIFY_FEATURE_NAMES = [
    'danceability',      # 0-1
    'energy',            # 0-1
    'key',               # 0-11 (normalized to 0-1)
    'loudness',          # -60 to 0 dB (normalized to 0-1)
    'mode',              # 0 or 1
    'speechiness',       # 0-1
    'acousticness',      # 0-1
    'instrumentalness',  # 0-1
    'liveness',          # 0-1
    'valence',           # 0-1
    'tempo',             # 0-250 BPM (normalized to 0-1)
    'duration_ms',       # normalized to 0-1
    'time_signature'     # 3-7 (normalized to 0-1)
]


if __name__ == "__main__":
    # Test the model
    model = FastMLP()
    print(f"Model architecture:")
    print(model)
    
    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, 1024)
    y = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

