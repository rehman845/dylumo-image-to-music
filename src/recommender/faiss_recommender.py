"""
FAISS-based Fast Music Recommendation System

Uses GPU-accelerated FAISS for instant similarity search over 1M songs.
"""

import numpy as np
import faiss
import pandas as pd
from pathlib import Path
from typing import Union, List, Tuple, Optional
import pickle
import torch


class FAISSRecommender:
    """
    Fast recommendation system using FAISS for similarity search.
    
    Supports:
        - GPU acceleration (if available)
        - Batch queries
        - Custom distance metrics
        - Efficient indexing of 1M+ songs
    """
    
    def __init__(
        self,
        music_features: Optional[np.ndarray] = None,
        music_metadata: Optional[pd.DataFrame] = None,
        use_gpu: bool = True,
        index_type: str = "L2"  # "L2" or "cosine"
    ):
        """
        Initialize FAISS recommender.
        
        Args:
            music_features: Array of shape (N, 13) with music features
            music_metadata: DataFrame with track info (track_id, name, artist, etc.)
            use_gpu: Whether to use GPU acceleration
            index_type: "L2" for Euclidean distance, "cosine" for cosine similarity
        """
        self.music_features = music_features
        self.music_metadata = music_metadata
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.index_type = index_type
        self.index = None
        self.gpu_index = None
        
        if music_features is not None:
            self.build_index(music_features)
    
    def build_index(self, music_features: np.ndarray):
        """
        Build FAISS index from music features.
        
        Args:
            music_features: Array of shape (N, 13) with music features
        """
        print(f"Building FAISS index for {len(music_features):,} songs...")
        
        # Normalize features (important for cosine similarity)
        if self.index_type == "cosine":
            faiss.normalize_L2(music_features)
        
        # Convert to float32 (FAISS requirement)
        music_features = music_features.astype('float32')
        self.music_features = music_features
        
        # Create index
        dimension = music_features.shape[1]
        
        if self.index_type == "L2":
            self.index = faiss.IndexFlatL2(dimension)
        else:  # cosine (using inner product on normalized vectors)
            self.index = faiss.IndexFlatIP(dimension)
        
        # Add vectors to index
        self.index.add(music_features)
        
        print(f"✓ Index built with {self.index.ntotal:,} vectors")
        
        # Move to GPU if available
        if self.use_gpu:
            print("Moving index to GPU...")
            res = faiss.StandardGpuResources()
            self.gpu_index = faiss.index_cpu_to_gpu(res, 0, self.index)
            print("✓ Index moved to GPU")
    
    def search(
        self,
        query_features: np.ndarray,
        k: int = 10,
        return_distances: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Search for similar songs.
        
        Args:
            query_features: Query vector(s) of shape (13,) or (N, 13)
            k: Number of nearest neighbors to return
            return_distances: Whether to return distances along with indices
            
        Returns:
            If return_distances=False: Array of indices (N, k)
            If return_distances=True: Tuple of (indices, distances)
        """
        if self.gpu_index is None and self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Ensure query is 2D
        if query_features.ndim == 1:
            query_features = query_features.reshape(1, -1)
        
        # Normalize if using cosine similarity
        if self.index_type == "cosine":
            faiss.normalize_L2(query_features)
        
        # Convert to float32
        query_features = query_features.astype('float32')
        
        # Search
        index_to_use = self.gpu_index if self.gpu_index is not None else self.index
        
        if return_distances:
            distances, indices = index_to_use.search(query_features, k)
            return indices, distances
        else:
            distances, indices = index_to_use.search(query_features, k)
            return indices
    
    def recommend(
        self,
        query_features: np.ndarray,
        k: int = 10
    ) -> pd.DataFrame:
        """
        Get recommendations with metadata.
        
        Args:
            query_features: Query vector(s) of shape (13,) or (N, 13)
            k: Number of recommendations
            
        Returns:
            DataFrame with recommended songs and metadata
        """
        indices = self.search(query_features, k=k)
        
        # Flatten if single query
        if indices.ndim == 2 and indices.shape[0] == 1:
            indices = indices[0]
        
        # Get metadata if available
        if self.music_metadata is not None:
            recommendations = self.music_metadata.iloc[indices].copy()
            return recommendations
        else:
            # Return just indices
            return pd.DataFrame({'track_index': indices.flatten()})
    
    def save_index(self, filepath: Union[str, Path]):
        """Save index to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save CPU index (GPU index can't be saved directly)
        if self.index is not None:
            faiss.write_index(self.index, str(filepath))
            print(f"✓ Index saved to {filepath}")
        
        # Save metadata separately
        if self.music_metadata is not None:
            metadata_path = filepath.with_suffix('.metadata.pkl')
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.music_metadata, f)
            print(f"✓ Metadata saved to {metadata_path}")
    
    def load_index(self, filepath: Union[str, Path]):
        """Load index from disk."""
        filepath = Path(filepath)
        
        # Load index
        self.index = faiss.read_index(str(filepath))
        print(f"✓ Index loaded from {filepath}")
        print(f"  Contains {self.index.ntotal:,} vectors")
        
        # Load metadata if available
        metadata_path = filepath.with_suffix('.metadata.pkl')
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                self.music_metadata = pickle.load(f)
            print(f"✓ Metadata loaded")
        
        # Move to GPU if requested
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.gpu_index = faiss.index_cpu_to_gpu(res, 0, self.index)
            print("✓ Index moved to GPU")
    
    @classmethod
    def from_csv(
        cls,
        csv_path: Union[str, Path],
        feature_columns: List[str],
        metadata_columns: Optional[List[str]] = None,
        n_samples: Optional[int] = None,
        use_gpu: bool = True
    ) -> 'FAISSRecommender':
        """
        Create recommender from CSV file.
        
        Args:
            csv_path: Path to CSV with music features
            feature_columns: List of column names for music features (13 features)
            metadata_columns: List of column names for metadata (track_id, name, etc.)
            n_samples: Number of songs to use (None = all, useful for 100K subset)
            use_gpu: Whether to use GPU
            
        Returns:
            FAISSRecommender instance
        """
        print(f"Loading music data from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Sample if requested
        if n_samples is not None and len(df) > n_samples:
            df = df.sample(n=n_samples, random_state=42)
            print(f"  Sampled {n_samples:,} songs")
        
        # Extract features
        music_features = df[feature_columns].values.astype('float32')
        
        # Extract metadata
        if metadata_columns:
            music_metadata = df[metadata_columns].copy()
        else:
            music_metadata = df.copy()
        
        print(f"✓ Loaded {len(music_features):,} songs")
        print(f"  Feature shape: {music_features.shape}")
        
        # Create recommender
        recommender = cls(
            music_features=music_features,
            music_metadata=music_metadata,
            use_gpu=use_gpu
        )
        
        return recommender


if __name__ == "__main__":
    # Test with dummy data
    print("Testing FAISSRecommender...")
    
    # Create dummy music features (1000 songs, 13 features)
    n_songs = 1000
    music_features = np.random.rand(n_songs, 13).astype('float32')
    
    # Create dummy metadata
    music_metadata = pd.DataFrame({
        'track_id': range(n_songs),
        'name': [f"Song {i}" for i in range(n_songs)],
        'artist': [f"Artist {i % 10}" for i in range(n_songs)]
    })
    
    # Create recommender
    recommender = FAISSRecommender(
        music_features=music_features,
        music_metadata=music_metadata,
        use_gpu=False  # Test on CPU first
    )
    
    # Test search
    query = np.random.rand(13).astype('float32')
    recommendations = recommender.recommend(query, k=5)
    
    print(f"\n✓ Search successful")
    print(f"  Query shape: {query.shape}")
    print(f"  Recommendations:")
    print(recommendations.head())

