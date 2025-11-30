"""
Music Recommender using FAISS for fast similarity search.

Takes predicted music features from FastMLP and finds the most similar
songs from the Spotify dataset using FAISS index.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Lazy import for FAISS
_faiss = None


def get_faiss():
    """Lazy load FAISS."""
    global _faiss
    if _faiss is None:
        try:
            import faiss
            _faiss = faiss
        except ImportError:
            raise ImportError("Please install faiss-cpu: pip install faiss-cpu")
    return _faiss


class MusicRecommender:
    """
    FAISS-based music recommendation system.
    
    Uses L2 distance to find songs with similar audio features
    to the predicted features from FastMLP.
    """
    
    def __init__(self):
        """Initialize the recommender."""
        self.index = None
        self.metadata = None
        self.feature_dim = None
        self.is_built = False
    
    def build_index(
        self,
        features: np.ndarray,
        metadata: pd.DataFrame,
        use_gpu: bool = False
    ):
        """
        Build FAISS index from music features.
        
        Args:
            features: Music feature matrix of shape (n_songs, feature_dim)
            metadata: DataFrame with song metadata (track_name, artist_name, etc.)
            use_gpu: Whether to use GPU for FAISS (requires faiss-gpu)
        """
        faiss = get_faiss()
        
        # Ensure features are float32 and contiguous
        features = np.ascontiguousarray(features.astype(np.float32))
        n_songs, feature_dim = features.shape
        
        print(f"[INFO] Building FAISS index for {n_songs:,} songs...")
        
        # Normalize features for cosine similarity
        faiss.normalize_L2(features)
        
        # Create index
        # Using IndexFlatIP for inner product (cosine similarity after normalization)
        self.index = faiss.IndexFlatIP(feature_dim)
        
        # Add vectors to index
        self.index.add(features)
        
        self.metadata = metadata.reset_index(drop=True)
        self.feature_dim = feature_dim
        self.is_built = True
        
        print(f"[OK] Index built with {self.index.ntotal:,} vectors")
    
    def search(
        self,
        query_features: np.ndarray,
        k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar songs.
        
        Args:
            query_features: Query feature vector(s) of shape (n_queries, feature_dim)
            k: Number of results to return
            
        Returns:
            Tuple of (distances, indices) arrays
        """
        if not self.is_built:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        faiss = get_faiss()
        
        # Ensure query is 2D and float32
        if query_features.ndim == 1:
            query_features = query_features.reshape(1, -1)
        query_features = np.ascontiguousarray(query_features.astype(np.float32))
        
        # Normalize query
        faiss.normalize_L2(query_features)
        
        # Search
        distances, indices = self.index.search(query_features, k)
        
        return distances, indices
    
    def recommend(
        self,
        predicted_features: np.ndarray,
        k: int = 10
    ) -> List[Dict]:
        """
        Get song recommendations based on predicted music features.
        
        Args:
            predicted_features: Predicted music features from FastMLP
            k: Number of recommendations to return
            
        Returns:
            List of dictionaries with song info and similarity scores
        """
        distances, indices = self.search(predicted_features, k)
        
        recommendations = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx >= 0 and idx < len(self.metadata):
                song_info = self.metadata.iloc[idx].to_dict()
                song_info['similarity_score'] = float(dist)
                song_info['rank'] = i + 1
                recommendations.append(song_info)
        
        return recommendations
    
    def save(self, path: str):
        """
        Save the FAISS index and metadata.
        
        Args:
            path: Base path for saving (will create .index and .meta files)
        """
        if not self.is_built:
            raise RuntimeError("Index not built. Nothing to save.")
        
        faiss = get_faiss()
        path = Path(path)
        
        # Save FAISS index
        index_path = path.with_suffix('.index')
        faiss.write_index(self.index, str(index_path))
        
        # Save metadata
        meta_path = path.with_suffix('.parquet')
        self.metadata.to_parquet(meta_path)
        
        print(f"[OK] Index saved to {index_path}")
        print(f"[OK] Metadata saved to {meta_path}")
    
    def load(self, path: str):
        """
        Load the FAISS index and metadata.
        
        Args:
            path: Base path for loading
        """
        faiss = get_faiss()
        path = Path(path)
        
        # Load FAISS index
        index_path = path.with_suffix('.index')
        self.index = faiss.read_index(str(index_path))
        
        # Load metadata
        meta_path = path.with_suffix('.parquet')
        self.metadata = pd.read_parquet(meta_path)
        
        self.feature_dim = self.index.d
        self.is_built = True
        
        print(f"[OK] Loaded index with {self.index.ntotal:,} vectors")


if __name__ == "__main__":
    # Test the recommender
    print("Testing MusicRecommender...")
    
    # Create dummy data
    n_songs = 1000
    feature_dim = 13
    
    features = np.random.randn(n_songs, feature_dim).astype(np.float32)
    metadata = pd.DataFrame({
        'track_name': [f'Song {i}' for i in range(n_songs)],
        'artist_name': [f'Artist {i % 100}' for i in range(n_songs)],
        'track_id': [f'id_{i}' for i in range(n_songs)]
    })
    
    # Build and test
    recommender = MusicRecommender()
    recommender.build_index(features, metadata)
    
    # Query
    query = np.random.randn(1, feature_dim).astype(np.float32)
    recommendations = recommender.recommend(query, k=5)
    
    print("\nTop 5 recommendations:")
    for rec in recommendations:
        print(f"  {rec['rank']}. {rec['track_name']} - {rec['artist_name']} "
              f"(score: {rec['similarity_score']:.4f})")

