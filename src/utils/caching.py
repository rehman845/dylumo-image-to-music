"""
Caching utilities for fast feature loading
"""

import pickle
import json
from pathlib import Path
from typing import Any, Optional
import numpy as np


class CacheManager:
    """
    Simple cache manager for storing and loading features.
    """
    
    def __init__(self, cache_dir: str = "data/cache"):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, key: str, data: Any, format: str = "pkl"):
        """
        Save data to cache.
        
        Args:
            key: Cache key (filename without extension)
            data: Data to cache
            format: "pkl" for pickle, "json" for JSON, "npy" for numpy
        """
        if format == "pkl":
            filepath = self.cache_dir / f"{key}.pkl"
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        elif format == "json":
            filepath = self.cache_dir / f"{key}.json"
            with open(filepath, 'w') as f:
                json.dump(data, f)
        elif format == "npy":
            filepath = self.cache_dir / f"{key}.npy"
            np.save(filepath, data)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def load(self, key: str, format: str = "pkl") -> Optional[Any]:
        """
        Load data from cache.
        
        Args:
            key: Cache key
            format: "pkl", "json", or "npy"
            
        Returns:
            Cached data or None if not found
        """
        if format == "pkl":
            filepath = self.cache_dir / f"{key}.pkl"
        elif format == "json":
            filepath = self.cache_dir / f"{key}.json"
        elif format == "npy":
            filepath = self.cache_dir / f"{key}.npy"
        else:
            raise ValueError(f"Unknown format: {format}")
        
        if not filepath.exists():
            return None
        
        try:
            if format == "pkl":
                with open(filepath, 'rb') as f:
                    return pickle.load(f)
            elif format == "json":
                with open(filepath, 'r') as f:
                    return json.load(f)
            elif format == "npy":
                return np.load(filepath)
        except Exception as e:
            print(f"Error loading cache {key}: {e}")
            return None
    
    def exists(self, key: str, format: str = "pkl") -> bool:
        """Check if cache entry exists."""
        if format == "pkl":
            filepath = self.cache_dir / f"{key}.pkl"
        elif format == "json":
            filepath = self.cache_dir / f"{key}.json"
        elif format == "npy":
            filepath = self.cache_dir / f"{key}.npy"
        else:
            raise ValueError(f"Unknown format: {format}")
        
        return filepath.exists()

