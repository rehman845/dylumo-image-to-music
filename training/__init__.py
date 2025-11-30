"""
DYLUMO Training Pipeline

Scripts:
    01_prepare_spotify.py - Prepare Spotify dataset
    02_prepare_emid.py    - Prepare EMID images/emotions
    03_extract_features.py - Extract CLIP features from images
    04_create_pairs.py    - Create training pairs
    05_train.py           - Train FastMLP model
"""

# Audio features used for music representation
AUDIO_FEATURES = [
    'danceability',      # 0-1
    'energy',            # 0-1
    'key',               # 0-11 -> normalized
    'loudness',          # -60 to 0 -> normalized
    'mode',              # 0 or 1
    'speechiness',       # 0-1
    'acousticness',      # 0-1
    'instrumentalness',  # 0-1
    'liveness',          # 0-1
    'valence',           # 0-1
    'tempo',             # 0-250 -> normalized
    'duration_ms',       # -> normalized
    'time_signature'     # 3-7 -> normalized
]

# Emotion categories from EMID dataset
EMOTION_CATEGORIES = [
    'anger',
    'amusement', 
    'fear',
    'sadness',
    'excitement',
    'awe',
    'contentment'
]

