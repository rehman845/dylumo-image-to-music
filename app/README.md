# DyLuMo Backend

Flask backend for Image-to-Music recommendation system.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python app.py
```

## API Endpoints

### `GET /health`
Health check endpoint
- Returns server status and model loading state

### `POST /recommend`
Get music recommendations for an image

**Request:**
- Content-Type: `multipart/form-data`
- Parameters:
  - `image`: Image file (required)
  - `top_k`: Number of recommendations (optional, default 10)

**Response:**
```json
{
  "status": "success",
  "emotion": {
    "predicted_emotion": "sadness",
    "confidence": 0.93,
    "all_probabilities": {...}
  },
  "recommendations": [
    {
      "rank": 1,
      "track_name": "Song Name",
      "artist_name": "Artist",
      "emotion": "sadness",
      "similarity_score": 0.983
    }
  ],
  "inference_time": "1.8s"
}
```

### `GET /emotions`
Get list of supported emotions

### `GET /stats`
Get system statistics

## Access

- **API**: http://localhost:5000
- **Frontend**: http://localhost:5000/ (served by Flask)

## Project Structure

```
backend/
├── app.py              # Main Flask application
├── inference.py        # Model inference logic
├── config.py           # Configuration
├── requirements.txt    # Dependencies
├── templates/
│   └── index.html      # Frontend HTML
└── static/
    └── script.js       # Frontend JavaScript
```

