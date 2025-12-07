"""
Flask Backend for DyLuMo - Image to Music Recommendation
Main application entry point
"""
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import io
import logging
from pathlib import Path

import config
from inference import ModelInference

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')
CORS(app)  # Enable CORS for frontend requests

# Initialize inference system
inference = ModelInference(config)

# Load model at startup
logger.info("="*60)
logger.info("DYLUMO BACKEND - STARTING UP")
logger.info("="*60)

try:
    inference.load_all()
    logger.info("✓ Backend ready!")
except Exception as e:
    logger.error(f"✗ Failed to load model: {e}")
    logger.error("Server will start but /recommend endpoint will fail")

logger.info("="*60)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS


def validate_image(file):
    """Validate uploaded image"""
    if not file:
        return False, "No file provided"
    
    if file.filename == '':
        return False, "No file selected"
    
    if not allowed_file(file.filename):
        return False, f"Invalid file type. Allowed: {', '.join(config.ALLOWED_EXTENSIONS)}"
    
    # Check file size
    file.seek(0, 2)  # Seek to end
    size = file.tell()
    file.seek(0)  # Reset
    
    if size > config.MAX_IMAGE_SIZE:
        return False, f"File too large. Max size: {config.MAX_IMAGE_SIZE / 1024 / 1024:.1f}MB"
    
    return True, "OK"


# ============================================================
# API ROUTES
# ============================================================

@app.route('/')
def home():
    """Homepage - serve frontend"""
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': inference.is_loaded(),
        'stats': inference.get_stats()
    })


@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Main endpoint: Get song recommendations for an image
    
    Request:
        - image: Image file (multipart/form-data)
        - top_k: Number of recommendations (optional, default 10)
    
    Response:
        JSON with emotion prediction and song recommendations
    """
    try:
        # Check if model is loaded
        if not inference.is_loaded():
            return jsonify({
                'status': 'error',
                'message': 'Model not loaded. Server starting up...'
            }), 503
        
        # Validate request
        if 'image' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No image file in request'
            }), 400
        
        file = request.files['image']
        
        # Validate file
        is_valid, message = validate_image(file)
        if not is_valid:
            return jsonify({
                'status': 'error',
                'message': message
            }), 400
        
        # Get top_k parameter
        top_k = request.form.get('top_k', config.DEFAULT_TOP_K, type=int)
        top_k = min(max(1, top_k), config.MAX_TOP_K)  # Clamp between 1 and MAX_TOP_K
        
        # Load image
        try:
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Invalid image file: {str(e)}'
            }), 400
        
        # Get predictions
        logger.info(f"Processing image: {file.filename} (top_k={top_k})")
        result = inference.predict(image, top_k=top_k)
        
        if result['status'] == 'error':
            return jsonify(result), 500
        
        logger.info(f"✓ Prediction complete: {result['emotion']['predicted_emotion']} "
                   f"({result['emotion']['confidence']:.1%})")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in /recommend: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Internal server error',
            'details': str(e)
        }), 500


@app.route('/emotions', methods=['GET'])
def get_emotions():
    """Get list of supported emotions"""
    if not inference.is_loaded():
        return jsonify({
            'status': 'error',
            'message': 'Model not loaded'
        }), 503
    
    return jsonify({
        'status': 'success',
        'emotions': inference.deployment_config['emotions']
    })


@app.route('/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    return jsonify(inference.get_stats())


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'status': 'error',
        'message': 'Internal server error'
    }), 500


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    logger.info(f"Starting Flask server on {config.HOST}:{config.PORT}")
    logger.info(f"Debug mode: {config.DEBUG}")
    logger.info("Access at: http://localhost:5000")
    logger.info("API Documentation:")
    logger.info("  GET  /health      - Health check")
    logger.info("  POST /recommend   - Get recommendations")
    logger.info("  GET  /emotions    - List emotions")
    logger.info("  GET  /stats       - System stats")
    logger.info("="*60)
    
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG
    )

