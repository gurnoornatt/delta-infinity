"""
app.py - Flask API Server for MemoryMark

RESTful API for GPU memory analysis. Provides endpoints for health checks,
model listing, and running memory analysis.

Documentation:
- Flask 3.x Quickstart: https://flask.palletsprojects.com/en/stable/quickstart/
- Flask-CORS: https://flask-cors.readthedocs.io/
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from datetime import datetime
from typing import Dict, Any
import memorymark

app = Flask(__name__)

# CORS Configuration
# Allow all origins for development/hackathon
# For production, restrict to specific domains
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})


@app.route('/analyze', methods=['POST'])
def analyze() -> tuple[Dict[str, Any], int]:
    """
    Run MemoryMark analysis on a specified model.

    Request Body:
        {
            "model_name": str  # One of: bert, gpt2, resnet
        }

    Returns:
        200: {
            "status": "success",
            "data": {
                "optimal_batch_size": int,
                "waste_gb": float,
                "speedup": float,
                ...
            }
        }
        400: {"status": "error", "error": str}
        500: {"status": "error", "error": str}
    """
    try:
        # Get request data
        data = request.get_json()

        # Validate request
        if not data:
            return jsonify({
                'status': 'error',
                'error': 'Request body is required'
            }), 400

        if 'model_name' not in data:
            return jsonify({
                'status': 'error',
                'error': 'model_name is required in request body'
            }), 400

        model_name = data['model_name']

        # Validate model name
        valid_models = ['bert', 'gpt2', 'resnet']
        if model_name not in valid_models:
            return jsonify({
                'status': 'error',
                'error': f'Invalid model_name. Must be one of: {", ".join(valid_models)}'
            }), 400

        # Run analysis (this takes 30-60 seconds on real GPU)
        # WARNING: This will download models and use GPU heavily
        results = memorymark.find_optimal_batch_size(model_name)

        # Return success
        return jsonify({
            'status': 'success',
            'data': results
        }), 200

    except Exception as e:
        # Log error (in production, use proper logging)
        print(f"Analysis error: {str(e)}")
        import traceback
        traceback.print_exc()

        return jsonify({
            'status': 'error',
            'error': f'Analysis failed: {str(e)}'
        }), 500


@app.route('/health', methods=['GET'])
def health() -> tuple[Dict[str, Any], int]:
    """
    Health check endpoint.

    Returns GPU status and availability.

    Returns:
        200: {
            "status": "healthy",
            "gpu_available": bool,
            "gpu_name": str,
            "gpu_memory_total_gb": float,
            "device": str,
            "timestamp": str
        }
        500: {"status": "unhealthy", "error": str}
    """
    try:
        # Detect device
        device = memorymark.get_device()
        gpu_available = device in ['cuda', 'mps']

        # Get GPU info
        if device == 'cuda':
            props = torch.cuda.get_device_properties(0)
            gpu_name = props.name
            gpu_memory_gb = round(props.total_memory / (1024 ** 3), 1)
        elif device == 'mps':
            gpu_name = "Apple Silicon (MPS)"
            gpu_memory_gb = 18.0  # Approximate for M-series
        else:
            gpu_name = None
            gpu_memory_gb = 0.0

        return jsonify({
            'status': 'healthy',
            'gpu_available': gpu_available,
            'gpu_name': gpu_name,
            'gpu_memory_total_gb': gpu_memory_gb,
            'device': device,
            'timestamp': datetime.now().isoformat() + 'Z'
        }), 200

    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500


@app.route('/models', methods=['GET'])
def list_models() -> tuple[Dict[str, Any], int]:
    """
    List available models for analysis.

    Returns:
        200: {
            "models": [
                {
                    "id": str,
                    "name": str,
                    "description": str,
                    "type": str
                },
                ...
            ]
        }
    """
    models = [
        {
            'id': 'bert',
            'name': 'BERT Base',
            'description': 'NLP model - 110M parameters',
            'type': 'nlp',
            'huggingface_id': 'google-bert/bert-base-uncased'
        },
        {
            'id': 'gpt2',
            'name': 'GPT-2',
            'description': 'Language model - 117M parameters',
            'type': 'nlp',
            'huggingface_id': 'openai-community/gpt2'
        },
        {
            'id': 'resnet',
            'name': 'ResNet-50',
            'description': 'Vision model - 25M parameters',
            'type': 'vision',
            'huggingface_id': 'microsoft/resnet-50'
        }
    ]

    return jsonify({'models': models}), 200


@app.route('/', methods=['GET'])
def root() -> tuple[Dict[str, Any], int]:
    """
    API root endpoint.

    Returns basic API information.
    """
    return jsonify({
        'name': 'MemoryMark API',
        'version': '1.0.0',
        'description': 'GPU Memory Waste Detection and Optimization',
        'endpoints': {
            'GET /': 'API information',
            'GET /health': 'Health check and GPU status',
            'GET /models': 'List available models',
            'POST /analyze': 'Run memory analysis (requires model_name in body)'
        },
        'documentation': 'https://github.com/yourusername/memorymark'
    }), 200


@app.errorhandler(404)
def not_found(error) -> tuple[Dict[str, Any], int]:
    """Handle 404 errors."""
    return jsonify({
        'status': 'error',
        'error': 'Endpoint not found',
        'message': 'Please check the API documentation for valid endpoints'
    }), 404


@app.errorhandler(405)
def method_not_allowed(error) -> tuple[Dict[str, Any], int]:
    """Handle 405 Method Not Allowed errors."""
    return jsonify({
        'status': 'error',
        'error': 'Method not allowed',
        'message': 'Please check the API documentation for allowed HTTP methods'
    }), 405


@app.errorhandler(500)
def internal_error(error) -> tuple[Dict[str, Any], int]:
    """Handle 500 Internal Server errors."""
    return jsonify({
        'status': 'error',
        'error': 'Internal server error',
        'message': 'An unexpected error occurred. Please try again later.'
    }), 500


if __name__ == '__main__':
    print("="*60)
    print("■ Starting MemoryMark API server...")
    print("="*60)
    print(f"Device: {memorymark.get_device()}")
    print(f"Running on: http://0.0.0.0:5001")
    print(f"Health check: http://0.0.0.0:5001/health")
    print(f"Models list: http://0.0.0.0:5001/models")
    print("="*60)
    print("NOTE: /analyze endpoint will download models and use GPU heavily")
    print("      Test on Lambda Labs GPU, not your local Mac!")
    print("="*60)

    # Run Flask app
    # debug=False for production
    # host='0.0.0.0' allows external connections
    # port=5001 (5000 often used by AirPlay on macOS)
    app.run(host='0.0.0.0', port=5001, debug=False)
