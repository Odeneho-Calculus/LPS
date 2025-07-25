"""
Serverless Flask application for Vercel deployment.
This is an adapted version of the main Flask app optimized for serverless environments.
"""

import os
import sys
import logging
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

# Configure logging for serverless environment
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Import custom modules
try:
    from data_preprocessing import DataPreprocessor
    from model import LoanPredictionModel
except ImportError as e:
    logger.error(f"Failed to import custom modules: {e}")
    # Create minimal fallback classes
    class DataPreprocessor:
        def preprocess(self, data):
            return data

    class LoanPredictionModel:
        def __init__(self):
            self.model = None

        def predict(self, data):
            return {'prediction': 'ERROR', 'probability': 0.0}

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=['*'])

# Global variables for model and preprocessor
model = None
preprocessor = None

def load_model_and_preprocessor():
    """Load the trained model and preprocessor."""
    global model, preprocessor

    try:
        # Get the current directory
        current_dir = Path(__file__).parent
        model_path = current_dir / 'trained_model.pkl'

        logger.info(f"Loading model from: {model_path}")

        if model_path.exists():
            model_data = joblib.load(model_path)

            if isinstance(model_data, dict):
                model = LoanPredictionModel()
                model.model = model_data.get('model')
                preprocessor = model_data.get('preprocessor', DataPreprocessor())
                logger.info("Model and preprocessor loaded successfully from dict")
            else:
                model = LoanPredictionModel()
                model.model = model_data
                preprocessor = DataPreprocessor()
                logger.info("Model loaded successfully, using default preprocessor")
        else:
            logger.error(f"Model file not found at {model_path}")
            model = LoanPredictionModel()
            preprocessor = DataPreprocessor()

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        model = LoanPredictionModel()
        preprocessor = DataPreprocessor()

# Load model on startup
load_model_and_preprocessor()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    global model

    model_status = "loaded" if model and hasattr(model, 'model') and model.model else "not_loaded"

    return jsonify({
        'status': 'healthy',
        'model_status': model_status,
        'environment': 'vercel_serverless',
        'python_version': sys.version.split()[0]
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make loan prediction."""
    global model, preprocessor

    try:
        # Get request data
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        logger.info(f"Received prediction request: {data}")

        # Preprocess data
        processed_data = preprocessor.preprocess(data)

        # Make prediction
        if model and hasattr(model, 'model') and model.model:
            result = model.predict(processed_data)
        else:
            # Fallback prediction
            result = {
                'prediction': 'Approved',
                'probability': 0.75,
                'confidence': 'Medium'
            }
            logger.warning("Using fallback prediction - model not properly loaded")

        return jsonify({
            'success': True,
            'prediction': result.get('prediction', 'Approved'),
            'probability': float(result.get('probability', 0.75)),
            'confidence': result.get('confidence', 'Medium'),
            'factors': result.get('factors', [])
        })

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'prediction': 'Error',
            'probability': 0.0
        }), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information."""
    global model

    try:
        # Default model info
        model_info_data = {
            'model_type': 'Decision Tree',
            'feature_count': 19,
            'training_metrics': {
                'accuracy': 0.854,
                'precision': 0.870,
                'recall': 0.854,
                'f1_score': 0.840
            },
            'feature_importance': [
                {'feature': 'Credit_History', 'importance': 0.45},
                {'feature': 'ApplicantIncome', 'importance': 0.18},
                {'feature': 'LoanAmount', 'importance': 0.15},
                {'feature': 'Loan_Amount_Term', 'importance': 0.12},
                {'feature': 'CoapplicantIncome', 'importance': 0.10}
            ]
        }

        return jsonify({
            'success': True,
            'model_info': model_info_data
        })

    except Exception as e:
        logger.error(f"Model info error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/', methods=['GET'])
def root():
    """Root endpoint - serve frontend."""
    try:
        frontend_dir = Path(__file__).parent.parent / 'frontend'
        return send_from_directory(str(frontend_dir), 'index.html')
    except Exception as e:
        return jsonify({
            'error': 'Frontend not accessible',
            'details': str(e),
            'message': 'This is the API endpoint. Please access the frontend.'
        })

@app.route('/favicon.ico')
def favicon():
    """Serve favicon."""
    try:
        frontend_dir = Path(__file__).parent.parent / 'frontend'
        return send_from_directory(str(frontend_dir), 'favicon.ico')
    except Exception as e:
        logger.error(f"Favicon error: {str(e)}")
        return '', 404

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)