"""
Professional Flask API for Loan Prediction System
Implements secure, scalable REST endpoints with comprehensive error handling.
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import sys
import traceback
import logging
from datetime import datetime
import json
from werkzeug.exceptions import BadRequest, InternalServerError
import warnings
warnings.filterwarnings('ignore')

# Add backend directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import LoanDataProcessor
from model import LoanPredictionModel

# Initialize Flask application
app = Flask(__name__,
           template_folder='../frontend',
           static_folder='../frontend')

# Enable CORS for all routes
CORS(app, origins=['*'])

# Configure safe logging that handles Unicode properly on Windows
import codecs
import locale

def safe_encode(text):
    """Safely encode text to avoid Unicode errors on Windows console."""
    if isinstance(text, str):
        try:
            # Try to encode with the console encoding
            text.encode(locale.getpreferredencoding())
            return text
        except UnicodeEncodeError:
            # Replace problematic Unicode characters with safe alternatives
            replacements = {
                '🚀': '[STARTUP]',
                '✅': '[SUCCESS]',
                '📊': '[INFO]',
                '🌐': '[SERVER]',
                '📍': '[ENDPOINTS]',
                '❌': '[ERROR]',
                '🔧': '[CONFIG]',
                '🎯': '[TARGET]',
                '📈': '[METRICS]',
                '🏋️': '[TRAINING]',
                '💾': '[SAVE]',
                '📋': '[REPORT]',
                '🔢': '[DATA]',
                '🎉': '[COMPLETE]'
            }
            for emoji, replacement in replacements.items():
                text = text.replace(emoji, replacement)
            return text
    return str(text)

class SafeStreamHandler(logging.StreamHandler):
    """Stream handler that safely encodes Unicode characters."""
    def emit(self, record):
        try:
            msg = self.format(record)
            msg = safe_encode(msg)
            stream = self.stream
            stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

# Remove existing handlers to avoid duplicates
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('loan_prediction_api.log', encoding='utf-8'),
        SafeStreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Override the built-in print function to handle Unicode safely
import builtins
_original_print = builtins.print

def safe_print(*args, **kwargs):
    """Safe print function that handles Unicode characters."""
    try:
        # Convert all arguments to safe strings
        safe_args = [safe_encode(str(arg)) for arg in args]
        _original_print(*safe_args, **kwargs)
    except Exception:
        # Fallback to original print with error handling
        try:
            _original_print(*args, **kwargs)
        except UnicodeEncodeError:
            # Last resort: print without problematic characters
            safe_args = [str(arg).encode('ascii', 'ignore').decode('ascii') for arg in args]
            _original_print(*safe_args, **kwargs)

# Replace the built-in print function
builtins.print = safe_print

# Global variables for model and preprocessor
model = None
preprocessor = None
model_info = {}

class APIError(Exception):
    """Custom exception for API errors."""
    def __init__(self, message, status_code=400, payload=None):
        super().__init__()
        self.message = message
        self.status_code = status_code
        self.payload = payload

@app.errorhandler(APIError)
def handle_api_error(error):
    """Handle custom API errors."""
    response = {'error': error.message}
    if error.payload:
        response.update(error.payload)
    return jsonify(response), error.status_code

@app.errorhandler(400)
def bad_request(error):
    """Handle bad request errors."""
    return jsonify({'error': 'Bad request', 'message': str(error)}), 400

@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

def load_or_train_model():
    """
    Load existing model or train a new one if none exists.
    """
    global model, preprocessor, model_info

    model_path = os.path.join(os.path.dirname(__file__), 'trained_model.pkl')
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'loan_dataset.csv')

    try:
        # Try to load existing model
        if os.path.exists(model_path):
            logger.info("Loading existing trained model...")
            model = LoanPredictionModel()
            if model.load_model(model_path):
                logger.info("[SUCCESS] Model loaded successfully")

                # Also load preprocessor for feature names
                preprocessor = LoanDataProcessor()
                if os.path.exists(data_path):
                    preprocessor.run_complete_pipeline(data_path)

                model_info = model.get_model_info()
                return True

        # Train new model if none exists
        logger.info("No existing model found. Training new model...")

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at: {data_path}")

        # Initialize preprocessor and process data
        preprocessor = LoanDataProcessor()
        X_train, X_test, y_train, y_test, encoders = preprocessor.run_complete_pipeline(data_path)

        # Initialize and train model
        model = LoanPredictionModel(model_type='decision_tree')

        # Hyperparameter tuning
        logger.info("Performing hyperparameter optimization...")
        model.hyperparameter_tuning(X_train, y_train, cv_folds=3, n_jobs=1)  # Reduced for faster startup

        # Train model
        model.train_model(X_train, y_train, preprocessor.feature_columns, encoders)

        # Evaluate model
        metrics, _, _ = model.evaluate_model(X_test, y_test)

        # Save model
        model.save_model(model_path)

        model_info = model.get_model_info()
        logger.info(f"[SUCCESS] Model trained successfully with accuracy: {metrics.get('accuracy', 0):.4f}")

        return True

    except Exception as e:
        logger.error(f"[ERROR] Error loading/training model: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def validate_input_data(data):
    """
    Validate and preprocess input data for prediction.
    """
    required_fields = [
        'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
        'Loan_Amount_Term', 'Credit_History', 'Property_Area'
    ]

    errors = []

    # Check required fields
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")

    if errors:
        raise APIError("Validation errors", 400, {'validation_errors': errors})

    # Validate data types and ranges
    try:
        # Numerical fields validation
        numerical_fields = {
            'ApplicantIncome': (0, 100000),
            'CoapplicantIncome': (0, 100000),
            'LoanAmount': (0, 10000),
            'Loan_Amount_Term': (1, 500),
            'Credit_History': (0, 1)
        }

        for field, (min_val, max_val) in numerical_fields.items():
            if field in data:
                try:
                    value = float(data[field])
                    if not (min_val <= value <= max_val):
                        errors.append(f"{field} must be between {min_val} and {max_val}")
                    data[field] = value
                except (ValueError, TypeError):
                    errors.append(f"{field} must be a valid number")

        # Categorical fields validation
        categorical_validations = {
            'Gender': ['Male', 'Female'],
            'Married': ['Yes', 'No'],
            'Dependents': ['0', '1', '2', '3+'],
            'Education': ['Graduate', 'Not Graduate'],
            'Self_Employed': ['Yes', 'No'],
            'Property_Area': ['Urban', 'Semiurban', 'Rural']
        }

        for field, valid_values in categorical_validations.items():
            if field in data and data[field] not in valid_values:
                errors.append(f"{field} must be one of: {', '.join(valid_values)}")

        if errors:
            raise APIError("Validation errors", 400, {'validation_errors': errors})

        return data

    except APIError:
        raise
    except Exception as e:
        raise APIError(f"Data validation error: {str(e)}")

def preprocess_for_prediction(input_data):
    """
    Preprocess input data to match model training format.
    """
    try:
        # Create DataFrame
        df = pd.DataFrame([input_data])

        # Handle Dependents
        df['Dependents'] = df['Dependents'].replace('3+', '3').astype(int)

        # Create derived features (matching training preprocessing)
        df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
        df['Income_Loan_Ratio'] = df['Total_Income'] / (df['LoanAmount'] + 1)
        df['Loan_Per_Month'] = df['LoanAmount'] / (df['Loan_Amount_Term'] / 12)
        df['Income_Per_Dependent'] = df['Total_Income'] / (df['Dependents'] + 1)
        df['High_Income'] = (df['Total_Income'] > df['Total_Income'].median()).astype(int)
        df['High_Loan_Amount'] = (df['LoanAmount'] > df['LoanAmount'].median()).astype(int)
        df['Log_LoanAmount'] = np.log1p(df['LoanAmount'])
        df['Log_Total_Income'] = np.log1p(df['Total_Income'])

        # Encode categorical variables using stored encoders
        categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']

        for col in categorical_cols:
            if col in model.label_encoders:
                encoder = model.label_encoders[col]
                try:
                    df[col] = encoder.transform(df[col])
                except ValueError as e:
                    # Handle unseen categories
                    logger.warning(f"Unseen category in {col}: {df[col].iloc[0]}")
                    df[col] = 0  # Default to first class

        # Select features in the same order as training
        if model.feature_names:
            feature_data = df[model.feature_names].values[0]
        else:
            # Fallback feature order
            feature_order = [
                'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                'Loan_Amount_Term', 'Credit_History', 'Property_Area',
                'Total_Income', 'Income_Loan_Ratio', 'Loan_Per_Month',
                'Income_Per_Dependent', 'High_Income', 'High_Loan_Amount',
                'Log_LoanAmount', 'Log_Total_Income'
            ]
            feature_data = df[feature_order].values[0]

        return feature_data

    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        raise APIError(f"Preprocessing error: {str(e)}")

# API Routes

@app.route('/')
def index():
    """Serve the main frontend interface."""
    return send_from_directory('../frontend', 'index.html')

@app.route('/favicon.ico')
def favicon():
    """Serve the favicon."""
    try:
        # Get the absolute path to the frontend directory
        frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'frontend')
        favicon_path = os.path.join(frontend_dir, 'favicon.ico')

        logger.info(f"[DEBUG] Looking for favicon at: {favicon_path}")
        logger.info(f"[DEBUG] Favicon exists: {os.path.exists(favicon_path)}")

        if os.path.exists(favicon_path):
            return send_from_directory(frontend_dir, 'favicon.ico')
        else:
            logger.error(f"[ERROR] Favicon not found at: {favicon_path}")
            from flask import abort
            abort(404)
    except Exception as e:
        logger.error(f"[ERROR] Failed to serve favicon: {str(e)}")
        # Return a 404 response if favicon can't be found
        from flask import abort
        abort(404)

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files from frontend directory."""
    try:
        # Get the absolute path to the frontend directory
        frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'frontend')
        return send_from_directory(frontend_dir, filename)
    except Exception as e:
        logger.error(f"[ERROR] Failed to serve static file {filename}: {str(e)}")
        from flask import abort
        abort(404)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    global model

    status = {
        'status': 'healthy' if model is not None else 'unhealthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None,
        'version': '1.0.0'
    }

    return jsonify(status), 200 if model is not None else 503

@app.route('/api/predict', methods=['POST'])
def predict_loan():
    """
    Main prediction endpoint for loan applications.

    Expected JSON payload:
    {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "1",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 5849,
        "CoapplicantIncome": 0,
        "LoanAmount": 128,
        "Loan_Amount_Term": 360,
        "Credit_History": 1,
        "Property_Area": "Urban"
    }
    """
    global model

    if model is None:
        raise APIError("Model not loaded", 503)

    try:
        # Get JSON data
        if not request.is_json:
            raise APIError("Request must contain JSON data")

        input_data = request.get_json()

        if not input_data:
            raise APIError("Empty request body")

        logger.info(f"Prediction request received: {json.dumps(input_data, indent=2)}")

        # Validate input
        validated_data = validate_input_data(input_data)

        # Preprocess for prediction
        processed_features = preprocess_for_prediction(validated_data)

        # Make prediction
        prediction_result = model.predict_single(processed_features, return_probability=True)

        # Add request timestamp and processing info
        response = {
            'request_id': datetime.now().strftime('%Y%m%d_%H%M%S_%f'),
            'timestamp': datetime.now().isoformat(),
            'input_data': validated_data,
            'prediction': prediction_result,
            'model_info': {
                'model_type': model.model_type,
                'accuracy': model.training_metrics.get('accuracy', 'N/A')
            }
        }

        logger.info(f"Prediction completed: {prediction_result['prediction_label']} "
                   f"(confidence: {prediction_result.get('confidence', 'N/A'):.3f})")

        return jsonify(response), 200

    except APIError:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        raise APIError(f"Prediction failed: {str(e)}", 500)

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """
    Get comprehensive model information and performance metrics.
    """
    global model, model_info

    if model is None:
        raise APIError("Model not loaded", 503)

    try:
        response = {
            'timestamp': datetime.now().isoformat(),
            'model_info': model_info,
            'api_version': '1.0.0',
            'endpoints': {
                'predict': '/api/predict',
                'model_info': '/api/model-info',
                'health': '/api/health'
            }
        }

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Model info error: {str(e)}")
        raise APIError(f"Failed to get model info: {str(e)}", 500)

@app.route('/api/feature-importance', methods=['GET'])
def get_feature_importance():
    """
    Get feature importance rankings for model interpretability.
    """
    global model

    if model is None:
        raise APIError("Model not loaded", 503)

    try:
        importance_dict = model.get_feature_importance_dict()

        response = {
            'timestamp': datetime.now().isoformat(),
            'feature_importance': importance_dict,
            'top_features': dict(list(importance_dict.items())[:10]) if importance_dict else {}
        }

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Feature importance error: {str(e)}")
        raise APIError(f"Failed to get feature importance: {str(e)}", 500)

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint for multiple loan applications.
    Expects JSON array of loan application objects.
    """
    global model

    if model is None:
        raise APIError("Model not loaded", 503)

    try:
        if not request.is_json:
            raise APIError("Request must contain JSON data")

        batch_data = request.get_json()

        if not isinstance(batch_data, list):
            raise APIError("Batch data must be an array of loan applications")

        if len(batch_data) > 100:  # Limit batch size
            raise APIError("Batch size limited to 100 applications")

        results = []
        errors = []

        for i, input_data in enumerate(batch_data):
            try:
                # Validate and preprocess
                validated_data = validate_input_data(input_data)
                processed_features = preprocess_for_prediction(validated_data)

                # Make prediction
                prediction_result = model.predict_single(processed_features, return_probability=True)

                results.append({
                    'index': i,
                    'input_data': validated_data,
                    'prediction': prediction_result
                })

            except Exception as e:
                errors.append({
                    'index': i,
                    'error': str(e),
                    'input_data': input_data
                })

        response = {
            'timestamp': datetime.now().isoformat(),
            'total_processed': len(batch_data),
            'successful_predictions': len(results),
            'errors': len(errors),
            'results': results,
            'error_details': errors
        }

        return jsonify(response), 200

    except APIError:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise APIError(f"Batch prediction failed: {str(e)}", 500)

# Application initialization
def initialize_app():
    """Initialize the Flask application."""
    logger.info("[STARTUP] Initializing Loan Prediction API...")

    # Load or train model
    if not load_or_train_model():
        logger.error("[ERROR] Failed to initialize model. API will not function properly.")
        return False

    logger.info("[SUCCESS] Loan Prediction API initialized successfully!")
    logger.info(f"[INFO] Model Info: {model_info}")

    return True

if __name__ == '__main__':
    # Initialize application
    if initialize_app():
        # Start development server
        logger.info("[SERVER] Starting development server...")
        logger.info("[INFO] API Endpoints:")
        logger.info("   GET  /api/health          - Health check")
        logger.info("   POST /api/predict         - Single prediction")
        logger.info("   GET  /api/model-info      - Model information")
        logger.info("   GET  /api/feature-importance - Feature rankings")
        logger.info("   POST /api/batch-predict   - Batch predictions")
        logger.info("   GET  /                    - Frontend interface")

        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,  # Set to False for production
            threaded=True
        )
    else:
        logger.error("[ERROR] Failed to start application")
        sys.exit(1)