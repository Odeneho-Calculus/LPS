"""
Vercel serverless function entry point for the Loan Prediction API.
This file imports the Flask app optimized for serverless deployment.
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import the Flask app from the same directory
try:
    from app import app

    # Configure for serverless deployment
    app.config['PROPAGATE_EXCEPTIONS'] = True
    app.config['JSON_SORT_KEYS'] = False

    # For development/testing
    if __name__ == '__main__':
        app.run(debug=False, host='0.0.0.0', port=5000)

except ImportError as e:
    # Fallback minimal Flask app if import fails
    from flask import Flask, jsonify

    app = Flask(__name__)

    @app.route('/api/health')
    def health():
        return jsonify({
            'status': 'error',
            'message': f'Import error: {str(e)}',
            'deployment': 'vercel-fallback'
        })

    @app.route('/')
    def root():
        return jsonify({
            'error': 'Application failed to initialize',
            'details': str(e)
        })

# Export the app for Vercel
handler = app