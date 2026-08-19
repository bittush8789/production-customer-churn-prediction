import os
import sys
from flask import Flask, send_from_directory, jsonify
from flask_cors import CORS

# Add root directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.routes import api_bp

def create_app() -> Flask:
    frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'frontend')
    
    app = Flask(__name__, static_folder=frontend_dir, static_url_path='')
    CORS(app)
    
    # Register API blueprint under /api and also root aliases
    app.register_blueprint(api_bp, url_prefix='/api')
    app.register_blueprint(api_bp, name='api_root') # Exposes /predict, /health directly
    
    @app.route('/')
    def serve_index():
        if os.path.exists(os.path.join(frontend_dir, 'index.html')):
            return send_from_directory(frontend_dir, 'index.html')
        return jsonify({
            "service": "Customer Churn Prediction API",
            "status": "online",
            "endpoints": [
                "GET /",
                "POST /predict",
                "GET /api/stats",
                "GET /api/sample",
                "GET /health"
            ]
        })
        
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Not Found", "message": "Resource not found"}), 404
        
    @app.errorhandler(500)
    def server_error(e):
        return jsonify({"error": "Internal Server Error", "message": "An unexpected error occurred"}), 500
        
    return app

app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"[*] Customer Churn Prediction Server running on http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)
