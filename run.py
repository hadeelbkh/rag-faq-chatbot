import os
import sys

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import and run the Flask app
from src.api.app import app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 