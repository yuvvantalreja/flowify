import sys
import os

# Add the application directory to the Python path
path = os.path.dirname(os.path.abspath(__file__))
if path not in sys.path:
    sys.path.append(path)

# Set up NLTK data path (important for PythonAnywhere)
import nltk
nltk.data.path.append(os.path.join(path, "nltk_data"))

# Download NLTK data if needed (only run this once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=os.path.join(path, "nltk_data"))
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', download_dir=os.path.join(path, "nltk_data"))

# Import the Flask application
from app import app as application 