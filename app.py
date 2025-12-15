# app.py - Main Flask Application
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Disease classes (example for tomato diseases)
DISEASE_CLASSES = {
    0: 'Healthy',
    1: 'Early Blight',
    2: 'Late Blight',
    3: 'Leaf Mold',
    4: 'Septoria Leaf Spot',
    5: 'Spider Mites',
    6: 'Target Spot',
    7: 'Yellow Leaf Curl Virus',
    8: 'Mosaic Virus',
    9: 'Bacterial Spot'
}

# Treatment recommendations
TREATMENTS = {
    'Healthy': {
        'treatment': 'No treatment needed',
        'prevention': 'Continue good agricultural practices, monitor regularly',
        'severity': 'None'
    },
    'Early Blight': {
        'treatment': 'Apply fungicides containing chlorothalonil or copper-based compounds. Remove affected leaves.',
        'prevention': 'Rotate crops, ensure proper spacing, avoid overhead irrigation',
        'severity': 'Medium'
    },
    'Late Blight': {
        'treatment': 'Apply fungicides immediately (chlorothalonil, mancozeb). Remove and destroy infected plants.',
        'prevention': 'Use resistant varieties, improve air circulation, avoid wet foliage',
        'severity': 'High'
    },
    'Leaf Mold': {
        'treatment': 'Apply fungicides, improve ventilation, reduce humidity',
        'prevention': 'Maintain low humidity, prune for air circulation, use resistant varieties',
        'severity': 'Medium'
    },
    'Septoria Leaf Spot': {
        'treatment': 'Remove infected leaves, apply fungicides (chlorothalonil, mancozeb)',
        'prevention': 'Crop rotation, mulching, avoid overhead watering',
        'severity': 'Medium'
    },
    'Spider Mites': {
        'treatment': 'Apply insecticidal soap or neem oil. Use miticides if severe.',
        'prevention': 'Regular monitoring, maintain humidity, natural predators',
        'severity': 'Medium'
    },
    'Target Spot': {
        'treatment': 'Apply fungicides, remove infected plant parts',
        'prevention': 'Good air circulation, avoid overhead irrigation, crop rotation',
        'severity': 'Medium'
    },
    'Yellow Leaf Curl Virus': {
        'treatment': 'Remove infected plants, control whitefly vectors with insecticides',
        'prevention': 'Use resistant varieties, control whitefly populations, use reflective mulches',
        'severity': 'High'
    },
    'Mosaic Virus': {
        'treatment': 'No cure available. Remove and destroy infected plants.',
        'prevention': 'Use virus-free seeds, control aphid vectors, remove weeds',
        'severity': 'High'
    },
    'Bacterial Spot': {
        'treatment': 'Apply copper-based bactericides, remove infected leaves',
        'prevention': 'Use disease-free seeds, crop rotation, avoid overhead irrigation',
        'severity': 'Medium'
    }
}

# Load or create a simple model (for demonstration)
# In production, you would load a pre-trained model
def create_simple_model():
    """
    Create a simple CNN model for demonstration
    In production, replace this with a trained model loaded from file:
    model = tf.keras.models.load_model('crop_disease_model.h5')
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(DISEASE_CLASSES), activation='softmax')
    ])
    return model

# Initialize model
try:
    # Try to load pre-trained model
    model = tf.keras.models.load_model('models/crop_disease_model.h5')
    print("Loaded pre-trained model")
except:
    # Create simple model for demonstration
    model = create_simple_model()
    print("Created demonstration model")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Resize image to model input size
    img = image.resize((224, 224))
    # Convert to array
    img_array = np.array(img)
    # Normalize pixel values
    img_array = img_array / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if file is allowed
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG'}), 400
        
        # Read and process image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        # Get disease name
        disease_name = DISEASE_CLASSES.get(predicted_class, 'Unknown')
        
        # Get treatment information
        treatment_info = TREATMENTS.get(disease_name, {
            'treatment': 'Consult agricultural expert',
            'prevention': 'General good practices',
            'severity': 'Unknown'
        })
        
        # Prepare response
        response = {
            'success': True,
            'disease': disease_name,
            'confidence': round(confidence * 100, 2),
            'treatment': treatment_info['treatment'],
            'prevention': treatment_info['prevention'],
            'severity': treatment_info['severity'],
            'all_predictions': {
                DISEASE_CLASSES[i]: round(float(predictions[0][i]) * 100, 2)
                for i in range(len(DISEASE_CLASSES))
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/diseases', methods=['GET'])
def get_diseases():
    """Get list of all detectable diseases"""
    return jsonify({
        'diseases': list(DISEASE_CLASSES.values()),
        'count': len(DISEASE_CLASSES)
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)