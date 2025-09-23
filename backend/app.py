from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import json
from database import PredictionDatabase

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables
brain_tumor_model = None
db = PredictionDatabase()

# Rate limiting
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)
limiter.init_app(app)

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model():
    """Load the trained brain tumor model"""
    global brain_tumor_model
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        model_path = os.path.join(project_root, 'trained_models', 'brain_tumor_model.keras')
        
        print(f"Looking for model at: {model_path}")
        
        if os.path.exists(model_path):
            brain_tumor_model = tf.keras.models.load_model(model_path)
            print("✓ Brain tumor model loaded successfully!")
            
            # Print model summary for debugging
            print("Model input shape:", brain_tumor_model.input_shape)
            print("Model output shape:", brain_tumor_model.output_shape)
            return True
        else:
            print("✗ Model file not found.")
            print(f"Please check if the model exists at: {model_path}")
            return False
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False

# ============================================================================
# IMAGE PROCESSING FUNCTIONS
# ============================================================================

def validate_image(image_file):
    """Basic image validation"""
    try:
        # Check file size (max 10MB)
        image_file.seek(0, 2)
        file_size = image_file.tell()
        image_file.seek(0)
        
        if file_size > 10 * 1024 * 1024:
            return False, "File size too large. Maximum 10MB allowed."
        
        # Try to open image
        image = Image.open(image_file)
        
        # Check format
        if image.format not in ['JPEG', 'PNG', 'JPG']:
            return False, "Unsupported format. Please use JPEG or PNG."
        
        # Check dimensions
        width, height = image.size
        if width < 50 or height < 50:
            return False, "Image too small. Minimum 50x50 pixels required."
        
        if width > 5000 or height > 5000:
            return False, "Image too large. Maximum 5000x5000 pixels allowed."
        
        # Verify image integrity
        image.verify()
        
        return True, "Image validation successful"
        
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"

def preprocess_image(image):
    """Preprocess image for model prediction"""
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    except Exception as e:
        raise Exception(f"Error preprocessing image: {e}")

# ============================================================================
# PREDICTION PROCESSING
# ============================================================================

def process_prediction(prediction):
    """Process model prediction and return formatted result"""
    # Based on training output: {'glioma_tumor': 0, 'meningioma_tumor': 1, 'no_tumor': 2, 'pituitary_tumor': 3}
    classes = ['meningioma_tumor', 'no_tumor']
    
    predicted_index = np.argmax(prediction[0])
    predicted_class = classes[predicted_index]
    confidence = float(prediction[0][predicted_index])
    
    # Check for close predictions (uncertainty detection)
    sorted_probs = sorted(prediction[0], reverse=True)
    top_prob = sorted_probs[0]
    second_prob = sorted_probs[1] if len(sorted_probs) > 1 else 0
    is_uncertain = (top_prob - second_prob) < 0.2  # Less than 20% difference
    
    probabilities = {}
    for i, class_name in enumerate(classes):
        probabilities[class_name] = round(float(prediction[0][i]) * 100, 2)
    
    tumor_detected = predicted_class != 'no_tumor'
    
    return {
        'class': predicted_class,
        'confidence': confidence,
        'probabilities': probabilities,
        'tumor_detected': tumor_detected,
        'is_uncertain': is_uncertain,
        'uncertainty_note': 'Prediction is uncertain - consider multiple analyses' if is_uncertain else None
    }

def get_prediction_message(result):
    """Generate user-friendly message based on prediction"""
    confidence_percent = round(result['confidence'] * 100, 2)
    
    if not result['tumor_detected']:
        if result['confidence'] > 0.8:
            return f"No tumor detected with {confidence_percent}% confidence. The brain scan appears normal."
        elif result['confidence'] > 0.6:
            return f"Likely no tumor detected ({confidence_percent}% confidence), but consider professional evaluation."
        else:
            return f"Low confidence prediction ({confidence_percent}%) - image quality may be poor or model is uncertain. Professional evaluation recommended."
    else:
        tumor_type = result['class'].replace('_', ' ').title()
        if result['confidence'] > 0.8:
            return f"{tumor_type} detected with {confidence_percent}% confidence. Please consult a medical professional immediately."
        elif result['confidence'] > 0.6:
            return f"Possible {tumor_type} detected ({confidence_percent}% confidence). Medical evaluation recommended."
        elif result['is_uncertain']:
            return f"Uncertain prediction: Possible {tumor_type} ({confidence_percent}% confidence). Results are ambiguous - multiple medical opinions recommended."
        else:
            return f"Low confidence detection of {tumor_type} ({confidence_percent}%). Image quality or model limitations may affect accuracy. Professional diagnosis essential."

def get_reliability_level(confidence):
    """Get reliability level based on confidence"""
    if confidence >= 0.9:
        return 'High'
    elif confidence >= 0.75:
        return 'Medium'
    else:
        return 'Low'

# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/')
def home():
    """Home endpoint with API information"""
    return jsonify({
        'message': 'Brain Tumor Detection API',
        'status': 'running',
        'model_loaded': brain_tumor_model is not None,
        'available_endpoints': {
            'health_check': '/api/health',
            'predict': '/api/predict (POST)',
            'debug_prediction': '/api/debug/prediction (POST)',
            'debug_class_order': '/api/debug/class-order (POST)',
            'classes': '/api/classes',
            'model_info': '/api/model/info',
            'analytics': '/api/analytics/summary',
            'history': '/api/predictions/history',
            'test_page': '/test'
        }
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'message': 'Brain Tumor Detection API is running',
        'model_loaded': brain_tumor_model is not None,
        'class_mapping': {
            'training_indices': {'meningioma_tumor': 0, 'no_tumor': 1},
            'api_classes': ['meningioma_tumor', 'no_tumor']
        }
    })

@app.route('/api/predict', methods=['POST'])
@limiter.limit("10 per minute")
def predict_brain_tumor():
    """Main prediction endpoint"""
    start_time = time.time()
    
    try:
        # Check if model is loaded
        if brain_tumor_model is None:
            return jsonify({
                'success': False, 
                'error': 'Model not loaded. Please check server logs.'
            }), 500

        # Check if image is provided
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({'success': False, 'error': 'No image selected'}), 400

        # Validate image
        is_valid, message = validate_image(image_file)
        if not is_valid:
            return jsonify({
                'success': False, 
                'error': message,
                'processing_time_ms': round((time.time() - start_time) * 1000, 2)
            }), 400
        
        # Process image
        image_file.seek(0)
        image = Image.open(image_file.stream)
        processed_image = preprocess_image(image)
        
        # Make prediction
        prediction = brain_tumor_model.predict(processed_image)
        result = process_prediction(prediction)
        
        # Check confidence threshold
        MIN_CONFIDENCE_THRESHOLD = 0.30  # In app.py
        
        if result['confidence'] < MIN_CONFIDENCE_THRESHOLD:
            return jsonify({
                'success': False,
                'error': 'Low confidence prediction - image may not be suitable for analysis',
                'confidence': round(result['confidence'] * 100, 2),
                'suggestion': 'Please upload a clearer brain scan image or try a different image',
                'probabilities': result['probabilities'],
                'processing_time_ms': round((time.time() - start_time) * 1000, 2),
                'note': 'Consider using the debug endpoint /api/debug/prediction for detailed analysis'
            }), 400
        
        # Add warning for uncertain predictions
        response_data = {
            'success': True,
            'prediction': result['class'],
            'confidence': round(result['confidence'] * 100, 2),
            'probabilities': result['probabilities'],
            'tumor_detected': result['tumor_detected'],
            'message': get_prediction_message(result),
            'reliability_level': get_reliability_level(result['confidence']),
            'processing_time_ms': round((time.time() - start_time) * 1000, 2)
        }
        
        # Add uncertainty warning if applicable
        if result['is_uncertain']:
            response_data['warning'] = True
            response_data['uncertainty_note'] = result['uncertainty_note']
            response_data['recommendation'] = 'Consider getting a second opinion or retaking the scan'
        
        # Save to database
        try:
            prediction_data = {
                'filename': image_file.filename,
                'prediction': result['class'],
                'confidence': result['confidence'],
                'probabilities': result['probabilities'],
                'tumor_detected': result['tumor_detected'],
                'reliability_level': get_reliability_level(result['confidence']),
                'processing_time_ms': (time.time() - start_time) * 1000,
                'ip_address': request.remote_addr
            }
            prediction_id = db.save_prediction(prediction_data)
            response_data['prediction_id'] = prediction_id
        except Exception as db_error:
            print(f"Database error: {db_error}")
            response_data['prediction_id'] = None
        
        return jsonify(response_data)
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({
            'success': False, 
            'error': str(e),
            'processing_time_ms': round((time.time() - start_time) * 1000, 2)
        }), 500

@app.route('/api/debug/prediction', methods=['POST'])
def debug_prediction():
    """Debug endpoint to see raw model output and detailed analysis"""
    if brain_tumor_model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    image_file.seek(0)
    image = Image.open(image_file.stream)
    processed_image = preprocess_image(image)
    
    # Get raw prediction
    raw_prediction = brain_tumor_model.predict(processed_image)
    
    # Our current class mapping
    classes = ['meningioma_tumor', 'no_tumor']
    
    # Show detailed breakdown
    detailed_probs = {}
    for i, class_name in enumerate(classes):
        detailed_probs[f"{i}_{class_name}"] = {
            'index': i,
            'class': class_name,
            'raw_probability': float(raw_prediction[0][i]),
            'percentage': round(float(raw_prediction[0][i]) * 100, 2)
        }
    
    predicted_index = int(np.argmax(raw_prediction[0]))
    
    # Additional analysis - Convert numpy types to Python types
    sorted_probs = sorted(raw_prediction[0], reverse=True)
    top_prob = float(sorted_probs[0])  # Convert to float
    second_prob = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
    prob_difference = float(top_prob - second_prob)  # Convert to float
    
    return jsonify({
        'filename': image_file.filename,
        'raw_prediction_array': [float(x) for x in raw_prediction[0]],
        'predicted_index': predicted_index,
        'predicted_class': classes[predicted_index],
        'confidence': float(raw_prediction[0][predicted_index]),
        'detailed_probabilities': detailed_probs,
        'analysis': {
            'top_probability': top_prob,
            'second_probability': second_prob,
            'probability_difference': prob_difference,
            'is_uncertain': bool(prob_difference < 0.2),  # Convert to Python bool
            'confidence_level': 'High' if top_prob > 0.8 else 'Medium' if top_prob > 0.6 else 'Low'
        },
        'class_mapping': {
            'training_indices': {'meningioma_tumor': 0, 'no_tumor': 1},
            'api_classes': classes
        }
    })

@app.route('/api/debug/class-order', methods=['POST'])
def debug_class_order():
    """Debug endpoint to check different class order interpretations"""
    if brain_tumor_model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    image_file.seek(0)
    image = Image.open(image_file.stream)
    processed_image = preprocess_image(image)
    
    # Get raw prediction
    raw_prediction = brain_tumor_model.predict(processed_image)
    
    # Different possible class orders to test
    possible_orders = [
        ['meningioma_tumor', 'no_tumor'],  # Current API order 
        ['no_tumor', 'meningioma_tumor'],  
    ]
    
    results = {}
    for i, class_order in enumerate(possible_orders):
        probabilities = {}
        for j, class_name in enumerate(class_order):
            probabilities[class_name] = round(float(raw_prediction[0][j]) * 100, 2)
        
        predicted_index = np.argmax(raw_prediction[0])
        predicted_class = class_order[predicted_index]
        
        results[f'order_{i+1}'] = {
            'description': f'Order {i+1}' + (' (Current API)' if i == 0 else ''),
            'class_order': class_order,
            'predicted_class': predicted_class,
            'confidence': round(float(raw_prediction[0][predicted_index]) * 100, 2),
            'probabilities': probabilities
        }
    
    return jsonify({
        'filename': image_file.filename,
        'raw_prediction': [round(float(x), 4) for x in raw_prediction[0]],
        'possible_interpretations': results,
        'training_class_indices': {'meningioma_tumor': 0, 'no_tumor': 1},
        'note': 'Compare results to see which interpretation makes sense for your test image'
    })

@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Get available tumor classes with descriptions"""
    classes = {
        'meningioma_tumor': {
            'name': 'Meningioma Tumor',
            'description': 'A tumor that arises from the meninges',
            'index': 0
        },
        'no_tumor': {
            'name': 'No Tumor',
            'description': 'Normal brain scan with no tumor detected',
            'index': 1
        }
    }
    return jsonify({'success': True, 'classes': classes})

@app.route('/api/model/info', methods=['GET'])
def get_model_info():
    """Get model information"""
    if brain_tumor_model is None:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 500
    
    return jsonify({
        'success': True,
        'model_loaded': True,
        'input_shape': str(brain_tumor_model.input_shape),
        'output_classes': 2,
        'model_type': 'CNN for Brain Tumor Classification',
        'class_indices': {'meningioma_tumor': 0, 'no_tumor': 1},
        'confidence_threshold': 0.55
    })

@app.route('/api/model/performance', methods=['GET'])
def get_model_performance():
    """Get model performance information"""
    if brain_tumor_model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    performance_info = {
        'model_info': {
            'architecture': 'Basic CNN (4 Conv2D layers)',
            'training_epochs': 15,
            'input_size': '224x224x3',
            'output_classes': 2
        },
        'known_limitations': {
            'training_epochs': 'Only 15 epochs - may need 30-50 for medical images',
            'architecture': 'Basic CNN - consider transfer learning (ResNet, VGG)',
            'class_similarity': 'Glioma and Meningioma tumors can look very similar',
            'dataset_size': 'Small dataset may lead to overfitting'
        },
        'accuracy_expectations': {
            'good_performance': '>85% on test set',
            'acceptable': '70-85% with uncertainty warnings',
            'poor': '<70% - needs model improvement'
        },
        'improvement_suggestions': {
            'immediate': [
                'Lower confidence threshold to 40%',
                'Add uncertainty warnings for close predictions',
                'Test with multiple images from same class'
            ],
            'long_term': [
                'Retrain with 30+ epochs',
                'Use transfer learning (ResNet50, VGG16)',
                'Collect more training data',
                'Implement data balancing'
            ]
        }
    }
    
    return jsonify(performance_info)

@app.route('/api/analytics/summary', methods=['GET'])
def get_analytics_summary():
    """Get prediction analytics summary"""
    try:
        summary = db.get_analytics_summary()
        return jsonify({
            'status': 'success',
            'data': summary
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/predictions/history', methods=['GET'])
def get_prediction_history():
    """Get recent prediction history"""
    try:
        limit = request.args.get('limit', 50, type=int)
        history = db.get_prediction_history(limit)
        
        formatted_history = []
        for row in history:
            formatted_history.append({
                'id': row[0],
                'filename': row[1],
                'prediction': row[2],
                'confidence': row[3],
                'probabilities': json.loads(row[4]),
                'tumor_detected': bool(row[5]),
                'reliability_level': row[6],
                'processing_time_ms': row[7],
                'created_at': row[8]
            })
        
        return jsonify({'success': True, 'history': formatted_history})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/test/accuracy', methods=['GET'])
def test_model_accuracy():
    """Test model accuracy with known dataset images"""
    if brain_tumor_model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Test results for verification
    test_results = {
        'message': 'Manual accuracy testing endpoint',
        'instructions': {
            'step_1': 'Upload images from your test dataset using /api/debug/prediction',
            'step_2': 'Compare predicted class with expected class from folder name',
            'step_3': 'Check if confidence levels are reasonable',
            'step_4': 'Look for patterns in misclassifications'
        },
        'expected_accuracy': {
            'good_model': '>80% accuracy on test set',
            'current_threshold': '40% confidence minimum',
            'uncertainty_warning': 'Predictions with <60% confidence should be flagged'
        },
        'troubleshooting': {
            'all_wrong': 'Check class mapping order',
            'low_confidence': 'Model needs more training epochs',
            'specific_class_wrong': 'Class-specific training data issue',
            'random_results': 'Model architecture or preprocessing issue'
        }
    }
    
    return jsonify(test_results)

@app.route('/api/debug/database', methods=['GET'])
def debug_database():
    """Debug database connection and tables"""
    try:
        # Test database connection
        summary = db.get_analytics_summary()
        return jsonify({
            'status': 'success',
            'database_working': True,
            'summary': summary
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'database_working': False,
            'error': str(e),
            'error_type': type(e).__name__
        })

@app.route('/test')
def test_page():
    """Enhanced test page for uploading images"""
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Brain Tumor Detection Test</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }
            .warning { background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .info { background-color: #d1ecf1; border: 1px solid #bee5eb; padding: 10px; margin: 10px 0; border-radius: 5px; }
            .debug { background-color: #f8d7da; border: 1px solid #f5c6cb; padding: 10px; margin: 10px 0; border-radius: 5px; }
            .upload-form { background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0; }
            button { background-color: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 5px; }
            button:hover { background-color: #0056b3; }
            .debug-btn { background-color: #dc3545; }
            .debug-btn:hover { background-color: #c82333; }
            .endpoint-list { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 10px; }
            .endpoint-item { background: #f8f9fa; padding: 10px; border-radius: 5px; border: 1px solid #dee2e6; }
        </style>
    </head>
    <body>
        <h1>Brain Tumor Detection API Test</h1>
        
        <div class="warning">
            <strong>⚠️ Medical Disclaimer:</strong> This tool is for educational purposes only. 
            Results should not be used for medical diagnosis without consulting a qualified healthcare professional.
        </div>
        
        <div class="info">
            <strong>📋 Upload Guidelines:</strong>
            <ul>
                <li>Upload brain MRI or CT scan images</li>
                <li>Supported formats: JPEG, PNG</li>
                <li>File size: Maximum 10MB</li>
                <li>Minimum confidence threshold: 55%</li>
            </ul>
        </div>
        
        <div class="debug">
            <strong>🔧 Debug Mode:</strong> Use debug endpoints to analyze prediction details and troubleshoot classification issues.
        </div>
        
        <div class="upload-form">
            <h2>Upload Brain Scan Image</h2>
            <form action="/api/predict" method="post" enctype="multipart/form-data">
                <input type="file" name="image" accept="image/jpeg,image/png" required>
                <br><br>
                <button type="submit">Analyze Brain Scan</button>
            </form>
            
            <h3>Debug Analysis</h3>
            <form action="/api/debug/prediction" method="post" enctype="multipart/form-data">
                <input type="file" name="image" accept="image/jpeg,image/png" required>
                <br><br>
                <button type="submit" class="debug-btn">Debug Prediction</button>
            </form>
            
            <form action="/api/debug/class-order" method="post" enctype="multipart/form-data">
                <input type="file" name="image" accept="image/jpeg,image/png" required>
                <br><br>
                <button type="submit" class="debug-btn">Test Class Orders</button>
            </form>
        </div>
        
        <h2>API Endpoints</h2>
        <div class="endpoint-list">
            <div class="endpoint-item">
                <strong><a href="/api/health">Health Check</a></strong>
                <p>Check API status and class mapping</p>
            </div>
            <div class="endpoint-item">
                <strong><a href="/api/classes">Classes Info</a></strong>
                <p>Get tumor class descriptions</p>
            </div>
            <div class="endpoint-item">
                <strong><a href="/api/model/info">Model Info</a></strong>
                <p>Get model details and configuration</p>
            </div>
            <div class="endpoint-item">
                <strong><a href="/api/analytics/summary">Analytics</a></strong>
                <p>Get prediction statistics</p>
            </div>
            <div class="endpoint-item">
                <strong><a href="/api/predictions/history">History</a></strong>
                <p>View recent predictions</p>
            </div>
        </div>
        
        <h3>Model Performance Testing</h3>
        <div class="info">
            <strong>📊 Accuracy Testing:</strong>
            <ul>
                <li><a href="/api/test/accuracy">View Testing Guidelines</a></li>
                <li><a href="/api/model/performance">Check Model Limitations</a></li>
                <li>Test multiple images from each class folder</li>
                <li>Compare predicted vs expected results</li>
            </ul>
        </div>
        
        <h3>Class Mapping Information</h3>
        <div class="info">
            <strong>Training Class Indices:</strong>
            <ul>
                <li>meningioma_tumor: 0</li>
                <li>no_tumor: 1</li>
            </ul>
        </div>
    </body>
    </html>
    '''
    return html

# ============================================================================
# APPLICATION STARTUP
# ============================================================================

if __name__ == '__main__':
    print("🧠 Starting Brain Tumor Detection API...")
    print("📁 Current working directory:", os.getcwd())
    
    # Load model on startup
    if load_model():
        print("✅ Server ready to accept predictions")
        print("🔧 Debug endpoints available:")
        print("   - /api/debug/prediction (detailed analysis)")
        print("   - /api/debug/class-order (test different class orders)")
    else:
        print("❌ Server starting without model - predictions will fail")
        print("📝 Please ensure the model file exists in the trained_models directory")
    
    # Run the Flask application
    print("🚀 Starting server on http://localhost:5000")
    print("📄 Test page available at http://localhost:5000/test")
    app.run(debug=True, host='0.0.0.0', port=5000)