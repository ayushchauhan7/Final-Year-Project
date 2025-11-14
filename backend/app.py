from flask import Flask, render_template, request, send_from_directory, jsonify
from flask_cors import CORS
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from io import BytesIO
import base64
from collections import Counter
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import datetime
from werkzeug.utils import secure_filename  # Import secure_filename
from sklearn.metrics import confusion_matrix, classification_report
from dotenv import load_dotenv
from bson import ObjectId

# Load environment variables
load_dotenv()

# Import authentication and database
from routes.auth_routes import auth_bp
from config.database import get_database
from utils.auth import token_required, optional_token, decode_token

# Initialize Flask app
app = Flask(__name__)

# Configuration from environment variables
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# CORS Configuration
allowed_origins = os.getenv('ALLOWED_ORIGINS', 'http://localhost:5173').split(',')
CORS(app, resources={
    r"/api/*": {
        "origins": allowed_origins,
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})  # Enable CORS for all routes

# Register authentication blueprint
app.register_blueprint(auth_bp, url_prefix='/api/auth')

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('models', exist_ok=True)

# Load the trained model
MODEL_PATH = 'models/brain_tumor_model.h5'
model = None

try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully from:", MODEL_PATH)
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("⚠️ Application will run but predictions will fail")

# Class labels
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Define the uploads folder (backwards compatibility)
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Store prediction history (simple in-memory storage)
prediction_history = []

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_tumor_information(prediction, confidence):
    """Generate medical information based on prediction and confidence"""
    confidence_value = float(confidence) if isinstance(confidence, (int, float)) else float(str(confidence).strip('%'))
    
    tumor_descriptions = {
        'glioma': {
            'name': 'Glioma Tumor',
            'description': 'Gliomas are tumors that originate from glial cells in the brain or spine.',
            'severity': 'High Risk',
            'color': 'danger',
            'icon': '⚠️',
            'details': [
                'Most common primary brain tumor in adults',
                'Can be slow-growing (low-grade) or fast-growing (high-grade)',
                'May cause headaches, seizures, and neurological symptoms',
                'Treatment options include surgery, radiation, and chemotherapy'
            ]
        },
        'meningioma': {
            'name': 'Meningioma Tumor',
            'description': 'Meningiomas develop from the meninges. Most are benign and slow-growing.',
            'severity': 'Moderate Risk',
            'color': 'warning',
            'icon': '⚡',
            'details': [
                'Usually benign (non-cancerous) and slow-growing',
                'More common in women than men',
                'May not require immediate treatment if small',
                'Treatment includes observation, surgery, or radiation'
            ]
        },
        'pituitary': {
            'name': 'Pituitary Tumor',
            'description': 'Pituitary tumors form in the pituitary gland. Most are benign adenomas.',
            'severity': 'Moderate Risk',
            'color': 'info',
            'icon': '🔬',
            'details': [
                'Usually benign (non-cancerous)',
                'Can affect hormone levels and bodily functions',
                'May cause vision problems if pressing on optic nerves',
                'Treatment includes medication, surgery, or radiation'
            ]
        },
        'notumor': {
            'name': 'No Tumor Detected',
            'description': 'The AI analysis indicates no signs of tumor in the MRI scan.',
            'severity': 'Low Risk',
            'color': 'success',
            'icon': '✅',
            'details': [
                'No abnormal growth detected',
                'Brain tissue appears within normal parameters',
                'Continue regular health monitoring',
                'Consult healthcare provider for any symptoms'
            ]
        }
    }
    
    # Determine tumor type
    tumor_type = 'notumor'
    for key in tumor_descriptions.keys():
        if key in prediction.lower():
            tumor_type = key
            break
    
    info = tumor_descriptions[tumor_type]
    
    # Confidence-based recommendations
    if confidence_value >= 90:
        confidence_level = 'Very High Confidence'
        if tumor_type != 'notumor':
            recommendations = [
                '🏥 Immediate Action Required: Schedule urgent consultation',
                '📋 Bring complete medical history to appointment',
                '🔬 Additional diagnostic tests may be recommended'
            ]
        else:
            recommendations = [
                '✅ No Immediate Concerns: Results indicate healthy brain tissue',
                '📅 Continue routine health check-ups',
                '🧠 Maintain brain health through proper diet and exercise'
            ]
    elif confidence_value >= 70:
        confidence_level = 'High Confidence'
        recommendations = [
            '🏥 Medical Consultation Advised: See a neurologist',
            '📋 Request additional imaging for confirmation',
            '📊 Compare with previous scans if available'
        ]
    elif confidence_value >= 50:
        confidence_level = 'Moderate Confidence'
        recommendations = [
            '🔍 Further Investigation Needed',
            '📋 Additional imaging recommended',
            '👨‍⚕️ Consultation with specialist advised'
        ]
    else:
        confidence_level = 'Low Confidence'
        recommendations = [
            '⚠️ Uncertain Results: AI analysis has low confidence',
            '🔄 Repeat MRI scan recommended',
            '👨‍⚕️ Professional radiologist review essential'
        ]
    
    return {
        **info,
        'confidenceLevel': confidence_level,
        'recommendations': recommendations,
        'tumorType': tumor_type
    }

def save_prediction_to_db(user_info, prediction_data):
    """Save prediction to MongoDB"""
    try:
        db = get_database()
        
        # Add user information
        prediction_data['userId'] = ObjectId(user_info['user_id'])
        prediction_data['username'] = user_info['username']
        prediction_data['createdAt'] = datetime.datetime.utcnow()
        
        # Insert into database
        result = db.predictions.insert_one(prediction_data)
        
        # Log the prediction
        db.audit_logs.insert_one({
            'userId': ObjectId(user_info['user_id']),
            'username': user_info['username'],
            'action': 'prediction',
            'ipAddress': request.remote_addr,
            'userAgent': request.headers.get('User-Agent'),
            'timestamp': datetime.datetime.utcnow(),
            'details': {
                'predictionId': str(result.inserted_id),
                'predictionType': prediction_data.get('predictionType'),
                'tumorType': prediction_data.get('tumorType')
            }
        })
        
        return str(result.inserted_id)
    except Exception as e:
        print(f"Error saving to database: {e}")
        return None

# Helper function to predict tumor type
def predict_tumor(image_path):
    """Predict tumor from image"""
    if model is None:
        raise Exception("Model not loaded")
    
    IMAGE_SIZE = 128
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions, axis=1)[0]

    if class_labels[predicted_class_index] == 'notumor':
        return "No Tumor", confidence_score, predictions[0]
    else:
        return f"Tumor: {class_labels[predicted_class_index]}", confidence_score, predictions[0]

def clean_for_json(obj):
    """Convert numpy types to Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: clean_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif hasattr(obj, 'item'):  # numpy types
        return obj.item()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj

# ============================================================================
# EXISTING ROUTES (Unchanged)
# ============================================================================

# Route for the main page (index.html)
@app.route('/', methods=['GET'])
def index():
    # Display API documentation as JSON response on GET request
    api_info = {
        "title": "Brain Tumor Detection API",
        "description": "Medical Image Analysis System for Brain Tumor Classification with Research Analytics & Authentication",
        "version": "2.0.0",  # Updated version with auth
        "authentication": {
            "required": "Yes - JWT Token required for predictions",
            "endpoints": {
                "/api/auth/register": "POST - Register new user",
                "/api/auth/login": "POST - Login and get JWT token",
                "/api/auth/verify": "GET - Verify token validity",
                "/api/auth/logout": "POST - Logout user"
            }
        },
        "endpoints": {
            "/": "GET - API Documentation",
            "/test": "GET/POST - Web Interface for Testing",
            "/api/predict": "POST - Analyze single brain scan image [PROTECTED]",
            "/api/predict/batch": "POST - Analyze multiple brain scan images [PROTECTED]",
            "/api/health": "GET - Health check and system status",
            "/api/classes": "GET - Available tumor classes",
            "/api/model/info": "GET - Model information and configuration",
            "/api/debug/prediction": "POST - Detailed prediction analysis",
            "/api/debug/class-order": "GET - Test different class interpretations",
            "/api/analytics/summary": "GET - Prediction statistics [PROTECTED]",
            "/api/predictions/history": "GET - Recent prediction history [PROTECTED]",
            "/api/results/charts": "GET - Generate research charts and visualizations",
            "/api/results/statistics": "GET - Detailed statistical analysis for research",
            "/uploads/<filename>": "GET - Serve uploaded files"
        },
        "usage": {
            "authentication_flow": "1. Register -> 2. Login -> 3. Use token in Authorization header",
            "web_interface": "Visit /test for file upload interface",
            "api_usage": "Use /api/* endpoints for programmatic access",
            "test_page": "/test serves as the web interface for testing",
            "batch_processing": "Use /api/predict/batch for multiple image analysis",
            "research_analytics": "Use /api/results/* endpoints for research visualizations and statistics"
        },
        "model_info": {
            "classes": class_labels,
            "input_size": "128x128 pixels",
            "model_type": "CNN with transfer learning"
        },
        "examples": {
            "register": "curl -X POST -H 'Content-Type: application/json' -d '{\"username\":\"user\",\"email\":\"user@example.com\",\"password\":\"password\",\"fullName\":\"Full Name\"}' http://localhost:5000/api/auth/register",
            "login": "curl -X POST -H 'Content-Type: application/json' -d '{\"username\":\"user\",\"password\":\"password\"}' http://localhost:5000/api/auth/login",
            "curl_predict": "curl -X POST -H 'Authorization: Bearer YOUR_TOKEN' -F 'image=@brain_scan.jpg' http://localhost:5000/api/predict",
            "curl_batch": "curl -X POST -H 'Authorization: Bearer YOUR_TOKEN' -F 'images=@scan1.jpg' -F 'images=@scan2.jpg' http://localhost:5000/api/predict/batch",
            "curl_health": "curl http://localhost:5000/api/health",
            "curl_charts": "curl http://localhost:5000/api/results/charts",
            "curl_statistics": "curl http://localhost:5000/api/results/statistics",
            "web_test": "Visit http://localhost:5000/test for web interface"
        },
        "research_features": {
            "available_charts": [
                "Class Distribution Bar Chart",
                "Confidence Distribution Histogram", 
                "Predictions Timeline Chart",
                "Method Usage Pie Chart",
                "Confidence Trend Scatter Plot"
            ],
            "statistical_metrics": [
                "Overall Performance Statistics",
                "Class-wise Confidence Analysis", 
                "Method Usage Statistics",
                "Performance Metrics Summary"
            ],
            "output_format": "Base64 encoded PNG images for charts, JSON for statistics"
        }
    }
    return jsonify(api_info)

# Route for the test interface (index.html)
@app.route('/test', methods=['GET', 'POST'])
def test_interface():
    if request.method == 'POST':
        # Handle file upload
        file = request.files['file']
        if file:
            # Save the file
            file_location = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_location)

            # Predict the tumor
            result, confidence, all_predictions = predict_tumor(file_location)
            
            # Store in history
            prediction_history.append({
                "timestamp": datetime.datetime.now().isoformat(),
                "filename": file.filename,
                "result": result,
                "confidence": float(confidence),  # Convert to Python float
                "method": "web_interface"
            })

            # Return result along with image path for display
            return render_template('index.html', result=result, confidence=f"{confidence*100:.2f}%", file_path=f'/uploads/{file.filename}')

    return render_template('index.html', result=None)

# Route to serve uploaded files
@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ============================================================================
# UPDATED API ENDPOINTS (With Authentication)
# ============================================================================

@app.route('/api/predict', methods=['POST'])
@token_required  # NEW: Authentication required
def predict():
    """Single image prediction - PROTECTED"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Get file size
        file_size = os.path.getsize(filepath)

        # Make prediction
        start_time = datetime.datetime.now()
        result, confidence, all_predictions = predict_tumor(filepath)
        end_time = datetime.datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        # Get tumor information
        confidence_percentage = float(confidence * 100)
        tumor_info = get_tumor_information(result, confidence_percentage)

        # Prepare prediction data for database
        prediction_data = {
            'predictionType': 'single',
            'filename': filename,
            'fileSize': file_size,
            'prediction': result,
            'tumorType': tumor_info['tumorType'],
            'confidence': float(confidence),
            'confidencePercentage': confidence_percentage,
            'confidenceLevel': tumor_info['confidenceLevel'],
            'severity': tumor_info['severity'],
            'medicalDescription': tumor_info['description'],
            'recommendations': tumor_info['recommendations'],
            'processingTime': f"{processing_time:.3f}s",
            'modelVersion': 'brain_tumor_model_v1',
            'analysisDate': datetime.datetime.utcnow(),
            'probabilities': {
                class_labels[i]: float(all_predictions[i]) 
                for i in range(len(class_labels))
            }
        }

        # Save to database
        prediction_id = save_prediction_to_db(request.current_user, prediction_data)

        # Add to prediction history (backwards compatibility)
        prediction_history.append({
            'prediction': result,
            'confidence': float(confidence),
            'result': result,
            'method': 'api',
            'timestamp': datetime.datetime.now().isoformat(),
            'filename': filename
        })

        # Prepare response
        response_data = {
            'prediction': result,
            'confidence': float(confidence),
            'confidence_percentage': round(confidence_percentage, 2),
            'tumorInfo': tumor_info,
            'all_predictions': {
                class_labels[i]: float(all_predictions[i]) 
                for i in range(len(class_labels))
            },
            'filename': filename,
            'timestamp': datetime.datetime.now().isoformat(),
            'processing_time': f"{processing_time:.3f}s",
            'predictionId': prediction_id
        }

        return jsonify(response_data)

    except Exception as e:
        print(f"ERROR in predict endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def api_health():
    """Health check with database status"""
    db_status = "connected"
    try:
        db = get_database()
        db.command('ping')
    except:
        db_status = "disconnected"
    
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "database": db_status,
        "timestamp": datetime.datetime.now().isoformat(),
        "upload_folder": UPLOAD_FOLDER,
        "upload_folder_exists": os.path.exists(UPLOAD_FOLDER)
    })

@app.route('/api/classes', methods=['GET'])
def api_classes():
    return jsonify({
        "classes": class_labels,
        "total_classes": len(class_labels),
        "description": {
            "glioma": "A type of brain tumor that starts in glial cells",
            "meningioma": "A tumor that arises from the meninges",
            "notumor": "No tumor detected in the scan",
            "pituitary": "A tumor in the pituitary gland"
        }
    })

@app.route('/api/model/info', methods=['GET'])
def api_model_info():
    return jsonify({
        "model_type": "Brain Tumor Classification CNN (VGG16 Transfer Learning)",
        "input_size": [128, 128, 3],
        "classes": class_labels,
        "total_parameters": model.count_params() if model and hasattr(model, 'count_params') else "Unknown",
        "model_format": "Keras H5",
        "preprocessing": "Normalization (0-1 range)",
        "framework": "TensorFlow/Keras"
    })

@app.route('/api/debug/prediction', methods=['POST'])
@optional_token  # NEW: Optional authentication
def api_debug_prediction():
    """Debug prediction - Optional authentication"""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        file = request.files['image']
        filename = secure_filename(file.filename)
        file_location = os.path.join(app.config['UPLOAD_FOLDER'], f"debug_{filename}")
        file.save(file_location)
        
        # Get detailed prediction info
        result, confidence, all_predictions = predict_tumor(file_location)
        
        user_info = None
        if hasattr(request, 'current_user'):
            user_info = request.current_user
        
        return jsonify({
            "filename": filename,
            "prediction": result,
            "confidence": float(confidence),
            "raw_predictions": all_predictions.tolist(),
            "class_probabilities": {
                class_labels[i]: {
                    "probability": float(all_predictions[i]),
                    "percentage": f"{all_predictions[i]*100:.2f}%"
                } for i in range(len(class_labels))
            },
            "predicted_class_index": int(np.argmax(all_predictions)),
            "debug_info": {
                "max_probability": float(np.max(all_predictions)),
                "min_probability": float(np.min(all_predictions)),
                "prediction_spread": float(np.max(all_predictions) - np.min(all_predictions))
            },
            "authenticated": user_info is not None,
            "user": user_info['username'] if user_info else 'anonymous'
        })
        
    except Exception as e:
        return jsonify({"error": str(e), "debug": True}), 500

@app.route('/api/debug/class-order', methods=['GET'])
def api_debug_class_order():
    return jsonify({
        "current_class_order": class_labels,
        "class_indices": {class_labels[i]: i for i in range(len(class_labels))},
        "note": "This shows the current class interpretation order used by the model"
    })

@app.route('/api/analytics/summary', methods=['GET'])
@token_required  # NEW: Authentication required
def api_analytics_summary():
    """Get user's analytics summary - PROTECTED"""
    try:
        db = get_database()
        user_id = ObjectId(request.current_user['user_id'])
        
        # Total predictions from database
        total_predictions = db.predictions.count_documents({'userId': user_id})
        
        # Tumor type distribution
        pipeline = [
            {'$match': {'userId': user_id, 'tumorType': {'$exists': True}}},
            {'$group': {'_id': '$tumorType', 'count': {'$sum': 1}}}
        ]
        tumor_distribution = list(db.predictions.aggregate(pipeline))
        
        # Recent activity
        recent = list(db.predictions.find(
            {'userId': user_id}
        ).sort('createdAt', -1).limit(5))
        
        for item in recent:
            item['_id'] = str(item['_id'])
            item['userId'] = str(item['userId'])
        
        return jsonify({
            'totalPredictions': total_predictions,
            'tumorDistribution': {item['_id']: item['count'] for item in tumor_distribution},
            'recentActivity': recent
        })
        
    except Exception as e:
        # Fallback to in-memory data if database fails
        if not prediction_history:
            return jsonify({
                "total_predictions": 0,
                "message": "No predictions made yet"
            })
        
        total = len(prediction_history)
        tumor_count = sum(1 for p in prediction_history if "No Tumor" not in p["result"])
        no_tumor_count = total - tumor_count
        
        recent_predictions = prediction_history[-5:] if len(prediction_history) >= 5 else prediction_history
        cleaned_predictions = []
        
        for pred in recent_predictions:
            cleaned_pred = {
                "timestamp": pred["timestamp"],
                "filename": pred["filename"],
                "result": pred["result"],
                "confidence": float(pred["confidence"]),
                "method": pred["method"]
            }
            cleaned_predictions.append(cleaned_pred)
        
        return jsonify(clean_for_json({
            "total_predictions": total,
            "tumor_detected": tumor_count,
            "no_tumor_detected": no_tumor_count,
            "tumor_detection_rate": f"{(tumor_count/total)*100:.1f}%" if total > 0 else "0%",
            "recent_predictions": cleaned_predictions
        }))

@app.route('/api/predictions/history', methods=['GET'])
@token_required  # NEW: Authentication required
def api_predictions_history():
    """Get user's prediction history - PROTECTED"""
    try:
        db = get_database()
        limit = int(request.args.get('limit', 20))
        
        predictions = list(db.predictions.find(
            {'userId': ObjectId(request.current_user['user_id'])},
            {'password': 0}
        ).sort('createdAt', -1).limit(limit))
        
        # Convert ObjectId to string
        for pred in predictions:
            pred['_id'] = str(pred['_id'])
            pred['userId'] = str(pred['userId'])
            if 'batchId' in pred:
                pred['batchId'] = str(pred['batchId'])
        
        return jsonify({
            'total': len(predictions),
            'predictions': predictions
        })
        
    except Exception as e:
        # Fallback to in-memory data
        limit = request.args.get('limit', 10, type=int)
        recent_predictions = prediction_history[-limit:] if prediction_history else []
        cleaned_predictions = []
        
        for pred in recent_predictions:
            cleaned_pred = {
                "timestamp": pred["timestamp"],
                "filename": pred["filename"],
                "result": pred["result"],
                "confidence": float(pred["confidence"]),
                "method": pred["method"]
            }
            cleaned_predictions.append(cleaned_pred)
        
        return jsonify(clean_for_json({
            "total_predictions": len(prediction_history),
            "recent_predictions": cleaned_predictions,
            "limit": limit
        }))

# Batch prediction - PROTECTED
@app.route('/api/predict/batch', methods=['POST'])
@token_required  # NEW: Authentication required
def api_predict_batch():
    """Batch prediction - PROTECTED"""
    try:
        files = request.files.getlist('images')
        
        if not files or len(files) == 0:
            return jsonify({"error": "No images provided"}), 400
        
        results = []
        tumor_types = []
        batch_id = ObjectId()  # Generate batch ID
        
        for file in files:
            if file.filename == '':
                continue
            
            filename = secure_filename(file.filename)
            file_location = os.path.join(app.config['UPLOAD_FOLDER'], f"batch_{filename}")
            file.save(file_location)
            
            file_size = os.path.getsize(file_location)
            
            # Predict
            start_time = datetime.datetime.now()
            result, confidence, all_predictions = predict_tumor(file_location)
            end_time = datetime.datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            confidence_percentage = float(confidence * 100)
            tumor_info = get_tumor_information(result, confidence_percentage)
            
            # Track tumor types
            if tumor_info['tumorType'] != 'notumor':
                tumor_types.append(tumor_info['tumorType'])
            
            # Save individual result to batch_results collection
            try:
                db = get_database()
                db.batch_results.insert_one({
                    'batchId': batch_id,
                    'userId': ObjectId(request.current_user['user_id']),
                    'username': request.current_user['username'],
                    'filename': filename,
                    'fileSize': file_size,
                    'prediction': result,
                    'tumorType': tumor_info['tumorType'],
                    'confidence': float(confidence),
                    'confidencePercentage': confidence_percentage,
                    'severity': tumor_info['severity'],
                    'processingTime': f"{processing_time:.3f}s",
                    'createdAt': datetime.datetime.utcnow()
                })
            except Exception as db_error:
                print(f"Error saving batch result: {db_error}")
            
            # Add to in-memory history
            prediction_history.append({
                "timestamp": datetime.datetime.now().isoformat(),
                "filename": filename,
                "result": result,
                "confidence": float(confidence),
                "method": "batch_api"
            })
            
            results.append({
                "filename": filename,
                "prediction": result,
                "confidence": f"{confidence*100:.2f}%",
                "confidence_percentage": confidence_percentage,
                "confidence_score": float(confidence),
                "processing_time": f"{processing_time:.3f}s",
                "probabilities": {
                    class_labels[i]: float(all_predictions[i]) 
                    for i in range(len(class_labels))
                }
            })
        
        # Calculate batch summary
        tumor_detected = sum(1 for r in results if "No Tumor" not in r["prediction"])
        tumor_type_counts = {}
        for tumor_type in tumor_types:
            tumor_type_counts[tumor_type] = tumor_type_counts.get(tumor_type, 0) + 1
        
        avg_confidence = sum(r["confidence_score"] for r in results) / len(results) if results else 0
        
        batch_summary = {
            "tumor_detected": tumor_detected,
            "no_tumor": sum(1 for r in results if "No Tumor" in r["prediction"]),
            "average_confidence": f"{avg_confidence*100:.2f}%",
            "by_tumor_type": tumor_type_counts
        }
        
        # Save batch summary to predictions collection
        try:
            batch_data = {
                'predictionType': 'batch',
                'batchId': batch_id,
                'totalImages': len(results),
                'batchSummary': batch_summary,
                'processingTime': sum(float(r['processing_time'].replace('s', '')) for r in results),
                'modelVersion': 'brain_tumor_model_v1',
                'analysisDate': datetime.datetime.utcnow()
            }
            save_prediction_to_db(request.current_user, batch_data)
        except Exception as db_error:
            print(f"Error saving batch summary: {db_error}")
        
        return jsonify({
            "total_images": len(results),
            "results": results,
            "batch_summary": batch_summary,
            "batchId": str(batch_id)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============================================================================
# CHART AND STATISTICS ENDPOINTS (Existing - Unchanged)
# ============================================================================

@app.route('/api/results/charts', methods=['GET'])
def api_results_charts():
    """Generate various charts and return as base64 encoded images"""
    try:
        if not prediction_history:
            return jsonify({"error": "No prediction data available"}), 404
        
        charts = {}
        
        # 1. Class Distribution Chart
        charts['class_distribution'] = generate_class_distribution_chart()
        
        # 2. Confidence Distribution
        charts['confidence_distribution'] = generate_confidence_distribution_chart()
        
        # 3. Predictions Timeline
        charts['predictions_timeline'] = generate_timeline_chart()
        
        # 4. Method Usage Statistics
        charts['method_usage'] = generate_method_usage_chart()
        
        # 5. Confidence vs Time Scatter
        charts['confidence_trend'] = generate_confidence_trend_chart()
        
        return jsonify({
            "charts": charts,
            "metadata": {
                "total_predictions": len(prediction_history),
                "generated_at": datetime.datetime.now().isoformat(),
                "chart_count": len(charts)
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def generate_class_distribution_chart():
    """Generate class distribution bar chart"""
    results = [pred['result'] for pred in prediction_history]
    result_counts = Counter(results)
    
    plt.figure(figsize=(10, 6))
    classes = list(result_counts.keys())
    counts = list(result_counts.values())
    
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
    bars = plt.bar(classes, counts, color=colors[:len(classes)])
    
    plt.title('Brain Tumor Prediction Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Prediction Classes', fontsize=12)
    plt.ylabel('Number of Predictions', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    chart_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return chart_base64

def generate_confidence_distribution_chart():
    """Generate confidence score distribution histogram"""
    confidences = [pred['confidence'] for pred in prediction_history]
    
    plt.figure(figsize=(10, 6))
    plt.hist(confidences, bins=20, color='#45b7d1', alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(confidences), color='red', linestyle='--', 
               label=f'Mean: {np.mean(confidences):.3f}')
    
    plt.title('Prediction Confidence Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Confidence Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    chart_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return chart_base64

def generate_timeline_chart():
    """Generate predictions over time line chart"""
    timestamps = [datetime.datetime.fromisoformat(pred['timestamp']) for pred in prediction_history]
    hourly_counts = Counter([ts.strftime('%Y-%m-%d %H:00') for ts in timestamps])
    sorted_hours = sorted(hourly_counts.keys())
    counts = [hourly_counts[hour] for hour in sorted_hours]
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(sorted_hours)), counts, marker='o', linewidth=2, markersize=6, color='#4ecdc4')
    plt.fill_between(range(len(sorted_hours)), counts, alpha=0.3, color='#4ecdc4')
    
    plt.title('Predictions Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Time Period', fontsize=12)
    plt.ylabel('Number of Predictions', fontsize=12)
    plt.xticks(range(0, len(sorted_hours), max(1, len(sorted_hours)//10)), 
              [sorted_hours[i] for i in range(0, len(sorted_hours), max(1, len(sorted_hours)//10))], 
              rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    chart_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return chart_base64

def generate_method_usage_chart():
    """Generate method usage pie chart"""
    methods = [pred['method'] for pred in prediction_history]
    method_counts = Counter(methods)
    
    plt.figure(figsize=(8, 8))
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
    
    wedges, texts, autotexts = plt.pie(method_counts.values(), 
                                      labels=method_counts.keys(), 
                                      autopct='%1.1f%%',
                                      colors=colors,
                                      startangle=90,
                                      explode=[0.05] * len(method_counts))
    
    plt.title('API Usage Methods', fontsize=16, fontweight='bold')
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    chart_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return chart_base64

def generate_confidence_trend_chart():
    """Generate confidence trend over time"""
    timestamps = [datetime.datetime.fromisoformat(pred['timestamp']) for pred in prediction_history]
    confidences = [pred['confidence'] for pred in prediction_history]
    results = [pred['result'] for pred in prediction_history]
    
    plt.figure(figsize=(12, 6))
    
    unique_results = list(set(results))
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
    
    for i, result in enumerate(unique_results):
        result_indices = [j for j, r in enumerate(results) if r == result]
        result_timestamps = [timestamps[j] for j in result_indices]
        result_confidences = [confidences[j] for j in result_indices]
        
        plt.scatter(result_timestamps, result_confidences, 
                   label=result, alpha=0.7, s=50, 
                   color=colors[i % len(colors)])
    
    plt.title('Prediction Confidence Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Confidence Score', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    chart_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return chart_base64

@app.route('/api/results/statistics', methods=['GET'])
def api_results_statistics():
    """Get detailed statistics for research analysis"""
    try:
        if not prediction_history:
            return jsonify({"error": "No prediction data available"}), 404
        
        confidences = [pred['confidence'] for pred in prediction_history]
        results = [pred['result'] for pred in prediction_history]
        methods = [pred['method'] for pred in prediction_history]
        
        stats = {
            "overall_statistics": {
                "total_predictions": len(prediction_history),
                "average_confidence": np.mean(confidences),
                "confidence_std": np.std(confidences),
                "min_confidence": np.min(confidences),
                "max_confidence": np.max(confidences),
                "median_confidence": np.median(confidences)
            },
            "class_statistics": dict(Counter(results)),
            "method_statistics": dict(Counter(methods)),
            "confidence_by_class": {},
            "performance_metrics": {
                "high_confidence_predictions": len([c for c in confidences if c > 0.8]),
                "low_confidence_predictions": len([c for c in confidences if c < 0.5]),
                "tumor_detection_rate": len([r for r in results if 'Tumor' in r and 'No Tumor' not in r]) / len(results)
            }
        }
        
        for result in set(results):
            class_confidences = [pred['confidence'] for pred in prediction_history if pred['result'] == result]
            if class_confidences:
                stats["confidence_by_class"][result] = {
                    "mean": np.mean(class_confidences),
                    "std": np.std(class_confidences),
                    "count": len(class_confidences)
                }
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============================================================================
# APPLICATION STARTUP
# ============================================================================

if __name__ == '__main__':
    print("🚀 Brain Tumor Detection API Starting...")
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    print(f"🔬 Model classes: {class_labels}")
    print("📡 API Endpoints available at: http://localhost:5000")
    
    # Test database connection
    try:
        db = get_database()
        print("✅ Database connection successful")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("⚠️ Application will run with limited functionality")
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)