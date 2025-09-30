from flask import Flask, render_template, request, send_from_directory, jsonify
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import datetime

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('../notebooks/brain_tumor_model.h5')

# Class labels
class_labels = ['glioma', 'meningioma' ,'notumor', 'pituitary']

# Define the uploads folder
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Store prediction history (simple in-memory storage)
prediction_history = []

# Helper function to predict tumor type
def predict_tumor(image_path):
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

# Route for the main page (index.html)
@app.route('/', methods=['GET'])
def index():
    # Display API documentation as JSON response on GET request
    api_info = {
        "title": "Brain Tumor Detection API",
        "description": "Medical Image Analysis System for Brain Tumor Classification",
        "version": "1.0.0",
        "endpoints": {
            "/": "GET - API Documentation",
            "/test": "GET/POST - Web Interface for Testing",
            "/api/predict": "POST - Analyze brain scan image",
            "/api/health": "GET - Health check and system status",
            "/api/classes": "GET - Available tumor classes",
            "/api/model/info": "GET - Model information and configuration",
            "/api/debug/prediction": "POST - Detailed prediction analysis",
            "/api/debug/class-order": "GET - Test different class interpretations",
            "/api/analytics/summary": "GET - Prediction statistics",
            "/api/predictions/history": "GET - Recent prediction history",
            "/uploads/<filename>": "GET - Serve uploaded files"
        },
        "usage": {
            "web_interface": "Visit /test for file upload interface",
            "api_usage": "Use /api/* endpoints for programmatic access",
            "test_page": "/test serves as the web interface for testing"
        },
        "model_info": {
            "classes": class_labels,
            "input_size": "128x128 pixels",
            "model_type": "CNN with transfer learning"
        },
        "examples": {
            "curl_predict": "curl -X POST -F 'image=@brain_scan.jpg' http://localhost:5000/api/predict",
            "curl_health": "curl http://localhost:5000/api/health",
            "web_test": "Visit http://localhost:5000/test for web interface"
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

# === NEW API ENDPOINTS ===

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No image selected"}), 400
        
        # Save file
        file_location = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_location)
        
        # Predict
        result, confidence, all_predictions = predict_tumor(file_location)
        
        # Store in history
        prediction_history.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "filename": file.filename,
            "result": result,
            "confidence": float(confidence),  # Convert to Python float
            "method": "api"
        })
        
        return jsonify({
            "prediction": result,
            "confidence": f"{confidence*100:.2f}%",
            "confidence_score": float(confidence),
            "probabilities": {
                class_labels[i]: float(all_predictions[i]) for i in range(len(class_labels))
            },
            "filename": file.filename
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def api_health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
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
        "model_type": "Brain Tumor Classification CNN",
        "input_size": [128, 128, 3],
        "classes": class_labels,
        "total_parameters": model.count_params() if hasattr(model, 'count_params') else "Unknown",
        "model_format": "Keras H5",
        "preprocessing": "Normalization (0-1 range)"
    })

@app.route('/api/debug/prediction', methods=['POST'])
def api_debug_prediction():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        file = request.files['image']
        file_location = os.path.join(app.config['UPLOAD_FOLDER'], f"debug_{file.filename}")
        file.save(file_location)
        
        # Get detailed prediction info
        result, confidence, all_predictions = predict_tumor(file_location)
        
        return jsonify({
            "filename": file.filename,
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
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e), "debug": True}), 500

@app.route('/api/debug/class-order', methods=['GET'])  # Changed from POST to GET
def api_debug_class_order():
    return jsonify({
        "current_class_order": class_labels,
        "class_indices": {class_labels[i]: i for i in range(len(class_labels))},
        "note": "This shows the current class interpretation order used by the model"
    })

@app.route('/api/analytics/summary', methods=['GET'])
def api_analytics_summary():
    if not prediction_history:
        return jsonify({
            "total_predictions": 0,
            "message": "No predictions made yet"
        })
    
    # Simple analytics - Convert any remaining numpy types
    total = len(prediction_history)
    tumor_count = sum(1 for p in prediction_history if "No Tumor" not in p["result"])
    no_tumor_count = total - tumor_count
    
    # Clean recent predictions for JSON serialization
    recent_predictions = prediction_history[-5:] if len(prediction_history) >= 5 else prediction_history
    cleaned_predictions = []
    
    for pred in recent_predictions:
        cleaned_pred = {
            "timestamp": pred["timestamp"],
            "filename": pred["filename"],
            "result": pred["result"],
            "confidence": float(pred["confidence"]),  # Ensure it's Python float
            "method": pred["method"]
        }
        cleaned_predictions.append(cleaned_pred)
    
    return jsonify(clean_for_json({
        "total_predictions": total,
        "tumor_detected": tumor_count,
        "no_tumor_detected": no_tumor_count,
        "tumor_detection_rate": f"{(tumor_count/total)*100:.1f}%" if total > 0 else "0%",
        "recent_predictions": prediction_history[-5:] if len(prediction_history) >= 5 else prediction_history
    }))

@app.route('/api/predictions/history', methods=['GET'])
def api_predictions_history():
    limit = request.args.get('limit', 10, type=int)
    
    # Clean all predictions for JSON serialization
    recent_predictions = prediction_history[-limit:] if prediction_history else []
    cleaned_predictions = []
    
    for pred in recent_predictions:
        cleaned_pred = {
            "timestamp": pred["timestamp"],
            "filename": pred["filename"],
            "result": pred["result"],
            "confidence": float(pred["confidence"]),  # Ensure it's Python float
            "method": pred["method"]
        }
        cleaned_predictions.append(cleaned_pred)
    
    return jsonify(clean_for_json({
        "total_predictions": len(prediction_history),
        "recent_predictions": prediction_history[-limit:] if prediction_history else [],
        "limit": limit
    }))

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

if __name__ == '__main__':
    print("🚀 Brain Tumor Detection API Starting...")
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    print(f"🔬 Model classes: {class_labels}")
    print("📡 API Endpoints available at: http://127.0.0.1:5000")
    app.run(debug=True)