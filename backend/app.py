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

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

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
        "description": "Medical Image Analysis System for Brain Tumor Classification with Research Analytics",
        "version": "2.0.0",  # Updated version
        "endpoints": {
            "/": "GET - API Documentation",
            "/test": "GET/POST - Web Interface for Testing",
            "/api/predict": "POST - Analyze single brain scan image",
            "/api/predict/batch": "POST - Analyze multiple brain scan images",
            "/api/health": "GET - Health check and system status",
            "/api/classes": "GET - Available tumor classes",
            "/api/model/info": "GET - Model information and configuration",
            "/api/debug/prediction": "POST - Detailed prediction analysis",
            "/api/debug/class-order": "GET - Test different class interpretations",
            "/api/analytics/summary": "GET - Prediction statistics",
            "/api/predictions/history": "GET - Recent prediction history",
            "/api/results/charts": "GET - Generate research charts and visualizations",  # New
            "/api/results/statistics": "GET - Detailed statistical analysis for research",  # New
            "/uploads/<filename>": "GET - Serve uploaded files"
        },
        "usage": {
            "web_interface": "Visit /test for file upload interface",
            "api_usage": "Use /api/* endpoints for programmatic access",
            "test_page": "/test serves as the web interface for testing",
            "batch_processing": "Use /api/predict/batch for multiple image analysis",
            "research_analytics": "Use /api/results/* endpoints for research visualizations and statistics"  # New
        },
        "model_info": {
            "classes": class_labels,
            "input_size": "128x128 pixels",
            "model_type": "CNN with transfer learning"
        },
        "examples": {
            "curl_predict": "curl -X POST -F 'image=@brain_scan.jpg' http://localhost:5000/api/predict",
            "curl_batch": "curl -X POST -F 'images=@scan1.jpg' -F 'images=@scan2.jpg' http://localhost:5000/api/predict/batch",
            "curl_health": "curl http://localhost:5000/api/health",
            "curl_charts": "curl http://localhost:5000/api/results/charts",  # New
            "curl_statistics": "curl http://localhost:5000/api/results/statistics",  # New
            "web_test": "Visit http://localhost:5000/test for web interface"
        },
        "research_features": {  # New section
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

# Add this new endpoint for multiple image upload
@app.route('/api/predict/batch', methods=['POST'])
def api_predict_batch():
    try:
        files = request.files.getlist('images')  # Get multiple files
        
        if not files or len(files) == 0:
            return jsonify({"error": "No images provided"}), 400
        
        results = []
        
        for file in files:
            if file.filename == '':
                continue
                
            # Save file
            file_location = os.path.join(app.config['UPLOAD_FOLDER'], f"batch_{file.filename}")
            file.save(file_location)
            
            # Predict
            result, confidence, all_predictions = predict_tumor(file_location)
            
            # Store in history
            prediction_history.append({
                "timestamp": datetime.datetime.now().isoformat(),
                "filename": file.filename,
                "result": result,
                "confidence": float(confidence),
                "method": "batch_api"
            })
            
            # Add to results
            results.append({
                "filename": file.filename,
                "prediction": result,
                "confidence": f"{confidence*100:.2f}%",
                "confidence_score": float(confidence),
                "probabilities": {
                    class_labels[i]: float(all_predictions[i]) for i in range(len(class_labels))
                }
            })
        
        return jsonify({
            "total_images": len(results),
            "results": results,
            "batch_summary": {
                "tumor_detected": sum(1 for r in results if "No Tumor" not in r["prediction"]),
                "no_tumor": sum(1 for r in results if "No Tumor" in r["prediction"]),
                "average_confidence": sum(r["confidence_score"] for r in results) / len(results) if results else 0
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# New Results and Analytics Endpoints
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
    # Extract results from prediction history
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
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Convert to base64
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
    
    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    chart_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return chart_base64

def generate_timeline_chart():
    """Generate predictions over time line chart"""
    # Convert timestamps and group by hour
    timestamps = [datetime.datetime.fromisoformat(pred['timestamp']) for pred in prediction_history]
    
    # Group predictions by hour
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
    
    # Convert to base64
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
    
    # Enhance text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    # Convert to base64
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
    
    # Create scatter plot with different colors for different results
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
    
    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    chart_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return chart_base64

# Statistics Summary Endpoint
@app.route('/api/results/statistics', methods=['GET'])
def api_results_statistics():
    """Get detailed statistics for research analysis"""
    try:
        if not prediction_history:
            return jsonify({"error": "No prediction data available"}), 404
        
        # Extract data
        confidences = [pred['confidence'] for pred in prediction_history]
        results = [pred['result'] for pred in prediction_history]
        methods = [pred['method'] for pred in prediction_history]
        
        # Calculate statistics
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
        
        # Confidence by class
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
    print("📡 API Endpoints available at: http://localhost:5000")
    app.run(debug=True)