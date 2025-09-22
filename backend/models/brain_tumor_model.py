import tensorflow as tf
import numpy as np
from PIL import Image
import os

class BrainTumorModel:
    def __init__(self, model_path):
        self.model = None
        self.classes = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load the trained model"""
        try:
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
                print(f"Model loaded from: {model_path}")
                return True
            else:
                print(f"Model file not found: {model_path}")
                return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def preprocess_image(self, image):
        """Preprocess image for prediction"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    
    def predict(self, image):
        """Make prediction on image"""
        if self.model is None:
            raise Exception("Model not loaded")
        
        processed_image = self.preprocess_image(image)
        prediction = self.model.predict(processed_image)
        
        return self.process_prediction(prediction)
    
    def process_prediction(self, prediction):
        """Process raw prediction to meaningful result"""
        predicted_index = np.argmax(prediction[0])
        predicted_class = self.classes[predicted_index]
        confidence = float(prediction[0][predicted_index])
        
        probabilities = {}
        for i, class_name in enumerate(self.classes):
            probabilities[class_name] = float(prediction[0][i])
        
        return {
            'class': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities,
            'tumor_detected': predicted_class != 'no_tumor'
        }