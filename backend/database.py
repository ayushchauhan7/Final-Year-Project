# database.py
import sqlite3
import json
from datetime import datetime
import os

class PredictionDatabase:
    def __init__(self, db_path='predictions.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                prediction TEXT,
                confidence REAL,
                probabilities TEXT,
                tumor_detected BOOLEAN,
                reliability_level TEXT,
                processing_time_ms REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ip_address TEXT
            )
        ''')
        
        # Create analytics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                endpoint TEXT,
                status_code INTEGER,
                processing_time_ms REAL,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_prediction(self, prediction_data):
        """Save prediction to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO predictions 
                (filename, prediction, confidence, probabilities, tumor_detected, 
                 reliability_level, processing_time_ms, ip_address)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                prediction_data['filename'],
                prediction_data['prediction'],
                prediction_data['confidence'],
                json.dumps(prediction_data['probabilities']),
                prediction_data['tumor_detected'],
                prediction_data['reliability_level'],
                prediction_data.get('processing_time_ms', 0),
                prediction_data.get('ip_address', '')
            ))
            
            prediction_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return prediction_id
        except Exception as e:
            print(f"Database save error: {e}")
            return None
    
    def get_prediction_history(self, limit=50):
        """Get recent predictions"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM predictions 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (limit,))
            
            results = cursor.fetchall()
            conn.close()
            
            return results
        except Exception as e:
            print(f"Database history error: {e}")
            return []
    
    def get_analytics_summary(self):
        """Get analytics summary"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Total predictions
            cursor.execute('SELECT COUNT(*) FROM predictions')
            total_predictions = cursor.fetchone()[0]
            
            # Tumor detection rate
            cursor.execute('SELECT COUNT(*) FROM predictions WHERE tumor_detected = 1')
            tumor_detections = cursor.fetchone()[0]
            
            # Average confidence
            cursor.execute('SELECT AVG(confidence) FROM predictions')
            avg_confidence = cursor.fetchone()[0] or 0
            
            # Predictions by type
            cursor.execute('''
                SELECT prediction, COUNT(*) 
                FROM predictions 
                GROUP BY prediction
            ''')
            predictions_by_type = dict(cursor.fetchall())
            
            conn.close()
            
            return {
                'total_predictions': total_predictions,
                'tumor_detections': tumor_detections,
                'tumor_detection_rate': tumor_detections / total_predictions if total_predictions > 0 else 0,
                'average_confidence': round(avg_confidence, 2),
                'predictions_by_type': predictions_by_type
            }
        except Exception as e:
            print(f"Database analytics error: {e}")
            return {
                'total_predictions': 0,
                'tumor_detections': 0,
                'tumor_detection_rate': 0,
                'average_confidence': 0,
                'predictions_by_type': {}
            }