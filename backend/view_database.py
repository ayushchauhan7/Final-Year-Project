import sqlite3
import json

def view_database():
    try:
        conn = sqlite3.connect('predictions.db')
        cursor = conn.cursor()
        
        # Check if tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print("Tables in database:", tables)
        
        # Count records in predictions table
        cursor.execute("SELECT COUNT(*) FROM predictions")
        count = cursor.fetchone()[0]
        print(f"Number of predictions: {count}")
        
        # Show recent predictions
        if count > 0:
            cursor.execute("SELECT * FROM predictions ORDER BY created_at DESC LIMIT 5")
            recent = cursor.fetchall()
            print("\nRecent predictions:")
            for row in recent:
                print(f"ID: {row[0]}, Prediction: {row[2]}, Confidence: {row[3]:.2%}")
        
        conn.close()
        
    except Exception as e:
        print(f"Database error: {e}")

if __name__ == "__main__":
    view_database()