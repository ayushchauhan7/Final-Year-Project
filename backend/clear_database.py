from database import PredictionDatabase
import os

def clear_old_predictions():
    """Clear old 4-class predictions for new 2-class system"""
    try:
        # Delete the entire database file to start fresh
        if os.path.exists('predictions.db'):
            os.remove('predictions.db')
            print("✅ Old database cleared!")
        
        # Create new database with fresh tables
        db = PredictionDatabase()
        print("✅ New database initialized for 2-class system!")
        
    except Exception as e:
        print(f"❌ Error clearing database: {e}")

if __name__ == "__main__":
    clear_old_predictions()