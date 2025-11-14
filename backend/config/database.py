import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Database:
    _instance = None
    _client = None
    _db = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Database, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._client is None:
            self.connect()
    
    def connect(self):
        """Connect to MongoDB Atlas"""
        try:
            mongodb_uri = os.getenv('MONGODB_URI')
            if not mongodb_uri:
                raise ValueError("MONGODB_URI not found in environment variables")
            
            self._client = MongoClient(
                mongodb_uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
                socketTimeoutMS=10000
            )
            
            # Test connection
            self._client.admin.command('ping')
            
            db_name = os.getenv('MONGODB_DB_NAME', 'brain_tumor_db')
            self._db = self._client[db_name]
            
            print(f"✅ Successfully connected to MongoDB Atlas: {db_name}")
            
            # Create indexes
            self.create_indexes()
            
            return self._db
            
        except ConnectionFailure as e:
            print(f"❌ Failed to connect to MongoDB: {e}")
            raise
        except Exception as e:
            print(f"❌ Database connection error: {e}")
            raise
    
    def create_indexes(self):
        """Create database indexes for better performance"""
        try:
            # Users collection indexes
            self._db.users.create_index("username", unique=True)
            self._db.users.create_index("email", unique=True)
            
            # Predictions collection indexes
            self._db.predictions.create_index("userId")
            self._db.predictions.create_index("username")
            self._db.predictions.create_index("createdAt")
            self._db.predictions.create_index([("userId", 1), ("createdAt", -1)])
            
            # Batch results indexes
            self._db.batch_results.create_index("batchId")
            self._db.batch_results.create_index("userId")
            
            # Audit logs indexes
            self._db.audit_logs.create_index("userId")
            self._db.audit_logs.create_index("timestamp")
            
            print("✅ Database indexes created successfully")
            
        except Exception as e:
            print(f"⚠️ Error creating indexes: {e}")
    
    def get_db(self):
        """Get database instance"""
        if self._db is None:
            self.connect()
        return self._db
    
    def close(self):
        """Close database connection"""
        if self._client:
            self._client.close()
            print("✅ Database connection closed")

# Singleton instance
db = Database()

# Helper function to get database
def get_database():
    return db.get_db()