# test_mongo.py
import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# Get connection string
uri = os.getenv('MONGODB_URI')
print(f"Connecting to: {uri[:50]}...")

try:
    # Connect
    client = MongoClient(uri)
    
    # Test connection
    client.admin.command('ping')
    print("✅ MongoDB connection successful!")
    
    # List databases
    print("📁 Available databases:")
    for db in client.list_database_names():
        print(f"   - {db}")
    
    # Create/use database
    db = client['vantage_point']
    
    # Test insert
    test_collection = db['test']
    result = test_collection.insert_one({"test": "Connection working!"})
    print(f"✅ Test document inserted with ID: {result.inserted_id}")
    
    # Clean up
    test_collection.delete_one({"_id": result.inserted_id})
    print("✅ Test cleanup complete")
    
    client.close()
    
except Exception as e:
    print(f"❌ Connection failed: {e}")