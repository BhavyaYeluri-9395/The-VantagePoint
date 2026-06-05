# init_database.py
import sqlite3
import os

def init_feedback_database():
    """Initialize the SQLite database for user feedback"""
    
    # Create database directory if it doesn't exist
    db_dir = "database"
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
        print(f"Created directory: {db_dir}")
    
    db_path = "database/feedback.db"
    
    # Connect to database (creates file if not exists)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create feedback table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            session_id TEXT,
            predicted_rating REAL,
            actual_rating REAL,
            rating_difference REAL,
            user_satisfaction INTEGER,
            movie_features TEXT,
            user_comments TEXT
        )
    ''')
    
    # Create model performance table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            model_version TEXT,
            mae REAL,
            rmse REAL,
            avg_user_satisfaction REAL,
            total_feedback INTEGER
        )
    ''')
    
    # Create user preferences table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_preferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            preferred_genres TEXT,
            avg_rating_preference REAL,
            cast_preference TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create sample data for testing (optional)
    cursor.execute('''
        INSERT OR IGNORE INTO feedback 
        (session_id, predicted_rating, actual_rating, rating_difference, user_satisfaction, movie_features)
        VALUES 
            ('test_session', 7.5, 8.0, 0.5, 4, '{"genre": "action"}'),
            ('test_session', 6.8, 7.2, 0.4, 5, '{"genre": "comedy"}')
    ''')
    
    conn.commit()
    conn.close()
    
    print(f"Database initialized at: {db_path}")
    print("   Tables created: feedback, model_performance, user_preferences")
    
    # Verify database was created
    if os.path.exists(db_path):
        file_size = os.path.getsize(db_path)
        print(f"   Database size: {file_size} bytes")
    return True

if __name__ == "__main__":
    print("Initializing VANTAGE POINT Feedback Database...")
    init_feedback_database()
    print("\nSetup complete! You can now run: python web_app.py")