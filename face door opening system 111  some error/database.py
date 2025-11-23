import sqlite3
import os
from datetime import datetime
import json

class DatabaseManager:
    """Manages the SQLite database for the face recognition door system"""
    
    def __init__(self, db_path="door_system.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP,
                access_count INTEGER DEFAULT 0
            )
        ''')
        
        # Create access_logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS access_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                event_type TEXT NOT NULL,
                person_name TEXT,
                details TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_user(self, name):
        """Add a new user to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "INSERT INTO users (name) VALUES (?)",
                (name,)
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            # User already exists
            return False
        finally:
            conn.close()
    
    def delete_user(self, name):
        """Delete a user from the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM users WHERE name = ?", (name,))
        conn.commit()
        conn.close()
        return cursor.rowcount > 0
    
    def get_all_users(self):
        """Retrieve all users from the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, name, created_at, last_seen, access_count FROM users ORDER BY name")
        users = cursor.fetchall()
        
        conn.close()
        return users
    
    def get_user(self, name):
        """Retrieve a specific user from the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, name, created_at, last_seen, access_count FROM users WHERE name = ?", (name,))
        user = cursor.fetchone()
        
        conn.close()
        return user
    
    def update_user_access(self, name):
        """Update user's last seen time and increment access count"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "UPDATE users SET last_seen = ?, access_count = access_count + 1 WHERE name = ?",
            (datetime.now(), name)
        )
        
        conn.commit()
        conn.close()
    
    def log_access_event(self, event_type, person_name=None, details=None):
        """Log an access event to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO access_logs (event_type, person_name, details) VALUES (?, ?, ?)",
            (event_type, person_name, details)
        )
        
        conn.commit()
        conn.close()
    
    def get_recent_access_logs(self, limit=50):
        """Retrieve recent access logs"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT id, timestamp, event_type, person_name, details FROM access_logs ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        )
        logs = cursor.fetchall()
        
        conn.close()
        return logs
    
    def get_user_access_logs(self, person_name):
        """Retrieve access logs for a specific user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT id, timestamp, event_type, details FROM access_logs WHERE person_name = ? ORDER BY timestamp DESC",
            (person_name,)
        )
        logs = cursor.fetchall()
        
        conn.close()
        return logs

# Global database instance
db_manager = DatabaseManager()

if __name__ == "__main__":
    # Test the database
    db = DatabaseManager()
    
    # Add some test users
    db.add_user("John Doe")
    db.add_user("Jane Smith")
    
    # Log some events
    db.log_access_event("SYSTEM_STARTED")
    db.log_access_event("AUTHORIZED_ACCESS", "John Doe")
    db.log_access_event("UNKNOWN_PERSON_DETECTED")
    db.log_access_event("DOOR_OPENED", "John Doe")
    db.log_access_event("DOOR_LOCKED")
    
    # Update user access
    db.update_user_access("John Doe")
    
    # Retrieve data
    print("Users:")
    for user in db.get_all_users():
        print(user)
    
    print("\nRecent Access Logs:")
    for log in db.get_recent_access_logs():
        print(log)