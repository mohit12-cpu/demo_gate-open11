#!/usr/bin/env python3
"""
Data migration script for the face recognition door system.
Migrates existing user data and logs to the new SQLite database.
"""

import os
import sqlite3
from datetime import datetime
from database import db_manager

def migrate_users():
    """Migrate existing users from known_faces directory to database"""
    known_faces_dir = 'known_faces'
    
    if not os.path.exists(known_faces_dir):
        print("No known_faces directory found. Skipping user migration.")
        return
    
    migrated_count = 0
    
    # Get all encoding files
    for file in os.listdir(known_faces_dir):
        if file.endswith('_encoding.npy'):
            # Extract username from filename
            username = file.replace('_encoding.npy', '')
            
            # Add user to database
            if db_manager.add_user(username):
                print(f"Migrated user: {username}")
                migrated_count += 1
            else:
                print(f"User {username} already exists in database")
    
    print(f"Migrated {migrated_count} users to database")

def migrate_logs():
    """Migrate existing logs from door_access.log to database"""
    log_file = 'door_access.log'
    
    if not os.path.exists(log_file):
        print("No door_access.log file found. Skipping log migration.")
        return
    
    migrated_count = 0
    
    try:
        with open(log_file, 'r') as f:
            # Skip header line
            next(f)
            
            for line in f:
                if line.strip():
                    parts = line.strip().split(',', 2)
                    if len(parts) == 3:
                        timestamp_str, event, person = parts
                        
                        # Parse timestamp
                        try:
                            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                        except ValueError:
                            timestamp = datetime.now()
                        
                        # Add to database
                        db_manager.log_access_event(event, person if person != "N/A" else None)
                        migrated_count += 1
        
        print(f"Migrated {migrated_count} log entries to database")
        
    except Exception as e:
        print(f"Error migrating logs: {e}")

def main():
    """Main migration function"""
    print("Starting data migration to SQLite database...")
    
    # Initialize database (creates tables if they don't exist)
    db_manager.init_database()
    
    # Migrate users
    migrate_users()
    
    # Migrate logs
    migrate_logs()
    
    print("Data migration completed!")

if __name__ == "__main__":
    main()