# Database Implementation for Face Recognition Door System

## Overview

This document describes the database implementation for the Face Recognition Door System. The system now uses SQLite to store user information and access logs, providing a centralized data management solution that complements the existing file-based storage.

## Database Schema

### Users Table

```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP,
    access_count INTEGER DEFAULT 0
);
```

### Access Logs Table

```sql
CREATE TABLE access_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    event_type TEXT NOT NULL,
    person_name TEXT,
    details TEXT
);
```

## Key Features

1. **Centralized User Management**: All registered users are stored in the database with additional metadata
2. **Access Tracking**: The system tracks when users were last seen and how many times they've accessed the system
3. **Enhanced Logging**: All access events are stored in the database with timestamps and details
4. **Data Persistence**: User and log data persists between system restarts
5. **Easy Querying**: Database queries provide efficient access to historical data

## Integration Points

### User Registration
- When a new user is registered via `register.py`, they are automatically added to the database
- The database prevents duplicate user registrations

### Access Control
- When an authorized user is recognized, their access count is incremented and last_seen timestamp is updated
- Unknown person detections are logged to the database

### Web Dashboard
- The web dashboard now retrieves user information and access logs from the database
- User deletion removes both file system data and database entries
- Access logs displayed in the dashboard come from the database

### Data Migration
- Existing users and logs from the file system are migrated to the database using `migrate_data.py`
- The migration script preserves all existing data while moving it to the new database format

## Database Manager API

The [DatabaseManager](file:///p:/face%20door%20opening%20system%20111/database.py#L7-L174) class in [database.py](file:///p:/face%20door%20opening%20system%20111/database.py) provides the following methods:

- `add_user(name)`: Add a new user to the database
- `delete_user(name)`: Remove a user from the database
- `get_all_users()`: Retrieve all registered users
- `get_user(name)`: Retrieve a specific user
- `update_user_access(name)`: Update user's last seen time and increment access count
- `log_access_event(event_type, person_name=None, details=None)`: Log an access event
- `get_recent_access_logs(limit=50)`: Retrieve recent access logs
- `get_user_access_logs(person_name)`: Retrieve access logs for a specific user

## Files

- [database.py](file:///p:/face%20door%20opening%20system%20111/database.py): Main database management module
- [migrate_data.py](file:///p:/face%20door%20opening%20system%20111/migrate_data.py): Script to migrate existing data to the database
- [test_database.py](file:///p:/face%20door%20opening%20system%20111/test_database.py): Test script for database functionality
- [door_system.db](file:///p:/face%20door%20opening%20system%20111/door_system.db): SQLite database file (created automatically)

## Benefits

1. **Improved Data Organization**: All user and access information is now stored in a structured format
2. **Enhanced Querying Capabilities**: Complex queries can be easily performed on the data
3. **Better Performance**: Database queries are more efficient than file parsing for large datasets
4. **Data Integrity**: Database constraints ensure data consistency
5. **Scalability**: The system can handle larger amounts of data more efficiently
6. **Backup and Recovery**: Database files can be easily backed up and restored

## Future Enhancements

1. **User Profiles**: Store additional user information such as photos, access permissions, etc.
2. **Advanced Analytics**: Implement more sophisticated access pattern analysis
3. **Audit Trails**: Track all system changes for security auditing
4. **Multi-door Support**: Extend the schema to support multiple door systems