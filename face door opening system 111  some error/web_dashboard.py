import os
import json
from flask import Flask, render_template, request, redirect, url_for, jsonify
import csv
from datetime import datetime
from database import db_manager
import face_recognition
import numpy as np
import base64
import io
from PIL import Image
import traceback

app = Flask(__name__)

# Configure maximum file upload size (16MB)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Path configurations
LOG_FILE = 'door_access.log'
KNOWN_FACES_DIR = 'known_faces'

@app.route('/')
def index():
    """Main dashboard page showing registered users"""
    users = get_registered_users()
    return render_template('index.html', users=users)

@app.route('/logs')
def logs():
    """API endpoint to get all logs as JSON"""
    logs = read_access_logs()
    return jsonify(logs)

@app.route('/users')
def users():
    """Page to manage registered users"""
    users = get_registered_users()
    return render_template('users.html', users=users)

@app.route('/register')
def register():
    """Page to register a new user with face capture"""
    return render_template('register.html')

@app.route('/register_user', methods=['POST'])
def register_user():
    """API endpoint to register a new user with face capture"""
    try:
        # Get form data
        user_name = request.form.get('name')
        image_data = request.form.get('image')
        
        if not user_name:
            return jsonify({"status": "error", "message": "User name is required"})
        
        if not image_data:
            return jsonify({"status": "error", "message": "Face image is required"})
        
        # Create known_faces directory if it doesn't exist
        if not os.path.exists(KNOWN_FACES_DIR):
            os.makedirs(KNOWN_FACES_DIR)
        
        # Check if user already exists
        existing_files = [f for f in os.listdir(KNOWN_FACES_DIR) if f.startswith(f"{user_name}_") and (f.endswith('.jpg') or f.endswith('_encoding.npy'))]
        if existing_files:
            return jsonify({"status": "error", "message": f"User {user_name} already exists. Please choose a different name or delete the existing user."})
        
        # Save the image directly to known_faces folder
        try:
            # Remove data URL prefix if present
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            # Decode base64 image data
            image_bytes = base64.b64decode(image_data)
            
            # Save image file directly (single image)
            image_path = os.path.join(KNOWN_FACES_DIR, f"{user_name}_1.jpg")
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            
            # Generate face encoding immediately
            if generate_single_user_encoding(user_name):
                # Add user to database
                db_manager.add_user(user_name)
                return jsonify({"status": "success", "message": f"User {user_name} registered successfully with face capture."})
            else:
                # Clean up the saved image if encoding fails
                if os.path.exists(image_path):
                    os.remove(image_path)
                return jsonify({"status": "error", "message": "Failed to generate face encoding. Please ensure a clear face image with good lighting."})
                
        except Exception as e:
            return jsonify({"status": "error", "message": f"Failed to save face image: {str(e)}"})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to register user: {str(e)}"})

def generate_single_user_encoding(name):
    """
    Generate face encoding for a user from a single captured image
    """
    try:
        # Look for the single image file
        image_file = f"{name}_1.jpg"
        image_path = os.path.join(KNOWN_FACES_DIR, image_file)
        
        if not os.path.exists(image_path):
            print(f"Error: Image file not found for user {name}")
            return False
        
        try:
            # Load image with PIL first to ensure it's valid
            image = Image.open(image_path)
            image = image.convert('RGB')  # Ensure RGB format
            image_array = np.array(image)
            
            # Then process with face_recognition
            face_encodings = face_recognition.face_encodings(image_array)
            
            if len(face_encodings) == 0:
                print(f"Warning: No faces found in {image_file}.")
                return False
            elif len(face_encodings) > 1:
                print(f"Warning: Multiple faces found in {image_file}. Using the first one.")
            
            # Use the first face encoding
            encoding = face_encodings[0]
            
            # Save the encoding as a numpy array
            encoding_path = os.path.join(KNOWN_FACES_DIR, f"{name}_encoding.npy")
            np.save(encoding_path, encoding)
            
            print(f"Face encoding saved as {encoding_path}")
            return True
        except Exception as e:
            print(f"Error processing image {image_file}: {e}")
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"Error generating encoding: {e}")
        traceback.print_exc()
        return False

@app.route('/add_user', methods=['POST'])
def add_user():
    """API endpoint to add a new user to the database"""
    try:
        # Get the user name from the form data
        user_name = request.form.get('name')
        
        if not user_name:
            return jsonify({"status": "error", "message": "User name is required"})
        
        # Check if user already exists
        if not os.path.exists(KNOWN_FACES_DIR):
            os.makedirs(KNOWN_FACES_DIR)
            
        existing_files = [f for f in os.listdir(KNOWN_FACES_DIR) if f.startswith(f"{user_name}_") and (f.endswith('.jpg') or f.endswith('_encoding.npy'))]
        if existing_files:
            # User already exists, return error
            return jsonify({"status": "error", "message": f"User {user_name} already exists"})
        
        # Add user to database
        db_manager.add_user(user_name)
        
        # Return success response
        return jsonify({"status": "success", "message": f"User {user_name} added successfully. Please capture face image for recognition.", "user_name": user_name})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to add user: {str(e)}"})

@app.route('/delete_user/<username>')
def delete_user(username):
    """Delete a registered user"""
    try:
        # Delete user image and encoding files
        if os.path.exists(KNOWN_FACES_DIR):
            for file in os.listdir(KNOWN_FACES_DIR):
                if file.startswith(f"{username}_") and (file.endswith('.jpg') or file.endswith('_encoding.npy')):
                    file_path = os.path.join(KNOWN_FACES_DIR, file)
                    os.remove(file_path)
            
        # Also delete user from database
        db_manager.delete_user(username)
            
        return jsonify({"status": "success", "message": f"User {username} deleted"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

def read_access_logs():
    """Read access logs from the database"""
    logs = []
    db_logs = db_manager.get_recent_access_logs(100)  # Get last 100 logs
    for db_log in db_logs:
        logs.append({
            'timestamp': db_log[1],  # timestamp
            'event': db_log[2],      # event_type
            'person': db_log[3] or "N/A"  # person_name
        })
    return logs

def get_registered_users():
    """Get list of registered users from database"""
    db_users = db_manager.get_all_users()
    users = []
    user_dict = {}  # Use a dictionary to avoid duplicates
    
    # First, process users from database
    for db_user in db_users:
        user_dict[db_user[1]] = {
            'name': db_user[1],
            'trained': False,  # Will be updated if encoding exists
            'created_at': db_user[2],
            'last_seen': db_user[3],
            'access_count': db_user[4]
        }
    
    # Then, check for image files and encodings
    if os.path.exists(KNOWN_FACES_DIR):
        encoding_files = [f for f in os.listdir(KNOWN_FACES_DIR) if f.endswith('_encoding.npy')]
        
        for file in os.listdir(KNOWN_FACES_DIR):
            if file.endswith('.jpg') and '_' in file and not file.startswith('.'):
                # Extract username from filename (before the _number.jpg part)
                parts = file.split('_')
                if len(parts) >= 2:
                    username = '_'.join(parts[:-1])
                    # Check if encoding file exists
                    encoding_file = f"{username}_encoding.npy"
                    trained = encoding_file in encoding_files
                    
                    if username in user_dict:
                        # Update existing user
                        user_dict[username]['trained'] = trained
                    else:
                        # Add new user from image files
                        user_dict[username] = {
                            'name': username,
                            'trained': trained,
                            'created_at': None,
                            'last_seen': None,
                            'access_count': 0
                        }
            elif file.endswith('_encoding.npy'):
                # Handle encoding files directly
                username = file.replace('_encoding.npy', '')
                if username not in user_dict:
                    # Add user from encoding file if not in database
                    user_dict[username] = {
                        'name': username,
                        'trained': True,
                        'created_at': None,
                        'last_seen': None,
                        'access_count': 0
                    }
                else:
                    # Ensure trained status is set correctly
                    user_dict[username]['trained'] = True
    
    # Convert dictionary to list
    users = list(user_dict.values())
    return users

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    app.run(host='0.0.0.0', port=5000, debug=True)