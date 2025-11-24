import face_recognition
import cv2
import os
import sys
import time
from datetime import datetime
import numpy as np
from database import db_manager

def capture_user_images(name, num_images=3):
    """
    Capture multiple user images and save them to known_faces directory
    """
    # Create known_faces directory if it doesn't exist
    if not os.path.exists('known_faces'):
        os.makedirs('known_faces')
    
    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0)
    
    if not video_capture.isOpened():
        print("Error: Cannot open webcam")
        return False
    
    captured_count = 0
    print(f"Capturing {num_images} images for {name}. Press 'c' to capture each image, 'q' to quit.")
    
    while captured_count < num_images:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Failed to grab frame from webcam")
            break
            
        # Display the resulting image
        cv2.imshow(f'Capture {captured_count+1}/{num_images} - Press "c" to capture, "q" to quit', frame)
        
        # Use a short delay to allow for key press detection
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            # Save the captured image
            filename = os.path.join('known_faces', f"{name}_{captured_count+1}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Image {captured_count+1} saved as {filename}")
            captured_count += 1
            
            if captured_count < num_images:
                print(f"Ready for image {captured_count+1}. Position yourself and press 'c' when ready.")
                # Wait a bit before next capture
                time.sleep(1)
        elif key == ord('q'):
            break
    
    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
    
    return captured_count > 0

def encode_user_faces(name):
    """
    Encode all images for a user and save the average encoding
    """
    image_files = []
    for file in os.listdir('known_faces'):
        if file.startswith(f"{name}_") and file.endswith('.jpg'):
            image_files.append(file)
    
    if not image_files:
        print(f"Error: No images found for user {name}")
        return False
    
    encodings = []
    for image_file in image_files:
        try:
            image_path = os.path.join('known_faces', image_file)
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)
            
            if len(face_encodings) == 0:
                print(f"Warning: No faces found in {image_file}. Skipping.")
                continue
            elif len(face_encodings) > 1:
                print(f"Warning: Multiple faces found in {image_file}. Using the first one.")
            
            encodings.append(face_encodings[0])
        except Exception as e:
            print(f"Error processing image {image_file}: {e}")
    
    if not encodings:
        print("Error: No valid face encodings generated")
        return False
    
    # Calculate the average encoding
    avg_encoding = np.mean(encodings, axis=0)
    
    # Save the average encoding as a numpy array
    encoding_path = os.path.join('known_faces', f"{name}_encoding.npy")
    np.save(encoding_path, avg_encoding)
    
    print(f"Average face encoding saved as {encoding_path}")
    return True

def register_user():
    """
    Main function to register a new user
    """
    print("=== Face Recognition Door System - User Registration ===")
    
    # Get user name
    name = input("Enter the name of the person to register: ").strip()
    
    if not name:
        print("Error: Name cannot be empty")
        return
    
    # Check if user already exists
    existing_files = [f for f in os.listdir('known_faces') if f.startswith(f"{name}_") and (f.endswith('.jpg') or f.endswith('_encoding.npy'))]
    if existing_files:
        response = input(f"User {name} already exists with {len(existing_files)} files. Overwrite? (y/n): ").strip().lower()
        if response != 'y':
            print("Registration cancelled")
            return
        # Delete existing files
        for file in existing_files:
            os.remove(os.path.join('known_faces', file))
    
    # Capture user images
    num_images = input("How many images to capture? (default: 3): ").strip()
    try:
        num_images = int(num_images) if num_images else 3
        num_images = max(1, min(10, num_images))  # Limit between 1 and 10
    except ValueError:
        num_images = 3
    
    if not capture_user_images(name, num_images):
        print("Failed to capture images")
        return
    
    # Encode user faces
    if not encode_user_faces(name):
        print("Failed to encode faces")
        return
    
    print(f"Successfully registered user: {name} with {num_images} images")
    
    # Add user to database
    db_manager.add_user(name)
    print(f"User {name} added to database")

if __name__ == "__main__":
    register_user()