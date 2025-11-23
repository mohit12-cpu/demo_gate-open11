import face_recognition
import cv2
import numpy as np
import os
import sys
import time
from datetime import datetime
import threading
import queue
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False

from database import db_manager

# Global queue for text-to-speech greetings
global_greeting_queue = queue.Queue()

# --- Logging System ---
class DoorLogger:
    """Handles logging of door access events"""
    def __init__(self, log_file="door_access.log"):
        self.log_file = log_file
        self.ensure_log_file()
    
    def ensure_log_file(self):
        """Create log file if it doesn't exist"""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                f.write("Timestamp,Event,Person\n")
    
    def log_event(self, event, person="N/A"):
        """Log an event with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp},{event},{person}\n"
        
        # Print to console
        print(f"[LOG] {timestamp} - {event} - {person}")
        
        # Write to file
        with open(self.log_file, 'a') as f:
            f.write(log_entry)
        
        # Also log to database
        db_manager.log_access_event(event, person)

# --- Email Notification System ---
class EmailNotifier:
    """Sends email notifications for security events"""
    def __init__(self, smtp_server=None, smtp_port=None, email=None, password=None, recipient=None):
        # Use provided values or fallback to environment variables or hardcoded defaults
        self.smtp_server = smtp_server or os.getenv('SMTP_SERVER') or 'smtp.gmail.com'
        self.smtp_port = smtp_port or int(os.getenv('SMTP_PORT', 587))
        self.email = email or os.getenv('SENDER_EMAIL') or 'facialrecognitionandattendance@gmail.com'
        self.password = password or os.getenv('SENDER_PASSWORD') or 'vrpo lozh zygn yzvw'
        self.recipient = recipient or os.getenv('RECIPIENT_EMAIL') or 'shresthamanjil29@gmail.com'
        self.enabled = True  # Always enable email notifications with defaults
        
        if not self.enabled:
            print("[EMAIL] Email notifications disabled. Set SMTP credentials in environment variables to enable.")
    
    def send_notification(self, subject, message, attachment_path=None):
        """Send an email notification with optional attachment"""
        if not self.enabled:
            return False
            
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email
            msg['To'] = self.recipient
            msg['Subject'] = subject
            
            msg.attach(MIMEText(message, 'plain'))
            
            # Add attachment if provided
            if attachment_path and os.path.exists(attachment_path):
                with open(attachment_path, "rb") as attachment:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment.read())
                
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {os.path.basename(attachment_path)}',
                )
                msg.attach(part)
            
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.email, self.password)
            text = msg.as_string()
            server.sendmail(self.email, self.recipient, text)
            server.quit()
            
            print(f"[EMAIL] Notification sent: {subject}")
            return True
        except Exception as e:
            print(f"[EMAIL] Failed to send notification: {e}")
            return False

# --- Simulated GPIO for door control ---
class SimulatedGPIO:
    """Simulates GPIO operations for door control"""
    def __init__(self):
        self.door_locked = True
        self.relay_pin = 18  # Simulated relay pin
    
    def setup(self, pin, mode):
        """Simulate GPIO setup"""
        print(f"[GPIO] Setting up pin {pin} as {mode}")
    
    def output(self, pin, value):
        """Simulate GPIO output"""
        if value == 1:  # HIGH
            self.door_locked = False
            print(f"[GPIO] Pin {pin} set to HIGH - Door UNLOCKED")
        else:  # LOW
            self.door_locked = True
            print(f"[GPIO] Pin {pin} set to LOW - Door LOCKED")
    
    def cleanup(self):
        """Simulate GPIO cleanup"""
        print("[GPIO] Cleaning up GPIO resources")

# --- Door Control System ---
class DoorController:
    """Controls the door locking mechanism"""
    def __init__(self, gpio_instance, logger, email_notifier=None):
        self.gpio = gpio_instance
        self.logger = logger
        self.email_notifier = email_notifier
        self.door_unlocked_time = None
        self.unlock_duration = 5  # seconds
        
        # Setup GPIO pin for relay
        if GPIO_AVAILABLE:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(18, GPIO.OUT)
            GPIO.output(18, GPIO.LOW)  # Ensure door is locked initially
        else:
            self.gpio.setup(18, "OUTPUT")
            self.gpio.output(18, 0)  # Ensure door is locked initially
    
    def unlock_door(self, person_name="Unknown"):
        """Unlock the door for a specified duration"""
        if GPIO_AVAILABLE:
            GPIO.output(18, GPIO.HIGH)
        else:
            self.gpio.output(18, 1)
        
        self.door_unlocked_time = time.time()
        self.logger.log_event("Door Opened", person_name)
        
        print(f"[DOOR] Door unlocked for {person_name}")
    
    def lock_door(self):
        """Lock the door"""
        if GPIO_AVAILABLE:
            GPIO.output(18, GPIO.LOW)
        else:
            self.gpio.output(18, 0)
        
        self.door_unlocked_time = None
        self.logger.log_event("Door Locked")
        print("[DOOR] Door locked")
    
    def check_door_status(self):
        """Check if door should be relocked based on time"""
        if self.door_unlocked_time:
            if time.time() - self.door_unlocked_time >= self.unlock_duration:
                self.lock_door()
    
    def cleanup(self):
        """Clean up GPIO resources"""
        self.lock_door()
        if GPIO_AVAILABLE:
            GPIO.cleanup()
        else:
            self.gpio.cleanup()

# --- User-Friendly Error for Library Issues ---
def handle_library_error(e):
    """Provides a clear, user-friendly error message for the common TypeError."""
    print("---")
    print("FATAL ERROR: A critical issue was detected with your face_recognition or dlib library installation.")
    print(f"Error Details: {e}")
    print("\nThis is a known issue and is not a bug in the Python script. It must be fixed by reinstalling the libraries.")
    print("\nPlease follow these steps exactly in your terminal:")
    print("1. Uninstall the libraries:")
    print("   pip uninstall -y face-recognition dlib")
    print("\n2. Reinstall dlib first, then face-recognition:")
    print("   pip install dlib")
    print("   pip install face-recognition")
    print("\nAfter reinstalling, run the script again.")
    print("---")
    sys.exit(1)

# --- Main Application ---
def main():
    # Initialize systems
    logger = DoorLogger()
    email_notifier = EmailNotifier()
    gpio = GPIO if GPIO_AVAILABLE else SimulatedGPIO()
    door_controller = DoorController(gpio, logger, email_notifier)
    
    logger.log_event("System Started")
    
    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("FATAL ERROR: Cannot open webcam. Is it connected and not in use by another application?")
        logger.log_event("Error", "Cannot open webcam")
        sys.exit(1)
    
    # Create arrays of known face encodings and their names
    known_face_encodings = []
    known_face_names = []
    greeted_this_session = set() # Set to track who has been greeted
    
    # Define allowed image extensions
    image_extensions = ('.jpg', '.jpeg', '.png')
    known_faces_dir = 'known_faces'
    
    print("Loading known faces...")
    # Load sample pictures and learn how to recognize them.
    if os.path.exists(known_faces_dir):
        for image_file in os.listdir(known_faces_dir):
            if image_file.lower().endswith(image_extensions) and not image_file.endswith('_encoding.npy'):
                # Skip individual face images, only load encoding files
                continue
            if image_file.endswith('_encoding.npy'):
                try:
                    # Extract name from encoding file name
                    name = image_file.replace('_encoding.npy', '')
                    print(f" > Loading encoding for {name}...")
                    
                    # Load precomputed encoding
                    encoding_file = os.path.join(known_faces_dir, image_file)
                    image_face_encoding = np.load(encoding_file)
                    
                    known_face_encodings.append(image_face_encoding)
                    known_face_names.append(name)
                except Exception as e:
                    print(f"   Error loading encoding {image_file}: {e}")
    else:
        print(f"Warning: Directory '{known_faces_dir}' not found. No known faces will be loaded.")
    
    if not known_face_encodings:
        print("\nWarning: No known faces were loaded. The system will only detect 'Unknown' faces.")
        print("Please add images to the 'known_faces' directory using register.py\n")
    
    print("...Done loading faces. Starting video stream.")
    
    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    last_unknown_capture_time = 0  # To track when we last captured an unknown person
    last_unknown_face_encoding = None  # To track encoding of last unknown person
    unknown_face_tolerance = 0.6  # Tolerance for comparing unknown faces
    
    try:
        while True:
            # Check if door should be relocked
            door_controller.check_door_status()
            
            # Grab a single frame of video
            ret, frame = video_capture.read()
            if not ret:
                print("Error: Failed to grab frame from webcam. Exiting.")
                logger.log_event("Error", "Failed to grab frame from webcam")
                break
            
            # Only process every other frame of video to save time
            if process_this_frame:
                # Resize frame of video to 1/4 size for faster face recognition processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                
                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                rgb_small_frame = np.ascontiguousarray(rgb_small_frame)
                
                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                
                face_names = []
                for face_encoding in face_encodings:
                    name = "Unknown"
                    if known_face_encodings:
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = known_face_names[best_match_index]
                    
                    face_names.append(name)
                    
                    # Handle door access
                    if name != "Unknown":
                        # If a known person is found and not yet greeted, greet them and unlock door
                        if name not in greeted_this_session:
                            greeted_this_session.add(name)
                            global_greeting_queue.put(name)
                            door_controller.unlock_door(name)
                            logger.log_event("Authorized Access", name)
                            # Update user access in database
                            db_manager.update_user_access(name)
                    else:
                        # Log unknown person and send email notification
                        logger.log_event("Unknown Person Detected")
                        print("[ALERT] Unknown person detected!")
                        
                        # Speak "Unknown person detected" using text-to-speech
                        global_greeting_queue.put("Unknown person detected")
                        
                        # Check if this is likely the same unknown person as before
                        should_capture = True
                        current_time = time.time()
                        
                        # If we have a previous unknown face encoding, compare with current
                        if last_unknown_face_encoding is not None:
                            # Calculate distance between current and previous unknown face
                            distance = face_recognition.face_distance([last_unknown_face_encoding], face_encoding)[0]
                            # If distance is small (faces are similar), don't capture again
                            if distance < unknown_face_tolerance:
                                should_capture = False
                                print("[INFO] Same unknown person detected, skipping capture")
                        
                        # Also apply time-based cooldown as backup
                        if current_time - last_unknown_capture_time < 5:  # 5 second cooldown
                            should_capture = False
                            print("[INFO] Cooldown period active, skipping capture")
                        
                        # Capture only one clear image per unknown person detection
                        if should_capture:
                            # Save encoding for comparison with next unknown face
                            last_unknown_face_encoding = face_encoding
                            
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            image_filename = f"unknown_person_{timestamp}.jpg"
                            image_path = os.path.join("captured_images", image_filename)
                            
                            # Save the current frame
                            cv2.imwrite(image_path, frame)
                            print(f"[CAPTURE] Image saved: {image_path}")
                            
                            # Send email notification with captured image
                            if email_notifier.enabled:
                                subject = "Security Alert - Unknown Person Detected"
                                message = f"An unknown person was detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\nImage attached."
                                email_notifier.send_notification(subject, message, image_path)
                                
                            # Update last capture time
                            last_unknown_capture_time = current_time
                
            process_this_frame = not process_this_frame
            
            # Check if no faces were detected and speak alert
            if len(face_locations) == 0:
                # No faces detected in frame
                print("[ALERT] No face detected!")
                global_greeting_queue.put("Unknown person detected")
            
            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            
            cv2.imshow('Face Recognition Door System - Press "q" to quit', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except TypeError as e:
        handle_library_error(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        logger.log_event("Error", f"Unexpected error: {e}")
    finally:
        # Release handle to the webcam and clean up GPIO
        video_capture.release()
        cv2.destroyAllWindows()
        door_controller.cleanup()
        logger.log_event("System Stopped")

def speak_greetings():
    import pyttsx3
    global global_greeting_queue
    engine = pyttsx3.init()
    while True:
        try:
            greeting = global_greeting_queue.get(timeout=1)
            if greeting == "QUIT":
                break
            engine.say(greeting)
            engine.runAndWait()
        except queue.Empty:
            continue


if __name__ == "__main__":
    # Create captured_images directory if it doesn't exist
    if not os.path.exists("captured_images"):
        os.makedirs("captured_images")
    
    # Initialize the greeting queue and thread
    global_greeting_queue = queue.Queue()
    greeting_thread = threading.Thread(target=speak_greetings, daemon=True)
    
    # Start the greeting thread
    greeting_thread.start()
    
    try:
        main()
    finally:
        global_greeting_queue.put("QUIT")
