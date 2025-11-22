# Face Recognition Door Opening System

A complete face recognition door access control system that can detect and recognize faces in real-time using a webcam. When an authorized face is recognized, the system triggers a door unlocking mechanism using either a relay module (on Raspberry Pi) or a simulated GPIO signal.

## Enhanced Features

- **Web Dashboard**: Monitor access logs and manage users through a web interface
- **Improved Face Recognition**: Capture multiple images per user for better accuracy
- **Email Notifications**: Receive alerts for security events with photo attachments
- **Database Storage**: Centralized SQLite database for user and access log management
- **Real-time Face Detection & Recognition**
- **User Registration System**
- **Door Unlocking Mechanism**
- **Comprehensive Logging**
- **Voice Greetings**
- **Cross-platform Compatibility**
- **Security Alerts**: Automatic photo capture and email notifications for unknown persons

## Technologies Used

- **OpenCV** - For face detection and video processing
- **face_recognition** - For face encoding and matching
- **NumPy** - For numerical computations
- **pyttsx3** - For voice greetings
- **Flask** - For web dashboard
- **RPi.GPIO** - For Raspberry Pi GPIO control (optional)

## Installation

### Prerequisites

- Python 3.6 or higher
- A webcam connected to your computer

### Install Dependencies

1. Clone or download this repository

```bash
gh repo clone mohit12-cpu/demo_gate11
```

2. Navigate to the project directory
3. Install the required packages:

```bash
pip install -r requirements.txt
```

**Note for Windows users:** If you encounter issues installing `dlib` (a dependency of `face-recognition`), you might need to install it separately:

```bash
pip install dlib
pip install face-recognition
```

**Note for Raspberry Pi users:** The `RPi.GPIO` library is pre-installed on Raspberry Pi OS. On other systems, it will fall back to simulation mode.

## Usage

### 1. Register New Users

To add new authorised users to the system:

```bash
python register.py
```

Follow the prompts to:
- Enter the name of the person to register
- Position the person in front of the camera
- Press 'c' to capture multiple images (for better recognition accuracy)
- The system will automatically encode the faces

### 2. Run the Door System

To start the face recognition door system:

```bash
python main.py
```

The system will:
- Load all registered users from the `known_faces` directory
- Start the webcam feed
- Detect and recognize faces in real-time
- Unlock the door for authorized users
- Log all access events to `door_access.log`
- Send email notifications for security events (if configured)

Press 'q' to quit the application.

### 3. Run the Web Dashboard

To monitor access logs and manage users through a web interface:

```bash
python run_dashboard.py
```

Then open your browser to http://localhost:5000

Features:
- View recent access logs with color-coded events
- See registered users and their status
- Delete users
- Add new users (integration with registration system)

### 4. Configure Email Notifications (Optional)

To enable email notifications for security events:

1. Set the following environment variables:
   - `SMTP_SERVER` - Your SMTP server (e.g., smtp.gmail.com)
   - `SENDER_EMAIL` - Your email address
   - `SENDER_PASSWORD` - Your email password or app-specific password
   - `RECIPIENT_EMAIL` - Email address to receive notifications

2. On Linux/macOS:
   ```bash
   export SMTP_SERVER=smtp.gmail.com
   export SENDER_EMAIL=your_email@gmail.com
   export SENDER_PASSWORD=your_password
   export RECIPIENT_EMAIL=recipient@gmail.com
   ```

3. On Windows:
   ```cmd
   set SMTP_SERVER=smtp.gmail.com
   set SENDER_EMAIL=your_email@gmail.com
   set SENDER_PASSWORD=your_password
   set RECIPIENT_EMAIL=recipient@gmail.com
   ```

## How It Works

### Face Recognition Process

1. **Registration**: When registering a user, the system captures multiple images and computes an average 128-dimensional face encoding
2. **Storage**: Encodings are stored as `.npy` files in the `known_faces` directory
3. **Recognition**: During operation, the system compares detected faces with stored encodings
4. **Matching**: If a match is found within tolerance, the person is recognized

### Door Control

- **Authorized Access**: When a known person is recognized, the door unlocks for 5 seconds
- **GPIO Simulation**: On non-Raspberry Pi systems, GPIO operations are simulated in the console
- **Raspberry Pi**: On Raspberry Pi, pin 18 controls a relay module

### Logging

All events are logged to `door_access.log` with timestamps:
- System start/stop
- Authorized access attempts
- Unknown person detections
- Door unlock/lock events
- Errors and warnings

### Email Notifications

When configured, the system sends email notifications for:
- Door unlock events
- Unknown person detections
- System start/stop events

## Hardware Setup (Raspberry Pi)

To connect a relay module to Raspberry Pi:

1. Connect the relay module to Raspberry Pi GPIO pins:
   - VCC to 5V (Pin 2)
   - GND to Ground (Pin 6)
   - IN to GPIO 18 (Pin 12)

2. Connect the door lock mechanism to the relay:
   - One wire from the lock to the relay's COM (Common) terminal
   - Another wire from the lock to the relay's NO (Normally Open) terminal
   - Power supply for the lock (following manufacturer specifications)

**Safety Note**: Ensure proper electrical isolation and follow all safety guidelines when working with electrical locks and high-voltage systems.

## Project Structure

```
├── main.py              # Main application script
├── register.py          # User registration script
├── web_dashboard.py     # Web dashboard application
├── run_dashboard.py     # Script to run the web dashboard
├── database.py          # Database management module
├── migrate_data.py      # Data migration script
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── door_system.db       # SQLite database file (created on first run)
├── known_faces/         # Directory for registered user faces
│   ├── .gitkeep         # Placeholder to keep directory in git
│   └── *_1.jpg ...      # User face images (multiple per user)
│   └── *_encoding.npy   # Precomputed average face encodings
├── captured_images/     # Directory for captured unknown person images
├── templates/           # HTML templates for web dashboard
│   ├── index.html       # Main dashboard page
│   └── users.html       # User management page
└── door_access.log      # Access log file (created on first run)
```

## Troubleshooting

### Common Issues

1. **"Failed to load HOG model" or similar dlib errors**:
   - Reinstall face_recognition and dlib:
   ```bash
   pip uninstall -y face-recognition dlib
   pip install dlib
   pip install face-recognition
   ```

2. **Webcam not detected**:
   - Ensure the webcam is properly connected
   - Check if another application is using the webcam
   - Try a different USB port

3. **Poor recognition accuracy**:
   - Ensure good lighting conditions
   - Use high-quality face images during registration
   - Capture multiple images from different angles during registration

### Performance Tips

- Use well-lit environments for better face detection
- Position the camera at eye level
- Ensure registered face images are clear and frontal
- Close other CPU-intensive applications for smoother operation
- Capture 3-5 images per user from slightly different angles

## License

This project is open source and available under the MIT License.

## Contributing


Contributions are welcome! Feel free to submit issues and pull requests.
