# Email Notification Features for Face Recognition Door System

## Overview

This document describes the enhanced email notification features implemented for the Face Recognition Door System. The system now automatically captures photos of unknown persons and sends them via email to the configured recipient.

## New Features

### 1. Automatic Photo Capture for Unknown Persons
- When an unknown person is detected, the system automatically captures a photo
- Photos are saved in the [captured_images/](file:///p:/face%20door%20opening%20system%20111/captured_images/) directory with timestamped filenames
- Each photo is named in the format: `unknown_person_YYYYMMDD_HHMMSS.jpg`

### 2. Email Notifications with Photo Attachments
- All security alerts now include captured images as email attachments
- Enhanced EmailNotifier class supports sending emails with file attachments
- Uses MIME encoding to properly attach image files to emails

### 3. Pre-configured Email Settings
- Gmail account pre-configured: facialrecognitionandattendance@gmail.com
- App-specific password: vrpo lozh zygn yzvw
- Recipient email: shresthamanjil29@gmail.com

## Technical Implementation

### Email Notifier Enhancements
The [EmailNotifier](file:///p:/face%20door%20opening%20system%20111/main.py#L51-L88) class in [main.py](file:///p:/face%20door%20opening%20system%20111/main.py) was enhanced with a new method signature:
```python
def send_notification(self, subject, message, attachment_path=None)
```

Key improvements:
- Added support for file attachments using MIMEBase
- Automatic encoding of binary files for email transmission
- Proper Content-Disposition headers for attachments

### Unknown Person Detection
Enhanced detection logic in the main face recognition loop:
1. When an unknown person is detected, capture the current video frame
2. Save the image to the [captured_images/](file:///p:/face%20door%20opening%20system%20111/captured_images/) directory
3. Send an email notification with the captured image attached
4. Include timestamp and event details in the email body

### File Management
- Created [captured_images/](file:///p:/face%20door%20opening%20system%20111/captured_images/) directory for storing captured photos
- Automatic filename generation with timestamps to prevent overwrites
- Images are captured in JPG format for optimal email transmission

## Security Events That Trigger Notifications

1. **Unknown Person Detection**
   - Captures current frame showing the unknown person
   - Sends email with attached photo to shresthamanjil29@gmail.com
   - Includes timestamp and event details in email body

2. **Authorized Access**
   - Sends email notification when known users are recognized
   - Includes user name and timestamp

3. **System Events**
   - Notifies on system start and stop events
   - Helps monitor system status and uptime

## Testing

Created test scripts to verify functionality:
- [test_email_simple.py](file:///p:/face%20door%20opening%20system%20111/test_email_simple.py): Standalone test of email with attachment functionality
- [test_email_attachment.py](file:///p:/face%20door%20opening%20system%20111/test_email_attachment.py): Integrated test using the EmailNotifier class

## Configuration

Email settings are configured through environment variables:
- `SMTP_SERVER`: smtp.gmail.com
- `SMTP_PORT`: 587
- `SENDER_EMAIL`: facialrecognitionandattendance@gmail.com
- `SENDER_PASSWORD`: vrpo lozh zygn yzvw
- `RECIPIENT_EMAIL`: shresthamanjil29@gmail.com

These are automatically set by the [setup_gmail.py](file:///p:/face%20door%20opening%20system%20111/setup_gmail.py) script.

## Benefits

1. **Enhanced Security**: Visual confirmation of unknown persons
2. **Immediate Alerts**: Real-time notifications with evidence
3. **Evidence Collection**: Automatic photo documentation of security events
4. **Easy Investigation**: Timestamped images for incident review
5. **Remote Monitoring**: Receive alerts anywhere with email access

## Future Enhancements

1. **Image Optimization**: Compress images for faster email transmission
2. **Batch Processing**: Send multiple images in a single email for multiple detections
3. **Cloud Storage**: Upload images to cloud storage for permanent retention
4. **Privacy Features**: Blur or anonymize non-relevant areas in captured images
5. **Customizable Alerts**: Configure which events trigger photo capture and emails