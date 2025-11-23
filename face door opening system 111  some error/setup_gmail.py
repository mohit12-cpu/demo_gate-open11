#!/usr/bin/env python3
"""
Setup script for configuring Gmail credentials for the Face Recognition Door System.
This script will set the required environment variables for email notifications.
"""

import os
import sys

def setup_gmail_credentials():
    """Setup Gmail credentials for the door system"""
    print("Face Recognition Door System - Gmail Setup")
    print("=" * 45)
    
    # Gmail SMTP settings
    smtp_server = "smtp.gmail.com"
    smtp_port = "587"
    
    # Provided credentials
    sender_email = "facialrecognitionandattendance@gmail.com"
    sender_password = "vrpo lozh zygn yzvw"  # App-specific password
    
    # Get recipient email
    recipient_email = input("Enter recipient email address (to receive notifications): ").strip()
    
    if not recipient_email:
        print("Error: Recipient email is required!")
        return False
    
    # Set environment variables
    os.environ['SMTP_SERVER'] = smtp_server
    os.environ['SMTP_PORT'] = smtp_port
    os.environ['SENDER_EMAIL'] = sender_email
    os.environ['SENDER_PASSWORD'] = sender_password
    os.environ['RECIPIENT_EMAIL'] = recipient_email
    
    # Also save to system environment variables (Windows)
    try:
        os.system(f'setx SMTP_SERVER "{smtp_server}"')
        os.system(f'setx SMTP_PORT "{smtp_port}"')
        os.system(f'setx SENDER_EMAIL "{sender_email}"')
        os.system(f'setx SENDER_PASSWORD "{sender_password}"')
        os.system(f'setx RECIPIENT_EMAIL "{recipient_email}"')
        print("\nEnvironment variables have been set system-wide.")
    except Exception as e:
        print(f"\nWarning: Could not set system environment variables: {e}")
        print("Environment variables will only be available in this session.")
    
    print("\nGmail configuration completed!")
    print(f"SMTP Server: {smtp_server}:{smtp_port}")
    print(f"Sender Email: {sender_email}")
    print(f"Recipient Email: {recipient_email}")
    print("\nEmail notifications are now enabled.")
    
    return True

def test_email_configuration():
    """Test the email configuration by sending a test email"""
    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        from datetime import datetime
        
        # Get credentials from environment
        smtp_server = os.getenv('SMTP_SERVER')
        smtp_port = int(os.getenv('SMTP_PORT', 587))
        sender_email = os.getenv('SENDER_EMAIL')
        sender_password = os.getenv('SENDER_PASSWORD')
        recipient_email = os.getenv('RECIPIENT_EMAIL')
        
        if not all([smtp_server, sender_email, sender_password, recipient_email]):
            print("Error: Missing email configuration. Please run setup first.")
            return False
        
        # Create test email
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = "Face Recognition Door System - Email"
        
        body = f"""
This is a email from your Face Recognition Door System.
Email notifications are now configured and working correctly.

Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

You will receive notifications for:
- Door unlock events
- Unknown person detections
- System start/stop events
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, recipient_email, text)
        server.quit()
        
        print(f"Test email sent successfully to {recipient_email}")
        return True
        
    except Exception as e:
        print(f"Failed to send test email: {e}")
        return False

if __name__ == "__main__":
    print("Face Recognition Door System - Email Setup")
    print("=" * 45)
    
    # Setup Gmail credentials
    if setup_gmail_credentials():
        # Ask if user wants to send a test email
        send_test = input("\nDo you want to send a test email? (y/n): ").strip().lower()
        if send_test == 'y':
            print("\nSending test email...")
            if test_email_configuration():
                print("Email configuration is working correctly!")
            else:
                print("There was an issue with the email configuration.")
        
        print("\nTo use email notifications:")
        print("1. Restart the door system application")
        print("2. Or run the following commands in your current terminal:")
        print("   set SMTP_SERVER=smtp.gmail.com")
        print("   set SMTP_PORT=587")
        print("   set SENDER_EMAIL=facialrecognitionandattendance@gmail.com")
        print("   set SENDER_PASSWORD=vrpo lozh zygn yzvw")
        print("   set RECIPIENT_EMAIL=your_email@example.com")
    else:
        print("Email setup failed!")
        sys.exit(1)