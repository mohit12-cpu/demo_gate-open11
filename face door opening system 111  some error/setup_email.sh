#!/bin/bash

echo "Setting up email notification environment variables"
echo "================================================"

read -p "Enter SMTP server (e.g., smtp.gmail.com): " smtp_server
read -p "Enter sender email address: " sender_email
read -s -p "Enter sender password (or app-specific password): " sender_password
echo
read -p "Enter recipient email address: " recipient_email

# Export variables for current session
export SMTP_SERVER="$smtp_server"
export SENDER_EMAIL="$sender_email"
export SENDER_PASSWORD="$sender_password"
export RECIPIENT_EMAIL="$recipient_email"

# Add to ~/.bashrc for persistence (Linux)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "export SMTP_SERVER=\"$smtp_server\"" >> ~/.bashrc
    echo "export SENDER_EMAIL=\"$sender_email\"" >> ~/.bashrc
    echo "export SENDER_PASSWORD=\"$sender_password\"" >> ~/.bashrc
    echo "export RECIPIENT_EMAIL=\"$recipient_email\"" >> ~/.bashrc
    echo "Environment variables added to ~/.bashrc"
# Add to ~/.bash_profile for persistence (macOS)
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "export SMTP_SERVER=\"$smtp_server\"" >> ~/.bash_profile
    echo "export SENDER_EMAIL=\"$sender_email\"" >> ~/.bash_profile
    echo "export SENDER_PASSWORD=\"$sender_password\"" >> ~/.bash_profile
    echo "export RECIPIENT_EMAIL=\"$recipient_email\"" >> ~/.bash_profile
    echo "Environment variables added to ~/.bash_profile"
fi

echo
echo "Environment variables have been set:"
echo "SMTP_SERVER=$smtp_server"
echo "SENDER_EMAIL=$sender_email"
echo "SENDER_PASSWORD=********"
echo "RECIPIENT_EMAIL=$recipient_email"
echo
echo "These variables are now available in your current session."