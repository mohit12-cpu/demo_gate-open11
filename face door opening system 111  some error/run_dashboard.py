#!/usr/bin/env python3
"""
Script to run the Face Recognition Door System web dashboard
"""

import os
import sys
import subprocess
import threading
import time

def main():
    print("Face Recognition Door System - Web Dashboard")
    print("=" * 50)
    
    # Check if required files exist
    if not os.path.exists('web_dashboard.py'):
        print("Error: web_dashboard.py not found!")
        return 1
    
    if not os.path.exists('templates'):
        print("Error: templates directory not found!")
        return 1
    
    # Run the web dashboard
    try:
        print("Starting web dashboard on http://localhost:5000")
        print("Press Ctrl+C to stop")
        subprocess.run([sys.executable, 'web_dashboard.py'])
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
    except Exception as e:
        print(f"Error running dashboard: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())