#!/usr/bin/env python3
"""
Simple script to run the PDF RAG Backend server
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False

def run_server():
    """Run the Flask server"""
    print("🚀 Starting PDF RAG Backend Server...")
    try:
        # Set environment variable for better error handling
        os.environ['FLASK_ENV'] = 'development'
        subprocess.run([sys.executable, "pdf_rag_backend_api.py"])
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")

if __name__ == "__main__":
    print("🔧 PDF RAG Backend Setup")
    print("=" * 40)
    
    # Check if requirements.txt exists
    if os.path.exists("requirements.txt"):
        install_requirements()
    
    # Run the server
    run_server()
