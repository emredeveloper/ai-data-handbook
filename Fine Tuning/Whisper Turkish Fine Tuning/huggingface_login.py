"""
Separate script to log in to Hugging Face Hub
"""

from huggingface_hub import login
import os

def huggingface_login():
    """
    Logs in to Hugging Face Hub
    """
    print("Logging in to Hugging Face Hub...")
    
    # If token exists in environment variable, use it
    token = os.getenv("HUGGINGFACE_TOKEN")
    
    if token:
        print("Using token from environment variable...")
        login(token=token)
        print("Login successful!")
    else:
        print("Token not found. Proceeding with manual login...")
        print("You can get your token from https://huggingface.co/settings/tokens")
        login()
        print("Login successful!")

if __name__ == "__main__":
    huggingface_login()
