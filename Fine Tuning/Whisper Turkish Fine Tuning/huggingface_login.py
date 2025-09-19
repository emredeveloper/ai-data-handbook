"""
Hugging Face Hub'a giriş yapmak için ayrı script
"""

from huggingface_hub import login
import os

def huggingface_login():
    """
    Hugging Face Hub'a giriş yapar
    """
    print("Hugging Face Hub'a giriş yapılıyor...")
    
    # Eğer token environment variable'da varsa kullan
    token = os.getenv("HUGGINGFACE_TOKEN")
    
    if token:
        print("Environment variable'dan token alınıyor...")
        login(token=token)
        print("Giriş başarılı!")
    else:
        print("Token bulunamadı. Manuel giriş yapılacak...")
        print("Token'ınızı https://huggingface.co/settings/tokens adresinden alabilirsiniz")
        login()
        print("Giriş başarılı!")

if __name__ == "__main__":
    huggingface_login()
