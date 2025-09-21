"""
SafeTensors modelini PyTorch formatına çeviren script
Windows dosya kilitleme sorunlarını çözen gelişmiş versiyon
"""

import torch
from transformers import WhisperForConditionalGeneration
from safetensors.torch import load_file
import os
import shutil
import tempfile
import time

def convert_safetensors_to_pytorch():
    model_dir = "./whisper-small-turkish"
    safetensors_path = os.path.join(model_dir, "model.safetensors")
    pytorch_path = os.path.join(model_dir, "pytorch_model.bin")
    
    # Dosya varlığını kontrol et
    if not os.path.exists(safetensors_path):
        print(f"ERROR: SafeTensors dosyasi bulunamadi: {safetensors_path}")
        return False
    
    if os.path.exists(pytorch_path):
        print("WARNING: PyTorch model dosyasi zaten mevcut!")
        response = input("Uzerine yazilsin mi? (y/N): ").lower().strip()
        if response != 'y' and response != 'yes':
            print("Islem iptal edildi.")
            return True
    
    print("SafeTensors'dan PyTorch formatina ceviriliyor...")
    
    # Windows dosya kilitleme sorunlarını önlemek için geçici dizin kullan
    temp_dir = None
    try:
        # Geçici dizin oluştur
        temp_dir = tempfile.mkdtemp(prefix="safetensors_convert_")
        temp_safetensors = os.path.join(temp_dir, "model.safetensors")
        temp_pytorch = os.path.join(temp_dir, "pytorch_model.bin")
        
        # SafeTensors dosyasını geçici dizine kopyala
        print("SafeTensors kopyalaniyor...")
        shutil.copy2(safetensors_path, temp_safetensors)
        
        # Kısa bekleme
        time.sleep(1)
        
        # SafeTensors dosyasını yükle
        print("SafeTensors yukleniyor...")
        state_dict = load_file(temp_safetensors)
        print("SafeTensors dosyasi yuklendi!")
        
        # PyTorch formatında geçici dizine kaydet
        print("PyTorch formatina ceviriliyor...")
        torch.save(state_dict, temp_pytorch)
        
        # Dosya boyutlarını kontrol et
        safetensors_size = os.path.getsize(temp_safetensors) / (1024*1024)
        pytorch_size = os.path.getsize(temp_pytorch) / (1024*1024)
        
        print(f"SafeTensors: {safetensors_size:.1f} MB")
        print(f"PyTorch: {pytorch_size:.1f} MB")
        
        # Kısa bekleme
        time.sleep(1)
        
        # Ana dizine kopyala
        print("Ana dizine kopyalaniyor...")
        shutil.copy2(temp_pytorch, pytorch_path)
        
        print("Donusum tamamlandi!")
        print("PyTorch formatinda kaydedildi!")
        
        # Model yükleme testi
        print("Model yukleme testi yapiliyor...")
        try:
            model = WhisperForConditionalGeneration.from_pretrained(model_dir)
            print("Model basariyla yuklendi ve dogrulandi!")
            del model  # Belleği temizle
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception as test_e:
            print(f"WARNING: Model yukleme testi basarisiz: {test_e}")
            print("Model dosyasi olusturuldu ama dogrulama yapilamadi.")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        print("TIP: Lutfen tum Python sureclerini kapatip tekrar deneyin.")
        return False
        
    finally:
        # Geçici dizini temizle
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"Gecici dizin temizlendi: {temp_dir}")
            except Exception as cleanup_e:
                print(f"Gecici dizin temizleme hatasi: {cleanup_e}")

def main():
    print("SafeTensors -> PyTorch Donusturucu")
    print("Windows dosya kilitleme sorunlarini cozen gelismis versiyon\n")
    
    success = convert_safetensors_to_pytorch()
    
    if success:
        print("\nDonusum basariyla tamamlandi!")
        print("Model artik PyTorch formatinda kullanima hazir.")
    else:
        print("\nDonusum basarisiz!")
        print("Lutfen hata mesajlarini kontrol edin ve tekrar deneyin.")

if __name__ == "__main__":
    main()