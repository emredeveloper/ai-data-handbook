"""
Model dosyalarını checkpoint'ten ana klasöre kopyalar
"""
import os
import shutil
from pathlib import Path

def copy_model_files():
    # Yolları tanımla
    checkpoint_dir = Path("whisper-small-turkish/checkpoint-600")
    target_dir = Path("whisper-small-turkish")
    
    # Kopyalanacak dosyalar
    files_to_copy = [
        "model.safetensors",
        "generation_config.json"
    ]
    
    print("🔄 Model dosyaları kopyalanıyor...")
    
    for file_name in files_to_copy:
        source_file = checkpoint_dir / file_name
        target_file = target_dir / file_name
        
        if source_file.exists():
            try:
                shutil.copy2(source_file, target_file)
                print(f"✅ {file_name} kopyalandı")
            except Exception as e:
                print(f"❌ {file_name} kopyalanamadı: {e}")
        else:
            print(f"⚠️ {file_name} bulunamadı")
    
    print("🎯 Kopyalama işlemi tamamlandı!")

if __name__ == "__main__":
    copy_model_files()
