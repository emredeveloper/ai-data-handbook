#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Librosa ile ses dosyası decode işlemini test et ve çıktıyı gör
(soundfile OPUS formatını desteklemiyor, librosa kullanıyoruz)
"""

# ÖNCE: torchcodec kullanımını devre dışı bırak (Windows FFmpeg sorunu için)
import os
import sys
os.environ["HF_DATASETS_AUDIO_BACKEND"] = "soundfile"

# Torchcodec import'unu engelle (monkey patch)
import importlib.util
if importlib.util.find_spec("torchcodec") is not None:
    # torchcodec modülünü kara listeye al
    sys.modules['torchcodec'] = None
    sys.modules['torchcodec.decoders'] = None

import librosa
import datasets
from datasets import Audio
import numpy as np

# datasets.features.audio modülünü patch'le
import datasets.features.audio as audio_module

# Orijinal decode_example fonksiyonunu kaydet
_original_decode_example = audio_module.Audio.decode_example

def patched_decode_example(self, value, token_per_repo_id=None):
    """Soundfile/librosa kullanan özel decode fonksiyonu"""
    try:
        import soundfile as sf
        import io
        
        # Sampling rate al (yoksa None)
        target_sr = getattr(self, 'sampling_rate', None)
        
        if value.get("bytes"):
            # Bytes varsa oku
            audio_bytes = value["bytes"]
            audio_array, sampling_rate = sf.read(io.BytesIO(audio_bytes))
        elif value.get("path"):
            # Path varsa oku (librosa OPUS destekler)
            audio_array, sampling_rate = librosa.load(value["path"], sr=target_sr, mono=True)
        else:
            # Zaten array varsa direkt dön
            return value
        
        # Stereo ise mono'ya çevir
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)
        
        return {"path": value.get("path"), "array": audio_array, "sampling_rate": sampling_rate}
    except Exception as e:
        print(f"⚠️  Audio decode hatası: {e}")
        import traceback
        traceback.print_exc()
        return value

# Patch'i uygula
audio_module.Audio.decode_example = patched_decode_example
print("✅ Audio decoder başarıyla patch'lendi (soundfile/librosa kullanılacak)")

print("📦 Yerel dosya kontrol ediliyor...")
local_audio_path = os.path.join(os.path.dirname(__file__), "iphone-air.mp3")

if os.path.exists(local_audio_path):
    print(f"\n🎵 Yerel dosya bulundu: {local_audio_path}")
    # Librosa ile 16kHz mono decode et
    audio_array, sampling_rate = librosa.load(local_audio_path, sr=16000, mono=True)
    print("\n✅ Yerel dosya decode edildi (librosa, 16kHz mono)")
    source_info = {"path": local_audio_path}
else:
    # Dataset'e geri düş
    print("📦 Veri seti yükleniyor...")
    dataset = datasets.load_dataset("ysdede/khanacademy-turkish", split="train")
    # Audio'yu 16kHz'e ayarla (Whisper için)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    # İlk örneği al (datasets otomatik decode eder)
    print("\n🎵 İlk ses dosyası bilgileri:")
    example = dataset[0]
    print(f"Metin: {example['transcription'][:100]}...")
    # Audio bilgilerini al (datasets tarafından decode edilmiş)
    audio_info = example['audio']
    audio_array = audio_info['array']
    sampling_rate = audio_info['sampling_rate']
    print(f"Path: {audio_info.get('path', 'cache içinde')}")
    print("\n✅ Datasets tarafından otomatik decode edildi!")
    source_info = audio_info
print(f"Path: {source_info.get('path', 'yerel dosya')}" )
print("\n✅ Datasets tarafından otomatik decode edildi!")

# Çıktıları göster
print("\n📊 DECODE SONUCU:")
print(f"  Ses array tipi: {type(audio_array)}")
print(f"  Array shape: {audio_array.shape}")
print(f"  Veri tipi: {audio_array.dtype}")
print(f"  Sampling rate: {sampling_rate} Hz")
print(f"  Süre: {len(audio_array) / sampling_rate:.2f} saniye")
print(f"  Min değer: {audio_array.min():.6f}")
print(f"  Max değer: {audio_array.max():.6f}")
print(f"  Ortalama: {audio_array.mean():.6f}")
print(f"  İlk 10 değer: {audio_array[:10]}")

# Bellek boyutu
memory_mb = audio_array.nbytes / 1024 / 1024
print(f"\n💾 Bellek kullanımı: {memory_mb:.2f} MB")

print("\n✅ İşte bu array Whisper processor'a gidiyor!")
print(f"   Array'deki {len(audio_array):,} adet sayısal değer → Mel-Spectrogram → Model")

