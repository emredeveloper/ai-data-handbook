import librosa
import numpy as np
from transformers import WhisperFeatureExtractor

# Ses dosyasının yolu (örnek: aynı klasördeki bir mp3 dosyası)
audio_path = "english.mp3"  # Kendi dosya yolunu yazabilirsiniz

# 1. Ses dosyasını yükle (librosa ile), 16kHz'e yeniden örnekler
audio_array, sr = librosa.load(audio_path, sr=16000)

# 2. Whisper feature extractor ile log-Mel spectrogram çıkar
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
input_features = feature_extractor(audio_array, sampling_rate=16000).input_features[0]

print("Input features shape:", np.array(input_features).shape)
