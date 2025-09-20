from transformers import pipeline

# Modeli yükle
pipe = pipeline("automatic-speech-recognition", model="emredeveloper/whisper-small-turkish")

# MP3 dosyasını transkribe et
result = pipe("iphone-air.mp3")

print("🎯 Transkripsiyon Sonucu:")
print("=" * 50)
print(result["text"])
print("=" * 50)
