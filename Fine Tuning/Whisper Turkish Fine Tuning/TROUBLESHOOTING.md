# 🛠️ Whisper Turkish Fine-tuning - Sorun Giderme Kılavuzu

## 🚨 Yaygın Sorunlar ve Çözümleri

### 1. Hugging Face Upload Hatası (I/O Error 1224)

**Hata Mesajı:**
```
❌ Model yükleme hatası: Error while serializing: I/O error: İstenen işlem, kullanıcıya eşleşmiş bölümü açık olan bir dosyada yürütülemez. (os error 1224)
```

**Sebep:** Windows dosya kilitleme sorunu - model dosyaları hala başka bir süreç tarafından kullanılıyor.

**Çözüm:**
1. **Tüm Python süreçlerini kapatın**
2. **Gelişmiş çözüm scripti kullanın:**
   ```bash
   python fix_and_upload.py
   ```
3. **Manuel çözüm:**
   ```bash
   python convert_to_pytorch.py
   python huggingface_push_model.py
   ```

### 2. SafeTensors vs PyTorch Format Sorunu

**Sorun:** Model SafeTensors formatında ama PyTorch formatı gerekiyor.

**Çözüm:**
```bash
python convert_to_pytorch.py
```

Bu script:
- Windows dosya kilitleme sorunlarını çözer
- Geçici dizin kullanarak güvenli dönüşüm yapar
- Model doğrulaması yapar

### 3. Model Dosyaları Eksik

**Kontrol edilecek dosyalar:**
- `config.json` ✅
- `model.safetensors` ✅
- `pytorch_model.bin` ✅
- `tokenizer.json` ✅
- `vocab.json` ✅
- `merges.txt` ✅

**Çözüm:** Fine-tuning scriptini tekrar çalıştırın.

### 4. Hugging Face Token Sorunu

**Hata:** Token bulunamadı veya geçersiz.

**Çözüm:**
1. https://huggingface.co/settings/tokens adresinden token alın
2. Token'ı girin:
   ```bash
   python huggingface_login.py
   ```

## 🔧 Gelişmiş Çözümler

### Otomatik Problem Çözücü

En kolay yöntem - tüm sorunları otomatik çözer:
```bash
python fix_and_upload.py
```

Bu script:
- ✅ Model dosyalarını kontrol eder
- ✅ Eksik PyTorch formatını oluşturur
- ✅ Windows dosya kilitleme sorunlarını çözer
- ✅ Modeli Hugging Face'e yükler
- ✅ Alternatif upload yöntemleri dener

### Manuel Adım Adım Çözüm

1. **Model durumunu kontrol edin:**
   ```bash
   dir whisper-small-turkish
   ```

2. **PyTorch formatına çevirin:**
   ```bash
   python convert_to_pytorch.py
   ```

3. **Hugging Face'e yükleyin:**
   ```bash
   python huggingface_push_model.py
   ```

## 🎯 Optimizasyonlar

### Windows İçin Özel Optimizasyonlar

Scriptler aşağıdaki Windows-özel optimizasyonları içerir:

1. **Geçici dizin kullanımı** - dosya kilitleme sorunlarını önler
2. **Dosya kopyalama stratejisi** - güvenli dosya işlemleri
3. **Rate limiting** - upload sırasında hız sınırlaması
4. **Alternatif upload yöntemleri** - ana yöntem başarısız olursa
5. **Automatic cleanup** - geçici dosyaları temizler

### Model Upload Stratejileri

1. **Birincil yöntem:** `model.push_to_hub()` ile toplu upload
2. **İkincil yöntem:** `HfApi.upload_file()` ile tek tek dosya upload
3. **SafeTensors önceliği:** Daha güvenli serialization
4. **Dosya parçalama:** Büyük dosyalar için 1GB parçalar

## 📊 Model Performansı

Fine-tuned model özellikleri:
- **Base Model:** openai/whisper-small
- **Language:** Turkish
- **Training Steps:** 500
- **Dataset:** Khan Academy Turkish (5000 train + 100 test)
- **Optimizations:** SpecAugment, LayerDrop, Turkish suppress tokens

## 🚀 Kullanım Örnekleri

### Pipeline ile Kullanım
```python
from transformers import pipeline

pipe = pipeline('automatic-speech-recognition', model='whisper-small-turkish')
result = pipe('ses_dosyasi.wav')
print(result['text'])
```

### Doğrudan Model Kullanımı
```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

processor = WhisperProcessor.from_pretrained('./whisper-small-turkish')
model = WhisperForConditionalGeneration.from_pretrained('./whisper-small-turkish')

# Ses işleme
inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    generated_ids = model.generate(inputs["input_features"])
transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

## 💡 İpuçları

1. **Dosya boyutları:** Model dosyaları ~2GB olacak
2. **Upload süresi:** İnternet hızınıza bağlı olarak 5-15 dakika
3. **RAM kullanımı:** Upload sırasında ~4GB RAM gerekli
4. **Disk alanı:** Geçici dosyalar için ek ~2GB alan gerekli

## 🆘 Hala Sorun mu Yaşıyorsunuz?

1. **Tüm Python süreçlerini kapatın**
2. **Bilgisayarı yeniden başlatın**
3. **fix_and_upload.py scriptini kullanın**
4. **İnternet bağlantınızı kontrol edin**
5. **Hugging Face token'ınızı doğrulayın**

---

*Bu kılavuz Windows 10/11 sistemleri için optimize edilmiştir.*
