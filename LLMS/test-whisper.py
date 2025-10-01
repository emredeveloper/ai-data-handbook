import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import json
import jiwer

# ============================================================================
# SES DOSYASI & REFERANS METİN (Opsiyonel)
# ============================================================================
audio_file = "audio.mp3"  # Buraya kendi ses dosyanı yaz

# WER hesaplamak için gerçek metni buraya yazabilirsin (opsiyonel)
reference_text = "Enflasyon kelimesini duyduğunuzda genelde kastedilen fiyat enflasyonudur. Yani bir mal ve hizmet sepetinin genel fiyat seviyesindeki yükselmedir."  # Örnek: "Apple telefonları hakkında herkes bir şeyler söyledi..."

# ============================================================================
# MODEL YÜKLEME
# ============================================================================
print("🚀 Model yükleniyor...")

# Config
with open("whisper_best_config.json", 'r', encoding='utf-8') as f:
    config = json.load(f)

# Model
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

print(f"✓ Hazır ({device})\n")

# ============================================================================
# TRANSKRİPSİYON
# ============================================================================
try:
    # Ses yükle
    audio, sr = librosa.load(audio_file, sr=16000)
    duration = len(audio) / sr
    print(f"📝 Transkribe ediliyor... ({duration:.1f}s)\n")
    
    # Input features (her iki test için aynı)
    input_features = processor(
        audio, 
        sampling_rate=16000, 
        return_tensors="pt"
    ).input_features.to(device)
    
    # ========================================================================
    # 1. BASELINE (Config Olmadan)
    # ========================================================================
    print("1️⃣  BASELINE (Standart Ayarlar)")
    print("─" * 80)
    
    baseline_ids = model.generate(
        input_features,
        language="tr",
        task="transcribe",
        max_length=448
    )
    baseline_text = processor.batch_decode(baseline_ids, skip_special_tokens=True)[0]
    
    print(baseline_text)
    print("─" * 80)
    
    # ========================================================================
    # 2. CONFIG İLE (Optimize Edilmiş)
    # ========================================================================
    print("\n2️⃣  OPTİMİZE EDİLMİŞ (Best Config)")
    print("─" * 80)
    
    gen_kwargs = config["generation_params"].copy()
    
    if config.get("use_prompt") and config.get("prompt_text"):
        prompt_ids = processor.get_prompt_ids(
            config["prompt_text"].strip(), 
            return_tensors="pt"
        )
        optimized_ids = model.generate(
            input_features,
            prompt_ids=prompt_ids.to(device),
            **gen_kwargs
        )
    else:
        optimized_ids = model.generate(input_features, **gen_kwargs)
    
    optimized_text = processor.batch_decode(optimized_ids, skip_special_tokens=True)[0]
    
    print(optimized_text)
    print("─" * 80)
    
    # ========================================================================
    # 3. KARŞILAŞTIRMA & WER ANALİZİ
    # ========================================================================
    print("\n📊 KARŞILAŞTIRMA")
    print("=" * 80)
    
    # Referans metin var mı kontrol et (boş string değil)
    has_reference = reference_text and reference_text.strip()
    
    if has_reference:
        # WER Hesaplama
        baseline_wer = jiwer.wer(reference_text, baseline_text) * 100
        optimized_wer = jiwer.wer(reference_text, optimized_text) * 100
        improvement = baseline_wer - optimized_wer
        
        print(f"\n🎯 WER (Referans Metne Göre):")
        print(f"   📌 Referans: {len(reference_text.split())} kelime")
        print(f"   🔵 Baseline:  {baseline_wer:.2f}%")
        print(f"   🟢 Optimize:  {optimized_wer:.2f}%")
        print(f"   📊 Fark:      {improvement:+.2f}%")
        
        if improvement > 2:
            print(f"   ✅ Optimize açık ara daha iyi!")
        elif improvement > 0:
            print(f"   ✅ Optimize biraz daha iyi")
        elif improvement < -2:
            print(f"   ⚠️  Baseline açık ara daha iyi!")
        elif improvement < 0:
            print(f"   ⚠️  Baseline biraz daha iyi")
        else:
            print(f"   ⚡ İkisi de aynı performans!")
    
    # Kelime sayıları
    baseline_words = len(baseline_text.split())
    optimized_words = len(optimized_text.split())
    
    print(f"\n📝 Kelime Sayısı:")
    print(f"   Baseline: {baseline_words} kelime")
    print(f"   Optimize: {optimized_words} kelime")
    if has_reference:
        ref_words = len(reference_text.split())
        print(f"   Referans: {ref_words} kelime")
    
    # Farklılıklar
    if baseline_text != optimized_text:
        baseline_set = set(baseline_text.lower().split())
        optimized_set = set(optimized_text.lower().split())
        
        only_baseline = baseline_set - optimized_set
        only_optimized = optimized_set - baseline_set
        
        if only_baseline or only_optimized:
            print(f"\n🔄 Farklı Kelimeler:")
            if only_baseline:
                words = ', '.join(list(only_baseline)[:15])
                print(f"   🔵 Sadece Baseline'da: {words}")
            if only_optimized:
                words = ', '.join(list(only_optimized)[:15])
                print(f"   🟢 Sadece Optimize'de: {words}")
    else:
        print(f"\n✅ İki sonuç tamamen aynı!")
    
    print("=" * 80)
    
except FileNotFoundError:
    print(f"❌ Dosya bulunamadı: {audio_file}")
except Exception as e:
    print(f"❌ Hata: {e}")
