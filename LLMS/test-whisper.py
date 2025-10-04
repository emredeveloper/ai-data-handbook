import os
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.utils import logging as hf_logging
import librosa
import json
import jiwer
from tokenizers import Tokenizer

# ============================================================================
# SES DOSYASI & REFERANS METİN (Opsiyonel)
# ============================================================================
audio_file = "iphone-air.mp3"  # Buraya kendi ses dosyanı yaz

# WER hesaplamak için gerçek metni buraya yazabilirsin (opsiyonel)
reference_text = "Apple telefonları tanıttı, herkes de bir şeyler söyledi, tamam ama bak şimdi. Anladık çok ince telefon yapmışsın, sanki başkası daha önce yapmamış gibi. Ben bu telefonu almaya kalksam yaklaşık 44 bin lira vergi vereceğim. Telefona 98 bin lira verdikten sonra benim sadece tek kanlarım olacak. Ben 11'den devam kardeş."  # Örnek: "Apple telefonları hakkında herkes bir şeyler söyledi..."

# ============================================================================
# MODEL YÜKLEME
# ============================================================================
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
hf_logging.set_verbosity_error()
print("🚀 Model yükleniyor...")

# Türkçe tokenizer yükle
print("🇹🇷 Türkçe tokenizer yükleniyor...")
try:
    turkish_tokenizer = Tokenizer.from_file("turkish_wordpiece.json")
    print("✓ Türkçe tokenizer yüklendi")
except FileNotFoundError:
    print("⚠️  Türkçe tokenizer bulunamadı, standart tokenizer kullanılacak")
    turkish_tokenizer = None

# Config yolu: öncelik ortam değişkeni WHISPER_OUTPUTS_DIR, sonra Whisper/outputs/, sonra mevcut klasör
def _resolve_config_path():
    env_dir = os.environ.get("WHISPER_OUTPUTS_DIR")
    if env_dir:
        p = os.path.join(env_dir, "whisper_best_config.json")
        if os.path.exists(p):
            return p
    here = os.path.dirname(__file__)
    p2 = os.path.join(here, "Whisper", "outputs", "whisper_best_config.json")
    if os.path.exists(p2):
        return p2
    p3 = os.path.join(here, "whisper_best_config.json")
    return p3

config_path = _resolve_config_path()
with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)
print(f"✓ Config: {config_path}")

# Model
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

print(f"✓ Hazır ({device})\n")

# ============================================================================
# TÜRKÇE TOKENIZER POST-PROCESSING
# ============================================================================
def show_turkish_tokenizer_output(text, turkish_tokenizer=None):
    """
    Türkçe tokenizer'ın ham çıktısını göster - düzenleme yok
    """
    if not turkish_tokenizer:
        return text, "Tokenizer yok"
    
    try:
        # Metni tokenize et
        tokens = turkish_tokenizer.encode(text)
        
        # Ham token çıktısı
        raw_tokens = tokens.tokens
        raw_text = turkish_tokenizer.decode(tokens.ids)
        
        return raw_text, raw_tokens
        
    except Exception as e:
        print(f"⚠️  Türkçe tokenizer hatası: {e}")
        return text, []

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
    
    # Türkçe tokenizer ham çıktısı
    baseline_tokenizer_text, baseline_tokens = show_turkish_tokenizer_output(baseline_text, turkish_tokenizer)
    
    print("🔵 Whisper Orijinal:")
    print(baseline_text)
    print("\n🇹🇷 Türkçe Tokenizer Ham Çıktısı:")
    print(baseline_tokenizer_text)
    print(f"\n🔍 Tokenlar: {baseline_tokens}")
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
    
    # Türkçe tokenizer ham çıktısı
    optimized_tokenizer_text, optimized_tokens = show_turkish_tokenizer_output(optimized_text, turkish_tokenizer)
    
    print("🔵 Whisper Orijinal:")
    print(optimized_text)
    print("\n🇹🇷 Türkçe Tokenizer Ham Çıktısı:")
    print(optimized_tokenizer_text)
    print(f"\n🔍 Tokenlar: {optimized_tokens}")
    print("─" * 80)
    
    # ========================================================================
    # 3. KARŞILAŞTIRMA & WER ANALİZİ
    # ========================================================================
    print("\n📊 KARŞILAŞTIRMA")
    print("=" * 80)
    
    # Referans metin var mı kontrol et (boş string değil)
    has_reference = reference_text and reference_text.strip()
    
    if has_reference:
        # WER Hesaplama - Ham çıktılar için
        baseline_wer = jiwer.wer(reference_text, baseline_text) * 100
        optimized_wer = jiwer.wer(reference_text, optimized_text) * 100
        improvement = baseline_wer - optimized_wer
        
        print(f"\n🎯 WER (Referans Metne Göre - Ham Çıktılar):")
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
    
    print(f"\n📝 Kelime Sayısı (Ham Çıktılar):")
    print(f"   Baseline: {baseline_words} kelime")
    print(f"   Optimize: {optimized_words} kelime")
    if has_reference:
        ref_words = len(reference_text.split())
        print(f"   Referans: {ref_words} kelime")
    
    # Farklılıklar - Ham çıktılar
    if baseline_text != optimized_text:
        baseline_set = set(baseline_text.lower().split())
        optimized_set = set(optimized_text.lower().split())
        
        only_baseline = baseline_set - optimized_set
        only_optimized = optimized_set - baseline_set
        
        if only_baseline or only_optimized:
            print(f"\n🔄 Farklı Kelimeler (Ham Çıktılar):")
            if only_baseline:
                words = ', '.join(list(only_baseline)[:15])
                print(f"   🔵 Sadece Baseline'da: {words}")
            if only_optimized:
                words = ', '.join(list(only_optimized)[:15])
                print(f"   🟢 Sadece Optimize'de: {words}")
    else:
        print(f"\n✅ İki sonuç tamamen aynı!")
    
    # Türkçe tokenizer ham çıktıları
    if turkish_tokenizer:
        print(f"\n🇹🇷 Türkçe Tokenizer Ham Çıktıları:")
        print(f"   🔵 Baseline tokenizer çıktısı: {baseline_tokenizer_text}")
        print(f"   🟢 Optimize tokenizer çıktısı: {optimized_tokenizer_text}")
        
        if baseline_tokenizer_text != optimized_tokenizer_text:
            print(f"   📊 Tokenizer çıktıları farklı")
        else:
            print(f"   📊 Tokenizer çıktıları aynı")
    
    print("=" * 80)
    
except FileNotFoundError:
    print(f"❌ Dosya bulunamadı: {audio_file}")
except Exception as e:
    print(f"❌ Hata: {e}")
