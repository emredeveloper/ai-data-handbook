"""
Manuel Audio Test - Gerçek Ses Verisi ile Test
"""

import torch
import numpy as np
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, Audio
import warnings
warnings.filterwarnings("ignore")

def load_real_audio_from_dataset():
    """Hugging Face verisetinden gerçek ses verisi yükle"""
    print("🎵 Hugging Face'den gerçek ses verisi yükleniyor...")
    
    try:
        # Khan Academy Türkçe dataset'ini yükle
        print("📥 Khan Academy Türkçe dataset'i yükleniyor...")
        dataset = load_dataset("ysdede/khanacademy-turkish")
        
        # Dataset yapısını kontrol et
        if "train" not in dataset:
            dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
            train_dataset = dataset["train"]
        else:
            train_dataset = dataset["train"]
        
        # Sütun isimlerini kontrol et
        if "transcription" in train_dataset.column_names:
            text_column = "transcription"
        elif "sentence" in train_dataset.column_names:
            text_column = "sentence"
        else:
            text_column = "text"
        
        print(f"Dataset sütunları: {train_dataset.column_names}")
        print(f"Dataset boyutu: {len(train_dataset)}")
        
        # Audio sampling rate'ini ayarla
        print("📡 Audio sampling rate ayarlanıyor...")
        train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
        
        # İlk birkaç örneği al
        test_samples = []
        for i in range(min(3, len(train_dataset))):
            try:
                print(f"📥 Örnek {i+1} yükleniyor...")
                sample = train_dataset[i]
                audio_data = sample["audio"]
                text_data = sample[text_column]
                
                print(f"Audio data tipi: {type(audio_data)}")
                print(f"Audio data keys: {audio_data.keys() if isinstance(audio_data, dict) else 'Not dict'}")
                
                test_samples.append({
                    "audio_array": audio_data["array"],
                    "sampling_rate": audio_data["sampling_rate"],
                    "text": text_data,
                    "index": i
                })
                print(f"✅ Örnek {i+1}: '{text_data[:50]}...' ({len(audio_data['array'])} sample)")
                
            except Exception as e:
                print(f"❌ Örnek {i+1} işlenemedi: {e}")
                print(f"Hata detayı: {type(e).__name__}: {str(e)}")
                continue
        
        if test_samples:
            print(f"✅ {len(test_samples)} gerçek audio örneği yüklendi!")
            return test_samples
        else:
            print("❌ Hiçbir örnek işlenemedi!")
            raise Exception("No samples processed")
        
    except Exception as e:
        print(f"❌ Khan Academy dataset hatası: {e}")
        print(f"Hata tipi: {type(e).__name__}")
        print("🔄 Common Voice Türkçe dataset'ini deneniyor...")
        
        try:
            # Fallback: Common Voice Türkçe
            dataset = load_dataset("mozilla-foundation/common_voice_13_0", "tr", split="train")
            dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
            
            test_samples = []
            for i in range(min(3, len(dataset))):
                try:
                    sample = dataset[i]
                    audio_data = sample["audio"]
                    text_data = sample["sentence"]
                    
                    test_samples.append({
                        "audio_array": audio_data["array"],
                        "sampling_rate": audio_data["sampling_rate"],
                        "text": text_data,
                        "index": i
                    })
                    print(f"✅ CV Örnek {i+1}: '{text_data[:50]}...' ({len(audio_data['array'])} sample)")
                except Exception as e:
                    print(f"❌ CV Örnek {i+1} işlenemedi: {e}")
                    continue
            
            if test_samples:
                return test_samples
            else:
                raise Exception("Common Voice samples failed")
            
        except Exception as e2:
            print(f"❌ Common Voice dataset hatası: {e2}")
            print("⚠️ Synthetic audio'ya geçiliyor...")
            return create_realistic_test_audio()

def create_realistic_test_audio():
    """Daha gerçekçi test audio'su oluştur - konuşma benzeri"""
    print("🎵 Gerçekçi test audio oluşturuluyor...")
    
    sample_rate = 16000
    
    # Türkçe sesler için gerçekçi frekans aralıkları
    test_samples = []
    
    # Örnek 1: Kısa cümle
    duration1 = 3.0
    t1 = np.linspace(0, duration1, int(sample_rate * duration1))
    
    # İnsan sesi frekans aralığında (85-255 Hz temel frekans)
    fundamental = 150  # Hz
    audio1 = np.zeros_like(t1)
    
    # Harmonikler ekle (konuşma benzeri)
    for harmonic in range(1, 6):
        freq = fundamental * harmonic
        amplitude = 0.3 / harmonic  # Her harmonik daha zayıf
        audio1 += amplitude * np.sin(2 * np.pi * freq * t1)
    
    # Modülasyon ekle (konuşma benzeri)
    modulation = 0.1 * np.sin(2 * np.pi * 5 * t1)  # 5 Hz modülasyon
    audio1 *= (1 + modulation)
    
    # Envelope ekle (başlangıç ve bitiş yumuşak)
    envelope1 = np.ones_like(t1)
    fade_samples = int(0.1 * sample_rate)  # 0.1 saniye fade
    envelope1[:fade_samples] = np.linspace(0, 1, fade_samples)
    envelope1[-fade_samples:] = np.linspace(1, 0, fade_samples)
    audio1 *= envelope1
    
    # Gürültü ekle
    audio1 += 0.02 * np.random.normal(0, 1, len(t1))
    
    test_samples.append({
        "audio_array": audio1.astype(np.float32),
        "sampling_rate": sample_rate,
        "text": "Merhaba, ben Türkçe konuşan bir test ses dosyasıyım.",
        "index": 0
    })
    
    # Örnek 2: Farklı ton
    duration2 = 2.5
    t2 = np.linspace(0, duration2, int(sample_rate * duration2))
    
    fundamental2 = 200  # Hz (daha yüksek ton)
    audio2 = np.zeros_like(t2)
    
    for harmonic in range(1, 5):
        freq = fundamental2 * harmonic
        amplitude = 0.25 / harmonic
        audio2 += amplitude * np.sin(2 * np.pi * freq * t2)
    
    # Farklı modülasyon
    modulation2 = 0.15 * np.sin(2 * np.pi * 3 * t2)
    audio2 *= (1 + modulation2)
    
    # Envelope
    envelope2 = np.ones_like(t2)
    fade_samples2 = int(0.1 * sample_rate)
    envelope2[:fade_samples2] = np.linspace(0, 1, fade_samples2)
    envelope2[-fade_samples2:] = np.linspace(1, 0, fade_samples2)
    audio2 *= envelope2
    
    # Gürültü
    audio2 += 0.015 * np.random.normal(0, 1, len(t2))
    
    test_samples.append({
        "audio_array": audio2.astype(np.float32),
        "sampling_rate": sample_rate,
        "text": "Bu ikinci Türkçe test cümlesi, farklı bir tonla söylenmiştir.",
        "index": 1
    })
    
    # Örnek 3: Daha uzun cümle
    duration3 = 4.0
    t3 = np.linspace(0, duration3, int(sample_rate * duration3))
    
    fundamental3 = 120  # Hz (daha düşük ton)
    audio3 = np.zeros_like(t3)
    
    for harmonic in range(1, 7):
        freq = fundamental3 * harmonic
        amplitude = 0.35 / harmonic
        # Frekans değişimi ekle (prosodi)
        freq_variation = freq * (1 + 0.05 * np.sin(2 * np.pi * 0.5 * t3))
        audio3 += amplitude * np.sin(2 * np.pi * freq_variation * t3)
    
    # Daha karmaşık modülasyon
    modulation3 = 0.2 * np.sin(2 * np.pi * 4 * t3) * np.exp(-t3/2)
    audio3 *= (1 + modulation3)
    
    # Envelope
    envelope3 = np.ones_like(t3)
    fade_samples3 = int(0.15 * sample_rate)
    envelope3[:fade_samples3] = np.linspace(0, 1, fade_samples3)
    envelope3[-fade_samples3:] = np.linspace(1, 0, fade_samples3)
    audio3 *= envelope3
    
    # Gürültü
    audio3 += 0.01 * np.random.normal(0, 1, len(t3))
    
    test_samples.append({
        "audio_array": audio3.astype(np.float32),
        "sampling_rate": sample_rate,
        "text": "Whisper modeli Türkçe konuşmaları çok başarılı bir şekilde yazıya çeviriyor.",
        "index": 2
    })
    
    print(f"✅ {len(test_samples)} gerçekçi audio örneği oluşturuldu")
    return test_samples

def create_fallback_audio():
    """Fallback: Basit ses verisi oluştur"""
    print("🎵 Fallback: Basit test audio oluşturuluyor...")
    
    sample_rate = 16000
    duration = 2.0
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Daha basit ve temiz ses
    test_samples = [
        {
            "audio_array": (
                0.3 * np.sin(2 * np.pi * 440 * t) * np.exp(-t/2) +
                0.05 * np.random.normal(0, 1, len(t))
            ).astype(np.float32),
            "sampling_rate": sample_rate,
            "text": "Test sesi bir.",
            "index": 0
        },
        {
            "audio_array": (
                0.3 * np.sin(2 * np.pi * 523 * t) * np.exp(-t/2) +
                0.05 * np.random.normal(0, 1, len(t))
            ).astype(np.float32),
            "sampling_rate": sample_rate,
            "text": "Test sesi iki.",
            "index": 1
        }
    ]
    
    print(f"✅ {len(test_samples)} basit audio örneği oluşturuldu")
    return test_samples

def test_single_audio(audio_data, expected_text, sample_index):
    """Tek bir ses örneğini test et"""
    print(f"\n🎯 Örnek {sample_index + 1} Test Ediliyor:")
    print(f"📝 Beklenen metin: '{expected_text[:100]}...'")
    print("-" * 50)
    
    audio_array = audio_data["audio_array"]
    sampling_rate = audio_data["sampling_rate"]
    
    results = {}
    
    # Orijinal model
    print("🔵 Orijinal Whisper Small ile test...")
    try:
        original_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        original_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
        
        # Audio'yu işle
        inputs = original_processor(
            audio_array, 
            sampling_rate=sampling_rate, 
            return_tensors="pt",
            padding=True,
            return_attention_mask=True
        )
        
        # Generate - optimized parameters
        with torch.no_grad():
            generated_ids = original_model.generate(
                inputs["input_features"],
                language="turkish",
                task="transcribe",
                max_length=448,
                num_beams=5,
                do_sample=False,
                temperature=0.0,
                use_cache=True,
                pad_token_id=original_processor.tokenizer.pad_token_id,
                eos_token_id=original_processor.tokenizer.eos_token_id,
                forced_decoder_ids=original_processor.get_decoder_prompt_ids(language="turkish", task="transcribe")
            )
        
        original_text = original_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        results["original"] = original_text
        print(f"🔵 Orijinal sonuç: '{original_text}'")
        
    except Exception as e:
        print(f"❌ Orijinal model hatası: {e}")
        results["original"] = ""
    
    # Fine-tuned model
    print("🟢 Fine-tuned Whisper Small ile test...")
    try:
        # Processor'ı orijinal model'den al
        finetuned_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        
        # Fine-tuned model'i yükle - en son checkpoint'i bul
        import os
        import glob
        
        checkpoint_dirs = glob.glob("./whisper-small-turkish/checkpoint-*")
        if checkpoint_dirs:
            # En yüksek numaralı checkpoint'i al
            latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split('-')[-1]))
            print(f"📁 En son checkpoint kullanılıyor: {latest_checkpoint}")
            finetuned_model = WhisperForConditionalGeneration.from_pretrained(latest_checkpoint)
        else:
            print("❌ Hiç checkpoint bulunamadı!")
            raise FileNotFoundError("No checkpoint found")
        
        # Audio'yu işle
        inputs = finetuned_processor(
            audio_array, 
            sampling_rate=sampling_rate, 
            return_tensors="pt",
            padding=True,
            return_attention_mask=True
        )
        
        # Generate - optimized parameters
        with torch.no_grad():
            generated_ids = finetuned_model.generate(
                inputs["input_features"],
                language="turkish",
                task="transcribe",
                max_length=448,
                num_beams=5,
                do_sample=False,
                temperature=0.0,
                use_cache=True,
                pad_token_id=finetuned_processor.tokenizer.pad_token_id,
                eos_token_id=finetuned_processor.tokenizer.eos_token_id,
                forced_decoder_ids=finetuned_processor.get_decoder_prompt_ids(language="turkish", task="transcribe")
            )
        
        finetuned_text = finetuned_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        results["finetuned"] = finetuned_text
        print(f"🟢 Fine-tuned sonuç: '{finetuned_text}'")
        
    except Exception as e:
        print(f"❌ Fine-tuned model hatası: {e}")
        results["finetuned"] = ""
    
    return results, expected_text

def calculate_similarity(text1, text2):
    """İki metin arasındaki benzerliği hesapla (basit)"""
    if not text1 or not text2:
        return 0.0
    
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if len(words1) == 0 and len(words2) == 0:
        return 1.0
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0

def test_models():
    print("🎯 Manuel Audio Test - Gerçek Ses Verisi ile")
    print("=" * 60)
    
    # Gerçek ses verilerini yükle
    test_samples = load_real_audio_from_dataset()
    
    if not test_samples:
        print("❌ Test verileri yüklenemedi!")
        return
    
    print(f"\n📊 {len(test_samples)} örnek ile test yapılacak...")
    
    all_results = []
    
    # Her örneği test et
    for i, sample in enumerate(test_samples):
        results, expected_text = test_single_audio(sample, sample["text"], i)
        all_results.append({
            "index": i,
            "expected": expected_text,
            "original": results.get("original", ""),
            "finetuned": results.get("finetuned", ""),
            "audio_length": len(sample["audio_array"])
        })
    
    # Genel sonuçları değerlendir
    print("\n" + "=" * 60)
    print("📊 GENEL SONUÇLAR:")
    print("=" * 60)
    
    original_similarities = []
    finetuned_similarities = []
    
    for i, result in enumerate(all_results):
        print(f"\n🎯 Örnek {i + 1}:")
        print(f"📝 Beklenen: '{result['expected'][:80]}...'")
        print(f"🔵 Orijinal: '{result['original'][:80]}...'")
        print(f"🟢 Fine-tuned: '{result['finetuned'][:80]}...'")
        
        # Benzerlik hesapla
        orig_sim = calculate_similarity(result['expected'], result['original'])
        fine_sim = calculate_similarity(result['expected'], result['finetuned'])
        
        original_similarities.append(orig_sim)
        finetuned_similarities.append(fine_sim)
        
        print(f"📈 Orijinal benzerlik: {orig_sim:.2%}")
        print(f"📈 Fine-tuned benzerlik: {fine_sim:.2%}")
        
        if fine_sim > orig_sim:
            print("✅ Fine-tuned model bu örnekte daha başarılı!")
        elif orig_sim > fine_sim:
            print("🔵 Orijinal model bu örnekte daha başarılı!")
        else:
            print("🤔 İki model de eşit başarılı!")
    
    # Ortalama performans
    avg_original = sum(original_similarities) / len(original_similarities) if original_similarities else 0
    avg_finetuned = sum(finetuned_similarities) / len(finetuned_similarities) if finetuned_similarities else 0
    
    print(f"\n🏆 ORTALAMA PERFORMANS:")
    print(f"🔵 Orijinal model ortalama benzerlik: {avg_original:.2%}")
    print(f"🟢 Fine-tuned model ortalama benzerlik: {avg_finetuned:.2%}")
    
    if avg_finetuned > avg_original:
        improvement = ((avg_finetuned - avg_original) / avg_original * 100) if avg_original > 0 else 0
        print(f"🎉 Fine-tuned model {improvement:.1f}% daha iyi performans gösteriyor!")
        print("✅ Fine-tuning başarılı!")
    elif avg_original > avg_finetuned:
        decline = ((avg_original - avg_finetuned) / avg_original * 100) if avg_original > 0 else 0
        print(f"⚠️ Fine-tuned model {decline:.1f}% daha kötü performans gösteriyor!")
        print("🔄 Fine-tuning parametrelerini gözden geçirmeniz gerekebilir.")
    else:
        print("🤔 İki model de benzer performans gösteriyor.")
    
    print("\n✅ Test tamamlandı!")

if __name__ == "__main__":
    test_models()
