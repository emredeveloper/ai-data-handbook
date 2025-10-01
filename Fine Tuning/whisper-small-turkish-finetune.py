#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ÖNEMLİ: En başta torchcodec'i devre dışı bırak (Windows FFmpeg sorunu için)
import os
import sys
os.environ["HF_DATASETS_AUDIO_BACKEND"] = "soundfile"

# Torchcodec import'unu engelle (monkey patch)
import importlib.util
if importlib.util.find_spec("torchcodec") is not None:
    # torchcodec modülünü kara listeye al - datasets kütüphanesi soundfile kullanacak
    sys.modules['torchcodec'] = None
    sys.modules['torchcodec.decoders'] = None

import torch
import datasets
from datasets import DatasetDict, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import evaluate
import numpy as np
from transformers import EarlyStoppingCallback
from transformers import DataCollatorForSeq2Seq
try:
    # Whisper ASR için önerilen collator
    from transformers import DataCollatorSpeechSeq2SeqWithPadding
except Exception:
    DataCollatorSpeechSeq2SeqWithPadding = None

# Basit Whisper collator: input_features ve labels için doğru padding uygular
class SimpleWhisperCollator:
    def __init__(self, processor, pad_to_multiple_of: int | None = None):
        self.processor = processor
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        # input_features listesi (mel spektrogramlar)
        input_features = [f["input_features"] for f in features]
        inputs_padded = self.processor.feature_extractor.pad(
            {"input_features": input_features},
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        # labels: token id listeleri
        labels = [f["labels"] for f in features]
        labels_padded = self.processor.tokenizer.pad(
            {"input_ids": labels},
            padding=True,
            return_tensors="pt",
        )

        labels_tensor = labels_padded["input_ids"]
        # Pad token'ları -100'e çevir (loss ignore index)
        labels_tensor[labels_tensor == self.processor.tokenizer.pad_token_id] = -100

        # Decoder input ids oluştur (labels'i sağa kaydır)
        import torch
        pad_id = self.processor.tokenizer.pad_token_id
        bos_id = self.processor.tokenizer.bos_token_id or pad_id
        dec_in = labels_padded["input_ids"].clone()
        dec_in[dec_in == -100] = pad_id
        decoder_input_ids = torch.full_like(dec_in, fill_value=pad_id)
        decoder_input_ids[:, 1:] = dec_in[:, :-1]
        decoder_input_ids[:, 0] = bos_id

        batch = {
            "input_features": inputs_padded["input_features"],
            "labels": labels_tensor,
            "decoder_input_ids": decoder_input_ids,
        }
        return batch
import os
import gc  # Garbage collector için
import json
from datetime import datetime
from pathlib import Path

# Audio decode backend'ini soundfile olarak ayarla (torchcodec/FFmpeg hatası için)
import datasets.config
try:
    datasets.config.AUDIO_DECODER = "soundfile"
except Exception:
    pass  # Eski datasets versiyonlarında bu ayar olmayabilir

# datasets.features.audio modülünü patch'le (torchcodec hatası için kesin çözüm)
import datasets.features.audio as audio_module
import librosa

def patched_decode_example(self, value, token_per_repo_id=None):
    """Librosa/soundfile kullanan özel decode fonksiyonu - OPUS desteği ile"""
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
            # Path varsa oku (librosa OPUS/MP3/WAV hepsini destekler)
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
        # Hata durumunda orijinal value'yu dön
        return value

# Patch'i uygula
audio_module.Audio.decode_example = patched_decode_example
print("✅ Audio decoder patch'lendi (librosa/soundfile - OPUS desteği aktif)")

def clear_memory():
    """Bellek temizleme fonksiyonu"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def print_gpu_memory():
    """GPU bellek kullanımını yazdır"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        print(f"GPU Bellek - Ayrılmış: {allocated:.2f} GB, Rezerve: {reserved:.2f} GB")

def check_data_quality(dataset, text_column="transcription", audio_column="audio", 
                       min_duration=0.5, max_duration=30, 
                       min_text_length=5, max_text_length=500,
                       sample_size=3000):
    """
    Veri setinin kalitesini kontrol eder ve istatistikleri yazdırır
    
    Args:
        dataset: Kontrol edilecek veri seti
        text_column: Metin sütunu adı
        audio_column: Ses sütunu adı
        min_duration: Minimum ses süresi (saniye)
        max_duration: Maksimum ses süresi (saniye)
        min_text_length: Minimum metin uzunluğu (karakter)
        max_text_length: Maksimum metin uzunluğu (karakter)
        sample_size: Kontrol edilecek örnek sayısı (None = tümü)
    
    Returns:
        Filtrelenmiş veri seti ve istatistikler
    """
    print("\n" + "="*60)
    print("VERİ KALİTESİ KONTROLÜ")
    print("="*60)
    
    total_samples = len(dataset)
    actual_size = min(sample_size, total_samples) if sample_size else total_samples
    durations = []
    text_lengths = []
    invalid_samples = []
    
    print(f"Toplam {total_samples} örnek - {actual_size} örnek kontrol ediliyor...")
    if actual_size < total_samples:
        print(f"⚡ Hızlı kontrol modu: İlk {actual_size} örnek analiz edilecek")
    print("NOT: Büyük veri setlerinde bu işlem uzun sürebilir.\n")
    
    # Her örneği kontrol et - decode=False ile hızlı erişim
    for idx in range(actual_size):
        if idx % 1000 == 0 and idx > 0:
            print(f"  İşlenen: {idx}/{actual_size}")
        
        try:
            # Batch olarak al (decode yapmadan)
            example = dataset[idx]
            
            # Ses süresini al (datasets tarafından decode edilmiş)
            audio_data = example[audio_column]
            duration = len(audio_data["array"]) / audio_data["sampling_rate"]
            
            durations.append(duration)
            
            # Metin uzunluğunu al
            text = example[text_column] if text_column in example else ""
            text_length = len(text.strip())
            text_lengths.append(text_length)
            
            # Geçersiz örnekleri işaretle
            if duration < min_duration or duration > max_duration:
                invalid_samples.append((idx, "duration", duration))
            elif text_length < min_text_length or text_length > max_text_length:
                invalid_samples.append((idx, "text_length", text_length))
            elif text_length == 0:
                invalid_samples.append((idx, "empty_text", 0))
                
        except Exception as e:
            invalid_samples.append((idx, "error", str(e)))
    
    # İstatistikleri hesapla
    if durations:
        avg_duration = np.mean(durations)
        min_dur = np.min(durations)
        max_dur = np.max(durations)
        std_duration = np.std(durations)
        
        avg_text_len = np.mean(text_lengths)
        min_text = np.min(text_lengths)
        max_text = np.max(text_lengths)
        std_text = np.std(text_lengths)
        
        print(f"\nToplam örnek sayısı: {total_samples}")
        print(f"\nSES İSTATİSTİKLERİ:")
        print(f"  Ortalama süre: {avg_duration:.2f} saniye")
        print(f"  Min süre: {min_dur:.2f} saniye")
        print(f"  Max süre: {max_dur:.2f} saniye")
        print(f"  Standart sapma: {std_duration:.2f} saniye")
        print(f"  Toplam süre: {sum(durations)/3600:.2f} saat")
        
        print(f"\nMETİN İSTATİSTİKLERİ:")
        print(f"  Ortalama uzunluk: {avg_text_len:.0f} karakter")
        print(f"  Min uzunluk: {min_text} karakter")
        print(f"  Max uzunluk: {max_text} karakter")
        print(f"  Standart sapma: {std_text:.2f} karakter")
        
        # Geçersiz örnekleri göster
        if invalid_samples:
            print(f"\n⚠️  GEÇERSİZ ÖRNEKLER: {len(invalid_samples)}")
            
            # Geçersizlik nedenlerini grupla
            reason_counts = {}
            for _, reason, _ in invalid_samples:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            
            for reason, count in reason_counts.items():
                print(f"  - {reason}: {count} örnek")
            
            # İlk 5 geçersiz örneği detaylı göster
            if len(invalid_samples) > 0:
                print(f"\n  İlk {min(5, len(invalid_samples))} geçersiz örnek:")
                for idx, reason, value in invalid_samples[:5]:
                    print(f"    Index {idx}: {reason} = {value}")
        else:
            print(f"\n✅ Tüm örnekler geçerli!")
        
        # Öneriler
        print(f"\nKALİTE ÖNERİLERİ:")
        very_short = sum(1 for d in durations if d < 5.0)
        very_long = sum(1 for d in durations if d > 20.0)
        
        if very_short > 0:
            print(f"  ⚠️  {very_short} örnek 5 saniyeden kısa (filtrelenecek)")
        if very_long > 0:
            print(f"  ⚠️  {very_long} örnek 20 saniyeden uzun (bellek sorunu olabilir)")
        
        short_text = sum(1 for t in text_lengths if t < 10)
        if short_text > 0:
            print(f"  ⚠️  {short_text} örnek 10 karakterden kısa metin içeriyor")
        
        if len(invalid_samples) / total_samples > 0.05:
            print(f"  ⚠️  Geçersiz örnek oranı yüksek (%{len(invalid_samples)/total_samples*100:.1f})")
            print(f"      Filtreleme yapılması önerilir!")
        
    print("="*60)
    
    return invalid_samples

def filter_dataset(dataset, invalid_indices, split_name="train"):
    """
    Geçersiz örnekleri veri setinden çıkarır
    
    Args:
        dataset: Filtrelenecek veri seti
        invalid_indices: Geçersiz örnek indeksleri listesi
        split_name: Veri seti bölümü adı (train/validation)
    
    Returns:
        Filtrelenmiş veri seti
    """
    if len(invalid_indices) == 0:
        print(f"✅ {split_name} veri setinde filtreleme gerekli değil.")
        return dataset
    
    # Geçersiz indeksleri çıkar
    invalid_ids = [idx for idx, _, _ in invalid_indices]
    
    # Geçerli örnekleri seç
    valid_indices = [i for i in range(len(dataset)) if i not in invalid_ids]
    filtered_dataset = dataset.select(valid_indices)
    
    print(f"✅ {split_name}: {len(invalid_indices)} örnek filtrelendi.")
    print(f"   Önceki boyut: {len(dataset)} → Yeni boyut: {len(filtered_dataset)}")
    
    return filtered_dataset

def analyze_text_audio_ratio(dataset, text_column="transcription", audio_column="audio", sample_size=1000):
    """
    Metin uzunluğu ile ses süresi arasındaki ilişkiyi analiz eder
    sample_size: Analiz edilecek örnek sayısı (hız için)
    """
    print("\n" + "="*60)
    print("METİN/SES ORAN ANALİZİ")
    print("="*60)
    
    # Hız için sadece sample al
    actual_size = min(sample_size, len(dataset))
    print(f"⚡ Hızlı analiz: {actual_size} / {len(dataset)} örnek analiz ediliyor...")
    
    ratios = []
    for idx in range(actual_size):
        try:
            example = dataset[idx]
            audio_data = example[audio_column]
            duration = len(audio_data["array"]) / audio_data["sampling_rate"]
            
            text = example[text_column] if text_column in example else ""
            text_length = len(text.strip())
            
            # Karakter/saniye oranı
            if duration > 0:
                ratio = text_length / duration
                ratios.append(ratio)
        except:
            pass
    
    if ratios:
        avg_ratio = np.mean(ratios)
        std_ratio = np.std(ratios)
        
        print(f"Ortalama karakter/saniye: {avg_ratio:.2f}")
        print(f"Standart sapma: {std_ratio:.2f}")
        print(f"Min: {np.min(ratios):.2f}, Max: {np.max(ratios):.2f}")
        
        # Aykırı değerleri tespit et
        outliers = [r for r in ratios if r < (avg_ratio - 2*std_ratio) or r > (avg_ratio + 2*std_ratio)]
        if outliers:
            print(f"\n⚠️  {len(outliers)} aykırı değer tespit edildi")
            print(f"   (Metin çok kısa/uzun veya ses çok kısa/uzun olabilir)")
        else:
            print(f"\n✅ Metin/ses oranı tutarlı görünüyor")
    
    print("="*60)

def generate_dataset_statistics(dataset, split_name="train", text_column="transcription", 
                                 audio_column="audio", output_dir="./whisper-small-turkish-khanacademy",
                                 sample_size=2000):
    """
    Veri seti hakkında kapsamlı istatistikler üretir ve kaydeder
    sample_size: Analiz edilecek örnek sayısı (None = tümü, hız için 2000 önerilir)
    """
    print("\n" + "="*70)
    print(f"KAPSAMLI VERİ SETİ İSTATİSTİKLERİ - {split_name.upper()}")
    print("="*70)
    
    # Veri toplama
    durations = []
    text_lengths = []
    word_counts = []
    char_per_second = []
    word_per_second = []
    sampling_rates = []
    
    # Türkçe özel karakterler
    turkish_chars = {'ç', 'ğ', 'ı', 'i', 'ö', 'ş', 'ü', 'Ç', 'Ğ', 'İ', 'Ö', 'Ş', 'Ü'}
    turkish_char_count = 0
    
    # Sample size belirleme (hız için)
    actual_size = min(sample_size, len(dataset)) if sample_size else len(dataset)
    print(f"Analiz ediliyor: {actual_size} / {len(dataset)} örnek...")
    if actual_size < len(dataset):
        print(f"⚡ Hızlı analiz modu: {actual_size} örnek üzerinden istatistik hesaplanacak")
    
    for idx in range(actual_size):
        if idx % 1000 == 0 and idx > 0:
            print(f"  İşlenen: {idx}/{actual_size}")
        
        try:
            example = dataset[idx]
            
            # Ses bilgileri (datasets tarafından decode edilmiş)
            audio_data = example[audio_column]
            duration = len(audio_data["array"]) / audio_data["sampling_rate"]
            sampling_rate = audio_data["sampling_rate"]
            
            durations.append(duration)
            sampling_rates.append(sampling_rate)
            
            # Metin bilgileri
            text = example[text_column] if text_column in example else ""
            text = text.strip()
            text_length = len(text)
            text_lengths.append(text_length)
            
            # Kelime sayısı
            words = text.split()
            word_count = len(words)
            word_counts.append(word_count)
            
            # Oran hesaplamaları
            if duration > 0:
                char_per_second.append(text_length / duration)
                word_per_second.append(word_count / duration)
            
            # Türkçe karakter analizi
            turkish_char_count += sum(1 for c in text if c in turkish_chars)
            
        except Exception as e:
            print(f"  Hata (örnek {idx}): {e}")
            continue
    
    # İstatistik hesaplamaları
    stats = {
        "dataset_info": {
            "split": split_name,
            "total_samples": len(dataset),
            "analyzed_samples": actual_size,
            "processed_samples": len(durations),
            "is_sampled": actual_size < len(dataset)
        },
        "audio_statistics": {
            "total_duration_hours": sum(durations) / 3600,
            "avg_duration_sec": np.mean(durations),
            "median_duration_sec": np.median(durations),
            "std_duration_sec": np.std(durations),
            "min_duration_sec": np.min(durations),
            "max_duration_sec": np.max(durations),
            "percentile_25": np.percentile(durations, 25),
            "percentile_75": np.percentile(durations, 75),
            "percentile_95": np.percentile(durations, 95)
        },
        "text_statistics": {
            "avg_text_length": np.mean(text_lengths),
            "median_text_length": np.median(text_lengths),
            "std_text_length": np.std(text_lengths),
            "min_text_length": int(np.min(text_lengths)),
            "max_text_length": int(np.max(text_lengths)),
            "total_characters": int(np.sum(text_lengths)),
            "percentile_25": np.percentile(text_lengths, 25),
            "percentile_75": np.percentile(text_lengths, 75),
            "percentile_95": np.percentile(text_lengths, 95)
        },
        "word_statistics": {
            "avg_words": np.mean(word_counts),
            "median_words": np.median(word_counts),
            "std_words": np.std(word_counts),
            "min_words": int(np.min(word_counts)),
            "max_words": int(np.max(word_counts)),
            "total_words": int(np.sum(word_counts)),
            "avg_word_length": np.mean(text_lengths) / max(np.mean(word_counts), 1)
        },
        "ratio_statistics": {
            "avg_char_per_sec": np.mean(char_per_second),
            "median_char_per_sec": np.median(char_per_second),
            "std_char_per_sec": np.std(char_per_second),
            "avg_word_per_sec": np.mean(word_per_second),
            "median_word_per_sec": np.median(word_per_second),
            "std_word_per_sec": np.std(word_per_second)
        },
        "turkish_language": {
            "turkish_char_count": turkish_char_count,
            "turkish_char_percentage": (turkish_char_count / max(sum(text_lengths), 1)) * 100,
            "avg_turkish_chars_per_text": turkish_char_count / len(dataset)
        },
        "duration_distribution": {
            "under_5_sec": sum(1 for d in durations if d < 5),
            "5_10_sec": sum(1 for d in durations if 5 <= d < 10),
            "10_15_sec": sum(1 for d in durations if 10 <= d < 15),
            "15_20_sec": sum(1 for d in durations if 15 <= d < 20),
            "20_30_sec": sum(1 for d in durations if 20 <= d < 30),
            "over_30_sec": sum(1 for d in durations if d >= 30)
        },
        "text_length_distribution": {
            "under_50_chars": sum(1 for t in text_lengths if t < 50),
            "50_100_chars": sum(1 for t in text_lengths if 50 <= t < 100),
            "100_200_chars": sum(1 for t in text_lengths if 100 <= t < 200),
            "200_300_chars": sum(1 for t in text_lengths if 200 <= t < 300),
            "over_300_chars": sum(1 for t in text_lengths if t >= 300)
        }
    }
    
    # Ekrana yazdır
    print("\n📊 SES İSTATİSTİKLERİ:")
    print(f"  Toplam süre: {stats['audio_statistics']['total_duration_hours']:.2f} saat")
    print(f"  Ortalama: {stats['audio_statistics']['avg_duration_sec']:.2f} sn")
    print(f"  Medyan: {stats['audio_statistics']['median_duration_sec']:.2f} sn")
    print(f"  Min/Max: {stats['audio_statistics']['min_duration_sec']:.2f} / {stats['audio_statistics']['max_duration_sec']:.2f} sn")
    print(f"  Std: {stats['audio_statistics']['std_duration_sec']:.2f} sn")
    print(f"  25%/75%/95% Percentile: {stats['audio_statistics']['percentile_25']:.2f} / {stats['audio_statistics']['percentile_75']:.2f} / {stats['audio_statistics']['percentile_95']:.2f} sn")
    
    print("\n📝 METİN İSTATİSTİKLERİ:")
    print(f"  Toplam karakter: {stats['text_statistics']['total_characters']:,}")
    print(f"  Ortalama uzunluk: {stats['text_statistics']['avg_text_length']:.0f} karakter")
    print(f"  Medyan uzunluk: {stats['text_statistics']['median_text_length']:.0f} karakter")
    print(f"  Min/Max: {stats['text_statistics']['min_text_length']} / {stats['text_statistics']['max_text_length']} karakter")
    print(f"  Std: {stats['text_statistics']['std_text_length']:.2f}")
    print(f"  25%/75%/95% Percentile: {stats['text_statistics']['percentile_25']:.0f} / {stats['text_statistics']['percentile_75']:.0f} / {stats['text_statistics']['percentile_95']:.0f}")
    
    print("\n📖 KELİME İSTATİSTİKLERİ:")
    print(f"  Toplam kelime: {stats['word_statistics']['total_words']:,}")
    print(f"  Ortalama kelime sayısı: {stats['word_statistics']['avg_words']:.1f}")
    print(f"  Medyan kelime sayısı: {stats['word_statistics']['median_words']:.1f}")
    print(f"  Min/Max: {stats['word_statistics']['min_words']} / {stats['word_statistics']['max_words']}")
    print(f"  Ortalama kelime uzunluğu: {stats['word_statistics']['avg_word_length']:.2f} karakter")
    
    print("\n📈 ORAN İSTATİSTİKLERİ:")
    print(f"  Karakter/saniye (ort): {stats['ratio_statistics']['avg_char_per_sec']:.2f}")
    print(f"  Karakter/saniye (medyan): {stats['ratio_statistics']['median_char_per_sec']:.2f}")
    print(f"  Kelime/saniye (ort): {stats['ratio_statistics']['avg_word_per_sec']:.2f}")
    print(f"  Kelime/saniye (medyan): {stats['ratio_statistics']['median_word_per_sec']:.2f}")
    
    print("\n🇹🇷 TÜRKÇE DİL ANALİZİ:")
    print(f"  Türkçe özel karakter: {stats['turkish_language']['turkish_char_count']:,}")
    print(f"  Türkçe karakter oranı: {stats['turkish_language']['turkish_char_percentage']:.2f}%")
    print(f"  Metin başına ort. Türkçe karakter: {stats['turkish_language']['avg_turkish_chars_per_text']:.2f}")
    
    print("\n⏱️  SES SÜRESİ DAĞILIMI:")
    for key, value in stats['duration_distribution'].items():
        percentage = (value / len(dataset)) * 100
        print(f"  {key}: {value} örnek ({percentage:.1f}%)")
    
    print("\n📊 METİN UZUNLUĞU DAĞILIMI:")
    for key, value in stats['text_length_distribution'].items():
        percentage = (value / len(dataset)) * 100
        print(f"  {key}: {value} örnek ({percentage:.1f}%)")
    
    # JSON dosyasına kaydet
    output_path = Path(output_dir) / "statistics"
    output_path.mkdir(parents=True, exist_ok=True)
    
    stats_file = output_path / f"dataset_statistics_{split_name}.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 İstatistikler kaydedildi: {stats_file}")
    
    # Ham verileri de kaydet (histogram için)
    raw_data = {
        "durations": durations,
        "text_lengths": text_lengths,
        "word_counts": word_counts,
        "char_per_second": char_per_second,
        "word_per_second": word_per_second
    }
    
    raw_data_file = output_path / f"raw_data_{split_name}.json"
    with open(raw_data_file, 'w', encoding='utf-8') as f:
        json.dump(raw_data, f, indent=2)
    
    print(f"💾 Ham veriler kaydedildi: {raw_data_file}")
    
    print("="*70 + "\n")
    
    return stats

class TrainingLogger:
    """
    Eğitim sürecini detaylı loglamak için sınıf
    """
    def __init__(self, output_dir="./whisper-small-turkish-khanacademy"):
        self.output_dir = Path(output_dir)
        self.log_dir = self.output_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Timestamp ile log dosyası
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"training_log_{self.timestamp}.txt"
        self.metrics_file = self.log_dir / f"training_metrics_{self.timestamp}.json"
        self.csv_file = self.log_dir / f"training_history_{self.timestamp}.csv"
        
        self.metrics_history = []
        self.start_time = None
        
        # CSV başlıklarını yaz
        with open(self.csv_file, 'w', encoding='utf-8') as f:
            f.write("step,loss,learning_rate,epoch,eval_loss,eval_wer,eval_cer,timestamp\n")
        
        self.log("="*70)
        self.log(f"EĞİTİM BAŞLATILDI - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("="*70)
    
    def log(self, message):
        """Console ve dosyaya log yaz"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        print(log_message)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + "\n")
    
    def log_metrics(self, metrics, step=None):
        """Metrikleri kaydet"""
        metrics_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            **metrics
        }
        self.metrics_history.append(metrics_entry)
        
        # JSON dosyasına kaydet
        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics_history, f, indent=2, ensure_ascii=False)
    
    def log_to_csv(self, step, loss=None, lr=None, epoch=None, 
                    eval_loss=None, eval_wer=None, eval_cer=None):
        """CSV dosyasına metrik kaydet"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.csv_file, 'a', encoding='utf-8') as f:
            f.write(f"{step},{loss},{lr},{epoch},{eval_loss},{eval_wer},{eval_cer},{timestamp}\n")
    
    def log_config(self, config):
        """Eğitim konfigürasyonunu kaydet"""
        config_file = self.log_dir / f"training_config_{self.timestamp}.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        self.log(f"Konfigürasyon kaydedildi: {config_file}")
    
    def log_data_info(self, train_size, val_size, invalid_train, invalid_val):
        """Veri seti bilgilerini kaydet"""
        info = {
            "train_size": train_size,
            "validation_size": val_size,
            "invalid_train_samples": invalid_train,
            "invalid_val_samples": invalid_val,
            "total_invalid": invalid_train + invalid_val
        }
        
        data_file = self.log_dir / f"data_info_{self.timestamp}.json"
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        
        self.log(f"Veri seti bilgileri:")
        self.log(f"  Train: {train_size} örnek")
        self.log(f"  Validation: {val_size} örnek")
        self.log(f"  Geçersiz (train): {invalid_train}")
        self.log(f"  Geçersiz (val): {invalid_val}")
    
    def start_training(self):
        """Eğitim başlangıcını kaydet"""
        self.start_time = datetime.now()
        self.log("\n🚀 EĞİTİM BAŞLIYOR...")
    
    def end_training(self, final_metrics=None):
        """Eğitim bitişini kaydet"""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        self.log("\n" + "="*70)
        self.log("✅ EĞİTİM TAMAMLANDI")
        self.log(f"Toplam süre: {duration}")
        
        if final_metrics:
            self.log("\nFinal Metrikler:")
            for key, value in final_metrics.items():
                self.log(f"  {key}: {value}")
        
        self.log("="*70)
        self.log(f"\nLog dosyaları:")
        self.log(f"  - Ana log: {self.log_file}")
        self.log(f"  - Metrikler: {self.metrics_file}")
        self.log(f"  - CSV: {self.csv_file}")

class ProgressCallback(EarlyStoppingCallback):
    """
    Eğitim ilerlemesini takip eden özel callback
    """
    def __init__(self, logger, early_stopping_patience=3):
        super().__init__(early_stopping_patience=early_stopping_patience)
        self.logger = logger
        self.best_metric = float('inf')
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Her log adımında çağrılır"""
        if logs:
            step = state.global_step
            
            # Training metrikleri
            if 'loss' in logs:
                self.logger.log(f"Step {step}: Loss = {logs['loss']:.4f}")
                self.logger.log_to_csv(
                    step=step,
                    loss=logs.get('loss'),
                    lr=logs.get('learning_rate'),
                    epoch=logs.get('epoch')
                )
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Her değerlendirmede çağrılır"""
        if metrics:
            step = state.global_step
            eval_loss = metrics.get('eval_loss', 'N/A')
            eval_wer = metrics.get('eval_wer', 'N/A')
            eval_cer = metrics.get('eval_cer', 'N/A')
            
            self.logger.log("\n" + "="*70)
            self.logger.log(f"📊 DEĞERLENDIRME - Step {step}")
            self.logger.log(f"  Loss: {eval_loss}")
            self.logger.log(f"  WER: {eval_wer:.4f}" if isinstance(eval_wer, float) else f"  WER: {eval_wer}")
            self.logger.log(f"  CER: {eval_cer:.4f}" if isinstance(eval_cer, float) else f"  CER: {eval_cer}")
            self.logger.log("="*70 + "\n")
            
            # Metrikleri kaydet
            self.logger.log_metrics(metrics, step=step)
            self.logger.log_to_csv(
                step=step,
                eval_loss=eval_loss if isinstance(eval_loss, (int, float)) else None,
                eval_wer=eval_wer if isinstance(eval_wer, float) else None,
                eval_cer=eval_cer if isinstance(eval_cer, float) else None
            )
            
            # En iyi metriği güncelle
            if isinstance(eval_wer, float) and eval_wer < self.best_metric:
                self.best_metric = eval_wer
                self.logger.log(f"🌟 YENİ EN İYİ MODEL! WER: {eval_wer:.4f}\n")
    
    def on_save(self, args, state, control, **kwargs):
        """Checkpoint kaydedildiğinde çağrılır"""
        self.logger.log(f"💾 Checkpoint kaydedildi: Step {state.global_step}")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Eğitim bittiğinde çağrılır"""
        self.logger.log(f"\n✅ Eğitim {state.global_step} step'te tamamlandı")
        self.logger.log(f"En iyi WER: {self.best_metric:.4f}")

def main():
    # Logger başlat
    logger = TrainingLogger(output_dir="./whisper-small-turkish-khanacademy")
    
    # GPU kullanımını kontrol et
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.log(f"Kullanılan cihaz: {device}")
    
    # GPU bellek bilgisi
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        logger.log(f"GPU: {gpu_name}")
        logger.log(f"Toplam GPU Bellek: {gpu_memory:.2f} GB")
        print_gpu_memory()

    # Veri setini yükle
    logger.log("\nVeri seti yükleniyor...")
    dataset = datasets.load_dataset("ysdede/khanacademy-turkish")
    
    # Veri seti yapısını kontrol et ve bilgi ver
    print(f"Veri seti sütunları: {dataset['train'].column_names if 'train' in dataset else dataset.column_names}")
    print(f"Veri seti yapısı: {dataset}")
    
    # NOT: decode=False path sorunlarına yol açıyor, datasets'in kendi decode'u zaten verimli
    # Sadece sampling rate'i ayarla
    logger.log("Audio sampling rate ayarlanıyor (Whisper için 16kHz)...")

# Eğer veri seti train/validation/test ayrımı yapmıyorsa, kendimiz ayıralım
    if 'validation' not in dataset and 'test' not in dataset:
        print("Veri seti train/validation olarak ayrılıyor...")
        if 'train' in dataset:
            # Sadece train var, onu bölelim
            train_val = dataset['train'].train_test_split(test_size=0.1, seed=42)
            dataset = DatasetDict({
                'train': train_val['train'],
                'validation': train_val['test']
            })
        else:
            # Dataset objesi ise, onu bölelim
            dataset = dataset.train_test_split(test_size=0.1, seed=42)
            dataset = DatasetDict({
                'train': dataset['train'],
                'validation': dataset['test']
            })
    elif 'test' in dataset and 'validation' not in dataset:
        # Test var ama validation yok, test'i validation olarak kullan
        print("Test seti validation olarak kullanılıyor...")
        dataset = DatasetDict({
            'train': dataset['train'],
            'validation': dataset['test']
        })

    # Ses örnekleme oranını kontrol et (Whisper 16kHz bekler)
    target_sampling_rate = 16000

    # Audio sütununu yapılandır
    dataset = dataset.cast_column("audio", Audio(sampling_rate=target_sampling_rate))
    
    # İstatistik dosyalarını kontrol et (cache)
    stats_dir = Path("./whisper-small-turkish-khanacademy/statistics")
    train_stats_file = stats_dir / "dataset_statistics_train.json"
    val_stats_file = stats_dir / "dataset_statistics_validation.json"
    
    stats_exist = train_stats_file.exists() and val_stats_file.exists()
    
    if stats_exist:
        logger.log("\n📊 İstatistik dosyaları bulundu, önbellek kullanılıyor...")
        print("✅ Kalite kontrolü ve istatistik analizi atlanıyor (dosyalar mevcut)")
        print(f"   Train: {train_stats_file}")
        print(f"   Validation: {val_stats_file}")
        
        # Boş invalid listesi (filtreleme yapılmayacak varsayımıyla)
        train_invalid = []
        val_invalid = []
        
        # İstatistikleri yükle
        with open(train_stats_file, 'r', encoding='utf-8') as f:
            train_stats = json.load(f)
        with open(val_stats_file, 'r', encoding='utf-8') as f:
            val_stats = json.load(f)
    else:
        # Veri kalitesi kontrolü - Train seti için
        print("\n>>> TRAIN VERİ SETİ KALİTESİ <<<")
        train_invalid = check_data_quality(
            dataset['train'], 
            text_column="transcription",  # Khan Academy dataseti için
            audio_column="audio",
            min_duration=5.0,    # En az 5 saniye
            max_duration=40.0,   # En fazla 40 saniye (güncellendi)
            min_text_length=5,   # En az 5 karakter
            max_text_length=500,  # En fazla 500 karakter
            sample_size=3000  # Hız için 3000 örnek kontrol et
        )
        
        # Veri kalitesi kontrolü - Validation seti için
        print("\n>>> VALIDATION VERİ SETİ KALİTESİ <<<")
        val_invalid = check_data_quality(
            dataset['validation'], 
            text_column="transcription",
            audio_column="audio",
            min_duration=5.0,    # En az 5 saniye
            max_duration=40.0,   # En fazla 40 saniye (güncellendi)
            min_text_length=5,   # En az 5 karakter
            max_text_length=500,  # En fazla 500 karakter
            sample_size=500  # Validation küçük, 500 yeterli
        )
        
        # Metin/Ses oran analizi (opsiyonel - kapatmak isterseniz yorum yapın)
        # analyze_text_audio_ratio(dataset['train'], text_column="transcription", audio_column="audio")
        print("\n⏭️  Metin/Ses oran analizi atlandı (hız için)")
        
        # Kapsamlı veri seti istatistikleri
        logger.log("\n📊 Kapsamlı veri seti istatistikleri oluşturuluyor...")
        train_stats = generate_dataset_statistics(
            dataset['train'], 
            split_name="train",
            text_column="transcription",
            audio_column="audio",
            output_dir="./whisper-small-turkish-khanacademy",
            sample_size=2000  # Hız için 2000 örnek
        )
        
        val_stats = generate_dataset_statistics(
            dataset['validation'], 
            split_name="validation",
            text_column="transcription",
            audio_column="audio",
            output_dir="./whisper-small-turkish-khanacademy",
            sample_size=500  # Validation küçük, 500 yeterli
        )
    
    # SÜRESİ 5-40 SANİYE ARASINDA OLMAYAN ÖRNEKLERİ FİLTRELE
    # Bu filtreleme her zaman yapılır (cache kullanılsa bile)
    print("\n🔍 Ses süresi filtresi uygulanıyor (5-40 saniye)...")
    
    def filter_by_duration(dataset, min_dur=5.0, max_dur=40.0, show_progress=True):
        """Süre bazlı filtreleme"""
        valid_indices = []
        filtered_count = 0
        total = len(dataset)
        
        for idx in range(total):
            if show_progress and idx % 5000 == 0 and idx > 0:
                print(f"  İşlenen: {idx}/{total}")
            
            try:
                audio_data = dataset[idx]["audio"]
                duration = len(audio_data["array"]) / audio_data["sampling_rate"]
                
                if min_dur <= duration <= max_dur:
                    valid_indices.append(idx)
                else:
                    filtered_count += 1
            except:
                filtered_count += 1
        
        return dataset.select(valid_indices), filtered_count
    
    train_before = len(dataset['train'])
    val_before = len(dataset['validation'])
    
    dataset['train'], train_filtered = filter_by_duration(dataset['train'], 5.0, 40.0)
    dataset['validation'], val_filtered = filter_by_duration(dataset['validation'], 5.0, 40.0)
    
    print(f"✅ Train: {train_filtered} örnek filtrelendi ({train_before} → {len(dataset['train'])})")
    print(f"✅ Validation: {val_filtered} örnek filtrelendi ({val_before} → {len(dataset['validation'])})")
    print(f"📊 Toplam kalan veri: {len(dataset['train'])} train + {len(dataset['validation'])} validation\n")
    
    # Bellek temizle
    clear_memory()

    # Processor ve modeli yükle
    print("\nProcessor ve model yükleniyor...")
    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-small", 
        language="Turkish", 
        task="transcribe"
    )

    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="Turkish", task="transcribe")
    model.config.suppress_tokens = []

    # Modeli GPU'ya taşı
    model.to(device)
    
    # Model yüklendikten sonra bellek durumu
    if device == "cuda":
        print("\nModel yüklendi.")
        print_gpu_memory()
        clear_memory()
        print("Bellek temizlendi.")
        print_gpu_memory()

    # Veri ön işleme fonksiyonu
    def prepare_dataset(batch):
        # Ses dosyası zaten datasets tarafından decode edilmiş
        audio = batch["audio"]
        
        # Input features'ı hesapla
        batch["input_features"] = processor.feature_extractor(
            audio["array"], 
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt"
        ).input_features[0]
        
        # Doğru sütun adını bul (transcription, sentence veya text)
        text_column = None
        if "transcription" in batch:
            text_column = batch["transcription"]
        elif "sentence" in batch:
            text_column = batch["sentence"]
        elif "text" in batch:
            text_column = batch["text"]
        else:
            raise ValueError("Veri setinde 'transcription', 'sentence' veya 'text' sütunu bulunamadı!")
        
        # Labels'leri encode et
        batch["labels"] = processor.tokenizer(
            text_column,
            padding=False,
            truncation=True,
            max_length=448  # Maksimum token uzunluğu
        ).input_ids
        
        return batch

    # Veri setini hazırla
    print("Veri seti hazırlanıyor...")
    tokenized_dataset = dataset.map(
        prepare_dataset,
        remove_columns=dataset["train"].column_names,
        num_proc=1  # Windows için tek işlem kullan
    )

    # Metric için hazırlık
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        
        # Label_ids'de -100 olanları ignore et
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        
        # Decode et
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        
        # WER ve CER hesapla
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        
        return {"wer": wer, "cer": cer}

    # Data collator (HF Whisper dokümantasyonuna göre)
    if DataCollatorSpeechSeq2SeqWithPadding is not None:
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=processor,
            model=model,
            pad_to_multiple_of=8 if device == "cuda" else None,
        )
    else:
        # Eski sürümler için kendi collator'ımız
        data_collator = SimpleWhisperCollator(
            processor=processor,
            pad_to_multiple_of=8 if device == "cuda" else None,
        )

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper-small-turkish-khanacademy",
        per_device_train_batch_size=8 if device == "cuda" else 2,  # GPU için 8, CPU için 2
        per_device_eval_batch_size=4 if device == "cuda" else 1,   # GPU için 4, CPU için 1
        gradient_accumulation_steps=2,  # Efektif batch size: 16 (GPU) veya 4 (CPU)
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=4000,  # Maksimum 4000 step, early stopping ile daha erken durabilir
        lr_scheduler_type="cosine",  # ⭐ Cosine decay: daha iyi yakınsama
        warmup_ratio=0.0,  # warmup_steps kullanıyoruz
        gradient_checkpointing=(device == "cuda"),  # CPU'da kapat (checkpoint backward hatasını önle)
        fp16=device == "cuda",  # GPU'da mixed precision training - %50 bellek tasarrufu
        eval_strategy="steps",
        eval_steps=500,  # Her 500 step'te bir değerlendirme
        save_steps=500,  # Her 500 step'te bir checkpoint kaydet
        logging_steps=100,  # Her 100 step'te bir log kaydet
        report_to=["tensorboard"],
        load_best_model_at_end=True,  # Eğitim sonunda en iyi modeli yükle
        metric_for_best_model="wer",  # En iyi model WER'e göre seçilir
        greater_is_better=False,  # WER için düşük daha iyi
        predict_with_generate=True,  # Değerlendirmede generate kullan
        generation_max_length=225,  # Maksimum 225 token generate et
        
        # ⭐ GENERATION KALİTE İYİLEŞTİRMELERİ (Whisper best practices)
        generation_num_beams=5,  # Beam search: 5 farklı yol dene (kalite artışı)
        generation_config=None,  # Manuel config için (aşağıda ayarlanacak)
        
        push_to_hub=False,
        seed=42,
        
        # ⭐ LABEL SMOOTHING: Overfitting'i azaltır
        label_smoothing_factor=0.1,  # %10 smoothing
        
        # Bellek optimizasyonları
        optim="adamw_torch",  # Varsayılan optimizer, bellek verimliliği için
        max_grad_norm=1.0,  # Gradient clipping - gradient patlamasını önler
        dataloader_num_workers=0,  # Windows için 0, Linux/Mac'te 2-4 olabilir
        dataloader_pin_memory=True if device == "cuda" else False,  # GPU için bellek optimizasyonu
        ddp_find_unused_parameters=False,  # Distributed training için bellek tasarrufu
        save_total_limit=2,  # Sadece son 2 checkpoint'i sakla - disk alanı tasarrufu
        remove_unused_columns=True,  # Kullanılmayan sütunları kaldır
    )
    
    # ⭐ GENERATION CONFIG: Kalite iyileştirmeleri
    from transformers import GenerationConfig
    
    generation_config = GenerationConfig(
        max_length=448,              # Whisper max decoder length
        num_beams=5,                 # Beam search: 5 yol
        length_penalty=1.0,          # Uzunluk cezası (1.0=nötr)
        no_repeat_ngram_size=3,      # 3-gram tekrarını engelle
        repetition_penalty=1.2,      # Tekrar cezası (>1: tekrarı azalt)
        # temperature parametresi sadece sampling ile kullanılır; beam search'te kaldırıldı
        do_sample=False,             # Beam search için False
        early_stopping=True,         # EOS görünce dur
        # Whisper'a özel
        forced_decoder_ids=model.config.forced_decoder_ids,
        suppress_tokens=model.config.suppress_tokens,
        begin_suppress_tokens=model.config.begin_suppress_tokens,
    )
    
    # Model'e ata
    model.generation_config = generation_config
    training_args.generation_config = generation_config
    
    logger.log("✅ Generation config ayarlandı (beam_search=5, repetition_penalty=1.2)")
    
    # Eğitim bilgilerini yazdır
    print("\n" + "="*60)
    print("EĞİTİM AYARLARI")
    print("="*60)
    print(f"Batch size (train): {training_args.per_device_train_batch_size}")
    print(f"Batch size (eval): {training_args.per_device_eval_batch_size}")
    print(f"Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    print(f"Efektif batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"Maksimum steps: {training_args.max_steps}")
    print(f"Evaluation strategy: Her {training_args.eval_steps} step'te bir")
    print(f"Early stopping patience: 3 evaluation (3 x {training_args.eval_steps} = 1500 step)")
    print(f"FP16: {'Aktif' if training_args.fp16 else 'Pasif'}")
    print(f"Veri seti boyutu (train): {len(tokenized_dataset['train'])}")
    print(f"Veri seti boyutu (validation): {len(tokenized_dataset['validation'])}")
    print("\nBELLEK OPTİMİZASYONLARI:")
    print(f"✓ Gradient Checkpointing: Aktif")
    print(f"✓ FP16 Training: {'Aktif' if training_args.fp16 else 'Pasif'}")
    print(f"✓ Gradient Clipping: {training_args.max_grad_norm}")
    print(f"✓ Dataloader Workers: {training_args.dataloader_num_workers}")
    print(f"✓ Pin Memory: {'Aktif' if training_args.dataloader_pin_memory else 'Pasif'}")
    print(f"✓ Checkpoint Limit: {training_args.save_total_limit}")
    
    print("\n⭐ KALİTE İYİLEŞTİRMELERİ:")
    print(f"✓ Beam Search: {generation_config.num_beams} beams")
    print(f"✓ Repetition Penalty: {generation_config.repetition_penalty}")
    print(f"✓ No Repeat N-gram: {generation_config.no_repeat_ngram_size}")
    print(f"✓ Label Smoothing: {training_args.label_smoothing_factor}")
    print(f"✓ LR Scheduler: {training_args.lr_scheduler_type}")
    print(f"✓ Temperature: {generation_config.temperature} (deterministik)")
    print("="*60 + "\n")

    # Progress callback ile trainer oluştur
    progress_callback = ProgressCallback(logger, early_stopping_patience=3)
    
    # Trainer oluştur
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor,  # ✅ tokenizer yerine processing_class (yeni API)
        callbacks=[progress_callback],
    )

    # Fine-tuning başlat
    logger.start_training()
    if device == "cuda":
        logger.log("Eğitim öncesi bellek durumu:")
        print_gpu_memory()
    
    trainer.train()
    
    # Eğitim sonrası bellek temizle
    clear_memory()
    if device == "cuda":
        logger.log("\nEğitim sonrası bellek durumu:")
        print_gpu_memory()

    # Modeli kaydet
    logger.log("Model kaydediliyor...")
    trainer.save_model()
    processor.save_pretrained("./whisper-small-turkish-khanacademy")

    # Final değerlendirme
    print("\n" + "="*60)
    print("FİNAL DEĞERLENDİRME")
    print("="*60)
    results = trainer.evaluate()
    print(f"Final WER (Word Error Rate): {results['eval_wer']:.4f}")
    print(f"Final CER (Character Error Rate): {results['eval_cer']:.4f}")
    print("="*60 + "\n")
    
    # Not: WER ve CER değerleri ne kadar düşükse o kadar iyi
    # WER < 0.10: Mükemmel
    # WER 0.10-0.20: Çok iyi
    # WER 0.20-0.30: İyi
    # WER > 0.30: İyileştirmeye ihtiyaç var
    
    # Logger'a final metrikleri kaydet
    logger.end_training(final_metrics={
        "final_wer": results['eval_wer'],
        "final_cer": results['eval_cer']
    })

    # Modeli test etmek için örnek fonksiyon
    def transcribe_audio(audio_path, use_beam_search=True):
        """
        Ses dosyasını transkribe et
        
        Args:
            audio_path: Ses dosyası yolu
            use_beam_search: Beam search kullan (daha iyi kalite)
        """
        import librosa  # OPUS format desteği için
        
        # Ses dosyasını yükle (librosa otomatik mono yapar)
        audio_data, sampling_rate = librosa.load(audio_path, sr=16000, mono=True)
        
        # Input features'ı hazırla
        input_features = processor.feature_extractor(
            audio_data,
            sampling_rate=sampling_rate,
            return_tensors="pt"
        ).input_features.to(device)
        
        # Tokenları oluştur (generation config ile)
        if use_beam_search:
            predicted_ids = model.generate(
                input_features,
                generation_config=generation_config  # ⭐ Beam search + quality improvements
            )
        else:
            # Greedy decoding (hızlı ama daha düşük kalite)
            predicted_ids = model.generate(input_features, num_beams=1)
        
        # Transkripsiyonu decode et
        transcription = processor.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
        
        return transcription

    # Örnek transkripsiyon
    logger.log("Eğitim tamamlandı! Model hazır.")


if __name__ == '__main__':
    main()

# =============================================================================
# BELLEK OPTİMİZASYONU İPUÇLARI
# =============================================================================
# 
# GPU belleği yetmiyorsa şu ayarları yapın:
#
# 1. Batch size'ı düşür:
#    per_device_train_batch_size=4 (veya 2, 1)
#
# 2. Gradient accumulation'ı artır:
#    gradient_accumulation_steps=4 (veya 8)
#    (Efektif batch size aynı kalır: 4x4=16)
#
# 3. Generation length'i azalt:
#    generation_max_length=150 (veya daha düşük)
#
# 4. Model boyutunu küçült:
#    "openai/whisper-tiny" veya "openai/whisper-base" kullan
#
# 5. Evaluation batch size'ı düşür:
#    per_device_eval_batch_size=1
#
# 6. CPU'da eğitim yapıyorsanız:
#    - Batch size=1 kullanın
#    - num_proc=1 kullanın (zaten ayarlı)
#    - Gradient accumulation artırın
#
# 7. Adafactor optimizer kullanın (daha az bellek):
#    optim="adafactor"
#
# =============================================================================
# VERİ KALİTESİ KONTROL İPUÇLARI
# =============================================================================
#
# 1. Veri kalitesi kontrolleri otomatik yapılır:
#    - Ses süresi kontrolü (0.5-30 saniye)
#    - Metin uzunluğu kontrolü (5-500 karakter)
#    - Boş metin kontrolü
#    - Metin/ses oranı analizi
#
# 2. Geçersiz örnekleri filtrelemek için:
#    FILTER_INVALID_SAMPLES = True yapın (316. satır)
#
# 3. Filtreleme eşiklerini ayarlamak için:
#    check_data_quality fonksiyonundaki parametreleri değiştirin:
#    - min_duration: Minimum ses süresi (şu an: 5.0 saniye)
#    - max_duration: Maksimum ses süresi (şu an: 30.0 saniye)
#    - min_text_length: Minimum metin uzunluğu (şu an: 5 karakter)
#    - max_text_length: Maksimum metin uzunluğu (şu an: 500 karakter)
#
# 4. İyi veri kalitesi özellikleri:
#    - Ortalama ses süresi: 5-15 saniye
#    - Ortalama metin uzunluğu: 50-200 karakter
#    - Karakter/saniye oranı: 20-40 (Türkçe için)
#    - Geçersiz örnek oranı: < %5
#
# 5. Veri kalitesi sorunları:
#    - Çok kısa sesler (< 5 sn): Yetersiz içerik, gürültü olabilir
#    - Çok uzun sesler (> 20 sn): Bellek sorunu çıkarabilir
#    - Çok kısa metinler (< 5 karakter): Etiketleme hatası olabilir
#    - Yüksek karakter/saniye oranı (> 60): Hızlı konuşma veya hatalı transkript
#    - Düşük karakter/saniye oranı (< 15): Yavaş konuşma veya sessizlik
#
# =============================================================================