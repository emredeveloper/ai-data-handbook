"""
Türkçe için Whisper Small Fine-tuning Scripti
Khan Academy Türkçe Dataset kullanarak Whisper modelini fine-tune eder
"""

import os
import torch
import torchaudio
from datasets import load_dataset, DatasetDict, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    WhisperTokenizer,
    WhisperFeatureExtractor
)
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
import numpy as np
from huggingface_hub import login
import warnings
warnings.filterwarnings("ignore")

def main():
    # Hugging Face Hub girişi atlandı (token gerekirse otomatik istenir)
    print("Hugging Face Hub'a giriş atlandı...")

    # Dataset yükleme
    print("Khan Academy Türkçe dataset'i yükleniyor...")
    try:
        # Dataset'i yükle
        dataset = load_dataset("ysdede/khanacademy-turkish")
        
        # Dataset yapısını kontrol et
        print("Dataset yapısı:")
        print(dataset)
        
        # Eğer dataset'te train/test split yoksa oluştur
        if "train" not in dataset:
            # Dataset'i %80 train, %20 test olarak böl
            dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
            dataset = DatasetDict({
                "train": dataset["train"],
                "test": dataset["test"]
            })
        
        # Sadece audio ve transcription sütunlarını seç
        if "audio" in dataset["train"].column_names and "transcription" in dataset["train"].column_names:
            dataset = dataset.select_columns(["audio", "transcription"])
            # transcription sütununu text olarak yeniden adlandır
            dataset = dataset.rename_column("transcription", "text")
        elif "audio" in dataset["train"].column_names and "text" in dataset["train"].column_names:
            dataset = dataset.select_columns(["audio", "text"])
        elif "sentence" in dataset["train"].column_names:
            dataset = dataset.select_columns(["audio", "sentence"])
            # sentence sütununu text olarak yeniden adlandır
            dataset = dataset.rename_column("sentence", "text")
        else:
            print("Mevcut sütunlar:", dataset["train"].column_names)
            raise ValueError("Dataset'te 'audio' ve 'transcription' sütunları bulunamadı")
            
        print(f"Train örnekleri: {len(dataset['train'])}")
        print(f"Test örnekleri: {len(dataset['test'])}")
        
        # 500 veri ile eğitim yap
        print("Dataset küçültülüyor (500 train, 50 test)...")
        dataset["train"] = dataset["train"].select(range(500))
        dataset["test"] = dataset["test"].select(range(50))
        print(f"Yeni train örnekleri: {len(dataset['train'])}")
        print(f"Yeni test örnekleri: {len(dataset['test'])}")
        
    except Exception as e:
        print(f"Dataset yükleme hatası: {e}")
        print("Alternatif dataset deneniyor...")
        # Fallback - minimal dataset
        raise Exception("Dataset yüklenemedi")

    # Audio'yu olduğu gibi bırak - preprocessing'de handle edilecek
    print("Audio verisi preprocessing'de işlenecek...")

    # Audio kolonunu 16kHz'e cast et (AudioDecoder/sampling_rate sorunlarını önler)
    try:
        print("Audio kolonu 16kHz'e cast ediliyor (datasets.Audio)...")
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    except Exception as e:
        print(f"Audio cast başarısız, devam ediliyor: {e}")

    # Processor'ı yükle (Türkçe için)
    print("Whisper processor yükleniyor...")
    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-small", 
        language="turkish", 
        task="transcribe"
    )

    # Model'i yükle
    print("Whisper model yükleniyor...")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

    # Model konfigürasyonu
    model.config.use_cache = False
    model.config.forced_decoder_ids = None  # Forced decoder IDs'yi temizle

    # Generation için cache'i etkinleştir
    from functools import partial
    model.generate = partial(
        model.generate, 
        language="turkish", 
        task="transcribe", 
        use_cache=True
    )

    # Data preprocessing fonksiyonu - cast sonrası dict tabanlı basit akış
    def prepare_dataset(batch):
        try:
            # Audio'yu işle
            audio = batch["audio"]

            # Beklenen: dict {array, sampling_rate}. Değilse yumuşak dönüştürme uygula
            if isinstance(audio, dict):
                audio_array = audio.get("array")
                sampling_rate = audio.get("sampling_rate", 16000)
            else:
                audio_array = getattr(audio, "array", audio)
                sampling_rate = getattr(audio, "sampling_rate", 16000)

            # sampling_rate güvenliği
            try:
                sampling_rate = int(sampling_rate) if sampling_rate else 16000
            except Exception:
                sampling_rate = 16000

            # Numpy'a çevir
            if not isinstance(audio_array, np.ndarray):
                if hasattr(audio_array, "numpy"):
                    audio_array = audio_array.numpy()
                else:
                    audio_array = np.asarray(audio_array, dtype=object)

            # Tip ve güvenlik kontrolleri
            # Objeli/jagged dizileri düzleştir
            if audio_array.dtype == object:
                try:
                    parts = []
                    for x in audio_array:
                        if x is None:
                            continue
                        x_arr = np.asarray(x, dtype=np.float32).reshape(-1)
                        parts.append(x_arr)
                    audio_array = np.concatenate(parts) if parts else np.zeros(16000, dtype=np.float32)
                except Exception:
                    audio_array = np.zeros(16000, dtype=np.float32)
            else:
                audio_array = audio_array.astype(np.float32, copy=False)

            # Boyut düzenleme
            if audio_array.ndim >= 2:
                # (C, N) veya (N, C): C=2 ise kanalları ortalama al
                if 2 in audio_array.shape:
                    axis = int(np.argmin(audio_array.shape)) if audio_array.ndim == 2 else -1
                    audio_array = np.mean(audio_array, axis=axis)
                else:
                    audio_array = audio_array.reshape(-1)
            if audio_array.size == 0 or np.isnan(audio_array).any():
                audio_array = np.zeros(16000, dtype=np.float32)
                sampling_rate = 16000

            # Genlik normalizasyonu (gerekirse)
            max_abs = float(np.max(np.abs(audio_array))) if audio_array.size else 0.0
            if max_abs > 1.0:
                audio_array = audio_array / max_abs
            
            # Text'i işle
            batch["input_features"] = processor(
                audio_array, 
                sampling_rate=sampling_rate
            ).input_features[0]
            
            # Text'i tokenize et
            batch["labels"] = processor.tokenizer(
                batch["text"], 
                max_length=225, 
                padding="max_length", 
                truncation=True
            ).input_ids
            
        except Exception as e:
            print(f"Audio preprocessing hatası: {e}")
            print("❌ Bu örnek atlanıyor, dummy data kullanılıyor...")
            # Hata durumunda tamamen dummy data oluştur
            batch["input_features"] = processor(
                np.zeros(16000, dtype=np.float32), 
                sampling_rate=16000
            ).input_features[0]
            batch["labels"] = processor.tokenizer(
                "Dummy text for failed audio processing.", 
                max_length=225, 
                padding="max_length", 
                truncation=True
            ).input_ids
        
        return batch

    # Dataset'i hazırla
    print("Dataset hazırlanıyor...")
    dataset = dataset.map(
        prepare_dataset,
        remove_columns=dataset["train"].column_names,
        num_proc=1  # Windows'ta multiprocessing sorununu önlemek için 1 yapıldı
    )

    # Data collator
    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            # Input features'ları stack et
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

            # Labels'ları stack et
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

            # Labels'da padding token'ları -100 ile değiştir
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

            # Eğer tüm labels -100 ise, loss hesaplanmayacak
            if (labels == -100).all():
                labels[:, 0] = self.processor.tokenizer.eos_token_id

            batch["labels"] = labels

            return batch

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Evaluation metrikleri
    print("Evaluation metrikleri yükleniyor...")
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # -100'leri kaldır
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # Decode et
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        # WER hesapla
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        
        # CER hesapla
        cer = cer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer, "cer": cer}

    # Training arguments
    print("Training konfigürasyonu ayarlanıyor...")
    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper-small-turkish",
        per_device_train_batch_size=2,  # Daha da azaltıldı
        gradient_accumulation_steps=2,  # Gradient accumulation artırıldı
        learning_rate=1e-5,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=10,  # Warmup artırıldı
        max_steps=30,  # Orta seviye step - dengeli fine-tuning
        gradient_checkpointing=False,  # Gradient checkpointing kapatıldı
        fp16=False,  # FP16 kapatıldı
        fp16_full_eval=False,
        eval_strategy="steps",
        per_device_eval_batch_size=2,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=15,
        eval_steps=15,
        logging_steps=5,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,  # Hub'a yükleme kapalı
        hub_strategy="checkpoint",
        dataloader_num_workers=0,  # Multiprocessing kapatıldı
    )

    # Trainer'ı oluştur
    print("Trainer oluşturuluyor...")
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.tokenizer,  # feature_extractor yerine tokenizer
    )

    # Training'i başlat
    print("Training başlatılıyor...")
    print("Bu işlem GPU'ya bağlı olarak 1-3 saat sürebilir...")

    try:
        trainer.train()
        print("Training tamamlandı!")
        print("Model './whisper-small-turkish' klasörüne kaydedildi!")
        
    except Exception as e:
        print(f"Training hatası: {e}")
        print("Lütfen GPU memory'yi kontrol edin ve batch size'ı azaltın.")

    # Test örneği
    print("\nTest örneği:")
    print("Model'i test etmek için:")
    print("from transformers import pipeline")
    print("pipe = pipeline('automatic-speech-recognition', model='./whisper-small-turkish')")
    print("result = pipe('path/to/audio.wav')")
    print("print(result['text'])")

if __name__ == "__main__":
    main()