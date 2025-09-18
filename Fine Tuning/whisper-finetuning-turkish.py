# Hata ayıklama ve versiyon kontrolü
import warnings
warnings.filterwarnings("ignore")

# Temel kütüphaneleri yükle
import torch
print(f"PyTorch version: {torch.__version__}")

# Torchvision kontrolü
try:
    import torchvision
    print(f"Torchvision version: {torchvision.__version__}")
except Exception as e:
    print(f"Torchvision uyarısı: {e}")

# Datasets
from datasets import load_dataset, DatasetDict

# Transformers - Hugging Face dokümantasyonuna göre doğru import
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import numpy as np
import time

# Model ve veri seti konfigürasyonu
MODEL_NAME = "openai/whisper-small"
DATASET_NAME = "ysdede/khanacademy-turkish"
LANGUAGE = "turkish"
TASK = "transcribe"

print(" ULTRA HIZLI WHISPER FINE-TUNING")
print("📊 Sadece 100 örnek + 50 step training!")
print("="*50)

print("Model ve processor yükleniyor...")
# Model bileşenlerini yükle - Hugging Face dokümantasyonuna göre
feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)
processor = WhisperProcessor.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)

# Model yükle
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

# Türkçe için tokenizer ayarlarını güncelle
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
tokenizer.set_prefix_tokens(language=LANGUAGE, task=TASK)

print("Veri seti yükleniyor...")
# Veri setini yükle
dataset = load_dataset(DATASET_NAME, split="train")

# ULTRA KÜÇÜK DATASET - Sadece 100 örnek!
print("⚡ ULTRA HIZLI MODU: Sadece 100 örnek kullanılıyor...")
dataset = dataset.select(range(min(100, len(dataset))))  # Sadece 100 örnek
dataset = dataset.train_test_split(test_size=0.2, seed=42)  # 80 train, 20 validation

print(f"✅ Train örnekleri: {len(dataset['train'])} (80 örnek)")
print(f"✅ Validation örnekleri: {len(dataset['test'])} (20 örnek)")
print("⚡ Bu ultra hızlı test içindir!")

# Veri preprocessing fonksiyonu - torchcodec olmadan
def prepare_dataset(batch):
    audio = batch["audio"]
    
    # Audio array'i doğrudan kullan
    audio_array = audio["array"]
    sampling_rate = audio["sampling_rate"]
    
    # Feature extraction - torchcodec olmadan
    try:
        batch["input_features"] = feature_extractor(
            audio_array,
            sampling_rate=sampling_rate
        ).input_features[0]
    except Exception as e:
        print(f"Feature extraction hatası: {e}")
        # Alternatif: basit mel spectrogram
        import librosa
        mel_spec = librosa.feature.melspectrogram(
            y=audio_array, 
            sr=sampling_rate, 
            n_mels=80
        )
        batch["input_features"] = mel_spec.T  # Transpose for Whisper format

    # Tokenize transcription
    batch["labels"] = tokenizer(batch["transcription"]).input_ids

    return batch

print("📝 Veri seti preprocessing...")
# Veri setini hazırla - torchcodec olmadan
try:
    dataset = dataset.map(
        prepare_dataset,
        remove_columns=dataset["train"].column_names,
        num_proc=0  # Multiprocessing'i tamamen kapat
    )
except Exception as e:
    print(f"Map hatası: {e}")
    # Alternatif: tek tek işle
    print("Alternatif yöntemle işleniyor...")
    train_data = []
    test_data = []
    
    for i, example in enumerate(dataset["train"]):
        try:
            processed = prepare_dataset(example)
            train_data.append(processed)
        except Exception as ex:
            print(f"Train örnek {i} hatası: {ex}")
            continue
    
    for i, example in enumerate(dataset["test"]):
        try:
            processed = prepare_dataset(example)
            test_data.append(processed)
        except Exception as ex:
            print(f"Test örnek {i} hatası: {ex}")
            continue
    
    # Dataset'i yeniden oluştur
    from datasets import Dataset
    dataset = {
        "train": Dataset.from_list(train_data),
        "test": Dataset.from_list(test_data)
    }

# Data collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

# Evaluation metrik
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

print("⚡ ULTRA HIZLI TRAINING AYARLARI:")
# Ultra hızlı training argümanları
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-turkish-ultra-fast",
    per_device_train_batch_size=16,  # Daha büyük batch size
    gradient_accumulation_steps=1,   # Accumulation yok
    learning_rate=5e-5,              # Daha yüksek LR
    warmup_steps=10,                 # Minimal warmup
    max_steps=50,                    # SADECE 50 STEP!
    gradient_checkpointing=False,    # Kapalı - daha hızlı
    fp16=torch.cuda.is_available(),
    eval_strategy="steps",
    per_device_eval_batch_size=16,
    predict_with_generate=True,
    generation_max_length=150,       # Daha kısa
    save_steps=25,                   # 2 kez kaydet
    eval_steps=25,                   # 2 kez evaluate
    logging_steps=5,                 # Sık log
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    save_total_limit=1,              # Sadece 1 checkpoint
    dataloader_num_workers=0,        # Tek thread
    remove_unused_columns=False,
)

# Trainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

# GPU bilgisi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"️  Cihaz: {device}")

if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    estimated_time = "2-3 dakika"
else:
    print("💻 CPU kullanıyorsunuz")
    estimated_time = "5-10 dakika"

print(f"⏰ Tahmini süre: {estimated_time}")
print(f"📊 {len(dataset['train'])} train örneği, batch size {training_args.per_device_train_batch_size}")
print(f"🔥 Toplam sadece {training_args.max_steps} step!")

print("\n" + "="*50)
print(" ULTRA HIZLI TRAINING BAŞLIYOR...")
print("="*50)

# Training başlat
start_time = time.time()
trainer.train()
end_time = time.time()

training_time = end_time - start_time
print(f"\n Training tamamlandı!")
print(f"⏰ Gerçek süre: {training_time/60:.1f} dakika")
print(f"⚡ Hız: {training_args.max_steps/training_time*60:.1f} step/dakika")

# Model kaydet
print("💾 Model kaydediliyor...")
trainer.save_model()
processor.save_pretrained("./whisper-turkish-ultra-fast")

print("✅ Model kaydedildi: './whisper-turkish-ultra-fast'")

# Hızlı değerlendirme
print("\n📊 HIZLI DEĞERLENDİRME:")
logs = trainer.state.log_history

if logs:
    final_train_loss = next((log['train_loss'] for log in reversed(logs) if 'train_loss' in log), None)
    final_eval_loss = next((log['eval_loss'] for log in reversed(logs) if 'eval_loss' in log), None)
    final_wer = next((log['eval_wer'] for log in reversed(logs) if 'eval_wer' in log), None)

    print(f"📉 Final Train Loss: {final_train_loss:.4f}" if final_train_loss else "Train Loss: -")
    print(f"📉 Final Eval Loss: {final_eval_loss:.4f}" if final_eval_loss else "Eval Loss: -")
    print(f"🎯 Final WER: {final_wer:.2f}%" if final_wer else "WER: -")

# Ultra hızlı test fonksiyonu
def ultra_fast_test():
    """En hızlı test - sadece 1 örnek"""
    print("\n⚡ ULTRA HIZLI TEST:")

    if len(dataset["test"]) > 0:
        sample = dataset["test"][0]

        # Model yükle
        model_path = "./whisper-turkish-ultra-fast"
        test_processor = WhisperProcessor.from_pretrained(model_path)
        test_model = WhisperForConditionalGeneration.from_pretrained(model_path)
        test_model.to(device)
        test_model.eval()

        # Original text
        original_labels = [label for label in sample["labels"] if label != -100]
        original_text = tokenizer.decode(original_labels, skip_special_tokens=True)

        # Predict
        input_tensor = torch.tensor(sample["input_features"]).unsqueeze(0).to(device)

        start_time = time.time()
        with torch.no_grad():
            generated_ids = test_model.generate(
                input_features=input_tensor,
                max_length=150,
                num_beams=1,  # En hızlı
                do_sample=False
            )

        predicted_text = test_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        inference_time = time.time() - start_time

        print(f" Gerçek    : '{original_text}'")
        print(f" Tahmin    : '{predicted_text}'")
        print(f"⏰ Inference : {inference_time:.3f} saniye")

        # Benzerlik
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, original_text.lower(), predicted_text.lower()).ratio()
        print(f"📊 Benzerlik : {similarity*100:.1f}%")

        return predicted_text
    else:
        print("❌ Test örneği yok!")
        return None

# Test çalıştır
ultra_fast_test()

print(f"\n" + "="*50)
print("🏁 ÖZET:")
print(f" Veri: {len(dataset['train'])} train + {len(dataset['test'])} test örneği")
print(f" Training: {training_args.max_steps} step")
print(f"⏰ Süre: {training_time/60:.1f} dakika")
print(f"💾 Model: ./whisper-turkish-ultra-fast")
print("⚡ Ultra hızlı test tamamlandı!")
print("="*50)

print("\n SONUÇ:")
print("✅ Bu ultra hızlı bir demo fine-tuning'dir")
print(" Gerçek kullanım için daha fazla veri ve step gerekir")
print(" Temel yapı çalışıyor, parametreleri artırabilirsiniz")
print("\nKullanım:")
print("• ultra_fast_test() - Hızlı test")
print("• Model yolu: ./whisper-turkish-ultra-fast")