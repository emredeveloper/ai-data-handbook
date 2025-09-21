"""
Whisper Model Evaluation Script
Evaluates fine-tuned Whisper models on Turkish datasets
"""

import os
import argparse
import evaluate
import numpy as np
import librosa
import unicodedata
import re
from tqdm import tqdm
from pathlib import Path
from transformers import pipeline, WhisperForConditionalGeneration
from datasets import load_dataset, Audio
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import warnings
warnings.filterwarnings("ignore")

# Reward Shaping import (dinamik yükleme - dosya adı tire içerdiği için modül olarak içe aktarılamıyor)
try:
    from whisper_finetuning_turkish import RewardShaping, turkish_text_preprocessing, advanced_audio_preprocessing  # noqa: F401
except Exception:
    import importlib.util
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ft_path = os.path.join(current_dir, "whisper-finetuning-turkish.py")
    if not os.path.exists(ft_path):
        # Alternatif: çalışma dizininden dene
        ft_path = os.path.join(os.getcwd(), "Fine Tuning", "Whisper Turkish Fine Tuning", "whisper-finetuning-turkish.py")
    spec = importlib.util.spec_from_file_location("whisper_ft_dynamic", ft_path)
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError("whisper-finetuning-turkish.py bulunamadı, RewardShaping import edilemedi")
    module = importlib.util.module_from_spec(spec)
    sys.modules["whisper_ft_dynamic"] = module
    spec.loader.exec_module(module)
    RewardShaping = module.RewardShaping
    turkish_text_preprocessing = module.turkish_text_preprocessing
    advanced_audio_preprocessing = module.advanced_audio_preprocessing

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

def is_target_text_in_range(ref):
    if ref.strip() == "ignore time segment in scoring":
        return False
    else:
        return ref.strip() != ""


def get_text(sample):
    if "text" in sample:
        return sample["text"]
    elif "sentence" in sample:
        return sample["sentence"]
    elif "normalized_text" in sample:
        return sample["normalized_text"]
    elif "transcript" in sample:
        return sample["transcript"]
    elif "transcription" in sample:
        return sample["transcription"]
    else:
        raise ValueError(
            f"Expected transcript column of either 'text', 'sentence', 'normalized_text' or 'transcript'. "
            f"Got sample keys: {sample.keys()}. Ensure a text column name is present in the dataset."
        )


def get_text_column_names(column_names):
    if "text" in column_names:
        return "text"
    elif "sentence" in column_names:
        return "sentence"
    elif "normalized_text" in column_names:
        return "normalized_text"
    elif "transcript" in column_names:
        return "transcript"
    elif "transcription" in column_names:
        return "transcription"


whisper_norm = BasicTextNormalizer()
def normalise(batch):
    # Gelişmiş Türkçe preprocessing uygula
    text = get_text(batch)
    processed_text = turkish_text_preprocessing(text)
    batch["norm_text"] = whisper_norm(processed_text)
    return batch


def data(dataset):
    for i, item in enumerate(dataset):
        # Pipeline'a direkt audio objesini gönder - transformers kendi çözer
        yield {
            "audio": item["audio"],
            "reference": get_text(item), 
            "norm_reference": item["norm_text"]
        }

def test_model(model_pipeline, model_name, dataset, generate_kwargs):
    """Tek bir modeli test et ve sonuçları döndür"""
    predictions = []
    references = []
    norm_predictions = []
    norm_references = []
    
    # Reward shaping için
    reward_shaper = RewardShaping()
    total_rewards = []
    detailed_rewards = []
    
    print(f"\n🧪 {model_name} testi başlıyor...")
    
    for item in tqdm(dataset, desc=f'{model_name} Progress'):
        # Audio preprocessing uygula
        audio_array = item["audio"]["array"]
        sampling_rate = item["audio"]["sampling_rate"]
        processed_audio = advanced_audio_preprocessing(audio_array, sampling_rate)
        
        # Processed audio ile test et
        processed_audio_dict = {
            "array": processed_audio,
            "sampling_rate": sampling_rate
        }
        
        result = model_pipeline(processed_audio_dict, generate_kwargs=generate_kwargs)
        
        # Gelişmiş text preprocessing
        prediction = turkish_text_preprocessing(result["text"])
        reference = turkish_text_preprocessing(get_text(item))
        
        predictions.append(prediction)
        references.append(reference)
        norm_predictions.append(whisper_norm(prediction))
        norm_references.append(item["norm_text"])
        
        # Reward hesapla
        total_reward, rewards = reward_shaper.compute_comprehensive_reward(
            prediction, reference, processed_audio
        )
        total_rewards.append(total_reward)
        detailed_rewards.append(rewards)
    
    # Metrikleri hesapla - güvenlik kontrolü ile
    try:
        wer = wer_metric.compute(references=references, predictions=predictions)
        wer = round(100 * wer, 2)
        # WER %100'ü geçemez, kontrol et
        if wer > 100:
            print(f"⚠️ WER %100'ü geçti: {wer}%, 100% olarak sınırlandırılıyor")
            wer = 100.0
    except Exception as e:
        print(f"❌ WER hesaplama hatası: {e}")
        wer = 100.0
    
    try:
        cer = cer_metric.compute(references=references, predictions=predictions)
        cer = round(100 * cer, 2)
        # CER %100'ü geçemez, kontrol et
        if cer > 100:
            print(f"⚠️ CER %100'ü geçti: {cer}%, 100% olarak sınırlandırılıyor")
            cer = 100.0
    except Exception as e:
        print(f"❌ CER hesaplama hatası: {e}")
        cer = 100.0
    
    try:
        norm_wer = wer_metric.compute(references=norm_references, predictions=norm_predictions)
        norm_wer = round(100 * norm_wer, 2)
        if norm_wer > 100:
            norm_wer = 100.0
    except Exception as e:
        print(f"❌ Norm WER hesaplama hatası: {e}")
        norm_wer = 100.0
    
    try:
        norm_cer = cer_metric.compute(references=norm_references, predictions=norm_predictions)
        norm_cer = round(100 * norm_cer, 2)
        if norm_cer > 100:
            norm_cer = 100.0
    except Exception as e:
        print(f"❌ Norm CER hesaplama hatası: {e}")
        norm_cer = 100.0
    
    # Başarı oranları (100 - hata oranı)
    success_rate = round(100 - wer, 2)
    norm_success_rate = round(100 - norm_wer, 2)
    
    # Ortalama reward'ları hesapla
    avg_total_reward = np.mean(total_rewards) if total_rewards else 0.0
    avg_rewards = {}
    if detailed_rewards:
        for key in detailed_rewards[0].keys():
            avg_rewards[f"avg_reward_{key}"] = np.mean([r[key] for r in detailed_rewards])
    
    return {
        'model_name': model_name,
        'wer': wer,
        'cer': cer,
        'norm_wer': norm_wer,
        'norm_cer': norm_cer,
        'success_rate': success_rate,
        'norm_success_rate': norm_success_rate,
        'predictions': predictions,
        'references': references,
        'avg_total_reward': avg_total_reward,
        **avg_rewards # Detaylı reward'ları da ekle
    }

def main(args):
    print("🚀 Whisper Model Karşılaştırma Testi Başlıyor...")
    print("="*60)
    
    # Dataset'i yükle ve hazırla
    print("📁 Dataset yükleniyor...")
    
    # Her iki veri setinden 10'ar veri al - modelin görmediği veriler
    print("📊 Her iki veri setinden 10'ar veri alınıyor (modelin görmediği veriler)...")
    
    # Veri setleri listesi (3 kaynak): her birinden 10 örnek
    test_datasets = [
        ("cubukcum/TurkishVoiceDataset", "default", "train"),
        ("ysdede/khanacademy-turkish", "default", "test"),
        ("ilkerkara/common_voice_13_0_tr_pseudo_labelled", "tr", "test")
    ]
    
    combined_test_data = []
    
    for dataset_name, config, split in test_datasets:
        print(f"📥 {dataset_name} veri setinden 10 veri alınıyor...")
        try:
            # Streaming ile sadece 10 veri al
            dataset = load_dataset(dataset_name, config, split=split, streaming=True)
            
            # Streaming dataset'ten 10 veri al
            dataset_list = []
            for i, item in enumerate(dataset):
                if i >= 10:
                    break
                dataset_list.append(item)
            
            print(f"✅ {dataset_name}: {len(dataset_list)} veri alındı")
            combined_test_data.extend(dataset_list)
            
        except Exception as e:
            print(f"❌ {dataset_name} streaming başarısız: {e}")
            print(f"🔄 {dataset_name} normal yükleme deneniyor...")
            try:
                dataset = load_dataset(dataset_name, config, split=split)
                dataset = dataset.select(range(min(10, len(dataset))))
                dataset_list = [item for item in dataset]
                print(f"✅ {dataset_name}: {len(dataset_list)} veri alındı (normal)")
                combined_test_data.extend(dataset_list)
            except Exception as e2:
                print(f"❌ {dataset_name} tamamen başarısız: {e2}")
    
    # Combined dataset oluştur
    from datasets import Dataset
    dataset = Dataset.from_list(combined_test_data)
    print(f"🎯 Toplam {len(dataset)} test verisi hazırlandı")

    text_column_name = get_text_column_names(dataset.column_names)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset = dataset.map(normalise, num_proc=2)
    dataset = dataset.filter(is_target_text_in_range, input_columns=[text_column_name], num_proc=2)
    
    print(f"✅ Filtrelemeden sonra {len(dataset)} veri kaldı")

    # Generate kwargs'ı ayarla - daha agresif parametreler
    generate_kwargs = {
        "task": "transcribe",
        "language": args.language,
        "max_length": 150,  # Daha kısa maksimum uzunluk
        "min_length": 5,    # Minimum uzunluk
        "num_beams": 1,     # Beam search kapalı
        "do_sample": True,  # Sampling açık
        "temperature": 0.7, # Daha düşük temperature
        "top_p": 0.9,      # Nucleus sampling
        "repetition_penalty": 1.3,  # Daha yüksek tekrar cezası
        "no_repeat_ngram_size": 2,  # 2-gram tekrarını engelle
        "length_penalty": 1.2,      # Uzunluk cezası
        "early_stopping": True,     # Erken durdurma
        "max_new_tokens": 100,      # Maksimum yeni token
    }

    # 1. Orijinal Whisper Modeli Test Et
    print("\n" + "="*60)
    print("🔵 ORIJINAL WHISPER SMALL MODELI")
    print("="*60)
    
    original_whisper = pipeline(
        "automatic-speech-recognition", 
        model="openai/whisper-small", 
        device=args.device
    )
    
    original_whisper.model.config.forced_decoder_ids = (
        original_whisper.tokenizer.get_decoder_prompt_ids(
            language=args.language, task="transcribe"
        )
    )
    
    original_results = test_model(original_whisper, "Orijinal Whisper", dataset, generate_kwargs)

    # 2. Fine-tuned Modeli Test Et
    print("\n" + "="*60)
    print("🟢 FINE-TUNED WHISPER MODELI")
    print("="*60)
    
    print(f"Loading fine-tuned model from checkpoint: {args.ckpt_dir}")
    
    # Fine-tuned pipeline oluştur
    finetuned_whisper = pipeline(
        "automatic-speech-recognition", 
        model="openai/whisper-small", 
        device=args.device
    )
    
    # Fine-tuned weights'i yükle - checkpoint-100'ü kullan
    checkpoint_100_path = os.path.join(args.ckpt_dir, "checkpoint-100")
    if os.path.exists(checkpoint_100_path):
        print(f"📁 Checkpoint-100 kullanılıyor: {checkpoint_100_path}")
        
        fine_tuned_model = WhisperForConditionalGeneration.from_pretrained(checkpoint_100_path)
        fine_tuned_model = fine_tuned_model.to(finetuned_whisper.device)
        finetuned_whisper.model = fine_tuned_model
    else:
        # Eğer checkpoint-100 yoksa en son checkpoint'i dene
        checkpoint_dirs = [d for d in os.listdir(args.ckpt_dir) if d.startswith('checkpoint-')]
        if checkpoint_dirs:
            # En yüksek numaralı checkpoint'i bul
            latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split('-')[1]))
            checkpoint_path = os.path.join(args.ckpt_dir, latest_checkpoint)
            print(f"⚠️ Checkpoint-100 bulunamadı, en son checkpoint kullanılıyor: {latest_checkpoint}")
            
            fine_tuned_model = WhisperForConditionalGeneration.from_pretrained(checkpoint_path)
            fine_tuned_model = fine_tuned_model.to(finetuned_whisper.device)
            finetuned_whisper.model = fine_tuned_model
        else:
            # Eğer hiç checkpoint yoksa ana klasörü dene
            print("⚠️ Hiç checkpoint bulunamadı, ana klasör deneniyor...")
            fine_tuned_model = WhisperForConditionalGeneration.from_pretrained(args.ckpt_dir)
            fine_tuned_model = fine_tuned_model.to(finetuned_whisper.device)
            finetuned_whisper.model = fine_tuned_model
    
    finetuned_whisper.model.config.forced_decoder_ids = (
        finetuned_whisper.tokenizer.get_decoder_prompt_ids(
            language=args.language, task="transcribe"
        )
    )
    
    print("✅ Fine-tuned weights loaded successfully!")
    
    finetuned_results = test_model(finetuned_whisper, "Fine-tuned Whisper", dataset, generate_kwargs)

    # 3. Karşılaştırma Sonuçları
    print("\n" + "="*80)
    print("📊 KARŞILAŞTIRMA SONUÇLARI")
    print("="*80)
    
    print(f"\n{'Metrik':<20} {'Orijinal':<15} {'Fine-tuned':<15} {'İyileşme':<15}")
    print("-" * 70)
    
    # WER karşılaştırması
    wer_improvement = round(original_results['wer'] - finetuned_results['wer'], 2)
    print(f"{'WER (%)':<20} {original_results['wer']:<15} {finetuned_results['wer']:<15} {wer_improvement:+.2f}")
    
    # CER karşılaştırması  
    cer_improvement = round(original_results['cer'] - finetuned_results['cer'], 2)
    print(f"{'CER (%)':<20} {original_results['cer']:<15} {finetuned_results['cer']:<15} {cer_improvement:+.2f}")
    
    # Başarı oranı karşılaştırması
    success_improvement = round(finetuned_results['success_rate'] - original_results['success_rate'], 2)
    print(f"{'Başarı Oranı (%)':<20} {original_results['success_rate']:<15} {finetuned_results['success_rate']:<15} {success_improvement:+.2f}")
    
    # Reward karşılaştırması
    if 'avg_total_reward' in finetuned_results:
        reward_improvement = round(finetuned_results['avg_total_reward'] - original_results.get('avg_total_reward', 0), 3)
        print(f"{'Reward':<20} {original_results.get('avg_total_reward', 0):<15.3f} {finetuned_results['avg_total_reward']:<15.3f} {reward_improvement:+.3f}")
    
    # Normalized metrikleri
    print(f"\n{'NORMALIZED':<20}")
    print("-" * 70)
    norm_wer_improvement = round(original_results['norm_wer'] - finetuned_results['norm_wer'], 2)
    print(f"{'Norm WER (%)':<20} {original_results['norm_wer']:<15} {finetuned_results['norm_wer']:<15} {norm_wer_improvement:+.2f}")
    
    norm_cer_improvement = round(original_results['norm_cer'] - finetuned_results['norm_cer'], 2)
    print(f"{'Norm CER (%)':<20} {original_results['norm_cer']:<15} {finetuned_results['norm_cer']:<15} {norm_cer_improvement:+.2f}")
    
    norm_success_improvement = round(finetuned_results['norm_success_rate'] - original_results['norm_success_rate'], 2)
    print(f"{'Norm Başarı (%)':<20} {original_results['norm_success_rate']:<15} {finetuned_results['norm_success_rate']:<15} {norm_success_improvement:+.2f}")

    # Genel Değerlendirme
    print("\n" + "="*80)
    print("🎯 GENEL DEĞERLENDİRME")
    print("="*80)
    
    if success_improvement > 0:
        print(f"✅ Fine-tuned model %{success_improvement} daha iyi performans gösteriyor!")
    elif success_improvement < 0:
        print(f"❌ Fine-tuned model %{abs(success_improvement)} daha kötü performans gösteriyor!")
    else:
        print("⚖️ Her iki model de aynı performansı gösteriyor!")

    # Sonuçları dosyaya kaydet
    save_comparison_results(args, original_results, finetuned_results)

def save_comparison_results(args, original_results, finetuned_results):
    """Karşılaştırma sonuçlarını dosyaya kaydet"""
    os.makedirs(args.output_dir, exist_ok=True)
    
    dset = args.dataset.replace('/', '_') + '_' + args.config + '_' + args.split
    comparison_file = os.path.join(args.output_dir, f"{dset}_comparison.txt")
    
    with open(comparison_file, 'w', encoding='utf-8') as f:
        f.write("WHISPER MODEL KARŞILAŞTIRMA SONUÇLARI\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Test Veri Setleri: TurkishVoiceDataset (10) + KhanAcademy (10)\n")
        f.write(f"Split: Modelin görmediği veriler (unseen data)\n")
        f.write(f"Test Veri Sayısı: {len(original_results['references'])}\n")
        f.write(f"Dil: {args.language}\n\n")
        
        f.write("ORIJINAL MODEL SONUÇLARI:\n")
        f.write(f"WER: {original_results['wer']}%\n")
        f.write(f"CER: {original_results['cer']}%\n")
        f.write(f"Başarı Oranı: {original_results['success_rate']}%\n\n")
        
        f.write("FINE-TUNED MODEL SONUÇLARI:\n")
        f.write(f"WER: {finetuned_results['wer']}%\n")
        f.write(f"CER: {finetuned_results['cer']}%\n")
        f.write(f"Başarı Oranı: {finetuned_results['success_rate']}%\n\n")
        
        wer_improvement = original_results['wer'] - finetuned_results['wer']
        success_improvement = finetuned_results['success_rate'] - original_results['success_rate']
        
        f.write("İYİLEŞME:\n")
        f.write(f"WER İyileşmesi: {wer_improvement:+.2f}%\n")
        f.write(f"Başarı Oranı İyileşmesi: {success_improvement:+.2f}%\n\n")
        
        f.write("DETAYLI SONUÇLAR:\n")
        f.write("-" * 50 + "\n")
        
        for i, (ref, orig_pred, ft_pred) in enumerate(zip(
            original_results['references'], 
            original_results['predictions'], 
            finetuned_results['predictions']
        )):
            f.write(f"\nÖRNEK {i+1}:\n")
            f.write(f"GERÇEK: {ref}\n")
            f.write(f"ORJİNAL: {orig_pred}\n")
            f.write(f"FINE-TUNED: {ft_pred}\n")
            f.write("-" * 30 + "\n")
    
    print(f"\n💾 Detaylı sonuçlar kaydedildi: {comparison_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # is_public_repo ve hf_model parametrelerini kaldırdık çünkü artık her iki modeli de test ediyoruz
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=False,
        default="./whisper-small-turkish",
        help="Folder with the pytorch_model.bin file",
    )
    parser.add_argument(
        "--temp_ckpt_folder",
        type=str,
        required=False,
        default="temp_dir",
        help="Path to create a temporary folder containing the model and related files needed for inference",
    )
    parser.add_argument(
        "--language",
        type=str,
        required=False,
        default="turkish",
        help="Language code for the transcription language, e.g. use 'turkish' for Turkish. This helps initialize the tokenizer.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        default="cubukcum/TurkishVoiceDataset",
        help="Dataset from huggingface to evaluate the model on. Example: cubukcum/TurkishVoiceDataset",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="default",
        help="Config of the dataset. Eg. 'default' for the Turkish split of Khan Academy",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=False,
        default="test",
        help="Split of the dataset for testing. Default: 'test' (unseen data). Use 'train' if no test split exists.",
    )
    parser.add_argument(
        "--device",
        type=int,
        required=False,
        default=0,
        help="The device to run the pipeline on. -1 for CPU, 0 for the first GPU (default) and so on.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=8,
        help="Number of samples to go through each streamed batch.",
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=False, 
        default="evaluation_results", 
        help="Output directory for the predictions and hypotheses generated.",
    )

    args = parser.parse_args()
    main(args)