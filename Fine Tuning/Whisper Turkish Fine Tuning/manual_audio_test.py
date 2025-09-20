"""
Whisper Model Evaluation Script
Evaluates fine-tuned Whisper models on Turkish datasets
"""

import os
import argparse
import evaluate
from tqdm import tqdm
from pathlib import Path
from transformers import pipeline, WhisperForConditionalGeneration
from datasets import load_dataset, Audio
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import warnings
warnings.filterwarnings("ignore")

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
    batch["norm_text"] = whisper_norm(get_text(batch))
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
    
    print(f"\n🧪 {model_name} testi başlıyor...")
    
    for item in tqdm(dataset, desc=f'{model_name} Progress'):
        result = model_pipeline(item["audio"], generate_kwargs=generate_kwargs)
        
        predictions.append(result["text"])
        references.append(get_text(item))
        norm_predictions.append(whisper_norm(result["text"]))
        norm_references.append(item["norm_text"])
    
    # Metrikleri hesapla
    wer = wer_metric.compute(references=references, predictions=predictions)
    wer = round(100 * wer, 2)
    cer = cer_metric.compute(references=references, predictions=predictions)
    cer = round(100 * cer, 2)
    norm_wer = wer_metric.compute(references=norm_references, predictions=norm_predictions)
    norm_wer = round(100 * norm_wer, 2)
    norm_cer = cer_metric.compute(references=norm_references, predictions=norm_predictions)
    norm_cer = round(100 * norm_cer, 2)
    
    # Başarı oranları (100 - hata oranı)
    success_rate = round(100 - wer, 2)
    norm_success_rate = round(100 - norm_wer, 2)
    
    return {
        'model_name': model_name,
        'wer': wer,
        'cer': cer,
        'norm_wer': norm_wer,
        'norm_cer': norm_cer,
        'success_rate': success_rate,
        'norm_success_rate': norm_success_rate,
        'predictions': predictions,
        'references': references
    }

def main(args):
    print("🚀 Whisper Model Karşılaştırma Testi Başlıyor...")
    print("="*60)
    
    # Dataset'i yükle ve hazırla
    print("📁 Dataset yükleniyor...")
    
    # Fine-tuning'de kullanılmayan test setini yükle
    try:
        dataset = load_dataset(
            args.dataset,
            args.config,
            split="test",  # Test setini kullan
        )
        print(f"✅ Test seti yüklendi: {len(dataset)} veri")
    except:
        # Eğer test seti yoksa train setinden son 10 veriyi al
        print("⚠️ Test seti bulunamadı, train setinin sonundan veri alınıyor...")
        full_dataset = load_dataset(
            args.dataset,
            args.config,
            split=args.split,
        )
        # Train setinin son 100 verisini al (genellikle fine-tuning'de kullanılmaz)
        start_idx = max(0, len(full_dataset) - 100)
        dataset = full_dataset.select(range(start_idx, len(full_dataset)))
        print(f"📊 Train setinin son {len(dataset)} verisi test için kullanılıyor")
    
    # Eğer test seti çok büyükse ilk 100 veriyi al
    if len(dataset) > 100:
        dataset = dataset.select(range(100))
        print(f"📊 Test için {len(dataset)} veri kullanılıyor")

    text_column_name = get_text_column_names(dataset.column_names)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset = dataset.map(normalise, num_proc=2)
    dataset = dataset.filter(is_target_text_in_range, input_columns=[text_column_name], num_proc=2)
    
    print(f"✅ Filtrelemeden sonra {len(dataset)} veri kaldı")

    # Generate kwargs'ı ayarla
    generate_kwargs = {
        "task": "transcribe",
        "language": args.language,
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
    
    # Fine-tuned weights'i yükle
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
        
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Split: test (fine-tuning'de kullanılmayan veriler)\n")
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
        default="ysdede/khanacademy-turkish",
        help="Dataset from huggingface to evaluate the model on. Example: ysdede/khanacademy-turkish",
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