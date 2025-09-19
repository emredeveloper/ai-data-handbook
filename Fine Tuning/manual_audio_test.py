"""
Manuel Audio Test - Gerçek Ses Verisi ile Test
"""

import torch
import numpy as np
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, Audio
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import os
import time
import builtins
import evaluate
import sacrebleu
import jiwer
import unicodedata
import re
import warnings
warnings.filterwarnings("ignore")

def load_real_audio_from_dataset():
    """Hugging Face verisetinden GERÇEK ve EĞİTİMDE KULLANILMAMIŞ ses verisi yükle"""
    print("🎵 Hugging Face'den gerçek ses verisi yükleniyor...")
    
    try:
        # Khan Academy Türkçe dataset'ini yükle
        print("📥 Khan Academy Türkçe dataset'i yükleniyor...")
        dataset = load_dataset("ysdede/khanacademy-turkish")
        
        # Split'i garanti altına al (seed=42 ile, eğitim scriptiyle uyumlu)
        if "train" not in dataset:
            dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
        
        train_dataset = dataset["train"]
        test_dataset = dataset["test"] if "test" in dataset else dataset["train"].train_test_split(test_size=0.2, seed=42)["test"]
        
        # Sütun isimlerini kontrol et (test split üzerinden)
        if "transcription" in test_dataset.column_names:
            text_column = "transcription"
        elif "sentence" in test_dataset.column_names:
            text_column = "sentence"
        else:
            text_column = "text"
        
        print(f"Dataset sütunları: {test_dataset.column_names}")
        print(f"Train boyutu: {len(train_dataset)} | Test boyutu: {len(test_dataset)}")
        
        # Audio sampling rate'ini ayarla (test split)
        print("📡 Audio sampling rate ayarlanıyor (test split)...")
        test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))
        
        # Eğitim sırasında eval için kullanılan ilk 100 test örneğini dışla; sonrasından 15 örnek al
        desired = 15
        exclude_eval_count = 100
        start_idx = exclude_eval_count if len(test_dataset) > exclude_eval_count + desired else 0
        end_idx = min(start_idx + desired, len(test_dataset))
        
        test_samples = []
        for j, i in enumerate(range(start_idx, end_idx), start=1):
            try:
                print(f"📥 Örnek {j} yükleniyor (test idx={i})...")
                sample = test_dataset[i]
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
                print(f"✅ Örnek {j}: '{text_data[:50]}...' ({len(audio_data['array'])} sample)")
                
            except Exception as e:
                print(f"❌ Örnek {j} işlenemedi: {e}")
                print(f"Hata detayı: {type(e).__name__}: {str(e)}")
                continue
        
        if test_samples:
            print(f"✅ {len(test_samples)} gerçek audio örneği yüklendi (eğitimde kullanılmayan)!")
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
            for i in range(min(15, len(dataset))):
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
    
    # Yardımcı: metin normalizasyonu
    def normalize_text(text: str) -> str:
        if not text:
            return ""
        text = text.lower()
        # Türkçe karakterleri sadeleştirme değil; sadece Unicode NFKC normalize
        text = unicodedata.normalize("NFKC", text)
        # Fazla boşluk, noktalama sadeleştirme
        text = re.sub(r"[\s]+", " ", text).strip()
        return text

    # Metrikler
    wer_metric = evaluate.load("wer")  # yedek
    cer_metric = evaluate.load("cer")  # yedek
    # sacreBLEU/chrF daha istikrarlı sonuç verir
    # BLEU: 0-100, chrF: 0-100 döner

    # jiwer normalizasyon zinciri (TR dostu basitleştirilmiş)
    jiwer_transform = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemovePunctuation()
    ])

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
        t0 = time.time()
        with torch.no_grad():
            gen_out = original_model.generate(
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
                forced_decoder_ids=original_processor.get_decoder_prompt_ids(language="turkish", task="transcribe"),
                return_dict_in_generate=True,
                output_scores=True
            )
        t1 = time.time()
        generated_ids = gen_out.sequences
        original_text = original_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        original_seq_score = float(gen_out.sequences_scores[0].cpu().item()) if hasattr(gen_out, "sequences_scores") and gen_out.sequences_scores is not None else None
        orig_latency = t1 - t0
        audio_duration = len(audio_array) / float(sampling_rate) if sampling_rate else 0.0
        audio_duration = audio_duration if audio_duration > 0 else 1e-6
        orig_rtf = orig_latency / audio_duration
        results["original"] = original_text
        results["original_seq_score"] = original_seq_score
        results["original_latency_s"] = orig_latency
        results["original_rtf"] = orig_rtf
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
        t0 = time.time()
        with torch.no_grad():
            gen_out = finetuned_model.generate(
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
                forced_decoder_ids=finetuned_processor.get_decoder_prompt_ids(language="turkish", task="transcribe"),
                return_dict_in_generate=True,
                output_scores=True
            )
        t1 = time.time()
        generated_ids = gen_out.sequences
        finetuned_text = finetuned_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        finetuned_seq_score = float(gen_out.sequences_scores[0].cpu().item()) if hasattr(gen_out, "sequences_scores") and gen_out.sequences_scores is not None else None
        fine_latency = t1 - t0
        audio_duration = len(audio_array) / float(sampling_rate) if sampling_rate else 0.0
        audio_duration = audio_duration if audio_duration > 0 else 1e-6
        fine_rtf = fine_latency / audio_duration
        results["finetuned"] = finetuned_text
        results["finetuned_seq_score"] = finetuned_seq_score
        results["finetuned_latency_s"] = fine_latency
        results["finetuned_rtf"] = fine_rtf
        print(f"🟢 Fine-tuned sonuç: '{finetuned_text}'")
        
    except Exception as e:
        print(f"❌ Fine-tuned model hatası: {e}")
        results["finetuned"] = ""
    
    # Metin normalizasyonu ve metrikler
    ref_norm = normalize_text(expected_text)
    orig_norm = normalize_text(results.get("original", ""))
    fine_norm = normalize_text(results.get("finetuned", ""))

    # Robust metrikler: jiwer + sacrebleu + custom CER
    def cer_custom(ref: str, hyp: str) -> float:
        # Levenshtein mesafesi (karakter bazlı)
        r, h = ref, hyp
        n, m = len(r), len(h)
        if n == 0:
            return 1.0 if m > 0 else 0.0
        dp = [[0]*(m+1) for _ in range(n+1)]
        for i in range(n+1):
            dp[i][0] = i
        for j in range(m+1):
            dp[0][j] = j
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = 0 if r[i-1] == h[j-1] else 1
                dp[i][j] = min(
                    dp[i-1][j] + 1,      # deletion
                    dp[i][j-1] + 1,      # insertion
                    dp[i-1][j-1] + cost  # substitution
                )
        return dp[n][m] / max(1, n)

    # jiwer WER (0-1), CER custom (0-1)
    # jiwer sürümleri arasında API farkları olduğundan, ön-normalize edilmiş metinlerle direkt çağırıyoruz
    wer_o = jiwer.wer(ref_norm, orig_norm) if ref_norm else 1.0
    wer_f = jiwer.wer(ref_norm, fine_norm) if ref_norm else 1.0
    cer_o = cer_custom(ref_norm, orig_norm) if ref_norm else 1.0
    cer_f = cer_custom(ref_norm, fine_norm) if ref_norm else 1.0

    # sacreBLEU ve chrF (0-100)
    bleu_o = sacrebleu.corpus_bleu([orig_norm], [[ref_norm]]).score if ref_norm else 0.0
    bleu_f = sacrebleu.corpus_bleu([fine_norm], [[ref_norm]]).score if ref_norm else 0.0
    chrf_o = sacrebleu.corpus_chrf([orig_norm], [[ref_norm]]).score if ref_norm else 0.0
    chrf_f = sacrebleu.corpus_chrf([fine_norm], [[ref_norm]]).score if ref_norm else 0.0

    metrics = {
        "wer_original": wer_o,
        "wer_finetuned": wer_f,
        "cer_original": cer_o,
        "cer_finetuned": cer_f,
        "bleu_original": bleu_o,
        "bleu_finetuned": bleu_f,
        "chrf_original": chrf_o,
        "chrf_finetuned": chrf_f,
        "ref_len": len(ref_norm.split()),
        "orig_len": len(orig_norm.split()),
        "fine_len": len(fine_norm.split()),
    }

    results["metrics"] = metrics
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
    # Rich console ve log yakalama
    console = Console(record=True)
    original_print = builtins.print
    def rich_print(*args, **kwargs):
        console.print(*args, **kwargs)
    builtins.print = rich_print

    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(logs_dir, f"manual_audio_test_{ts}.txt")

    console.print(Panel.fit("🎯 Manuel Audio Test - Gerçek Ses Verisi ile", style="bold cyan"))
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
    wer_o_list, wer_f_list = [], []
    cer_o_list, cer_f_list = [], []
    bleu_o_list, bleu_f_list = [], []
    chrf_o_list, chrf_f_list = [], []
    rtf_o_list, rtf_f_list = [], []
    
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
        m = result.get("metrics", {})
        wer_o_list.append(m.get("wer_original", 1.0))
        wer_f_list.append(m.get("wer_finetuned", 1.0))
        cer_o_list.append(m.get("cer_original", 1.0))
        cer_f_list.append(m.get("cer_finetuned", 1.0))
        bleu_o_list.append(m.get("bleu_original", 0.0))
        bleu_f_list.append(m.get("bleu_finetuned", 0.0))
        chrf_o_list.append(m.get("chrf_original", 0.0))
        chrf_f_list.append(m.get("chrf_finetuned", 0.0))
        rtf_o_list.append(result.get("original_rtf", float("nan")))
        rtf_f_list.append(result.get("finetuned_rtf", float("nan")))
        
        print(f"📈 Orijinal benzerlik: {orig_sim:.2%}")
        print(f"📈 Fine-tuned benzerlik: {fine_sim:.2%}")
        print(f"   WER(O/F): {m.get('wer_original', 1.0):.2%} / {m.get('wer_finetuned', 1.0):.2%} | CER(O/F): {m.get('cer_original', 1.0):.2%} / {m.get('cer_finetuned', 1.0):.2%}")
        print(f"   BLEU(O/F): {m.get('bleu_original', 0.0):.3f} / {m.get('bleu_finetuned', 0.0):.3f} | chrF(O/F): {m.get('chrf_original', 0.0):.2f} / {m.get('chrf_finetuned', 0.0):.2f}")
        print(f"   RTF(O/F): {result.get('original_rtf', float('nan')):.3f} / {result.get('finetuned_rtf', float('nan')):.3f}")
        
        if fine_sim > orig_sim:
            print("✅ Fine-tuned model bu örnekte daha başarılı!")
        elif orig_sim > fine_sim:
            print("🔵 Orijinal model bu örnekte daha başarılı!")
        else:
            print("🤔 İki model de eşit başarılı!")
    
    # Özet tablo
    table = Table(title="Özet Tablo", show_lines=False)
    table.add_column("Örnek", style="bold")
    table.add_column("Orijinal %", justify="right")
    table.add_column("Fine-tuned %", justify="right")
    table.add_column("WER O/F", justify="right")
    table.add_column("CER O/F", justify="right")
    table.add_column("BLEU O/F", justify="right")
    table.add_column("chrF O/F", justify="right")
    table.add_column("RTF O/F", justify="right")
    table.add_column("En iyi", style="green")
    for idx, (orig_sim, fine_sim, wer_o, wer_f, cer_o, cer_f, bleu_o, bleu_f, chrf_o, chrf_f, rtf_o, rtf_f) in enumerate(
        zip(original_similarities, finetuned_similarities, wer_o_list, wer_f_list, cer_o_list, cer_f_list, bleu_o_list, bleu_f_list, chrf_o_list, chrf_f_list, rtf_o_list, rtf_f_list), start=1):
        winner = "Fine-tuned" if fine_sim > orig_sim else ("Orijinal" if orig_sim > fine_sim else "Eşit")
        table.add_row(
            str(idx),
            f"{orig_sim*100:.2f}%",
            f"{fine_sim*100:.2f}%",
            f"{wer_o:.2%}/{wer_f:.2%}",
            f"{cer_o:.2%}/{cer_f:.2%}",
            f"{bleu_o:.3f}/{bleu_f:.3f}",
            f"{chrf_o:.2f}/{chrf_f:.2f}",
            f"{rtf_o:.2f}/{rtf_f:.2f}",
            winner
        )
    console.print(table)

    # Ortalama performans
    avg_original = sum(original_similarities) / len(original_similarities) if original_similarities else 0
    avg_finetuned = sum(finetuned_similarities) / len(finetuned_similarities) if finetuned_similarities else 0
    avg_wer_o = sum(wer_o_list) / len(wer_o_list) if wer_o_list else 1.0
    avg_wer_f = sum(wer_f_list) / len(wer_f_list) if wer_f_list else 1.0
    avg_cer_o = sum(cer_o_list) / len(cer_o_list) if cer_o_list else 1.0
    avg_cer_f = sum(cer_f_list) / len(cer_f_list) if cer_f_list else 1.0
    avg_bleu_o = sum(bleu_o_list) / len(bleu_o_list) if bleu_o_list else 0.0
    avg_bleu_f = sum(bleu_f_list) / len(bleu_f_list) if bleu_f_list else 0.0
    avg_chrf_o = sum(chrf_o_list) / len(chrf_o_list) if chrf_o_list else 0.0
    avg_chrf_f = sum(chrf_f_list) / len(chrf_f_list) if chrf_f_list else 0.0
    avg_rtf_o = sum(x for x in rtf_o_list if not np.isnan(x)) / max(1, sum(0 if np.isnan(x) else 1 for x in rtf_o_list))
    avg_rtf_f = sum(x for x in rtf_f_list if not np.isnan(x)) / max(1, sum(0 if np.isnan(x) else 1 for x in rtf_f_list))
    
    print(f"\n🏆 ORTALAMA PERFORMANS:")
    print(f"🔵 Orijinal model ortalama benzerlik: {avg_original:.2%}")
    print(f"🟢 Fine-tuned model ortalama benzerlik: {avg_finetuned:.2%}")
    print(f"   WER(O/F): {avg_wer_o:.2%} / {avg_wer_f:.2%} | CER(O/F): {avg_cer_o:.2%} / {avg_cer_f:.2%}")
    print(f"   BLEU(O/F): {avg_bleu_o:.3f} / {avg_bleu_f:.3f} | chrF(O/F): {avg_chrf_o:.2f} / {avg_chrf_f:.2f}")
    print(f"   RTF(O/F): {avg_rtf_o:.3f} / {avg_rtf_f:.3f}")
    
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

    # Log kaydet ve print'i geri yükle
    try:
        console.print(f"📄 Log kaydedildi: {log_path}")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(console.export_text())
    finally:
        builtins.print = original_print

if __name__ == "__main__":
    test_models()
