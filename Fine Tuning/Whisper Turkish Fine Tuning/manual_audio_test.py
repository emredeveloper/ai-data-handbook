"""
Manual Audio Test - Test with Real Audio Data
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
    """Load REAL and NOT USED IN TRAINING audio data from a Hugging Face dataset"""
    print("🎵 Loading real audio data from Hugging Face...")
    
    try:
        # Load Khan Academy Turkish dataset
        print("📥 Loading Khan Academy Turkish dataset...")
        dataset = load_dataset("ysdede/khanacademy-turkish")
        
        # Ensure split (with seed=42 to align with training script)
        if "train" not in dataset:
            dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
        
        train_dataset = dataset["train"]
        test_dataset = dataset["test"] if "test" in dataset else dataset["train"].train_test_split(test_size=0.2, seed=42)["test"]
        
        # Check column names (on test split)
        if "transcription" in test_dataset.column_names:
            text_column = "transcription"
        elif "sentence" in test_dataset.column_names:
            text_column = "sentence"
        else:
            text_column = "text"
        
        print(f"Dataset columns: {test_dataset.column_names}")
        print(f"Train size: {len(train_dataset)} | Test size: {len(test_dataset)}")
        
        # Set audio sampling rate (test split)
        print("📡 Setting audio sampling rate (test split)...")
        test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))
        
        # Exclude first 100 test samples used for eval during training; then take 15 samples
        desired = 15
        exclude_eval_count = 100
        start_idx = exclude_eval_count if len(test_dataset) > exclude_eval_count + desired else 0
        end_idx = min(start_idx + desired, len(test_dataset))
        
        test_samples = []
        for j, i in enumerate(range(start_idx, end_idx), start=1):
            try:
                print(f"📥 Loading sample {j} (test idx={i})...")
                sample = test_dataset[i]
                audio_data = sample["audio"]
                text_data = sample[text_column]
                
                print(f"Audio data type: {type(audio_data)}")
                print(f"Audio data keys: {audio_data.keys() if isinstance(audio_data, dict) else 'Not dict'}")
                
                test_samples.append({
                    "audio_array": audio_data["array"],
                    "sampling_rate": audio_data["sampling_rate"],
                    "text": text_data,
                    "index": i
                })
                print(f"✅ Sample {j}: '{text_data[:50]}...' ({len(audio_data['array'])} samples)")
                
            except Exception as e:
                print(f"❌ Sample {j} could not be processed: {e}")
                print(f"Error detail: {type(e).__name__}: {str(e)}")
                continue
        
        if test_samples:
            print(f"✅ Loaded {len(test_samples)} real audio samples (not used in training)!")
            return test_samples
        else:
            print("❌ No samples could be processed!")
            raise Exception("No samples processed")
        
    except Exception as e:
        print(f"❌ Khan Academy dataset error: {e}")
        print(f"Error type: {type(e).__name__}")
        print("🔄 Trying Common Voice Turkish dataset...")
        
        try:
            # Fallback: Common Voice Turkish
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
                    print(f"✅ CV Sample {i+1}: '{text_data[:50]}...' ({len(audio_data['array'])} samples)")
                except Exception as e:
                    print(f"❌ CV Sample {i+1} could not be processed: {e}")
                    continue
            
            if test_samples:
                return test_samples
            else:
                raise Exception("Common Voice samples failed")
            
        except Exception as e2:
            print(f"❌ Common Voice dataset error: {e2}")
            print("⚠️ Switching to synthetic audio...")
            return create_realistic_test_audio()

def create_realistic_test_audio():
    """Create more realistic test audio - speech-like"""
    print("🎵 Creating realistic test audio...")
    
    sample_rate = 16000
    
    # Realistic frequency ranges for Turkish speech
    test_samples = []
    
    # Sample 1: Short sentence
    duration1 = 3.0
    t1 = np.linspace(0, duration1, int(sample_rate * duration1))
    
    # Within human voice frequency range (85-255 Hz fundamental)
    fundamental = 150  # Hz
    audio1 = np.zeros_like(t1)
    
    # Add harmonics (speech-like)
    for harmonic in range(1, 6):
        freq = fundamental * harmonic
        amplitude = 0.3 / harmonic  # each harmonic weaker
        audio1 += amplitude * np.sin(2 * np.pi * freq * t1)
    
    # Add modulation (speech-like)
    modulation = 0.1 * np.sin(2 * np.pi * 5 * t1)  # 5 Hz modulation
    audio1 *= (1 + modulation)
    
    # Add envelope (soft attack/release)
    envelope1 = np.ones_like(t1)
    fade_samples = int(0.1 * sample_rate)  # 0.1 second fade
    envelope1[:fade_samples] = np.linspace(0, 1, fade_samples)
    envelope1[-fade_samples:] = np.linspace(1, 0, fade_samples)
    audio1 *= envelope1
    
    # Add noise
    audio1 += 0.02 * np.random.normal(0, 1, len(t1))
    
    test_samples.append({
        "audio_array": audio1.astype(np.float32),
        "sampling_rate": sample_rate,
        "text": "Hello, I am a test audio file speaking Turkish.",
        "index": 0
    })
    
    # Sample 2: Different pitch
    duration2 = 2.5
    t2 = np.linspace(0, duration2, int(sample_rate * duration2))
    
    fundamental2 = 200  # Hz (higher pitch)
    audio2 = np.zeros_like(t2)
    
    for harmonic in range(1, 5):
        freq = fundamental2 * harmonic
        amplitude = 0.25 / harmonic
        audio2 += amplitude * np.sin(2 * np.pi * freq * t2)
    
    # Different modulation
    modulation2 = 0.15 * np.sin(2 * np.pi * 3 * t2)
    audio2 *= (1 + modulation2)
    
    # Envelope
    envelope2 = np.ones_like(t2)
    fade_samples2 = int(0.1 * sample_rate)
    envelope2[:fade_samples2] = np.linspace(0, 1, fade_samples2)
    envelope2[-fade_samples2:] = np.linspace(1, 0, fade_samples2)
    audio2 *= envelope2
    
    # Noise
    audio2 += 0.015 * np.random.normal(0, 1, len(t2))
    
    test_samples.append({
        "audio_array": audio2.astype(np.float32),
        "sampling_rate": sample_rate,
        "text": "This is the second Turkish test sentence, spoken with a different pitch.",
        "index": 1
    })
    
    # Sample 3: Longer sentence
    duration3 = 4.0
    t3 = np.linspace(0, duration3, int(sample_rate * duration3))
    
    fundamental3 = 120  # Hz (lower pitch)
    audio3 = np.zeros_like(t3)
    
    for harmonic in range(1, 7):
        freq = fundamental3 * harmonic
        amplitude = 0.35 / harmonic
        # Add frequency variation (prosody)
        freq_variation = freq * (1 + 0.05 * np.sin(2 * np.pi * 0.5 * t3))
        audio3 += amplitude * np.sin(2 * np.pi * freq_variation * t3)
    
    # More complex modulation
    modulation3 = 0.2 * np.sin(2 * np.pi * 4 * t3) * np.exp(-t3/2)
    audio3 *= (1 + modulation3)
    
    # Envelope
    envelope3 = np.ones_like(t3)
    fade_samples3 = int(0.15 * sample_rate)
    envelope3[:fade_samples3] = np.linspace(0, 1, fade_samples3)
    envelope3[-fade_samples3:] = np.linspace(1, 0, fade_samples3)
    audio3 *= envelope3
    
    # Noise
    audio3 += 0.01 * np.random.normal(0, 1, len(t3))
    
    test_samples.append({
        "audio_array": audio3.astype(np.float32),
        "sampling_rate": sample_rate,
        "text": "The Whisper model transcribes Turkish speech very successfully.",
        "index": 2
    })
    
    print(f"✅ {len(test_samples)} realistic audio samples created")
    return test_samples

def create_fallback_audio():
    """Fallback: Create simple audio data"""
    print("🎵 Fallback: Creating simple test audio...")
    
    sample_rate = 16000
    duration = 2.0
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Simpler and cleaner audio
    test_samples = [
        {
            "audio_array": (
                0.3 * np.sin(2 * np.pi * 440 * t) * np.exp(-t/2) +
                0.05 * np.random.normal(0, 1, len(t))
            ).astype(np.float32),
            "sampling_rate": sample_rate,
            "text": "Test audio one.",
            "index": 0
        },
        {
            "audio_array": (
                0.3 * np.sin(2 * np.pi * 523 * t) * np.exp(-t/2) +
                0.05 * np.random.normal(0, 1, len(t))
            ).astype(np.float32),
            "sampling_rate": sample_rate,
            "text": "Test audio two.",
            "index": 1
        }
    ]
    
    print(f"✅ {len(test_samples)} simple audio samples created")
    return test_samples

def test_single_audio(audio_data, expected_text, sample_index):
    """Test a single audio sample"""
    print(f"\n🎯 Testing Sample {sample_index + 1}:")
    print(f"📝 Expected text: '{expected_text[:100]}...'")
    print("-" * 50)
    
    audio_array = audio_data["audio_array"]
    sampling_rate = audio_data["sampling_rate"]
    
    results = {}
    
    # Helper: text normalization
    def normalize_text(text: str) -> str:
        if not text:
            return ""
        text = text.lower()
        # Do not simplify Turkish characters; only apply Unicode NFKC normalization
        text = unicodedata.normalize("NFKC", text)
        # Simplify excessive spaces and punctuation
        text = re.sub(r"[\s]+", " ", text).strip()
        return text

    # Metrics
    wer_metric = evaluate.load("wer")  # backup
    cer_metric = evaluate.load("cer")  # backup
    # sacreBLEU/chrF provide more stable results
    # BLEU: 0-100, chrF: 0-100

    # jiwer normalization chain (simplified and TR-friendly)
    jiwer_transform = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemovePunctuation()
    ])

    # Original model
    print("🔵 Testing with Original Whisper Small...")
    try:
        original_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        original_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
        
        # Process audio
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
        print(f"🔵 Original result: '{original_text}'")
        
    except Exception as e:
        print(f"❌ Original model error: {e}")
        results["original"] = ""
    
    # Fine-tuned model
    print("🟢 Testing with Fine-tuned Whisper Small...")
    try:
        # Use processor from original model
        finetuned_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        
        # Load fine-tuned model - find latest checkpoint
        import os
        import glob
        
        checkpoint_dirs = glob.glob("./whisper-small-turkish/checkpoint-*")
        if checkpoint_dirs:
            # Take the highest-numbered checkpoint
            latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split('-')[-1]))
            print(f"📁 Using latest checkpoint: {latest_checkpoint}")
            finetuned_model = WhisperForConditionalGeneration.from_pretrained(latest_checkpoint)
        else:
            print("❌ No checkpoints found!")
            raise FileNotFoundError("No checkpoint found")
        
        # Process audio
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
        print(f"🟢 Fine-tuned result: '{finetuned_text}'")
        
    except Exception as e:
        print(f"❌ Fine-tuned model error: {e}")
        results["finetuned"] = ""
    
    # Text normalization and metrics
    ref_norm = normalize_text(expected_text)
    orig_norm = normalize_text(results.get("original", ""))
    fine_norm = normalize_text(results.get("finetuned", ""))

    # Robust metrics: jiwer + sacrebleu + custom CER
    def cer_custom(ref: str, hyp: str) -> float:
        # Levenshtein distance (character-based)
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

    # jiwer WER (0-1), custom CER (0-1)
    # Due to API differences across jiwer versions, call directly with pre-normalized texts
    wer_o = jiwer.wer(ref_norm, orig_norm) if ref_norm else 1.0
    wer_f = jiwer.wer(ref_norm, fine_norm) if ref_norm else 1.0
    cer_o = cer_custom(ref_norm, orig_norm) if ref_norm else 1.0
    cer_f = cer_custom(ref_norm, fine_norm) if ref_norm else 1.0

    # sacreBLEU and chrF (0-100)
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
    """Compute similarity between two texts (simple)"""
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
    # Rich console and log capture
    console = Console(record=True)
    original_print = builtins.print
    def rich_print(*args, **kwargs):
        console.print(*args, **kwargs)
    builtins.print = rich_print

    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(logs_dir, f"manual_audio_test_{ts}.txt")

    console.print(Panel.fit("🎯 Manual Audio Test - With Real Audio Data", style="bold cyan"))
    print("=" * 60)
    
    # Load real audio data
    test_samples = load_real_audio_from_dataset()
    
    if not test_samples:
        print("❌ Test data could not be loaded!")
        return
    
    print(f"\n📊 Testing with {len(test_samples)} samples...")
    
    all_results = []
    
    # Test each sample
    for i, sample in enumerate(test_samples):
        results, expected_text = test_single_audio(sample, sample["text"], i)
        all_results.append({
            "index": i,
            "expected": expected_text,
            "original": results.get("original", ""),
            "finetuned": results.get("finetuned", ""),
            "audio_length": len(sample["audio_array"])
        })
    
    # Evaluate overall results
    print("\n" + "=" * 60)
    print("📊 OVERALL RESULTS:")
    print("=" * 60)
    
    original_similarities = []
    finetuned_similarities = []
    wer_o_list, wer_f_list = [], []
    cer_o_list, cer_f_list = [], []
    bleu_o_list, bleu_f_list = [], []
    chrf_o_list, chrf_f_list = [], []
    rtf_o_list, rtf_f_list = [], []
    
    for i, result in enumerate(all_results):
        print(f"\n🎯 Sample {i + 1}:")
        print(f"📝 Expected: '{result['expected'][:80]}...'")
        print(f"🔵 Original: '{result['original'][:80]}...'")
        print(f"🟢 Fine-tuned: '{result['finetuned'][:80]}...'")
        
        # Compute similarity
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
        
        print(f"📈 Original similarity: {orig_sim:.2%}")
        print(f"📈 Fine-tuned similarity: {fine_sim:.2%}")
        print(f"   WER(O/F): {m.get('wer_original', 1.0):.2%} / {m.get('wer_finetuned', 1.0):.2%} | CER(O/F): {m.get('cer_original', 1.0):.2%} / {m.get('cer_finetuned', 1.0):.2%}")
        print(f"   BLEU(O/F): {m.get('bleu_original', 0.0):.3f} / {m.get('bleu_finetuned', 0.0):.3f} | chrF(O/F): {m.get('chrf_original', 0.0):.2f} / {m.get('chrf_finetuned', 0.0):.2f}")
        print(f"   RTF(O/F): {result.get('original_rtf', float('nan')):.3f} / {result.get('finetuned_rtf', float('nan')):.3f}")
        
        if fine_sim > orig_sim:
            print("✅ Fine-tuned model performs better on this sample!")
        elif orig_sim > fine_sim:
            print("🔵 Original model performs better on this sample!")
        else:
            print("🤔 Both models perform equally well!")
    
    # Summary table
    table = Table(title="Summary Table", show_lines=False)
    table.add_column("Sample", style="bold")
    table.add_column("Original %", justify="right")
    table.add_column("Fine-tuned %", justify="right")
    table.add_column("WER O/F", justify="right")
    table.add_column("CER O/F", justify="right")
    table.add_column("BLEU O/F", justify="right")
    table.add_column("chrF O/F", justify="right")
    table.add_column("RTF O/F", justify="right")
    table.add_column("Best", style="green")
    for idx, (orig_sim, fine_sim, wer_o, wer_f, cer_o, cer_f, bleu_o, bleu_f, chrf_o, chrf_f, rtf_o, rtf_f) in enumerate(
        zip(original_similarities, finetuned_similarities, wer_o_list, wer_f_list, cer_o_list, cer_f_list, bleu_o_list, bleu_f_list, chrf_o_list, chrf_f_list, rtf_o_list, rtf_f_list), start=1):
        winner = "Fine-tuned" if fine_sim > orig_sim else ("Original" if orig_sim > fine_sim else "Equal")
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

    # Average performance
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
    
    print(f"\n🏆 AVERAGE PERFORMANCE:")
    print(f"🔵 Original model average similarity: {avg_original:.2%}")
    print(f"🟢 Fine-tuned model average similarity: {avg_finetuned:.2%}")
    print(f"   WER(O/F): {avg_wer_o:.2%} / {avg_wer_f:.2%} | CER(O/F): {avg_cer_o:.2%} / {avg_cer_f:.2%}")
    print(f"   BLEU(O/F): {avg_bleu_o:.3f} / {avg_bleu_f:.3f} | chrF(O/F): {avg_chrf_o:.2f} / {avg_chrf_f:.2f}")
    print(f"   RTF(O/F): {avg_rtf_o:.3f} / {avg_rtf_f:.3f}")
    
    if avg_finetuned > avg_original:
        improvement = ((avg_finetuned - avg_original) / avg_original * 100) if avg_original > 0 else 0
        print(f"🎉 Fine-tuned model performs {improvement:.1f}% better!")
        print("✅ Fine-tuning successful!")
    elif avg_original > avg_finetuned:
        decline = ((avg_original - avg_finetuned) / avg_original * 100) if avg_original > 0 else 0
        print(f"⚠️ Fine-tuned model performs {decline:.1f}% worse!")
        print("🔄 You may need to review the fine-tuning parameters.")
    else:
        print("🤔 Both models show similar performance.")
    
    print("\n✅ Test completed!")

    # Save log and restore print
    try:
        console.print(f"📄 Log saved: {log_path}")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(console.export_text())
    finally:
        builtins.print = original_print

if __name__ == "__main__":
    test_models()
