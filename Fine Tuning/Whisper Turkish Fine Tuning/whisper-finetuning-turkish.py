"""
Whisper Fine-tuning Script for Turkish
Fine-tunes the Whisper model using various Turkish datasets with flexible configuration
"""

import os
import torch
import argparse
import csv
import evaluate
import numpy as np
import librosa
import unicodedata
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import DatasetDict, Audio, load_dataset, concatenate_datasets
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers import (
    WhisperFeatureExtractor, 
    WhisperTokenizer, 
    WhisperProcessor, 
    WhisperForConditionalGeneration, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    TrainerCallback
)
import warnings
warnings.filterwarnings("ignore")

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

###############################     ADVANCED PREPROCESSING     ########################

def advanced_audio_preprocessing(audio_array, sampling_rate):
    """Gelişmiş audio preprocessing"""
    try:
        # Noise reduction
        audio_array = librosa.effects.preemphasis(audio_array)
        
        # Volume normalization
        audio_array = librosa.util.normalize(audio_array)
        
        # Silence trimming
        audio_array, _ = librosa.effects.trim(audio_array, top_db=20)
        
        # Dynamic range compression
        audio_array = np.clip(audio_array, -0.1, 0.1)
        
        return audio_array
    except Exception as e:
        print(f"Audio preprocessing error: {e}")
        return audio_array

def turkish_text_preprocessing(text):
    """Türkçe metin preprocessing - İyileştirilmiş versiyon"""
    if not text or len(text.strip()) < 2:
        return " "
    
    # Unicode normalization
    text = unicodedata.normalize('NFD', text)
    
    # Gereksiz boşlukları temizle
    text = " ".join(text.split())
    
    # Türkçe karakterleri koru
    text = text.strip()
    
    # Sayıları yazıya çevir (basit)
    text = convert_numbers_to_words(text)
    
    # Kısaltmaları genişlet
    text = expand_turkish_abbreviations(text)
    
    # Noktalama standardizasyonu
    text = standardize_turkish_punctuation(text)
    
    # Yeni: Anlamsız karakterleri temizle
    text = re.sub(r'[^\w\sçğıöşüÇĞIÖŞÜ.,!?;:\-]', '', text)
    
    # Yeni: Tekrar eden kelimeleri temizle
    words = text.split()
    cleaned_words = []
    prev_word = ""
    for word in words:
        if word != prev_word:  # Tekrar eden kelimeyi atla
            cleaned_words.append(word)
        prev_word = word
    text = " ".join(cleaned_words)
    
    # Yeni: Çok kısa kelimeleri temizle (1-2 harf)
    words = text.split()
    words = [word for word in words if len(word) > 2 or word in ['ve', 'da', 'de', 'ki', 'mi', 'mı', 'mu', 'mü']]
    text = " ".join(words)
    
    return text

def convert_numbers_to_words(text):
    """Sayıları yazıya çevir"""
    number_map = {
        '0': 'sıfır', '1': 'bir', '2': 'iki', '3': 'üç', '4': 'dört',
        '5': 'beş', '6': 'altı', '7': 'yedi', '8': 'sekiz', '9': 'dokuz'
    }
    
    for digit, word in number_map.items():
        text = text.replace(digit, word)
    
    return text

def expand_turkish_abbreviations(text):
    """Türkçe kısaltmaları genişlet"""
    abbreviations = {
        'vs.': 'versus',
        'vb.': 've benzeri',
        'vs': 'versus',
        'vb': 've benzeri',
        'dr.': 'doktor',
        'prof.': 'profesör',
        'mrb': 'merhaba',
        'slm': 'selam'
    }
    
    for abbr, full in abbreviations.items():
        text = text.replace(abbr, full)
    
    return text

def standardize_turkish_punctuation(text):
    """Türkçe noktalama standardizasyonu"""
    # Çoklu nokta temizleme
    text = re.sub(r'\.{2,}', '.', text)
    
    # Çoklu ünlem temizleme
    text = re.sub(r'!{2,}', '!', text)
    
    # Çoklu soru işareti temizleme
    text = re.sub(r'\?{2,}', '?', text)
    
    # Gereksiz boşlukları temizle
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def audio_augmentation(audio_array, sampling_rate, augmentation_prob=0.4):
    """Audio augmentation (genelleme için zenginleştirilmiş)"""
    if np.random.random() > augmentation_prob:
        return audio_array
    
    try:
        # Speed perturbation
        if np.random.random() < 0.5:
            speed_factor = np.random.uniform(0.9, 1.1)
            audio_array = librosa.effects.time_stretch(audio_array, rate=speed_factor)
        
        # Pitch shifting
        if np.random.random() < 0.3:
            pitch_shift = np.random.randint(-2, 3)
            audio_array = librosa.effects.pitch_shift(audio_array, sr=sampling_rate, n_steps=pitch_shift)
        
        # Background noise addition (SNR kontrollü)
        if np.random.random() < 0.5:
            target_snr_db = np.random.uniform(5, 25)  # 5-25 dB arası
            sig_power = np.mean(audio_array ** 2) + 1e-9
            snr_linear = 10 ** (target_snr_db / 10)
            noise_power = sig_power / snr_linear
            noise = np.random.normal(0, np.sqrt(noise_power), size=len(audio_array))
            audio_array = audio_array + noise

        # Random gain / soft clipping
        if np.random.random() < 0.4:
            gain = np.random.uniform(0.7, 1.3)
            audio_array = audio_array * gain
            # soft clipping
            audio_array = np.tanh(audio_array)

        # Simple reverb (exponential decay IR)
        if np.random.random() < 0.3:
            ir_len = int(0.03 * sampling_rate)  # ~30 ms
            decay = np.exp(-np.linspace(0, 6, ir_len))
            ir = decay * np.random.uniform(0.8, 1.2, size=ir_len)
            audio_array = np.convolve(audio_array, ir, mode='same')

        # Simple band-pass via FFT mask (approx)
        if np.random.random() < 0.3:
            low = np.random.uniform(100, 300)
            high = np.random.uniform(3000, 6000)
            n = len(audio_array)
            spec = np.fft.rfft(audio_array)
            freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate)
            mask = (freqs >= low) & (freqs <= high)
            spec = spec * mask
            audio_array = np.fft.irfft(spec, n=n)
        
        return audio_array
    except Exception as e:
        print(f"Audio augmentation error: {e}")
        return audio_array

###############################     REWARD SHAPING     ########################

class RewardShaping:
    """Reward shaping for better Turkish learning"""
    
    def __init__(self):
        self.weights = {
            'accuracy': 0.30,        # Azaltıldı
            'fluency': 0.20,         # Azaltıldı
            'turkish_quality': 0.35, # Artırıldı (en önemli)
            'length_consistency': 0.10, # Azaltıldı
            'audio_alignment': 0.0   # Etkisizleştirildi
        }
    
    def compute_comprehensive_reward(self, prediction, reference, audio_features=None):
        """Kapsamlı reward hesaplama"""
        rewards = {}
        
        # 1. Accuracy Reward (WER-based)
        rewards['accuracy'] = self.compute_accuracy_reward(prediction, reference)
        
        # 2. Fluency Reward
        rewards['fluency'] = self.compute_fluency_reward(prediction)
        
        # 3. Turkish Quality Reward
        rewards['turkish_quality'] = self.compute_turkish_quality_reward(prediction)
        
        # 4. Length Consistency Reward
        rewards['length_consistency'] = self.compute_length_reward(prediction, reference)
        
        # 5. Audio Alignment Reward
        if audio_features is not None:
            rewards['audio_alignment'] = self.compute_alignment_reward(prediction, audio_features)
        else:
            rewards['audio_alignment'] = 0.0
        
        # Weighted combination
        total_reward = sum(self.weights[key] * rewards[key] for key in rewards.keys())
        
        return total_reward, rewards
    
    def compute_accuracy_reward(self, prediction, reference):
        """Doğruluk reward'ı (WER tabanlı)"""
        try:
            # Basit edit distance hesaplama
            pred_words = prediction.lower().split()
            ref_words = reference.lower().split()
            
            if len(ref_words) == 0:
                return 0.0
            
            # Levenshtein distance
            distance = self.levenshtein_distance(pred_words, ref_words)
            wer = distance / len(ref_words)
            
            # WER'i reward'a çevir (0-1 arası)
            return max(0.0, 1.0 - wer)
        except:
            return 0.0
    
    def compute_fluency_reward(self, prediction):
        """Akıcılık reward'ı"""
        try:
            words = prediction.split()
            if len(words) < 2:
                return 0.0
            
            # Kelime uzunluk varyansı (çok kısa/uzun kelimeler cezalandırılır)
            word_lengths = [len(word) for word in words]
            length_variance = np.var(word_lengths)
            
            # Varyans ne kadar düşükse o kadar akıcı
            fluency_score = max(0.0, 1.0 - (length_variance / 10.0))
            
            return min(1.0, fluency_score)
        except:
            return 0.0
    
    def compute_turkish_quality_reward(self, prediction):
        """Türkçe kalite reward'ı - Geliştirilmiş versiyon"""
        try:
            score = 0.0
            prediction_lower = prediction.lower()
            words = prediction_lower.split()
            
            if len(words) == 0:
                return 0.0
            
            # 1. Türkçe karakter kullanımı (daha yüksek ağırlık)
            turkish_chars = sum(1 for c in prediction if c in 'çğıöşüÇĞIÖŞÜ')
            if len(prediction) > 0:
                turkish_char_ratio = turkish_chars / len(prediction)
                score += turkish_char_ratio * 0.6  # Artırıldı
            
            # 2. Türkçe kelime varlığı (genişletilmiş liste)
            turkish_words = [
                'bir', 'bu', 'şu', 'o', 'ben', 'sen', 'biz', 'siz', 'onlar',
                've', 'ile', 'için', 'olan', 'var', 'yok', 'çok', 'daha',
                'en', 'da', 'de', 'ki', 'mi', 'mı', 'mu', 'mü',
                'gibi', 'kadar', 'sonra', 'önce', 'şimdi', 'her', 'bütün',
                'tüm', 'hiç', 'bazı', 'birçok', 'çok', 'az', 'fazla'
            ]
            turkish_word_count = sum(1 for word in words if word in turkish_words)
            turkish_word_ratio = turkish_word_count / len(words)
            score += turkish_word_ratio * 0.4  # Artırıldı
            
            # 3. Türkçe gramer yapısı (basit kontroller)
            # -mek/-mak mastar eki
            infinitive_count = sum(1 for word in words if word.endswith(('mek', 'mak')))
            if len(words) > 0:
                infinitive_ratio = infinitive_count / len(words)
                score += infinitive_ratio * 0.2
            
            # -yor, -iyor, -uyor, -üyor şimdiki zaman
            present_tense_count = sum(1 for word in words if any(word.endswith(ending) for ending in ['yor', 'iyor', 'uyor', 'üyor']))
            if len(words) > 0:
                present_tense_ratio = present_tense_count / len(words)
                score += present_tense_ratio * 0.2
            
            # 4. Anlamsız kelime cezası (İngilizce kelimeler)
            english_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
            english_word_count = sum(1 for word in words if word in english_words)
            if len(words) > 0:
                english_word_ratio = english_word_count / len(words)
                score -= english_word_ratio * 0.5  # Cezalandır
            
            # 5. Tekrar eden kelime cezası
            if len(words) > 1:
                unique_words = len(set(words))
                repetition_penalty = 1.0 - (unique_words / len(words))
                score -= repetition_penalty * 0.3
            
            # 6. Çok kısa veya çok uzun cümle cezası
            if len(words) < 2:
                score -= 0.5  # Çok kısa cümle
            elif len(words) > 20:
                score -= 0.2  # Çok uzun cümle
            
            return max(0.0, min(1.0, score))
        except:
            return 0.0
    
    def compute_length_reward(self, prediction, reference):
        """Uzunluk tutarlılık reward'ı"""
        try:
            pred_len = len(prediction.split())
            ref_len = len(reference.split())
            
            if ref_len == 0:
                return 0.0
            
            length_ratio = pred_len / ref_len
            
            # Çok kısa veya çok uzun cezalandır
            if length_ratio < 0.3:
                return -0.5  # Çok kısa
            elif length_ratio > 2.0:
                return -0.5  # Çok uzun
            elif 0.7 <= length_ratio <= 1.3:
                return 1.0   # Mükemmel uzunluk
            else:
                return 0.5   # Kabul edilebilir uzunluk
        except:
            return 0.0
    
    def compute_alignment_reward(self, prediction, audio_features):
        """Audio-text alignment reward'ı"""
        try:
            # Basit alignment kontrolü
            pred_len = len(prediction.split())
            audio_duration = len(audio_features) / 16000  # 16kHz sampling rate
            
            # Saniyede kelime sayısı (normal: 2-4 kelime/saniye)
            words_per_second = pred_len / max(audio_duration, 0.1)
            
            if 1.5 <= words_per_second <= 4.0:
                return 1.0
            elif 1.0 <= words_per_second <= 5.0:
                return 0.5
            else:
                return 0.0
        except:
            return 0.0
    
    def levenshtein_distance(self, s1, s2):
        """Levenshtein distance hesaplama"""
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

###############################     CUSTOM CALLBACKS     ########################

class WERImprovementCallback(TrainerCallback):
    """WER iyileştirmesini takip eden callback"""
    
    def __init__(self):
        self.best_wer = float('inf')
        self.initial_wer = None
        self.initial_reward = None
        self.wer_history = []
        
    def on_evaluate(self, args, state, control, model, logs=None, **kwargs):
        if logs is not None and 'eval_wer' in logs:
            current_wer = logs['eval_wer']
            current_reward = logs.get('eval_avg_reward', 0.0)
            # Ek metrikler
            reward_accuracy = logs.get('eval_reward_accuracy', None)
            reward_fluency = logs.get('eval_reward_fluency', None)
            reward_turkish_quality = logs.get('eval_reward_turkish_quality', None)
            reward_length = logs.get('eval_reward_length_consistency', None)

            # Adım bazlı CSV log'u (ablation/etki izlemesi için)
            try:
                csv_path = os.path.join(args.output_dir, 'eval_metrics_log.csv')
                file_exists = os.path.exists(csv_path)
                with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(['global_step','eval_wer','eval_avg_reward','reward_accuracy','reward_fluency','reward_turkish_quality','reward_length'])
                    writer.writerow([
                        int(state.global_step),
                        float(current_wer),
                        float(current_reward),
                        float(reward_accuracy) if reward_accuracy is not None else '',
                        float(reward_fluency) if reward_fluency is not None else '',
                        float(reward_turkish_quality) if reward_turkish_quality is not None else '',
                        float(reward_length) if reward_length is not None else '',
                    ])
            except Exception as e:
                print(f"CSV log yazımı hatası: {e}")
            
            # İlk evaluation'da initial değerleri kaydet
            if self.initial_wer is None:
                self.initial_wer = current_wer
                self.initial_reward = current_reward
                print(f"\n🎯 Initial WER: {current_wer:.2f}% | Initial Reward: {current_reward:.3f}")
            
            # WER geçmişini güncelle
            self.wer_history.append(current_wer)
            
            # En iyi WER'i güncelle
            if current_wer < self.best_wer:
                self.best_wer = current_wer
                improvement = self.initial_wer - current_wer
                reward_improvement = current_reward - self.initial_reward
                print(f"🎉 New best WER: {current_wer:.2f}% (WER Improvement: +{improvement:.2f}% | Reward: {current_reward:.3f} | Reward Improvement: +{reward_improvement:.3f})")
            else:
                # Son 3 evaluation'daki ortalama WER
                if len(self.wer_history) >= 3:
                    recent_avg = sum(self.wer_history[-3:]) / 3
                    improvement = self.initial_wer - recent_avg
                    reward_improvement = current_reward - self.initial_reward
                    print(f"📊 Current WER: {current_wer:.2f}% (Recent avg: {recent_avg:.2f}% | WER Improvement: +{improvement:.2f}% | Reward: {current_reward:.3f} | Reward Improvement: +{reward_improvement:.3f})")
                else:
                    improvement = self.initial_wer - current_wer
                    reward_improvement = current_reward - self.initial_reward
                    print(f"📊 Current WER: {current_wer:.2f}% (WER Improvement: +{improvement:.2f}% | Reward: {current_reward:.3f} | Reward Improvement: +{reward_improvement:.3f})")
            
            # Reward detaylarını göster
            if 'eval_reward_accuracy' in logs:
                print(f"   📈 Reward Details - Accuracy: {logs.get('eval_reward_accuracy', 0):.3f} | "
                      f"Fluency: {logs.get('eval_reward_fluency', 0):.3f} | "
                      f"Turkish: {logs.get('eval_reward_turkish_quality', 0):.3f} | "
                      f"Length: {logs.get('eval_reward_length_consistency', 0):.3f}")
    
    def on_train_end(self, args, state, control, model, logs=None, **kwargs):
        if self.initial_wer is not None:
            final_wer_improvement = self.initial_wer - self.best_wer
            final_reward_improvement = logs.get('eval_avg_reward', 0.0) - self.initial_reward if self.initial_reward else 0.0
            
            print(f"\n🏆 TRAINING COMPLETED!")
            print(f"🎯 Initial WER: {self.initial_wer:.2f}% | Initial Reward: {self.initial_reward:.3f}")
            print(f"🏅 Best WER: {self.best_wer:.2f}% | Final Reward: {logs.get('eval_avg_reward', 0.0):.3f}")
            print(f"📈 WER Improvement: +{final_wer_improvement:.2f}% | Reward Improvement: +{final_reward_improvement:.3f}")
            
            # Son reward detayları
            if logs:
                print(f"📊 Final Reward Details:")
                print(f"   Accuracy: {logs.get('eval_reward_accuracy', 0):.3f}")
                print(f"   Fluency: {logs.get('eval_reward_fluency', 0):.3f}")
                print(f"   Turkish Quality: {logs.get('eval_reward_turkish_quality', 0):.3f}")
                print(f"   Length Consistency: {logs.get('eval_reward_length_consistency', 0):.3f}")
                print(f"   Audio Alignment: {logs.get('eval_reward_audio_alignment', 0):.3f}")
                # Son toplu CSV özeti
                try:
                    summary_csv = os.path.join(args.output_dir, 'eval_summary.csv')
                    with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(['initial_wer','best_wer','final_wer_improvement','final_avg_reward'])
                        writer.writerow([
                            float(self.initial_wer),
                            float(self.best_wer),
                            float(final_wer_improvement),
                            float(logs.get('eval_avg_reward', 0.0)),
                        ])
                except Exception as e:
                    print(f"CSV özet yazımı hatası: {e}")

###############################     DATA COLLATOR DEFINITION     ########################

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        # attention mask üret (WhisperFeatureExtractor bunu destekler ve generate sırasında uyarıları azaltır)
        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt",
            return_attention_mask=True,
        )

        # get the tokenized label sequences (tek çağrıda encode + pad)
        labels_text = [f.get("labels_text", " ") for f in features]
        labels_batch = self.processor.tokenizer(
            labels_text,
            padding=True,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

#######################     ARGUMENT PARSING        #########################

def parse_arguments():
    parser = argparse.ArgumentParser(description='Fine-tuning script for Whisper Models of various sizes.')
    parser.add_argument(
        '--model_name', 
        type=str, 
        required=False, 
        default='openai/whisper-small', 
        help='Huggingface model name to fine-tune. Eg: openai/whisper-small'
    )
    parser.add_argument(
        '--language', 
        type=str, 
        required=False, 
        default='turkish', 
        help='Language the model is being adapted to in lowercase.'
    )
    parser.add_argument(
        '--sampling_rate', 
        type=int, 
        required=False, 
        default=16000, 
        help='Sampling rate of audios.'
    )
    parser.add_argument(
        '--num_proc', 
        type=int, 
        required=False, 
        default=0, 
        help='Number of parallel jobs to run. Set to 0 for Windows compatibility (no multiprocessing).'
    )
    parser.add_argument(
        '--train_strategy', 
        type=str, 
        required=False, 
        default='steps', 
        help='Training strategy. Choose between steps and epoch.'
    )
    parser.add_argument(
        '--learning_rate', 
        type=float, 
        required=False, 
        default=5e-6, 
        help='Learning rate for the fine-tuning process. Lower for Turkish optimization.'
    )
    parser.add_argument(
        '--warmup', 
        type=int, 
        required=False, 
        default=500, 
        help='Number of warmup steps. Increased for gradual learning rate increase.'
    )
    parser.add_argument(
        '--train_batchsize', 
        type=int, 
        required=False, 
        default=8, 
        help='Batch size during the training phase. Increased for better performance.'
    )
    parser.add_argument(
        '--eval_batchsize', 
        type=int, 
        required=False, 
        default=16, 
        help='Batch size during the evaluation phase. Increased for faster evaluation.'
    )
    parser.add_argument(
        '--num_epochs', 
        type=int, 
        required=False, 
        default=10, 
        help='Number of epochs to train for.'
    )
    parser.add_argument(
        '--num_steps', 
        type=int, 
        required=False, 
        default=1000, 
        help='Number of steps to train for. Optimized for 1k steps.'
    )
    parser.add_argument(
        '--resume_from_ckpt', 
        type=str, 
        required=False, 
        default=None, 
        help='Path to a trained checkpoint to resume training from.'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        required=False, 
        default='./whisper-small-turkish', 
        help='Output directory for the checkpoints generated.'
    )
    parser.add_argument(
        '--train_datasets', 
        type=str, 
        nargs='+', 
        required=False, 
        default=['cubukcum/TurkishVoiceDataset', 'ysdede/khanacademy-turkish'], 
        help='List of datasets to be used for training.'
    )
    parser.add_argument(
        '--train_dataset_configs', 
        type=str, 
        nargs='+', 
        required=False, 
        default=['default', 'default'], 
        help="List of training dataset configs. Eg. 'hi' for the Hindi part of Common Voice",
    )
    parser.add_argument(
        '--train_dataset_splits', 
        type=str, 
        nargs='+', 
        required=False, 
        default=['train', 'train'], 
        help="List of training dataset splits. Eg. 'train' for the train split of Common Voice",
    )
    parser.add_argument(
        '--train_dataset_text_columns', 
        type=str, 
        nargs='+', 
        required=False, 
        default=['transcription', 'transcription'], 
        help="Text column name of each training dataset. Eg. 'sentence' for Common Voice",
    )
    parser.add_argument(
        '--eval_datasets', 
        type=str, 
        nargs='+', 
        required=False, 
        default=['ysdede/khanacademy-turkish', 'cubukcum/TurkishVoiceDataset', 'ilkerkara/common_voice_13_0_tr_pseudo_labelled'], 
        help='List of datasets to be used for evaluation.'
    )
    parser.add_argument(
        '--eval_dataset_configs', 
        type=str, 
        nargs='+', 
        required=False, 
        default=['default', 'default', 'tr'], 
        help="List of evaluation dataset configs. Eg. 'hi_in' for the Hindi part of Google Fleurs",
    )
    parser.add_argument(
        '--eval_dataset_splits', 
        type=str, 
        nargs='+', 
        required=False, 
        default=['test', 'train', 'test'], 
        help="List of evaluation dataset splits. Using 'test' for unseen data evaluation.",
    )
    parser.add_argument(
        '--eval_dataset_text_columns', 
        type=str, 
        nargs='+', 
        required=False, 
        default=['transcription', 'transcription', 'sentence'], 
        help="Text column name of each evaluation dataset. Eg. 'transcription' for Google Fleurs",
    )
    parser.add_argument(
        '--max_train_samples', 
        type=int, 
        required=False, 
        default=5000, 
        help='Maximum number of training samples to use.'
    )
    # Yeni: Üç eğitim verisini oranlarla belirt
    parser.add_argument(
        '--train_mix_datasets', 
        type=str,
        nargs='+',
        required=False,
        default=['ysdede/khanacademy-turkish', 'cubukcum/TurkishVoiceDataset', 'mozilla-foundation/common_voice_13_0'],
        help='Training datasets list for mixing (3 datasets).'
    )
    parser.add_argument(
        '--train_mix_configs', 
        type=str,
        nargs='+',
        required=False,
        default=['default', 'default', 'tr'],
        help='Configs for training datasets list (aligned by index).'
    )
    parser.add_argument(
        '--train_mix_splits', 
        type=str,
        nargs='+',
        required=False,
        default=['train', 'train', 'train'],
        help='Splits for training datasets list.'
    )
    parser.add_argument(
        '--train_mix_text_columns', 
        type=str,
        nargs='+',
        required=False,
        default=['transcription', 'transcription', 'sentence'],
        help='Text columns for each training dataset before renaming to sentence.'
    )
    parser.add_argument(
        '--train_mix_ratios', 
        type=float,
        nargs='+',
        required=False,
        default=[0.6, 0.3, 0.1],
        help='Ratios for each training dataset (sum must be 1.0).'
    )
    parser.add_argument(
        '--max_eval_samples', 
        type=int, 
        required=False, 
        default=200, 
        help='Maximum number of evaluation samples to use.'
    )

    return parser.parse_args()

def validate_arguments(args):
    if args.train_strategy not in ['steps', 'epoch']:
        raise ValueError('The train strategy should be either steps and epoch.')

    if len(args.train_datasets) == 0:
        raise ValueError('No train dataset has been passed')
    if len(args.eval_datasets) == 0:
        raise ValueError('No evaluation dataset has been passed')

    # Validation for list arguments
    list_args = [
        ('train_datasets', 'train_dataset_configs'),
        ('train_datasets', 'train_dataset_splits'),
        ('train_datasets', 'train_dataset_text_columns'),
        ('eval_datasets', 'eval_dataset_configs'),
        ('eval_datasets', 'eval_dataset_splits'),
        ('eval_datasets', 'eval_dataset_text_columns')
    ]
    
    for arg1, arg2 in list_args:
        if len(getattr(args, arg1)) != len(getattr(args, arg2)):
            raise ValueError(f"Ensure that the number of entries in {arg1} equals {arg2}. "
                           f"Received {len(getattr(args, arg1))} for {arg1} and {len(getattr(args, arg2))} for {arg2}.")

def main():
    # Parse arguments
    args = parse_arguments()
    validate_arguments(args)
    
    # Initialize Rich console
    console = Console()
    
    # Welcome panel
    console.print(Panel.fit(
        "[bold cyan]🎯 Whisper Turkish Fine-tuning[/bold cyan]\n"
        "[dim]Fine-tuning Whisper model for Turkish speech recognition[/dim]",
        border_style="cyan"
    ))
    
    # Print arguments
    console.print('\n[bold blue]📋 Configuration:[/bold blue]')
    config_table = Table()
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="green")
    for key, value in vars(args).items():
        config_table.add_row(str(key), str(value))
    console.print(config_table)
    
    # Global configuration
    gradient_checkpointing = False  # GPU'da gradient checkpointing kapatıldı
    freeze_feature_encoder = False
    freeze_encoder = False
    do_normalize_eval = True
    do_lower_case = False
    do_remove_punctuation = False
    normalizer = BasicTextNormalizer()

    #############################       MODEL LOADING       #####################################

    console.print("\n[bold blue]🔧 Loading model components...[/bold blue]")
    
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_name)
    tokenizer = WhisperTokenizer.from_pretrained(args.model_name, language=args.language, task="transcribe")
    processor = WhisperProcessor.from_pretrained(args.model_name, language=args.language, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that config.decoder_start_token_id is correctly defined")

    if freeze_feature_encoder:
        model.freeze_feature_encoder()

    if freeze_encoder:
        model.freeze_encoder()
        model.model.encoder.gradient_checkpointing = False

    model.config.forced_decoder_ids = None
    model.config.use_cache = False  # Training için cache kapatıldı
    
    # 🎯 WHISPER-SPECIFIC OPTIMIZATIONS (Hugging Face Dokümantasyonundan)
    console.print("[blue]🔧 Applying Whisper-specific optimizations...[/blue]")
    
    # 1. SpecAugment Activation (Dokümantasyondan)
    model.config.apply_spec_augment = True
    model.config.mask_time_prob = 0.05  # %5 time masking
    model.config.mask_time_length = 10  # 10 frame mask uzunluğu
    model.config.mask_time_min_masks = 2  # Minimum 2 mask
    model.config.mask_feature_prob = 0.0  # Feature masking kapalı (audio için)
    model.config.mask_feature_length = 10
    model.config.mask_feature_min_masks = 0
    console.print("[green]✅ SpecAugment activated (time masking: 5%)[/green]")
    
    # 2. LayerDrop Regularization (Dokümantasyondan)
    model.config.encoder_layerdrop = 0.1  # %10 encoder layer dropout
    model.config.decoder_layerdrop = 0.1  # %10 decoder layer dropout
    console.print("[green]✅ LayerDrop activated (10% encoder/decoder)[/green]")
    
    # 3. Advanced Dropout Configuration
    model.config.dropout = 0.15  # Default 0.1'den artırıldı
    model.config.attention_dropout = 0.15  # Default 0.1'den artırıldı
    model.config.activation_dropout = 0.15  # Default 0.1'den artırıldı
    
    # 4. Generation Parameters (Dokümantasyondan)
    model.config.max_source_positions = 1500  # Max audio frames
    model.config.max_target_positions = 448   # Max text tokens
    console.print("[green]✅ Generation parameters optimized[/green]")
    
    # 5. Turkish-specific Suppress Tokens (Genişletilmiş)
    model.config.suppress_tokens = [
        1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63,
        90, 91, 92, 93, 359, 503, 522, 542, 873, 893, 902, 918, 922, 931,
        1350, 1853, 1982, 2460, 2627, 3246, 3253, 3268, 3536, 3846, 3961,
        4183, 4667, 6585, 6647, 7273, 9061, 9383, 10428, 10929, 11938,
        # Ek Türkçe karakterler için suppress tokens
        50257, 50258, 50259, 50260, 50261, 50262, 50263, 50264, 50265
    ]  # Türkçe için optimize edilmiş unwanted tokens
    model.config.begin_suppress_tokens = [220, 50256]  # Dokümantasyondan
    console.print("[green]✅ Turkish-specific suppress tokens configured (extended)[/green]")
    
    # 6. Audio Processing Optimization
    model.config.median_filter_width = 7  # Dokümantasyondan
    
    console.print("[green]🎯 Whisper optimization completed![/green]")

    # Eval/generate davranışı için varsayılan generation config ayarları (Trainer eval'de kullanılacak)
    try:
        gen_conf = model.generation_config
        # Whisper özel: transcribe ve dil
        setattr(gen_conf, "task", "transcribe")
        setattr(gen_conf, "language", args.language)
        # Tekrar ve uzunluk kısıtları
        gen_conf.no_repeat_ngram_size = 2
        gen_conf.repetition_penalty = 1.2
        gen_conf.length_penalty = 1.2
        gen_conf.do_sample = False  # Eval sırasında deterministik
        gen_conf.max_new_tokens = 100
        gen_conf.min_length = 5
        gen_conf.num_beams = 5
        model.generation_config = gen_conf
        console.print("[green]✅ Generation config updated for evaluation (beams=5, repetition/length constraints)[/green]")
    except Exception as _:
        console.print("[yellow]⚠️ Generation config could not be updated; proceeding with defaults[/yellow]")

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_disable()
    
    # Device optimization
    if not torch.cuda.is_available():
        torch.set_num_threads(min(os.cpu_count(), 8))  # Windows'ta çok fazla thread sorun çıkarabilir
        console.print(f"[blue]💻 CPU kullanılıyor - {min(os.cpu_count(), 8)} threads[/blue]")
    else:
        console.print("[green]🚀 GPU kullanılıyor[/green]")
        console.print(f"[green]GPU: {torch.cuda.get_device_name(0)}[/green]")
    
    # Training mode
    model.train()

    console.print("[green]✅ Model components loaded![/green]")

    ############################        DATASET LOADING AND PREP        ##########################

    def load_all_datasets(split):    
        combined_dataset = []
        if split == 'train':
            # Üç veri setini oranlara göre paylaştır
            ratios = args.train_mix_ratios
            if abs(sum(ratios) - 1.0) > 1e-6 or len(ratios) != 3:
                raise ValueError("train_mix_ratios toplamı 1.0 olmalı ve 3 değer içermeli")

            samples_list = [int(r * args.max_train_samples) for r in ratios]
            # Yuvarlamadan doğan farkı son elemana ekle
            diff = args.max_train_samples - sum(samples_list)
            samples_list[-1] += diff

            dataset_configs = [
                (args.train_mix_datasets[i], args.train_mix_configs[i], args.train_mix_splits[i], samples_list[i], args.train_mix_text_columns[i])
                for i in range(3)
            ]

            for dataset_name, config, split_name, samples_needed, text_col in dataset_configs:
                console.print(f"[blue]📥 Loading train dataset: {dataset_name} ({samples_needed} samples)[/blue]")
                
                # Streaming ile sadece ihtiyacımız olan veriyi al
                try:
                    dataset = load_dataset(dataset_name, config, split=split_name, streaming=True, trust_remote_code=True)
                    
                    # Streaming dataset'ten sadece ihtiyacımız olan veriyi al
                    dataset_list = []
                    for j, item in enumerate(dataset):
                        if j >= samples_needed:
                            break
                        dataset_list.append(item)
                    
                    # List'i dataset'e çevir
                    from datasets import Dataset
                    dataset = Dataset.from_list(dataset_list)
                    console.print(f"[yellow]📊 {dataset_name}: {len(dataset)} veri alındı (streaming)[/yellow]")
                    
                except Exception as e:
                    console.print(f"[yellow]⚠️ Streaming başarısız, normal yükleme: {e}[/yellow]")
                    dataset = load_dataset(dataset_name, config, split=split_name, trust_remote_code=True)
                    
                    # İhtiyacımız olan veriyi al
                    if len(dataset) > samples_needed:
                        dataset = dataset.select(range(samples_needed))
                        console.print(f"[yellow]📊 {dataset_name}: {samples_needed} veri seçildi[/yellow]")
                    else:
                        console.print(f"[yellow]📊 {dataset_name}: Tüm {len(dataset)} veri kullanılıyor[/yellow]")
                
                dataset = dataset.cast_column("audio", Audio(args.sampling_rate))
                if text_col != "sentence":
                    dataset = dataset.rename_column(text_col, "sentence")
                dataset = dataset.remove_columns(set(dataset.features.keys()) - set(["audio", "sentence"]))
                combined_dataset.append(dataset)
        elif split == 'eval':
            # Streaming ile sadece gerekli kadar örnek al
            per_ds_limit = max(1, (args.max_eval_samples // max(1, len(args.eval_datasets))) if args.max_eval_samples > 0 else 100)
            for i, ds in enumerate(args.eval_datasets):
                console.print(f"[blue]📥 Loading eval dataset (streaming): {ds}[/blue]")
                try:
                    stream = load_dataset(ds, args.eval_dataset_configs[i], split=args.eval_dataset_splits[i], streaming=True, trust_remote_code=True)
                    sampled = []
                    for j, item in enumerate(stream):
                        if j >= per_ds_limit:
                            break
                        sampled.append(item)
                    from datasets import Dataset
                    dataset = Dataset.from_list(sampled)
                    console.print(f"[green]✅ {ds}: {len(dataset)} eval samples (streaming)")
                except Exception as e:
                    console.print(f"[yellow]⚠️ Streaming failed for {ds}: {e}. Falling back to normal load with select()[/yellow]")
                    dataset = load_dataset(ds, args.eval_dataset_configs[i], split=args.eval_dataset_splits[i], trust_remote_code=True)
                    if len(dataset) > per_ds_limit:
                        dataset = dataset.select(range(per_ds_limit))
                    console.print(f"[green]✅ {ds}: {len(dataset)} eval samples (fallback)")

                dataset = dataset.cast_column("audio", Audio(args.sampling_rate))
                if args.eval_dataset_text_columns[i] != "sentence":
                    dataset = dataset.rename_column(args.eval_dataset_text_columns[i], "sentence")
                dataset = dataset.remove_columns(set(dataset.features.keys()) - set(["audio", "sentence"]))
                combined_dataset.append(dataset)
        
        ds_to_return = concatenate_datasets(combined_dataset)
        ds_to_return = ds_to_return.shuffle(seed=22)
        return ds_to_return

    console.print('\n[bold blue]📊 DATASET PREPARATION IN PROGRESS...[/bold blue]')
    raw_dataset = DatasetDict()
    raw_dataset["train"] = load_all_datasets('train')
    raw_dataset["eval"] = load_all_datasets('eval')
    
    # Limit dataset sizes (use all data if limit is set to -1)
    if args.max_train_samples > 0:
        console.print(f"[yellow]📊 Limiting train to {args.max_train_samples} samples...[/yellow]")
        raw_dataset["train"] = raw_dataset["train"].select(range(min(args.max_train_samples, len(raw_dataset["train"]))))
    else:
        console.print(f"[green]📊 Using all {len(raw_dataset['train'])} training samples...[/green]")
        
    if args.max_eval_samples > 0:
        console.print(f"[yellow]📊 Limiting eval to {args.max_eval_samples} samples...[/yellow]")
        raw_dataset["eval"] = raw_dataset["eval"].select(range(min(args.max_eval_samples, len(raw_dataset["eval"]))))
    else:
        console.print(f"[green]📊 Using all {len(raw_dataset['eval'])} evaluation samples...[/green]")
    
    # Dataset info
    table = Table(title="Dataset Information")
    table.add_column("Split", style="cyan")
    table.add_column("Samples", justify="right", style="green")
    table.add_row("Train", str(len(raw_dataset['train'])))
    table.add_row("Eval", str(len(raw_dataset['eval'])))
    console.print(table)

    def prepare_dataset(batch):
        # load and (possibly) resample audio data to 16kHz
        audio = batch["audio"]
        
        # Gelişmiş audio preprocessing
        audio_array = advanced_audio_preprocessing(audio["array"], audio["sampling_rate"])
        
        # Audio augmentation (training sırasında)
        if args.train_strategy == 'steps':
            audio_array = audio_augmentation(audio_array, audio["sampling_rate"])

        # compute log-Mel input features from processed audio array 
        batch["input_features"] = processor.feature_extractor(audio_array, sampling_rate=audio["sampling_rate"]).input_features[0]
        # compute input length of audio sample in seconds
        batch["input_length"] = len(audio_array) / audio["sampling_rate"]
        
        # Gelişmiş text preprocessing
        transcription = batch["sentence"]
        
        # Türkçe text preprocessing
        transcription = turkish_text_preprocessing(transcription)
        
        # Küçük harfe çevir (Türkçe için)
        if do_lower_case:
            transcription = transcription.lower()
        
        # Noktalama işaretlerini normalize et
        if do_remove_punctuation:
            transcription = normalizer(transcription).strip()
        
        # Boş transkripsiyonları kontrol et
        if not transcription or len(transcription) < 2:
            transcription = " "
        
        # encode'i collator'a bırak: burada sadece metni taşı
        batch["labels_text"] = transcription
        return batch

    max_label_length = model.config.max_length
    min_input_length = 0.0
    max_input_length = 30.0
    def is_in_length_range(length, labels_text):
        # labels_text'i hızlıca tokenize ederek uzunluk kontrolü yap
        try:
            token_ids = processor.tokenizer(labels_text).input_ids if isinstance(labels_text, str) else []
            label_len = len(token_ids)
        except Exception:
            label_len = 0
        return min_input_length < length < max_input_length and 0 < label_len < max_label_length

    console.print("\n[bold blue]⚙️ Preparing dataset...[/bold blue]")
    # Windows uyumluluğu: num_proc=0 ise çoklu işlemeyi kapat (None geçir)
    effective_num_proc = args.num_proc if isinstance(args.num_proc, int) and args.num_proc > 0 else None
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Processing audio data...", total=None)
        raw_dataset = raw_dataset.map(prepare_dataset, num_proc=effective_num_proc)
        progress.update(task, description="Filtering by length...")
        raw_dataset = raw_dataset.filter(
            is_in_length_range,
            input_columns=["input_length", "labels_text"],
            num_proc=effective_num_proc,
        ) 
        progress.update(task, description="[green]✅ Dataset prepared!")
    console.print("[green]✅ Dataset preparation completed![/green]")

    ###############################     DATA COLLATOR AND METRIC DEFINITION     ########################

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    console.print("[green]✅ Data collator created![/green]")

    console.print("\n[bold blue]📊 Loading evaluation metrics...[/bold blue]")
    metric = evaluate.load("wer")
    console.print("[green]✅ Metrics loaded![/green]")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # WER için (normalizer uygulayarak) metinler
        wer_pred = pred_str
        wer_ref = label_str
        if do_normalize_eval:
            wer_pred = [turkish_text_preprocessing(p) for p in wer_pred]
            wer_ref = [turkish_text_preprocessing(l) for l in wer_ref]
            wer_pred = [normalizer(p) for p in wer_pred]
            wer_ref = [normalizer(l) for l in wer_ref]

        # Reward için (normalizer UYGULAMADAN) sadece Türkçe preprocessing uygula
        reward_pred = [turkish_text_preprocessing(p) for p in pred_str]
        reward_ref = [turkish_text_preprocessing(l) for l in label_str]

        # Temel WER hesaplama (normalizer'lı metinlerle)
        wer = 100 * metric.compute(predictions=wer_pred, references=wer_ref)
        
        # Reward shaping ile gelişmiş metrikler
        reward_shaper = RewardShaping()
        total_rewards = []
        detailed_rewards = []
        
        for pred, ref in zip(reward_pred, reward_ref):
            total_reward, rewards = reward_shaper.compute_comprehensive_reward(pred, ref)
            total_rewards.append(total_reward)
            detailed_rewards.append(rewards)
        
        # Ortalama reward'ları hesapla
        avg_total_reward = np.mean(total_rewards) if total_rewards else 0.0
        
        # Detaylı reward'ları hesapla
        avg_rewards = {}
        if detailed_rewards:
            for key in detailed_rewards[0].keys():
                avg_rewards[f"reward_{key}"] = np.mean([r[key] for r in detailed_rewards])
        
        # Sonuçları birleştir
        metrics = {"wer": wer, "avg_reward": avg_total_reward}
        metrics.update(avg_rewards)
        
        return metrics

    ###############################     TRAINING ARGS AND TRAINING      ############################

    console.print("\n[bold blue]⚙️ Setting up training configuration...[/bold blue]")
    
    use_fp16 = torch.cuda.is_available()  # Use FP16 only if GPU is available
    use_cpu = not torch.cuda.is_available()
    
    if use_cpu:
        console.print("[blue]💻 CPU kullanılıyor[/blue]")
        console.print(f"[green]🔧 CPU Cores: {os.cpu_count()}[/green]")
    else:
        console.print("[green]🚀 GPU kullanılıyor[/green]")

    if args.train_strategy == 'epoch':
        training_args = Seq2SeqTrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.train_batchsize,
            gradient_accumulation_steps=4,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup,
            gradient_checkpointing=gradient_checkpointing,
            fp16=use_fp16,
            eval_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=args.num_epochs,
            save_total_limit=5,  # Daha az checkpoint sakla
            per_device_eval_batch_size=args.eval_batchsize,
            predict_with_generate=True,
            generation_max_length=448,  # Dokümantasyondan max_target_positions
            generation_num_beams=5,  # Eval sırasında beam=5 (uyarıyı kaldırır)
            remove_unused_columns=False,  # Audio data için gerekli
            logging_steps=25,
            report_to=["tensorboard"],
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            # Overfitting önleme parametreleri - Türkçe için optimize
            weight_decay=0.005,  # Daha düşük L2 regularization
            lr_scheduler_type="cosine_with_restarts",  # Cosine with restarts
            max_grad_norm=0.5,  # Daha agresif gradient clipping
            # Ek optimizasyon parametreleri - Türkçe için
            adam_beta1=0.9,  # Adam optimizer beta1
            adam_beta2=0.98,  # Daha düşük beta2 (Türkçe için)
            adam_epsilon=1e-6,  # Daha düşük epsilon
            warmup_ratio=0.15,  # Daha uzun warmup (Türkçe için)
            # Evaluation ve generation parametreleri
            eval_accumulation_steps=1,  # Evaluation sırasında gradient accumulation
            eval_delay=0,  # Evaluation gecikmesi
            include_inputs_for_metrics=False,  # Metrics için input'ları dahil etme
            # Data loading optimizasyonları
            dataloader_num_workers=0,  # Windows uyumluluğu için 0
            dataloader_pin_memory=False,  # Windows uyumluluğu için False
            dataloader_drop_last=False,  # Son batch'i atlama
            # Memory ve performance optimizasyonları
            dataloader_persistent_workers=False,  # Persistent workers
            # Logging ve monitoring
            logging_first_step=True,  # İlk adımı logla
            logging_nan_inf_filter=True,  # NaN/Inf değerleri filtrele
            # Model saving optimizasyonları
            save_safetensors=True,  # SafeTensors formatında kaydet
            save_only_model=False,  # Sadece modeli kaydet
            # Resume ve checkpoint
            resume_from_checkpoint=args.resume_from_ckpt,
        )

    elif args.train_strategy == 'steps':
        training_args = Seq2SeqTrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.train_batchsize,
            gradient_accumulation_steps=4 if use_cpu else 2,  # Daha küçük accumulation
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup,
            gradient_checkpointing=gradient_checkpointing,
            fp16=use_fp16,
            eval_strategy="steps",
            eval_steps=100,  # Her 100 adımda bir evaluation
            save_strategy="steps",
            save_steps=100,  # Her 100 adımda bir save
            max_steps=args.num_steps,
            save_total_limit=5,  # Daha az checkpoint sakla
            per_device_eval_batch_size=args.eval_batchsize,
            predict_with_generate=True,
            generation_max_length=448,  # Dokümantasyondan max_target_positions
            generation_num_beams=5,  # Eval sırasında beam=5 (uyarıyı kaldırır)
            remove_unused_columns=False,  # Audio data için gerekli
            logging_steps=max(args.num_steps // 20, 10),
            report_to=["tensorboard"],
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            # Overfitting önleme parametreleri - Türkçe için optimize
            weight_decay=0.005,  # Daha düşük L2 regularization
            lr_scheduler_type="cosine_with_restarts",  # Cosine with restarts
            max_grad_norm=0.5,  # Daha agresif gradient clipping
            # Ek optimizasyon parametreleri - Türkçe için
            adam_beta1=0.9,  # Adam optimizer beta1
            adam_beta2=0.98,  # Daha düşük beta2 (Türkçe için)
            adam_epsilon=1e-6,  # Daha düşük epsilon
            warmup_ratio=0.15,  # Daha uzun warmup (Türkçe için)
            # Evaluation ve generation parametreleri
            eval_accumulation_steps=1,  # Evaluation sırasında gradient accumulation
            eval_delay=0,  # Evaluation gecikmesi
            include_inputs_for_metrics=False,  # Metrics için input'ları dahil etme
            # Data loading optimizasyonları
            dataloader_num_workers=0,  # Windows uyumluluğu için 0
            dataloader_pin_memory=False,  # Windows uyumluluğu için False
            dataloader_drop_last=False,  # Son batch'i atlama
            # Memory ve performance optimizasyonları
            dataloader_persistent_workers=False,  # Persistent workers
            # Logging ve monitoring
            logging_first_step=True,  # İlk adımı logla
            logging_nan_inf_filter=True,  # NaN/Inf değerleri filtrele
            # Model saving optimizasyonları
            save_safetensors=True,  # SafeTensors formatında kaydet
            save_only_model=False,  # Sadece modeli kaydet
            # Resume ve checkpoint
            resume_from_checkpoint=args.resume_from_ckpt,
        )

    # Callback'leri ekle
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3,  # 3 evaluation boyunca iyileşme yoksa dur
        early_stopping_threshold=0.001  # Minimum iyileşme threshold
    )
    
    wer_improvement_callback = WERImprovementCallback()

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=raw_dataset["train"],
        eval_dataset=raw_dataset["eval"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
        callbacks=[early_stopping_callback, wer_improvement_callback],  # Her iki callback'i ekle
    )

    processor.save_pretrained(training_args.output_dir)
    console.print("[green]✅ Trainer created![/green]")

    # Training configuration summary
    config_table = Table(title="Final Training Configuration")
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="green")
    config_table.add_row("Model", args.model_name)
    config_table.add_row("Language", args.language)
    config_table.add_row("Strategy", args.train_strategy)
    if args.train_strategy == 'steps':
        config_table.add_row("Max Steps", str(args.num_steps))
    else:
        config_table.add_row("Epochs", str(args.num_epochs))
    config_table.add_row("Learning Rate", str(args.learning_rate))
    config_table.add_row("Train Batch Size", str(args.train_batchsize))
    config_table.add_row("Eval Batch Size", str(args.eval_batchsize))
    config_table.add_row("Gradient Accumulation", str(training_args.gradient_accumulation_steps))
    config_table.add_row("Max Grad Norm", str(training_args.max_grad_norm))
    config_table.add_row("Adam Beta1", str(training_args.adam_beta1))
    config_table.add_row("Adam Beta2", str(training_args.adam_beta2))
    config_table.add_row("Warmup Ratio", str(training_args.warmup_ratio))
    config_table.add_row("Save SafeTensors", str(training_args.save_safetensors))
    config_table.add_row("FP16", str(use_fp16))
    config_table.add_row("Device", "GPU" if not use_cpu else "CPU")
    config_table.add_row("Train Samples", str(len(raw_dataset["train"])))
    config_table.add_row("Eval Samples", str(len(raw_dataset["eval"])))
    console.print(config_table)

    # Start training
    console.print("\n[bold green]🚀 Starting training...[/bold green]")
    if use_cpu:
        console.print(f"[dim]{args.num_steps} adım CPU eğitimi zaman alabilir...[/dim]")
    else:
        console.print(f"[dim]GPU ile {args.num_steps} adım eğitim başlıyor...[/dim]")

    try:
        console.print('\n[bold blue]TRAINING IN PROGRESS...[/bold blue]')
        trainer.train()
        console.print('\n[bold green]DONE TRAINING[/bold green]')
        console.print(f"[green]✅ Model saved to '{args.output_dir}' directory![/green]")
        
    except Exception as e:
        console.print(f"\n[red]❌ Training error: {e}[/red]")
        console.print("[yellow]💡 Eğitim sırasında hata oluştu. Lütfen sistem kaynaklarını kontrol edin.[/yellow]")

    # Test example
    console.print("\n[bold blue]📝 Test example:[/bold blue]")
    console.print(Panel(
        "[bold]To test the model:[/bold]\n"
        "from transformers import pipeline\n"
        f"pipe = pipeline('automatic-speech-recognition', model='{args.output_dir}')\n"
        "result = pipe('path/to/audio.wav')\n"
        "print(result['text'])",
        title="Usage Example",
        border_style="blue"
    ))
    
    console.print("\n[bold green]🎯 Fine-tuning script completed![/bold green]")

if __name__ == "__main__":
    main()