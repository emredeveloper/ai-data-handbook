"""
Whisper Production Kullanımı
Test sonuçlarına göre optimize edilmiş ayarlarla kullanım
"""

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import json
from pathlib import Path

class WhisperTurkishASR:
    """
    Test sonuçlarıyla optimize edilmiş Whisper Turkish ASR
    """
    
    def __init__(self, model_name="openai/whisper-small", config_file=None):
        """
        Args:
            model_name: Whisper model adı
            config_file: Önceden kaydedilmiş config dosyası (opsiyonel)
        """
        print(f"Model yükleniyor: {model_name}")
        
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        
        # Default config (test sonuçlarına göre güncellenebilir)
        self.config = self._load_config(config_file) if config_file else self._default_config()
        
        print(f"✓ Model hazır! Cihaz: {self.device}")
        print(f"✓ Aktif ayarlar: {self.config['name']}")
    
    def _default_config(self):
        """Default konfigürasyon"""
        return {
            "name": "Default - Few-Shot Long + Beam Search",
            "use_prompt": True,
            "prompt_type": "long",
            "prompt_text": """
Yerin yakınında yer çekimi ivmesi 9,8 metre bölü saniye kare ve aşağı yönlüdür. Bu, Newton'ın evrensel çekim yasasından türetilmiştir. Newton'ın birinci kanununa göre, bir cisim üzerine net kuvvet etki etmiyorsa o cismin hızı sabittir. Sabit hız derken, hem büyüklük hem de yön olarak sabit olduğunu kastediyoruz. İkinci kanunda ise net kuvvet kütle ile ivmenin çarpımına eşittir. Dolayısıyla, bir cisme etki eden kuvvet arttıkça ivme de artar. Kompozisyonda yer alan her bir öğe, kabın formunun yuvarlak olması ile uyum sağlayacak şekilde yerleştirilmiştir.
""",
            "generation_params": {
                "max_length": 448,
                "language": "tr",
                "task": "transcribe",
                "num_beams": 8,
                "early_stopping": True,
                "length_penalty": 1.2,
                "no_repeat_ngram_size": 3,
                "num_return_sequences": 1
            }
        }
    
    def _load_config(self, config_file):
        """Config dosyasından ayarları yükle"""
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_config(self, output_file="whisper_best_config.json"):
        """Mevcut ayarları kaydet"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        print(f"✓ Config kaydedildi: {output_file}")
    
    def update_config(self, 
                     use_prompt=None, 
                     prompt_type=None, 
                     prompt_text=None,
                     beam_search=None,
                     num_beams=None,
                     aggressive_mode=False):
        """
        Config'i güncelle
        
        Args:
            use_prompt: Few-shot prompt kullan (True/False)
            prompt_type: Prompt tipi ("short", "long", "domain_specific")
            prompt_text: Özel prompt metni
            beam_search: Beam search kullan (True/False)
            num_beams: Beam sayısı (5, 8, 12, vb.)
            aggressive_mode: Agresif beam search (True/False)
        """
        if use_prompt is not None:
            self.config["use_prompt"] = use_prompt
        
        if prompt_type:
            self.config["prompt_type"] = prompt_type
        
        if prompt_text:
            self.config["prompt_text"] = prompt_text
        
        if beam_search is not None:
            if beam_search:
                self.config["generation_params"]["num_beams"] = num_beams or 8
                self.config["generation_params"]["early_stopping"] = True
                self.config["generation_params"]["length_penalty"] = 1.2
                self.config["generation_params"]["no_repeat_ngram_size"] = 3
                
                if aggressive_mode:
                    self.config["generation_params"]["num_beams"] = num_beams or 12
                    self.config["generation_params"]["num_beam_groups"] = 3
                    self.config["generation_params"]["diversity_penalty"] = 0.5
                    self.config["generation_params"]["length_penalty"] = 1.5
            else:
                # Standard decoding
                self.config["generation_params"].pop("num_beams", None)
                self.config["generation_params"].pop("early_stopping", None)
                self.config["generation_params"].pop("num_beam_groups", None)
                self.config["generation_params"].pop("diversity_penalty", None)
        
        if num_beams and "num_beams" in self.config["generation_params"]:
            self.config["generation_params"]["num_beams"] = num_beams
        
        print(f"✓ Config güncellendi")
    
    def transcribe(self, audio_path_or_array, sampling_rate=None, advanced_technique=None):
        """
        Ses dosyasını transkribe et
        
        Args:
            audio_path_or_array: Ses dosya yolu (str) veya ses array (numpy)
            sampling_rate: Sampling rate (sadece array için gerekli)
            advanced_technique: İleri teknik kullan
                - "vad": Voice Activity Detection (hızlı)
                - "chunked": Uzun sesler için
                - "self_consistency": En yüksek doğruluk
                - None: Standard
        
        Returns:
            str: Transkripsiyon metni
        """
        # İleri teknikler
        if advanced_technique == "vad":
            if isinstance(audio_path_or_array, str):
                return self._transcribe_with_vad(audio_path_or_array)
        elif advanced_technique == "chunked":
            if isinstance(audio_path_or_array, str):
                return self._transcribe_chunked(audio_path_or_array)
        elif advanced_technique == "self_consistency":
            if isinstance(audio_path_or_array, str):
                return self._transcribe_self_consistency(audio_path_or_array)
        
        # Standard transkripsiyon
        # Ses verisi yükle
        if isinstance(audio_path_or_array, str):
            audio_array, sampling_rate = librosa.load(audio_path_or_array, sr=16000)
        else:
            audio_array = audio_path_or_array
            if sampling_rate != 16000:
                audio_array = librosa.resample(
                    audio_array, 
                    orig_sr=sampling_rate, 
                    target_sr=16000
                )
        
        # Input features
        input_features = self.processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(self.device)
        
        # Generation kwargs
        gen_kwargs = self.config["generation_params"].copy()
        
        # Prompt kullan
        if self.config["use_prompt"] and self.config.get("prompt_text"):
            prompt_ids = self.processor.get_prompt_ids(
                self.config["prompt_text"].strip(), 
                return_tensors="pt"
            )
            generated_ids = self.model.generate(
                input_features,
                prompt_ids=prompt_ids.to(self.device) if prompt_ids is not None else None,
                **gen_kwargs
            )
        else:
            generated_ids = self.model.generate(
                input_features,
                **gen_kwargs
            )
        
        # Decode
        transcription = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )[0]
        
        return transcription
    
    def batch_transcribe(self, audio_paths, show_progress=True, advanced_technique=None):
        """
        Birden fazla ses dosyasını transkribe et
        
        Args:
            audio_paths: Ses dosya yolları listesi
            show_progress: İlerleme göster
            advanced_technique: Tüm dosyalar için ileri teknik
        
        Returns:
            list: Transkripsiyon metinleri
        """
        results = []
        
        if show_progress:
            from tqdm import tqdm
            audio_paths = tqdm(audio_paths, desc="Transkribe ediliyor")
        
        for audio_path in audio_paths:
            transcription = self.transcribe(audio_path, advanced_technique=advanced_technique)
            results.append(transcription)
        
        return results
    
    # ========================================================================
    # İLERİ TEKNİKLER
    # ========================================================================
    
    def _transcribe_with_vad(self, audio_path):
        """VAD ile transkripsiyon (sessiz bölümleri atla)"""
        import numpy as np
        
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Basit energy-based VAD
        frame_length = int(0.025 * sr)
        hop_length = int(0.010 * sr)
        
        energy = np.array([
            np.sum(audio[i:i+frame_length]**2) 
            for i in range(0, len(audio)-frame_length, hop_length)
        ])
        
        threshold = np.percentile(energy, 40)
        speech_frames = energy > threshold
        
        # Konuşma segmentlerini bul
        segments = []
        in_speech = False
        start = 0
        
        for i, is_speech in enumerate(speech_frames):
            if is_speech and not in_speech:
                start = i * hop_length
                in_speech = True
            elif not is_speech and in_speech:
                end = i * hop_length
                segments.append((start, end))
                in_speech = False
        
        if in_speech:
            segments.append((start, len(audio)))
        
        # Her segment için transkribe et
        transcriptions = []
        for start, end in segments:
            segment_audio = audio[start:end]
            if len(segment_audio) > sr * 0.5:
                text = self.transcribe(segment_audio, sampling_rate=sr)
                if text.strip():
                    transcriptions.append(text)
        
        return " ".join(transcriptions)
    
    def _transcribe_chunked(self, audio_path, chunk_length_s=30, overlap_s=5):
        """Chunk-based transkripsiyon (uzun sesler için)"""
        audio, sr = librosa.load(audio_path, sr=16000)
        duration = len(audio) / sr
        
        if duration <= chunk_length_s:
            return self.transcribe(audio, sampling_rate=sr)
        
        chunk_samples = int(chunk_length_s * sr)
        overlap_samples = int(overlap_s * sr)
        stride = chunk_samples - overlap_samples
        
        transcriptions = []
        
        for start in range(0, len(audio), stride):
            end = min(start + chunk_samples, len(audio))
            chunk = audio[start:end]
            
            if len(chunk) < sr * 1.0:
                continue
            
            text = self.transcribe(chunk, sampling_rate=sr)
            transcriptions.append(text)
            
            if end >= len(audio):
                break
        
        return " ".join(transcriptions)
    
    def _transcribe_self_consistency(self, audio_path, num_samples=3):
        """Self-consistency ile transkripsiyon (en yüksek doğruluk)"""
        from collections import Counter
        
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Farklı seed/temperature ile N kez transkribe et
        samples = []
        
        for i in range(num_samples):
            # Config'i geçici olarak değiştir
            original_params = self.config["generation_params"].copy()
            
            self.config["generation_params"]["do_sample"] = True
            self.config["generation_params"]["temperature"] = 0.7 + (i * 0.1)
            self.config["generation_params"]["top_p"] = 0.9
            
            text = self.transcribe(audio, sampling_rate=sr)
            samples.append(text)
            
            # Config'i geri yükle
            self.config["generation_params"] = original_params
        
        # Majority voting
        word_votes = []
        max_len = max(len(s.split()) for s in samples)
        
        for i in range(max_len):
            words_at_position = []
            for sample in samples:
                words = sample.split()
                if i < len(words):
                    words_at_position.append(words[i])
            
            if words_at_position:
                most_common = Counter(words_at_position).most_common(1)[0][0]
                word_votes.append(most_common)
        
        return " ".join(word_votes)


# ============================================================================
# KULLANIM ÖRNEKLERİ
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*80)
    print("WHISPER TURKISH ASR - PRODUCTION KULLANIMI")
    print("="*80 + "\n")
    
    # 1. DEFAULT AYARLARLA KULLANIM
    print("### 1. Default Ayarlarla Başlat ###\n")
    asr = WhisperTurkishASR()
    
    # Örnek ses dosyası transkribe et (dosya yolu ver)
    # transcription = asr.transcribe("ses_dosyasi.wav")
    # print(f"Sonuç: {transcription}\n")
    
    # 2. TEST SONUÇLARINDAN EN İYİSİNİ SEÇ
    print("\n### 2. Test Sonuçlarına Göre Ayarları Güncelle ###\n")
    
    # Örnek: Test sonucunda "long prompt + beam search (8)" en iyiyse:
    asr.update_config(
        use_prompt=True,
        prompt_type="long",
        beam_search=True,
        num_beams=8
    )
    
    # Veya: "domain_specific + aggressive beam" en iyiyse:
    # asr.update_config(
    #     use_prompt=True,
    #     prompt_type="domain_specific",
    #     beam_search=True,
    #     num_beams=12,
    #     aggressive_mode=True
    # )
    
    # 3. AYARLARI KAYDET
    print("\n### 3. En İyi Ayarları Kaydet ###\n")
    asr.save_config("whisper_best_config.json")
    
    # 4. KAYDEDİLMİŞ CONFIG'LE YENİ INSTANCE OLUŞTUR
    print("\n### 4. Kaydedilmiş Config ile Yükle ###\n")
    # asr_prod = WhisperTurkishASR(config_file="whisper_best_config.json")
    # transcription = asr_prod.transcribe("yeni_ses.wav")
    
    # 5. TOPLU TRANSKRIPSIYON
    print("\n### 5. Birden Fazla Dosya ###\n")
    # audio_files = ["ses1.wav", "ses2.wav", "ses3.wav"]
    # results = asr.batch_transcribe(audio_files)
    
    print("\n" + "="*80)
    print("✓ Production kodu hazır!")
    print("="*80 + "\n")
    
    print("KULLANIM ÖRNEKLERİ:")
    print("""
╔════════════════════════════════════════════════════════════════════╗
║                      1. STANDART KULLANIM                           ║
╚════════════════════════════════════════════════════════════════════╝

# Model yükle
asr = WhisperTurkishASR()

# Test sonuçlarına göre ayarla
asr.update_config(
    use_prompt=True,
    prompt_type="long",
    beam_search=True,
    num_beams=8
)

# Transkribe et
text = asr.transcribe("ses_dosyasi.wav")
print(text)


╔════════════════════════════════════════════════════════════════════╗
║                  2. İLERİ TEKNİKLERLE KULLANIM                      ║
╚════════════════════════════════════════════════════════════════════╝

# Normal ses için VAD (hızlı)
text = asr.transcribe("normal.wav", advanced_technique="vad")

# Uzun ses için Chunked (1+ saat ses)
text = asr.transcribe("long.wav", advanced_technique="chunked")

# Maksimum doğruluk için Self-Consistency
text = asr.transcribe("critical.wav", advanced_technique="self_consistency")


╔════════════════════════════════════════════════════════════════════╗
║                       3. TOPLU İŞLEME                               ║
╚════════════════════════════════════════════════════════════════════╝

audio_files = ["ses1.wav", "ses2.wav", "ses3.wav"]

# Normal toplu işleme
results = asr.batch_transcribe(audio_files)

# VAD ile toplu işleme (hızlı)
results = asr.batch_transcribe(audio_files, advanced_technique="vad")


╔════════════════════════════════════════════════════════════════════╗
║                   4. CONFIG KAYDET VE YÜKLE                         ║
╚════════════════════════════════════════════════════════════════════╝

# En iyi ayarları kaydet
asr.save_config("best_config.json")

# Sonra yükle
asr_prod = WhisperTurkishASR(config_file="best_config.json")
text = asr_prod.transcribe("ses.wav")


╔════════════════════════════════════════════════════════════════════╗
║                     HANGİ TEKNİĞİ KULLAN?                           ║
╚════════════════════════════════════════════════════════════════════╝

✓ Kısa ses (< 30s)           → Standard (few-shot + beam)
✓ Uzun ses (> 1 saat)        → advanced_technique="chunked"
✓ Hız önemli                 → advanced_technique="vad"
✓ Doğruluk kritik            → advanced_technique="self_consistency"
✓ Çok hassas transkripsiyon  → Tüm teknikleri kombine et

    """)

