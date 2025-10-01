"""
YouTube Altyazı vs Whisper Karşılaştırması
"""

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import json
from youtube_transcript_api import YouTubeTranscriptApi
import os
import numpy as np
import subprocess
import warnings

# Uyarıları kapat
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# pytube'u optional yap
try:
    from pytube import YouTube
    HAS_PYTUBE = True
except:
    HAS_PYTUBE = False

# ============================================================================
# YOUTUBE VİDEO
# ============================================================================
youtube_url = "https://youtube.com/shorts/nGcJeron1e0?si=1UFZa7Rzale5f7EZ"  # YouTube linki

# ============================================================================
# 1. YOUTUBE ALTYAZISI ÇEKME
# ============================================================================
def get_youtube_transcript(url):
    """YouTube'dan resmi altyazıyı çek"""
    try:
        # Video ID'yi al (Shorts destekli)
        if "shorts/" in url:
            # YouTube Shorts: https://youtube.com/shorts/VIDEO_ID
            video_id = url.split("shorts/")[1].split("?")[0]
        elif "v=" in url:
            # Normal video: https://youtube.com/watch?v=VIDEO_ID
            video_id = url.split("v=")[1].split("&")[0]
        elif "youtu.be" in url:
            # Kısa link: https://youtu.be/VIDEO_ID
            video_id = url.split("/")[-1].split("?")[0]
        else:
            # Direkt ID
            video_id = url
        
        print(f"📺 Video ID: {video_id}")
        
        # Yeni API kullanımı (v1.2.2)
        try:
            # Önce API instance oluştur
            ytt_api = YouTubeTranscriptApi()
            
            # Türkçe altyazı çek
            try:
                fetched_transcript = ytt_api.fetch(video_id, languages=['tr'])
                print("✓ Türkçe altyazı bulundu")
            except:
                # İngilizce dene
                try:
                    fetched_transcript = ytt_api.fetch(video_id, languages=['en'])
                    print("✓ İngilizce altyazı bulundu")
                except:
                    # Herhangi bir dil dene
                    fetched_transcript = ytt_api.fetch(video_id)
                    print("✓ Altyazı bulundu")
            
            # Altyazıyı metin olarak birleştir
            full_text = " ".join([snippet.text for snippet in fetched_transcript])
            
            # Raw data formatına çevir (eski kod uyumluluğu için)
            transcript_data = fetched_transcript.to_raw_data()
            
        except Exception as e:
            # Eski API'yi dene (fallback)
            try:
                transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=['tr'])
                print("✓ Türkçe altyazı bulundu (eski API)")
                full_text = " ".join([entry['text'] for entry in transcript_data])
            except:
                try:
                    transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
                    print("✓ İngilizce altyazı bulundu (eski API)")
                    full_text = " ".join([entry['text'] for entry in transcript_data])
                except:
                    raise Exception(f"Hiç altyazı bulunamadı: {e}")
        
        return full_text, transcript_data
        
    except Exception as e:
        print(f"❌ Altyazı çekilemedi: {e}")
        return None, None

# ============================================================================
# 2. YOUTUBE SES İNDİRME
# ============================================================================
def download_youtube_audio(url, output_path="audio.mp3"):
    """YouTube'dan sesi indir (yt-dlp veya pytube)"""
    
    # Önce yt-dlp dene (daha güvenilir)
    try:
        print("\n🎵 Ses indiriliyor (yt-dlp)...")
        
        cmd = [
            'yt-dlp',
            '-x',  # Sadece ses
            '--audio-format', 'mp3',
            '--audio-quality', '0',  # En iyi kalite
            '-o', output_path.replace('.mp3', ''),  # Çıktı dosyası
            url,
            '--no-warnings',
            '--quiet'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Dosya oluşturuldu mu kontrol et
        possible_files = [output_path, output_path.replace('.mp3', '') + '.mp3']
        for file in possible_files:
            if os.path.exists(file):
                print(f"✓ Ses indirildi: {file}")
                
                # Video bilgisi al
                info_cmd = ['yt-dlp', '--get-title', '--get-duration', url]
                info = subprocess.run(info_cmd, capture_output=True, text=True)
                lines = info.stdout.strip().split('\n')
                title = lines[0] if len(lines) > 0 else "Unknown"
                duration_str = lines[1] if len(lines) > 1 else "0:00"
                
                # Duration'ı saniyeye çevir
                try:
                    parts = duration_str.split(':')
                    if len(parts) == 2:
                        duration = int(parts[0]) * 60 + int(parts[1])
                    else:
                        duration = 0
                except:
                    duration = 0
                
                return file, title, duration
        
        raise Exception("Dosya oluşturulamadı")
        
    except FileNotFoundError:
        print("⚠️  yt-dlp bulunamadı, pytube deneniyor...")
        
    except Exception as e:
        print(f"⚠️  yt-dlp hatası: {e}")
    
    # pytube dene (fallback)
    if HAS_PYTUBE:
        try:
            print("🎵 Ses indiriliyor (pytube)...")
            
            yt = YouTube(url, use_oauth=False, allow_oauth_cache=False)
            audio_stream = yt.streams.filter(only_audio=True).order_by('abr').desc().first()
            
            if audio_stream:
                audio_stream.download(filename=output_path)
                print(f"✓ Ses indirildi: {output_path}")
                return output_path, yt.title, yt.length
                
        except Exception as e:
            print(f"❌ pytube hatası: {e}")
    
    # Her iki yöntem de başarısız
    print("\n❌ Ses indirilemedi!")
    print("\n💡 Çözüm:")
    print("   pip install yt-dlp")
    print("   veya manuel indir:")
    print(f"   yt-dlp -x --audio-format mp3 {url}")
    
    return None, None, None

# ============================================================================
# 3. WHISPER TRANSKRİPSİYON
# ============================================================================
def transcribe_with_whisper(audio_path, config_path="whisper_best_config.json", use_config=True):
    """Whisper ile transkribe et"""
    if use_config:
        print("\n🤖 Whisper transkribe ediyor (Config'li)...")
    else:
        print("\n🤖 Whisper transkribe ediyor (Baseline)...")
    
    # Model yükle
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # Ses yükle (ilk 60 saniye - test için)
    audio, sr = librosa.load(audio_path, sr=16000, duration=60)
    print(f"✓ İlk 60 saniye yüklendi")
    
    # Transkribe et
    input_features = processor(
        audio, 
        sampling_rate=16000, 
        return_tensors="pt"
    ).input_features.to(device)
    
    if use_config:
        # Config yükle
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        gen_kwargs = config["generation_params"].copy()
        
        if config.get("use_prompt") and config.get("prompt_text"):
            prompt_ids = processor.get_prompt_ids(
                config["prompt_text"].strip(), 
                return_tensors="pt"
            )
            generated_ids = model.generate(
                input_features,
                prompt_ids=prompt_ids.to(device),
                **gen_kwargs
            )
        else:
            generated_ids = model.generate(input_features, **gen_kwargs)
    else:
        # Baseline (standart ayarlar)
        generated_ids = model.generate(
            input_features,
            language="tr",
            task="transcribe",
            max_length=448
        )
    
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return text

# ============================================================================
# 4. KARŞILAŞTIRMA (WER Hesaplama)
# ============================================================================
def calculate_wer(reference, hypothesis):
    """Word Error Rate hesapla"""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    # Levenshtein distance
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
    
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1].lower() == hyp_words[j-1].lower():
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion = d[i][j-1] + 1
                deletion = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)
    
    wer = (d[len(ref_words)][len(hyp_words)] / len(ref_words)) * 100
    return wer

def compare_texts(youtube_text, whisper_text):
    """İki metni karşılaştır"""
    print("\n" + "="*80)
    print("📊 KARŞILAŞTIRMA SONUÇLARI")
    print("="*80)
    
    # İlk 60 saniyelik YouTube altyazısını al (Whisper da 60 saniye yaptık)
    youtube_words = youtube_text.split()[:100]  # İlk 100 kelime
    youtube_sample = " ".join(youtube_words)
    
    # WER hesapla
    wer = calculate_wer(youtube_sample, whisper_text)
    
    # Benzerlik oranı
    similarity = max(0, 100 - wer)
    
    print(f"\n📈 Benzerlik Oranı: %{similarity:.1f}")
    print(f"📉 WER (Hata Oranı): %{wer:.1f}")
    
    print(f"\n{'─'*80}")
    print("📺 YOUTUBE ALTYAZISI (İlk 100 kelime):")
    print(f"{'─'*80}")
    print(youtube_sample)
    
    print(f"\n{'─'*80}")
    print("🤖 WHISPER TRANSKRİPSİYONU:")
    print(f"{'─'*80}")
    print(whisper_text)
    
    print(f"\n{'─'*80}")
    print("💡 YORUM:")
    print(f"{'─'*80}")
    
    if wer < 15:
        print("✅ Mükemmel! Whisper YouTube altyazısıyla neredeyse aynı.")
    elif wer < 30:
        print("👍 İyi! Küçük farklılıklar var ama genel olarak doğru.")
    elif wer < 50:
        print("⚠️  Orta. Bazı kelimeler yanlış tanınmış.")
    else:
        print("❌ Zayıf. YouTube altyazısı çok daha iyi.")
    
    print("\n" + "="*80)

# ============================================================================
# ANA PROGRAM
# ============================================================================
if __name__ == "__main__":
    
    print("\n" + "="*80)
    print("🎬 YOUTUBE vs WHISPER KARŞILAŞTIRMA")
    print("="*80 + "\n")
    
    # YouTube URL kontrolü
    if "ORNEK_VIDEO_ID" in youtube_url:
        print("⚠️  YouTube URL'sini değiştir!")
        print("\nÖrnek:")
        print('  youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"')
        print("\nVeya test için:")
        
        # Test modu
        print("\n📝 Test modu: Örnek verilerle çalışıyor...\n")
        
        youtube_text = "Merhaba ben bir test videosuyum. Bu videoda Türkçe konuşma testi yapıyoruz."
        whisper_text = "Merhaba ben bir test videosuyum bu videoda türkçe konuşma testi yapıyoruz"
        
        wer = calculate_wer(youtube_text, whisper_text)
        print(f"YouTube: {youtube_text}")
        print(f"Whisper: {whisper_text}")
        print(f"\n📊 WER: %{wer:.1f} | Benzerlik: %{max(0, 100-wer):.1f}")
        
    else:
        try:
            # 1. YouTube altyazısını çek
            youtube_text, transcript_data = get_youtube_transcript(youtube_url)
            
            if youtube_text:
                print(f"✓ Altyazı uzunluğu: {len(youtube_text.split())} kelime")
            
            # 2. Sesi indir
            audio_path, title, duration = download_youtube_audio(youtube_url)
            
            if audio_path and title:
                print(f"✓ Video: {title}")
                print(f"✓ Süre: {duration//60}:{duration%60:02d}")
            
            # 3. Whisper ile transkribe et (hem config'li hem baseline)
            whisper_config_text = None
            whisper_baseline_text = None
            
            if audio_path:
                # Config'li transkripsiyon
                whisper_config_text = transcribe_with_whisper(audio_path, use_config=True)
                
                # Baseline transkripsiyon
                whisper_baseline_text = transcribe_with_whisper(audio_path, use_config=False)
            
            # 4. Karşılaştır
            if youtube_text and whisper_config_text and whisper_baseline_text:
                # Her iki Whisper sonucu ile karşılaştır
                print("\n" + "="*80)
                print("📊 KARŞILAŞTIRMA SONUÇLARI")
                print("="*80)
                
                # YouTube vs Config'li
                wer_config = calculate_wer(youtube_text, whisper_config_text)
                similarity_config = max(0, 100 - wer_config)
                
                # YouTube vs Baseline
                wer_baseline = calculate_wer(youtube_text, whisper_baseline_text)
                similarity_baseline = max(0, 100 - wer_baseline)
                
                # İyileşme
                improvement = wer_baseline - wer_config
                
                print(f"\n🎯 WER (YouTube Altyazısına Göre):")
                print(f"   🔵 Baseline:  {wer_baseline:.1f}%")
                print(f"   🟢 Config'li: {wer_config:.1f}%")
                print(f"   📊 İyileşme:  {improvement:+.1f}%")
                
                if improvement > 2:
                    print(f"   ✅ Config açık ara daha iyi!")
                elif improvement > 0:
                    print(f"   ✅ Config biraz daha iyi")
                elif improvement < -2:
                    print(f"   ⚠️  Baseline açık ara daha iyi!")
                elif improvement < 0:
                    print(f"   ⚠️  Baseline biraz daha iyi")
                else:
                    print(f"   ⚡ İkisi de aynı performans!")
                
                print(f"\n{'─'*80}")
                print("📺 YOUTUBE ALTYAZISI:")
                print(f"{'─'*80}")
                print(youtube_text)
                
                print(f"\n{'─'*80}")
                print("🔵 BASELINE WHISPER:")
                print(f"{'─'*80}")
                print(whisper_baseline_text)
                
                print(f"\n{'─'*80}")
                print("🟢 CONFIG'Lİ WHISPER:")
                print(f"{'─'*80}")
                print(whisper_config_text)
                
                print("\n" + "="*80)
                
            elif youtube_text and whisper_config_text:
                # Sadece config'li varsa
                compare_texts(youtube_text, whisper_config_text)
            elif whisper_config_text and not youtube_text:
                # Sadece Whisper sonucu varsa onu göster
                print("\n" + "="*80)
                print("🤖 WHISPER TRANSKRİPSİYONU")
                print("="*80)
                print(whisper_config_text)
                print("="*80)
                print("\n⚠️  YouTube altyazısı bulunamadı, karşılaştırma yapılamadı")
            
            # Temizlik
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
                print("\n🧹 İndirilen ses dosyası silindi")
                
        except Exception as e:
            print(f"\n❌ Hata: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("✓ Tamamlandı!")
    print("="*80 + "\n")

