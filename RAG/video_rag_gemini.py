"""
Video RAG (Retrieval-Augmented Generation) with Gemini API
Bu kod video dosyalarını analiz edip içerik üzerinden soru-cevap sistemi oluşturur.
"""

import os
import json
import time
import tempfile
from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.table import Table
    from rich.progress import Progress, TaskID
    from rich.markdown import Markdown
    from rich.prompt import Prompt
    from rich import print as rprint
except ImportError:
    print("Rich kütüphanesi bulunamadı. Yüklemek için:")
    print("pip install rich")
    exit(1)

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
except ImportError:
    print("Google GenerativeAI kütüphanesi bulunamadı. Yüklemek için:")
    print("pip install google-generativeai")
    exit(1)

try:
    import cv2
    import numpy as np
except ImportError:
    print("OpenCV kütüphanesi bulunamadı. Yüklemek için:")
    print("pip install opencv-python")
    exit(1)

try:
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError:
    print("Sentence transformers veya FAISS bulunamadı. Yüklemek için:")
    print("pip install sentence-transformers faiss-cpu")
    exit(1)


class VideoRAGSystem:
    def __init__(self, api_key: str, cache_dir: str = "./video_cache"):
        """
        Video RAG sistemi başlatılır
        
        Args:
            api_key (str): Gemini API anahtarı
            cache_dir (str): Video analizlerinin cache edileceği dizin
        """
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.console = Console()
        
        # Gemini API'yi yapılandır
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Embedding modeli
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Video analiz cache'i
        self.video_segments = []
        self.embeddings = None
        self.faiss_index = None
        
        self.console.print(Panel.fit("🎬 Video RAG sistemi başlatıldı!", style="bold green"))
    
    def extract_frames(self, video_path: str, interval_seconds: int = 30) -> List[np.ndarray]:
        """
        Video dosyasından belirli aralıklarla frame'ler çıkarır
        
        Args:
            video_path (str): Video dosya yolu
            interval_seconds (int): Frame çıkarma aralığı (saniye)
            
        Returns:
            List[np.ndarray]: Çıkarılan frame'ler
        """
        self.console.print(f"🎞️ Video'dan frame'ler çıkarılıyor: [cyan]{video_path}[/cyan]")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * interval_seconds)
        
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                frames.append(frame)
                
            frame_count += 1
        
        cap.release()
        self.console.print(f"✅ Toplam [bold green]{len(frames)}[/bold green] frame çıkarıldı")
        return frames
    
    def save_frame_as_temp(self, frame: np.ndarray) -> str:
        """
        Frame'i geçici dosya olarak kaydeder
        
        Args:
            frame (np.ndarray): OpenCV frame
            
        Returns:
            str: Geçici dosya yolu
        """
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        cv2.imwrite(temp_file.name, frame)
        return temp_file.name
    
    def analyze_frame_with_gemini(self, frame_path: str, timestamp: float) -> Dict[str, Any]:
        """
        Gemini API ile frame'i analiz eder
        
        Args:
            frame_path (str): Frame dosya yolu
            timestamp (float): Video'daki zaman damgası
            
        Returns:
            Dict[str, Any]: Analiz sonucu
        """
        try:
            # Resmi Gemini'ye yükle
            uploaded_file = genai.upload_file(frame_path)
            
            # Frame analizi için prompt
            prompt = """
            Bu video frame'ini detaylı analiz et ve şu bilgileri ver:
            1. Sahnede neler oluyor?
            2. Görünen nesneler, insanlar, aktiviteler
            3. Metin varsa oku
            4. Önemli detaylar
            5. Bu sahnenin genel özeti
            
            Türkçe ve detaylı cevap ver.
            """
            
            response = self.model.generate_content([uploaded_file, prompt])
            
            # Geçici dosyayı sil
            os.unlink(frame_path)
            
            return {
                "timestamp": timestamp,
                "analysis": response.text,
                "frame_id": f"frame_{int(timestamp)}"
            }
            
        except Exception as e:
            self.console.print(f"❌ Frame analizi hatası: [red]{e}[/red]")
            return {
                "timestamp": timestamp,
                "analysis": f"Analiz hatası: {str(e)}",
                "frame_id": f"frame_{int(timestamp)}"
            }
    
    def get_video_hash(self, video_path: str) -> str:
        """Video dosyasının hash'ini hesaplar"""
        hash_md5 = hashlib.md5()
        with open(video_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def save_cache(self, video_hash: str, segments: List[Dict[str, Any]]):
        """Analiz sonuçlarını cache'e kaydeder"""
        cache_file = self.cache_dir / f"{video_hash}.json"
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
    
    def load_cache(self, video_hash: str) -> Optional[List[Dict[str, Any]]]:
        """Cache'den analiz sonuçlarını yükler"""
        cache_file = self.cache_dir / f"{video_hash}.json"
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def process_video(self, video_path: str, frame_interval: int = 30) -> List[Dict[str, Any]]:
        """
        Video'yu işler ve analiz sonuçlarını döner
        
        Args:
            video_path (str): Video dosya yolu
            frame_interval (int): Frame çıkarma aralığı (saniye)
            
        Returns:
            List[Dict[str, Any]]: Video segment analizleri
        """
        # Video hash kontrolü
        video_hash = self.get_video_hash(video_path)
        cached_segments = self.load_cache(video_hash)
        
        if cached_segments:
            self.console.print("📁 Cache'den yükleniyor...", style="yellow")
            self.video_segments = cached_segments
            return cached_segments
        
        self.console.print(f"🎬 Video işleniyor: [cyan]{video_path}[/cyan]")
        
        # Frame'leri çıkar
        frames = self.extract_frames(video_path, frame_interval)
        
        segments = []
        
        with Progress() as progress:
            task = progress.add_task("🔍 Frame analizi...", total=len(frames))
            
            for i, frame in enumerate(frames):
                timestamp = i * frame_interval
                progress.update(task, description=f"🔍 Frame {i+1}/{len(frames)} analiz ediliyor (⏱️ {timestamp}s)")
                
                # Frame'i geçici dosya olarak kaydet
                temp_frame_path = self.save_frame_as_temp(frame)
                
                # Gemini ile analiz et
                analysis = self.analyze_frame_with_gemini(temp_frame_path, timestamp)
                segments.append(analysis)
                
                # API rate limit için bekle
                time.sleep(2)
                progress.advance(task)
        
        # Cache'e kaydet
        self.save_cache(video_hash, segments)
        self.video_segments = segments
        
        return segments
    
    def create_embeddings(self):
        """Video segment'lerinin embedding'lerini oluşturur"""
        if not self.video_segments:
            self.console.print("❌ Önce video'yu işlemelisiniz!", style="red")
            return
        
        self.console.print("🧠 Embedding'ler oluşturuluyor...", style="blue")
        
        # Tüm analizleri birleştir
        texts = [segment["analysis"] for segment in self.video_segments]
        
        # Embedding'leri hesapla
        self.embeddings = self.embedding_model.encode(texts)
        
        # FAISS index oluştur
        dimension = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.faiss_index.add(self.embeddings)
        
        self.console.print(f"✅ [bold green]{len(texts)}[/bold green] segment için embedding oluşturuldu")
    
    def search_relevant_segments(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Sorguyla ilgili segment'leri arar
        
        Args:
            query (str): Arama sorgusu
            top_k (int): Kaç tane segment döndürülecek
            
        Returns:
            List[Dict[str, Any]]: İlgili segment'ler
        """
        if self.faiss_index is None:
            self.console.print("❌ Önce embedding'ler oluşturulmalı!", style="red")
            return []
        
        # Sorgu embedding'i
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Benzer segment'leri ara
        scores, indices = self.faiss_index.search(query_embedding, top_k)
        
        relevant_segments = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            segment = self.video_segments[idx].copy()
            segment["similarity_score"] = float(score)
            relevant_segments.append(segment)
        
        return relevant_segments
    
    def answer_question(self, question: str, top_k: int = 3) -> str:
        """
        Video içeriği temelinde soruya cevap verir
        
        Args:
            question (str): Kullanıcı sorusu
            top_k (int): Kaç segment kullanılacak
            
        Returns:
            str: Gemini'nin cevabı
        """
        if not self.video_segments:
            return "Önce bir video işlemelisiniz!"
        
        # İlgili segment'leri bul
        relevant_segments = self.search_relevant_segments(question, top_k)
        
        if not relevant_segments:
            return "İlgili içerik bulunamadı."
        
        # Kullanılan zaman damgalarını topla
        timestamps_used = [int(segment['timestamp']) for segment in relevant_segments]
        
        # Context oluştur
        context_parts = []
        for segment in relevant_segments:
            context_parts.append(
                f"Zaman: {segment['timestamp']}s\n"
                f"İçerik: {segment['analysis']}\n"
                f"Benzerlik Skoru: {segment['similarity_score']:.3f}\n"
            )
        
        context = "\n---\n".join(context_parts)
        
        # Gemini'ye soruyu sor
        prompt = f"""
        Video içeriği hakkında şu soruyu cevaplayın:

        Soru: {question}

        Video'dan ilgili bölümler:
        {context}

        Lütfen soruyu video içeriğine dayanarak Türkçe cevaplayın. 
        Cevabınızın başında hangi saniyelerden bilgi aldığınızı belirtin (örn: "10.saniye ve 15.saniye verilerine göre").
        Eğer video'da ilgili bilgi yoksa bunu açıkça belirtin.
        """
        
        try:
            response = self.model.generate_content(prompt)
            
            # Cevabı rich formatında hazırla
            answer_with_timestamps = response.text
            
            # Kullanılan zaman damgalarını cevabın başına ekle
            timestamp_info = f"📍 **Kullanılan veriler:** {', '.join([f'{t}.saniye' for t in sorted(timestamps_used)])}\n\n"
            
            return timestamp_info + answer_with_timestamps
            
        except Exception as e:
            return f"Cevap oluşturma hatası: {str(e)}"
    
    def interactive_chat(self):
        """Interaktif sohbet modu"""
        # Video segment tablosu göster
        self.show_video_summary()
        
        # Chat başlığı
        self.console.print("\n" + "="*60)
        self.console.print(Panel.fit(
            "🤖 Video RAG Sohbet Modu\n\n"
            "Video hakkında sorular sorabilirsiniz.\n"
            "Çıkmak için 'quit', 'çık' veya 'exit' yazın.",
            style="bold blue",
            title="💬 Sohbet"
        ))
        
        while True:
            try:
                question = Prompt.ask("\n[bold cyan]🤔 Sorunuz[/bold cyan]").strip()
                
                if question.lower() in ['quit', 'çık', 'exit']:
                    self.console.print(Panel.fit("👋 Sohbet sonlandırıldı. Görüşmek üzere!", style="bold red"))
                    break
                
                if not question:
                    continue
                
                # Cevap hazırlanıyor animasyonu
                with self.console.status("[bold green]🧠 Cevap hazırlanıyor..."):
                    answer = self.answer_question(question)
                
                # Cevabı güzel formatta göster
                self.console.print("\n" + "─" * 60)
                self.console.print(Panel(
                    Markdown(answer),
                    title="🤖 Gemini'nin Cevabı",
                    title_align="left",
                    style="white"
                ))
                
            except KeyboardInterrupt:
                self.console.print("\n\n👋 Sohbet sonlandırıldı.")
                break
            except Exception as e:
                self.console.print(f"❌ Hata: [red]{e}[/red]")
    
    def show_video_summary(self):
        """Video segment özetini tablo halinde gösterir"""
        if not self.video_segments:
            return
        
        table = Table(title="📹 Video Segment Özeti")
        table.add_column("⏱️ Zaman", style="cyan", no_wrap=True)
        table.add_column("📝 Kısa Özet", style="white")
        table.add_column("🆔 Frame ID", style="dim")
        
        for segment in self.video_segments:
            # Kısa özet (ilk 100 karakter)
            short_summary = segment['analysis'][:100] + "..." if len(segment['analysis']) > 100 else segment['analysis']
            
            table.add_row(
                f"{int(segment['timestamp'])}s",
                short_summary,
                segment['frame_id']
            )
        
        self.console.print(table)


def get_video_duration(video_path: str) -> float:
    """Video süresini saniye cinsinden döner"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps
    cap.release()
    return duration

def main():
    """Ana fonksiyon"""
    console = Console()
    
    # Başlık
    console.print(Panel.fit(
        "🎬 Gemini Video RAG Sistemi\n\n"
        "Video dosyalarını analiz edip içerikten soru-cevap yapar",
        style="bold magenta",
        title="🚀 Video RAG"
    ))
    
    # API key alma
    api_key = "google.generativeai.api_key"
    if not api_key:
        console.print("❌ API anahtarı gereklidir!", style="red")
        return
    
    # Video dosyası yolu
    video_path = Prompt.ask("🎥 Video dosyası yolunu girin").strip()
    if not os.path.exists(video_path):
        console.print("❌ Video dosyası bulunamadı!", style="red")
        return
    
    # Video süresini tespit et
    duration = get_video_duration(video_path)
    console.print(f"\n⏱️ Video süresi: [bold green]{duration:.1f} saniye[/bold green]")
    
    # Önerilen frame aralığı hesapla - daha sık frame çıkarmak için
    if duration <= 30:
        suggested_interval = max(1, int(duration / 15))  # En az 15 frame için
    elif duration <= 120:
        suggested_interval = max(2, int(duration / 25))  # Orta süre için
    else:
        suggested_interval = max(3, int(duration / 40))   # Uzun videolar için
    
    console.print(f"💡 Önerilen frame aralığı: [bold yellow]{suggested_interval} saniye[/bold yellow] (yaklaşık [cyan]{int(duration/suggested_interval)} frame[/cyan])")
    
    # Frame aralığı - direkt önerilen değeri kullan
    interval_input = Prompt.ask(f"⚙️ Frame çıkarma aralığı (saniye)", default=str(suggested_interval)).strip()
    if interval_input:
        try:
            interval = int(interval_input)
        except ValueError:
            interval = suggested_interval
    else:
        interval = suggested_interval
    
    # RAG sistemini başlat
    rag_system = VideoRAGSystem(api_key)
    
    try:
        # Video'yu işle
        console.print(f"\n🎬 Video işleme başlatılıyor...")
        segments = rag_system.process_video(video_path, interval)
        
        # Embedding'leri oluştur
        rag_system.create_embeddings()
        
        console.print(f"\n✅ Video başarıyla işlendi! [bold green]{len(segments)}[/bold green] segment analiz edildi.")
        
        # Interaktif mod
        rag_system.interactive_chat()
        
    except KeyboardInterrupt:
        console.print("\n\n⚠️ İşlem iptal edildi.")
    except Exception as e:
        console.print(f"❌ Hata: [red]{e}[/red]")


if __name__ == "__main__":
    main()
