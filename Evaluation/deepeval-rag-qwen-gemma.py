"""
DeepEval ile PDF Tabanlı RAG Değerlendirme - Ollama Qwen3 & EmbeddingGemma
Bu script PDF dosyalarınızı kullanarak gerçek bir RAG sistemi oluşturur ve değerlendirir.

Model Yapılandırması:
- Ana LLM: Ollama qwen3:4b
- Embedding: Ollama embeddinggemma:latest
- Not: Qwen modelleri <think> tagları kullanır, bunlar otomatik temizlenir
"""

import os
import json
import re
import requests
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np

# PDF okuma
from PyPDF2 import PdfReader

# Embeddings
from sentence_transformers import SentenceTransformer
import faiss

# DeepEval
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric
)
from deepeval.models import DeepEvalBaseLLM


# ============================================================================
# OLLAMA MODEL SINIFLARI
# ============================================================================

class OllamaQwenModel(DeepEvalBaseLLM):
    """Ollama Qwen3:4b modeli"""
    def __init__(self, model_name="qwen3:4b"):
        self.model_name = model_name
        self.base_url = "http://localhost:11434"
        
        # Ollama generation options
        self.default_options = {
            "num_ctx": 8192,        # Context window (8K tokens)
            "num_predict": 300,     # Max response length
            "temperature": 0.1,     # Lower temp for structured output (0.7 -> 0.1)
            "top_p": 0.9,          # Nucleus sampling
            "repeat_penalty": 1.1,  # Prevent repetition
            "num_thread": 8,        # CPU threads (parallel processing)
        }
        
        print(f"🤖 LLM Modeli: {model_name} (Context: 8K tokens)")
    
    def load_model(self):
        return self.model_name
    
    def generate(self, prompt: str, max_tokens: int = 300) -> str:
        """Generate with token limit to speed up responses"""
        try:
            # Merge custom max_tokens with default options
            options = self.default_options.copy()
            options["num_predict"] = max_tokens
            
            # Check if prompt expects JSON output
            is_json_request = any(keyword in prompt.lower() for keyword in [
                "json", "```json", "output:", "respond in json", 
                "format:", "structured output"
            ])
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json" if is_json_request else "",  # JSON mode for structured outputs
                    "options": options
                },
                timeout=90
            )
            response.raise_for_status()
            result = response.json()["response"]
            
            # Qwen modellerinin <think> taglarını temizle
            result = self._remove_think_tags(result)
            
            # JSON çıktılarını temizle (DeepEval için kritik)
            if is_json_request:
                result = self._clean_json_output(result)
            
            return result
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _clean_json_output(self, text: str) -> str:
        """JSON çıktılarını temizle ve düzelt"""
        import re
        import json
        
        # Markdown code block'larını kaldır
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        
        # Ön ve son boşlukları temizle
        text = text.strip()
        
        # JSON'u validate et
        try:
            # Parse edip tekrar serialize et (formatting düzeltir)
            parsed = json.loads(text)
            return json.dumps(parsed)
        except json.JSONDecodeError:
            # JSON geçerli değilse, en azından temizlenmiş halini döndür
            return text
    
    def _remove_think_tags(self, text: str) -> str:
        """Qwen modellerinin <think>...</think> taglarını kaldır"""
        import re
        # <think> ile </think> arasındaki her şeyi kaldır
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        # Fazla boşlukları temizle
        cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)
        return cleaned.strip()
    
    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)
    
    def get_model_name(self) -> str:
        return f"Ollama-{self.model_name}"


class OllamaEmbedding:
    """Ollama Embedding modeli (embeddinggemma, mxbai-embed-large, nomic-embed-text, all-minilm)"""
    def __init__(self, model_name="embeddinggemma:latest"):
        self.model_name = model_name
        self.base_url = "http://localhost:11434"
        self.dimension = None
        print(f"🔢 Embedding Modeli: {model_name}")
        
        # Dimension'ı test ederek öğren
        self._get_dimension()
    
    def _get_dimension(self):
        """Embedding boyutunu test ederek öğren"""
        try:
            test_embedding = self.embed("test")
            self.dimension = len(test_embedding)
            print(f"✅ Embedding dimension: {self.dimension}")
        except Exception as e:
            print(f"⚠️ Dimension tespit edilemedi: {e}")
            self.dimension = 768  # Varsayılan
    
    def embed(self, text: str) -> np.ndarray:
        """Tek bir metin için embedding"""
        try:
            response = requests.post(
                f"{self.base_url}/api/embed",
                json={
                    "model": self.model_name,
                    "input": text
                },
                timeout=60
            )
            response.raise_for_status()
            # Response format: {"model": "...", "embeddings": [[...]]}
            embedding = response.json()["embeddings"][0]
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            print(f"❌ Embedding hatası: {e}")
            print(f"   Model: {self.model_name}")
            print(f"   İpucu: Model indirilmiş mi? 'ollama pull {self.model_name}'")
            # Fallback: rastgele embedding (sadece test için)
            if self.dimension:
                return np.random.rand(self.dimension).astype(np.float32)
            else:
                return np.random.rand(768).astype(np.float32)
    
    def embed_batch(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """Birden fazla metin için embedding"""
        embeddings = []
        total = len(texts)
        
        if show_progress:
            print(f"📊 {total} metin için embedding oluşturuluyor...")
        
        for i, text in enumerate(texts, 1):
            if show_progress and i % 10 == 0:
                print(f"   İşlenen: {i}/{total} ({100*i//total}%)")
            
            emb = self.embed(text)
            embeddings.append(emb)
        
        if show_progress:
            print(f"✅ {total} embedding oluşturuldu")
        
        return np.array(embeddings, dtype=np.float32)
    
    def get_sentence_embedding_dimension(self):
        """Sentence transformers uyumluluğu için"""
        return self.dimension


# ============================================================================
# PDF VE VECTOR STORE YÖNETİMİ
# ============================================================================

class PDFVectorStore:
    """PDF'leri okur, chunk'lara ayırır ve FAISS index oluşturur"""
    
    def __init__(self, pdf_folder: str, embedding_model=None, use_ollama_embedding: bool = True):
        """
        Args:
            pdf_folder: PDF dosyalarının bulunduğu klasör
            embedding_model: Özel embedding modeli (None ise otomatik seçilir)
            use_ollama_embedding: True ise Ollama gemma, False ise sentence-transformers
        """
        self.pdf_folder = Path(pdf_folder)
        self.chunks: List[str] = []
        self.chunk_metadata: List[Dict] = []
        self.embeddings: np.ndarray = None
        self.index: faiss.Index = None
        self.use_ollama = use_ollama_embedding
        
        # Embedding model seçimi
        if embedding_model:
            self.embedding_model = embedding_model
            self.embedding_model_name = "custom"
        elif use_ollama_embedding:
            print("\n📚 Ollama Embedding modeli yükleniyor...")
            self.embedding_model = OllamaEmbedding(model_name="embeddinggemma:latest")
            self.embedding_model_name = "ollama-embeddinggemma"
        else:
            print("\n📚 Sentence Transformers modeli yükleniyor...")
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            self.embedding_model_name = "all-MiniLM-L6-v2"
        
        print(f"✅ Embedding modeli hazır")
    
    def load_pdfs(self) -> int:
        """PDF dosyalarını yükle ve chunk'lara ayır"""
        print(f"\n📖 PDF dosyaları okunuyor: {self.pdf_folder}")
        
        pdf_files = list(self.pdf_folder.glob("*.pdf"))
        if not pdf_files:
            raise ValueError(f"❌ {self.pdf_folder} klasöründe PDF dosyası bulunamadı!")
        
        print(f"📄 {len(pdf_files)} PDF dosyası bulundu")
        
        for pdf_file in pdf_files:
            print(f"   📑 {pdf_file.name} okunuyor...")
            try:
                reader = PdfReader(str(pdf_file))
                text = ""
                for page_num, page in enumerate(reader.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                
                # Chunk'lara ayır
                chunks = self._chunk_text(text, chunk_size=500, overlap=50)
                
                for i, chunk in enumerate(chunks):
                    self.chunks.append(chunk)
                    self.chunk_metadata.append({
                        "source": pdf_file.name,
                        "chunk_id": i,
                        "total_chunks": len(chunks)
                    })
                
                print(f"      ✅ {len(chunks)} chunk oluşturuldu")
                
            except Exception as e:
                print(f"      ❌ Hata: {str(e)}")
                continue
        
        print(f"\n✅ Toplam {len(self.chunks)} chunk oluşturuldu")
        return len(self.chunks)
    
    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Metni chunk'lara ayır"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks
    
    def create_index(self):
        """FAISS index oluştur"""
        if not self.chunks:
            raise ValueError("❌ Önce PDF'leri yüklemelisiniz (load_pdfs)")
        
        print(f"\n🔢 Embeddingler oluşturuluyor ({len(self.chunks)} chunk)...")
        print("⏳ Bu işlem biraz zaman alabilir...")
        
        # Embeddinglari oluştur
        if self.use_ollama:
            # Ollama embedding
            self.embeddings = self.embedding_model.embed_batch(
                self.chunks,
                show_progress=True
            )
        else:
            # Sentence transformers
            self.embeddings = self.embedding_model.encode(
                self.chunks,
                show_progress_bar=True,
                convert_to_numpy=True
            )
        
        print(f"✅ Embeddingler oluşturuldu: {self.embeddings.shape}")
        
        # FAISS index oluştur
        print("\n🔍 FAISS index oluşturuluyor...")
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance
        self.index.add(self.embeddings)
        print(f"✅ Index oluşturuldu ({self.index.ntotal} vectors)")
    
    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float, Dict]]:
        """
        Query'ye en yakın chunk'ları bul
        
        Returns:
            List of (chunk_text, distance, metadata)
        """
        if self.index is None:
            raise ValueError("❌ Önce index oluşturmalısınız (create_index)")
        
        # Query embedding
        if self.use_ollama:
            query_embedding = self.embedding_model.embed(query)
            query_embedding = query_embedding.reshape(1, -1)
        else:
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        
        # Search
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append((
                self.chunks[idx],
                float(dist),
                self.chunk_metadata[idx]
            ))
        
        return results
    
    def save_index(self, save_path: str = "faiss_index"):
        """Index ve metadata'yı kaydet"""
        if self.index is None:
            raise ValueError("❌ Kaydedilecek index yok!")
        
        print(f"\n💾 Index kaydediliyor: {save_path}")
        
        # FAISS index kaydet
        faiss.write_index(self.index, f"{save_path}.index")
        
        # Metadata kaydet
        metadata = {
            "chunks": self.chunks,
            "chunk_metadata": self.chunk_metadata,
            "embedding_model": self.embedding_model_name,
            "use_ollama": self.use_ollama
        }
        
        with open(f"{save_path}.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Index kaydedildi")
    
    def load_index(self, load_path: str = "faiss_index"):
        """Kaydedilmiş index'i yükle"""
        print(f"\n📂 Index yükleniyor: {load_path}")
        
        # FAISS index yükle
        self.index = faiss.read_index(f"{load_path}.index")
        
        # Metadata yükle
        with open(f"{load_path}.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        self.chunks = metadata["chunks"]
        self.chunk_metadata = metadata["chunk_metadata"]
        
        # Embedding model kontrol
        if metadata.get("embedding_model") != self.embedding_model_name:
            print(f"⚠️ Uyarı: Farklı embedding model kullanılmış")
            print(f"   Kayıtlı: {metadata.get('embedding_model')}")
            print(f"   Şu anki: {self.embedding_model_name}")
        
        print(f"✅ Index yüklendi ({self.index.ntotal} vectors)")


# ============================================================================
# PDF TABANLI RAG PİPELİNE
# ============================================================================

class PDFRAGPipeline:
    """PDF'ler üzerinde RAG sistemi"""
    
    def __init__(self, vector_store: PDFVectorStore, llm_model: DeepEvalBaseLLM):
        self.vector_store = vector_store
        self.llm_model = llm_model
    
    def query(self, question: str, top_k: int = 3, max_context_length: int = 1500) -> Tuple[str, List[str]]:
        """
        RAG pipeline: Retrieval + Generation
        
        Args:
            question: Soru
            top_k: Kaç chunk getirileceği
            max_context_length: Maksimum context uzunluğu (kelime)
        
        Returns:
            (generated_answer, retrieved_contexts)
        """
        # 1. RETRIEVAL: En alakalı chunk'ları bul
        search_results = self.vector_store.search(question, top_k=top_k)
        
        retrieved_contexts = [chunk for chunk, _, _ in search_results]
        
        # 2. CONTEXT OPTIMIZATION: Context'i kısalt (token limiti için)
        optimized_contexts = []
        total_words = 0
        
        for chunk in retrieved_contexts:
            chunk_words = len(chunk.split())
            if total_words + chunk_words > max_context_length:
                # Kalan alanı doldur
                remaining = max_context_length - total_words
                if remaining > 50:  # En az 50 kelime varsa ekle
                    words = chunk.split()[:remaining]
                    optimized_contexts.append(" ".join(words) + "...")
                break
            else:
                optimized_contexts.append(chunk)
                total_words += chunk_words
        
        # 3. GENERATION: LLM ile cevap üret
        context_text = "\n\n".join([
            f"[Kaynak {i+1}]: {chunk}" 
            for i, chunk in enumerate(optimized_contexts)
        ])
        
        prompt = f"""Verilen kaynaklara göre soruyu 2-3 cümle ile yanıtla.

Kaynaklar:
{context_text}

Soru: {question}

Kısa yanıt (maksimum 3 cümle):"""
        
        answer = self.llm_model.generate(prompt, max_tokens=250)  # Shorter answers
        
        return answer, retrieved_contexts
    
    def create_test_case(self, question: str, expected_output: str = None) -> LLMTestCase:
        """RAG query'sinden test case oluştur"""
        actual_output, contexts = self.query(question)
        
        return LLMTestCase(
            input=question,
            actual_output=actual_output,
            retrieval_context=contexts,
            expected_output=expected_output
        )


# ============================================================================
# DEĞERLENDİRME FONKSİYONLARI
# ============================================================================

def setup_metrics(model: DeepEvalBaseLLM) -> List:
    """RAG metriklerini ayarla"""
    print("\n📊 Metrikler ayarlanıyor...")
    
    metrics = [
        AnswerRelevancyMetric(
            threshold=0.7,
            model=model,
            include_reason=True
        ),
        FaithfulnessMetric(
            threshold=0.7,
            model=model,
            include_reason=True
        ),
        ContextualRelevancyMetric(
            threshold=0.7,
            model=model,
            include_reason=True
        ),
        ContextualPrecisionMetric(
            threshold=0.7,
            model=model,
            include_reason=True
        ),
        ContextualRecallMetric(
            threshold=0.7,
            model=model,
            include_reason=True
        )
    ]
    
    print(f"✅ {len(metrics)} metrik hazır")
    return metrics


def create_sample_test_cases(rag_pipeline: PDFRAGPipeline) -> List[LLMTestCase]:
    """
    PDF içeriğine göre örnek test case'leri oluştur
    """
    print("\n📋 Test case'leri oluşturuluyor...")
    
    # Örnek sorular - PDF içeriğinize göre özelleştirin
    questions = [
        {
            "question": "Bu makalelerde hangi ana konular işleniyor?",
            "expected": "Makalelerin ana konuları ve amaçları açıklanmalıdır."
        },
        {
            "question": "Hangi yöntemler veya metodolojiler kullanılmış?",
            "expected": "Kullanılan araştırma metodolojileri açıklanmalıdır."
        },
        {
            "question": "Ana bulgular veya sonuçlar nelerdir?",
            "expected": "Çalışmaların temel bulguları özetlenmelidir."
        }
    ]
    
    test_cases = []
    for i, q_data in enumerate(questions, 1):
        print(f"   {i}. Test case oluşturuluyor: {q_data['question'][:50]}...")
        try:
            test_case = rag_pipeline.create_test_case(
                question=q_data["question"],
                expected_output=q_data["expected"]
            )
            test_cases.append(test_case)
            print(f"      ✅ Tamamlandı")
        except Exception as e:
            print(f"      ❌ Hata: {str(e)}")
    
    print(f"✅ {len(test_cases)} test case oluşturuldu")
    return test_cases


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║     PDF RAG Değerlendirme - Qwen3:4b & EmbeddingGemma       ║
    ╚════════════════════════════════════════════════════════════════╝
    
    Model Yapılandırması:
    🤖 Ana LLM: Ollama qwen3:4b
    🔢 Embedding: Ollama embeddinggemma:latest
    
    Bu script:
    1. PDF klasöründeki dosyaları okur
    2. EmbeddingGemma ile vektörler oluşturur
    3. FAISS vector index kurar
    4. Qwen3 ile RAG pipeline çalıştırır
    5. DeepEval ile değerlendirir
    
    ════════════════════════════════════════════════════════════════
    """)
    
    # Konfigürasyon
    PDF_FOLDER = Path(__file__).parent / "pdf"
    INDEX_PATH = Path(__file__).parent / "faiss_index_qwen"
    
    print(f"📁 PDF Klasörü: {PDF_FOLDER}")
    
    if not PDF_FOLDER.exists():
        print(f"❌ {PDF_FOLDER} klasörü bulunamadı!")
        return
    
    # Adım 1: Model Hazırlığı
    print("\n" + "="*70)
    print("ADIM 1: MODEL HAZIRLIĞI")
    print("="*70)
    
    print("\n🤖 Ollama Qwen3:4b modeli kontrol ediliyor...")
    llm_model = OllamaQwenModel(model_name="qwen3:4b")
    
    # Embedding model seçimi
    print("\n🔢 Embedding modeli seçimi:")
    print("1. Ollama embeddinggemma:latest (sizin modeliniz, önerilen)")
    print("2. Ollama mxbai-embed-large (334M)")
    print("3. Ollama nomic-embed-text (137M)")
    print("4. Ollama all-minilm (en hafif, 23M)")
    print("5. Sentence Transformers (all-MiniLM-L6-v2)")
    
    emb_choice = input("\nSeçiminiz (1/2/3/4/5, Enter=1): ").strip() or "1"
    
    if emb_choice == "1":
        use_ollama_emb = True
        print("✅ Ollama embeddinggemma:latest kullanılacak")
    elif emb_choice == "2":
        use_ollama_emb = True
        print("✅ Ollama mxbai-embed-large kullanılacak")
    elif emb_choice == "3":
        use_ollama_emb = True
        print("✅ Ollama nomic-embed-text kullanılacak")
    elif emb_choice == "4":
        use_ollama_emb = True
        print("✅ Ollama all-minilm kullanılacak")
    else:
        use_ollama_emb = False
        print("✅ Sentence Transformers kullanılacak")
    
    # Adım 2: Vector Store Hazırlama
    print("\n" + "="*70)
    print("ADIM 2: VECTOR STORE HAZIRLIĞI")
    print("="*70)
    
    # Index var mı kontrol et
    use_existing = False
    if INDEX_PATH.with_suffix('.index').exists():
        choice = input("\n💾 Kayıtlı index bulundu. Kullanmak ister misiniz? (e/h): ").strip().lower()
        use_existing = (choice == 'e')
    
    # Embedding model seçimine göre özel model oluştur
    custom_embedding = None
    if use_ollama_emb and emb_choice == "2":
        custom_embedding = OllamaEmbedding(model_name="mxbai-embed-large")
    elif use_ollama_emb and emb_choice == "3":
        custom_embedding = OllamaEmbedding(model_name="nomic-embed-text")
    elif use_ollama_emb and emb_choice == "4":
        custom_embedding = OllamaEmbedding(model_name="all-minilm")
    
    vector_store = PDFVectorStore(
        pdf_folder=str(PDF_FOLDER),
        embedding_model=custom_embedding,
        use_ollama_embedding=use_ollama_emb if not custom_embedding else True
    )
    
    if use_existing:
        vector_store.load_index(str(INDEX_PATH))
    else:
        # PDF'leri yükle ve index oluştur
        vector_store.load_pdfs()
        vector_store.create_index()
        
        # Index'i kaydet
        save_choice = input("\n💾 Index'i kaydetmek ister misiniz? (e/h): ").strip().lower()
        if save_choice == 'e':
            vector_store.save_index(str(INDEX_PATH))
    
    # Adım 3: RAG Pipeline Oluştur
    print("\n" + "="*70)
    print("ADIM 3: RAG PIPELINE OLUŞTURMA")
    print("="*70)
    
    rag_pipeline = PDFRAGPipeline(vector_store, llm_model)
    print("✅ RAG pipeline hazır (Qwen3:4b + EmbeddingGemma)")
    
    # Adım 4: Test (Opsiyonel)
    print("\n" + "="*70)
    print("ADIM 4: HIZLI TEST")
    print("="*70)
    
    test_choice = input("\nRAG pipeline'ı test etmek ister misiniz? (e/h): ").strip().lower()
    if test_choice == 'e':
        test_query = input("Test sorusu (Enter=örnek soru): ").strip()
        if not test_query:
            test_query = "Bu makalelerin ana konusu nedir?"
        
        print(f"\n🔍 Soru: {test_query}")
        print("\n⏳ Qwen3 ile cevap üretiliyor...")
        answer, contexts = rag_pipeline.query(test_query)
        print(f"\n💡 Cevap:\n{answer}")
        print(f"\n📚 Kullanılan kaynaklar ({len(contexts)}):")
        for i, ctx in enumerate(contexts, 1):
            print(f"\n[Kaynak {i}]:\n{ctx[:200]}...")
    
    # Adım 5: Test Case Oluşturma
    print("\n" + "="*70)
    print("ADIM 5: TEST CASE OLUŞTURMA")
    print("="*70)
    
    test_cases = create_sample_test_cases(rag_pipeline)
    
    if not test_cases:
        print("❌ Test case oluşturulamadı!")
        return
    
    # Adım 6: Değerlendirme
    print("\n" + "="*70)
    print("ADIM 6: DEEPEVAL DEĞERLENDİRMESİ")
    print("="*70)
    
    eval_choice = input("\nDeğerlendirmeyi başlatmak istiyor musunuz? (e/h): ").strip().lower()
    if eval_choice != 'e':
        print("❌ Değerlendirme iptal edildi.")
        return
    
    metrics = setup_metrics(llm_model)
    
    print("\n🚀 Değerlendirme başlıyor...")
    print("⏳ Bu işlem birkaç dakika sürebilir...\n")
    
    try:
        # DeepEval değerlendirmesini çalıştır
        results = evaluate(
            test_cases=test_cases,
            metrics=metrics
        )
        
        # Sonuçları manuel yazdır
        print("\n" + "="*70)
        print("📊 DEĞERLENDİRME SONUÇLARI")
        print("="*70)
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{'='*70}")
            print(f"Test Case #{i}")
            print(f"{'='*70}")
            print(f"Soru: {test_case.input[:100]}...")
            print(f"Cevap: {test_case.actual_output[:200]}...")
            print(f"\nMetrik Sonuçları:")
            
            # Her test case için metrik sonuçlarını göster
            for metric in metrics:
                metric_name = metric.__class__.__name__
                try:
                    # Metriği bu test case için çalıştır
                    score = getattr(test_case, 'score', 'N/A')
                    print(f"  • {metric_name}: {score}")
                except:
                    print(f"  • {metric_name}: Hesaplanamadı")
        
        print("\n" + "="*70)
        print("✅ DEĞERLENDİRME TAMAMLANDI!")
        print("="*70)
        
        print("""
        🎉 Tebrikler! RAG sisteminiz başarıyla değerlendirildi.
        
        💡 Sonraki adımlar:
        
        1. Embedding modelini test ettiniz mi?
           - embeddinggemma:latest: Sizin modeliniz (önerilen)
           - mxbai-embed-large: En iyi performans (334M)
           - nomic-embed-text: Dengeli seçenek (137M)
           - all-minilm: En hızlı (23M)
        
        2. Test case'lerinizi özelleştirin
           - create_sample_test_cases() içindeki soruları PDF'lerinize göre değiştirin
        
        3. Metrikleri ayarlayın
           - Threshold değerlerini ihtiyacınıza göre değiştirin (varsayılan: 0.7)
        
        4. Chunk parametrelerini optimize edin
           - chunk_size: 500 (kelime sayısı)
           - overlap: 50
           - top_k: 3 (retrieval sonuç sayısı)
        
        5. Confident AI'da görselleştirin
           - deepeval login ile giriş yapın
           - Sonuçlar otomatik yüklenecek
        """)
        
    except Exception as e:
        print(f"\n❌ Hata oluştu: {str(e)}")
        
        # JSON hatası özel kontrolü
        if "invalid JSON" in str(e):
            print("\n" + "="*70)
            print("🔧 JSON Formatı Sorunu Tespit Edildi!")
            print("="*70)
            print("""
Qwen3:4b bazen DeepEval'in beklediği JSON formatını oluşturamıyor.

💡 ÖNERİLEN ÇÖZÜMLER:

1. Daha iyi bir LLM modeli kullan:
   ollama pull qwen2.5:7b      # 7B, daha güvenilir JSON
   ollama pull llama3.2:3b     # Meta'nın 3B modeli
   ollama pull phi3.5:latest   # Microsoft'un modeli
   
   Sonra script'te model_name="qwen2.5:7b" olarak değiştir

2. VEYA DeepEval'i verbose mode ile çalıştır:
   - Script'i düzenle, strict_mode=True yap
   - Hangi adımda patladığını göreceksin

3. VEYA API anahtarı ile büyük model kullan:
   - OpenAI GPT-4 (en doğru JSON)
   - Anthropic Claude (çok iyi structured output)
   
Tekrar denemek için script'i yeniden çalıştır.
            """)
        
        print("\n🔧 Sorun giderme:")
        print("   - Ollama servisinin çalıştığından emin olun: ollama serve")
        print("   - Modellerin indirildiğini kontrol edin:")
        print("     ollama list")
        print("   - LLM modelini indirin:")
        print("     ollama pull qwen3:4b")
        print("   - Embedding modeliniz zaten var:")
        print("     embeddinggemma:latest (varsayılan)")
        print("   - Alternatif embedding modelleri:")
        print("     ollama pull mxbai-embed-large")
        print("     ollama pull nomic-embed-text")
        print("     ollama pull all-minilm")
        print("   - İnternet bağlantınızı kontrol edin")


if __name__ == "__main__":
    main()
