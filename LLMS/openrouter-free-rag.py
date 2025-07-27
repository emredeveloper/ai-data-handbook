import json
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from openai import OpenAI
from google import genai
import PyPDF2
import os
import time

# DeepEval imports
try:
    from deepeval import evaluate
    from deepeval.metrics import AnswerRelevancyMetric, ContextualRelevancyMetric, ContextualRecallMetric, FaithfulnessMetric
    from deepeval.test_case import LLMTestCase
    DEEPEVAL_AVAILABLE = True
    print("✅ DeepEval başarıyla yüklendi!")
except ImportError as e:
    print(f"⚠️ DeepEval import hatası: {e}")
    print("🔄 Alternatif import deneniyor...")
    try:
        from deepeval import evaluate
        from deepeval.metrics import AnswerRelevancyMetric, ContextualRelevancyMetric, ContextualRecallMetric, FaithfulnessMetric
        from deepeval import LLMTestCase
        DEEPEVAL_AVAILABLE = True
        print("✅ DeepEval alternatif import ile yüklendi!")
    except ImportError as e2:
        print(f"❌ DeepEval import başarısız: {e2}")
        DEEPEVAL_AVAILABLE = False

@dataclass
class Document:
    id: str
    content: str
    metadata: Dict[str, Any]
    embeddings: List[float] = None

@dataclass
class GraphNode:
    id: str
    content: str
    node_type: str  # 'document', 'entity', 'concept'
    embeddings: List[float] = None
    metadata: Dict[str, Any] = None

class ModernGraphRAG:
    def __init__(self, openrouter_api_key: str, gemini_api_key: str = None):
        # OpenRouter client (LLM için)
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
        )
        
        # Gemini client (embedding için)
        if gemini_api_key:
            self.gemini_client = genai.Client(api_key=gemini_api_key)
        else:
            # API key yoksa None olarak bırak, fallback kullanılacak
            self.gemini_client = None
        
        # Graph bileşenleri
        self.knowledge_graph = nx.Graph()
        self.documents = {}
        self.entity_nodes = {}
        self.concept_nodes = {}
        
        # Embedding ve retrieval bileşenleri
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.document_embeddings = {}
        self.all_texts = []  # Tüm metinleri sakla
        self.tfidf_fitted = False
        
        # RAG parametreleri
        self.top_k = 3  # Daha az belge al
        self.similarity_threshold = 0.7
        
        # DeepEval için test case'leri sakla
        self.test_cases = []
        
    def load_pdf_document(self, pdf_path: str, chunk_size: int = 500):  # Chunk boyutunu küçült
        """PDF dosyasını oku ve chunk'lara böl"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                full_text = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    full_text += f"\n--- Sayfa {page_num + 1} ---\n{page_text}\n"
                
                # Metni chunk'lara böl
                chunks = self._split_text_into_chunks(full_text, chunk_size)
                
                print(f"✅ PDF yüklendi: {pdf_path}")
                print(f"✅ Toplam sayfa: {len(pdf_reader.pages)}")
                print(f"✅ Toplam chunk: {len(chunks)}")
                
                # Her chunk'ı belge olarak ekle
                for i, chunk in enumerate(chunks):
                    doc_id = f"pdf_chunk_{i+1}"
                    metadata = {
                        "source": pdf_path,
                        "chunk_id": i+1,
                        "total_chunks": len(chunks),
                        "type": "pdf_document"
                    }
                    self.add_document(doc_id, chunk, metadata)
                    
                # TF-IDF vectorizer'ı fit et
                self._fit_tfidf()
                    
                return len(chunks)
                
        except Exception as e:
            print(f"❌ PDF okuma hatası: {e}")
            return 0
    
    def _split_text_into_chunks(self, text: str, chunk_size: int) -> List[str]:
        """Metni anlamlı chunk'lara böl"""
        # Önce paragraflara böl
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) < chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"
        
        # Son chunk'ı ekle
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
        
    def add_document(self, doc_id: str, content: str, metadata: Dict = None):
        """Belgeyi sisteme ekle ve graph'a entegre et"""
        # Belgeyi kaydet
        doc = Document(
            id=doc_id,
            content=content,
            metadata=metadata or {}
        )
        self.documents[doc_id] = doc
        
        # Graph node oluştur
        graph_node = GraphNode(
            id=doc_id,
            content=content,
            node_type='document',
            metadata=metadata or {}
        )
        
        # Graph'a ekle
        self.knowledge_graph.add_node(doc_id, **graph_node.__dict__)
        
        # Entity ve concept extraction
        self._extract_entities_and_concepts(doc_id, content)
        
        # Metni sakla (TF-IDF için)
        self.all_texts.append(content)
        
    def _fit_tfidf(self):
        """TF-IDF vectorizer'ı fit et"""
        try:
            # Tüm metinleri kullanarak TF-IDF fit et
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.all_texts)
            
            # Her belge için embedding oluştur
            for i, doc_id in enumerate(self.documents.keys()):
                embedding = tfidf_matrix[i].toarray()[0]
                self.document_embeddings[doc_id] = embedding
                print(f"📝 {doc_id} için TF-IDF embedding oluşturuldu")
            
            self.tfidf_fitted = True
            print("✅ TF-IDF vectorizer fit edildi")
            
        except Exception as e:
            print(f"❌ TF-IDF fit hatası: {e}")
    
    def _extract_entities_and_concepts(self, doc_id: str, content: str):
        """Belgeden entity ve concept'leri çıkar"""
        # Basit entity extraction (gerçek uygulamada NER kullanılabilir)
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
        
        for entity in entities[:5]:  # İlk 5 entity'yi al (daha az)
            if entity not in self.entity_nodes:
                entity_node = GraphNode(
                    id=f"entity_{entity}",
                    content=entity,
                    node_type='entity'
                )
                self.entity_nodes[entity] = entity_node
                self.knowledge_graph.add_node(f"entity_{entity}", **entity_node.__dict__)
            
            # Belge ile entity arasında edge oluştur
            self.knowledge_graph.add_edge(doc_id, f"entity_{entity}", weight=1.0)
    
    def hybrid_search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Hybrid search: semantic + graph-based retrieval"""
        # 1. Semantic search
        semantic_results = self._semantic_search(query, top_k)
        
        # 2. Graph-based search
        graph_results = self._graph_search(query, top_k)
        
        # 3. Results fusion
        combined_results = self._fuse_results(semantic_results, graph_results)
        
        return combined_results[:top_k]
    
    def _semantic_search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """TF-IDF tabanlı arama"""
        if not self.tfidf_fitted:
            print("❌ TF-IDF henüz fit edilmemiş!")
            return []
            
        try:
            # Query için TF-IDF embedding oluştur
            query_vector = self.tfidf_vectorizer.transform([query])
            
            # Similarity hesapla
            similarities = []
            for doc_id, doc_embedding in self.document_embeddings.items():
                if doc_embedding is not None:
                    # Cosine similarity hesapla
                    similarity = cosine_similarity(query_vector, [doc_embedding])[0][0]
                    similarities.append((doc_id, similarity))
            
            # Top-k döndür
            return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
            
        except Exception as e:
            print(f"❌ TF-IDF search hatası: {e}")
            return []
    
    def _graph_search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """Graph-based retrieval"""
        # Query'deki entity'leri bul
        query_entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        
        graph_scores = defaultdict(float)
        
        for entity in query_entities:
            entity_node_id = f"entity_{entity}"
            if entity_node_id in self.knowledge_graph:
                # Entity'ye bağlı belgeleri bul
                neighbors = list(self.knowledge_graph.neighbors(entity_node_id))
                for neighbor in neighbors:
                    if neighbor in self.documents:
                        # Graph distance'a göre score hesapla
                        try:
                            distance = nx.shortest_path_length(
                                self.knowledge_graph, 
                                entity_node_id, 
                                neighbor
                            )
                            score = 1.0 / (1.0 + distance)
                            graph_scores[neighbor] += score
                        except nx.NetworkXNoPath:
                            continue
        
        # Score'ları normalize et ve sırala
        results = [(doc_id, score) for doc_id, score in graph_scores.items()]
        return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]
    
    def _fuse_results(self, semantic_results: List[Tuple[str, float]], 
                     graph_results: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Semantic ve graph sonuçlarını birleştir"""
        combined_scores = defaultdict(float)
        
        # Semantic results
        for doc_id, score in semantic_results:
            combined_scores[doc_id] += score * 0.6  # Semantic weight
        
        # Graph results
        for doc_id, score in graph_results:
            combined_scores[doc_id] += score * 0.4  # Graph weight
        
        # Sırala ve döndür
        results = [(doc_id, score) for doc_id, score in combined_scores.items()]
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def generate_response(self, query: str) -> str:
        """Modern RAG ile yanıt oluştur"""
        try:
            # 1. Hybrid search ile relevant belgeleri bul
            relevant_docs = self.hybrid_search(query, self.top_k)
            
            if not relevant_docs:
                return "Üzgünüm, sorunuzla ilgili yeterli bilgi bulamadım."
            
            # 2. Context oluştur
            context = self._build_context(relevant_docs)
            
            # 3. Graph-aware prompt oluştur
            prompt = self._create_graph_aware_prompt(query, context)
            
            # 4. LLM ile yanıt oluştur (timeout ile)
            completion = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://my-rag-app.com",
                    "X-Title": "Modern Graph RAG System",
                },
                model="qwen/qwen3-coder:free",
                messages=[
                    {
                        "role": "system",
                        "content": "Sen modern bir RAG sistemi için asistan. Verilen context'i kullanarak doğru ve kapsamlı yanıtlar ver. PDF içeriğine dayalı yanıtlar ver."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=500,  # Daha kısa yanıt
                timeout=30  # 30 saniye timeout
            )
            return completion.choices[0].message.content
            
        except Exception as e:
            return f"Yanıt oluşturma hatası: {e}"
    
    def _build_context(self, relevant_docs: List[Tuple[str, float]]) -> str:
        """Relevant belgelerden context oluştur"""
        context_parts = []
        
        for doc_id, score in relevant_docs:
            if doc_id in self.documents:
                doc = self.documents[doc_id]
                context_parts.append(f"Belge {doc_id} (Relevance: {score:.3f}):\n{doc.content[:300]}...")  # Daha kısa context
        
        return "\n\n".join(context_parts)
    
    def _create_graph_aware_prompt(self, query: str, context: str) -> str:
        """Graph bilgilerini içeren prompt oluştur"""
        prompt = f"""
Context Bilgileri:
{context}

Kullanıcı Sorusu: {query}

Lütfen yukarıdaki context bilgilerini kullanarak kullanıcının sorusunu yanıtla. 
PDF içeriğine dayalı olarak doğru ve kapsamlı yanıtlar ver.
"""
        return prompt

    def get_graph_info(self) -> Dict[str, Any]:
        """Graph hakkında bilgi döndür"""
        return {
            "total_nodes": self.knowledge_graph.number_of_nodes(),
            "total_edges": self.knowledge_graph.number_of_edges(),
            "documents": len(self.documents),
            "entities": len(self.entity_nodes),
            "concepts": len(self.concept_nodes)
        }

    # DeepEval entegrasyonu
    def add_test_case(self, query: str, expected_answer: str = None, context: str = None):
        """Test case ekle"""
        if not DEEPEVAL_AVAILABLE:
            print("❌ DeepEval yüklü değil!")
            return
            
        # Gerçek yanıtı al
        actual_answer = self.generate_response(query)
        
        # Context'i al
        relevant_docs = self.hybrid_search(query, self.top_k)
        actual_context = self._build_context(relevant_docs)
        
        # Test case oluştur
        test_case = LLMTestCase(
            input=query,
            actual_output=actual_answer,
            expected_output=expected_answer,
            context=actual_context
        )
        
        self.test_cases.append(test_case)
        print(f"✅ Test case eklendi: {query}")
        
    def evaluate_rag_system(self):
        """RAG sistemini DeepEval ile değerlendir"""
        if not DEEPEVAL_AVAILABLE:
            print("❌ DeepEval yüklü değil!")
            return
            
        if not self.test_cases:
            print("❌ Test case yok! Önce test case'ler ekleyin.")
            return
            
        print("🔍 RAG Sistemi Değerlendiriliyor...")
        
        # Metrikleri tanımla
        metrics = [
            AnswerRelevancyMetric(threshold=0.7),
            ContextualRelevancyMetric(threshold=0.7),
            ContextualRecallMetric(threshold=0.7),
            FaithfulnessMetric(threshold=0.7)
        ]
        
        # Değerlendirme yap
        results = evaluate(
            test_cases=self.test_cases,
            metrics=metrics
        )
        
        # Sonuçları göster
        print("\n📊 Değerlendirme Sonuçları:")
        print("=" * 50)
        
        for metric_name, score in results.items():
            status = "✅ PASS" if score >= 0.7 else "❌ FAIL"
            print(f"{metric_name}: {score:.3f} {status}")
            
        return results

def interactive_qa():
    """İnteraktif soru-cevap sistemi"""
    # API anahtarları
    OPENROUTER_API_KEY = "openrouter"
    GEMINI_API_KEY = "gemini"  # Gemini API key yoksa None bırakın
    
    # RAG sistemi oluştur
    rag = ModernGraphRAG(OPENROUTER_API_KEY, GEMINI_API_KEY)
    
    # PDF dosyasını yükle - doğru yol
    pdf_path = "../pdf_docs/2502.12134v1.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF dosyası bulunamadı: {pdf_path}")
        print(f"🔍 Mevcut dizin: {os.getcwd()}")
        print(f"📁 PDF dizini kontrol ediliyor...")
        
        # Alternatif yolları dene
        alternative_paths = [
            "pdf_docs/2502.12134v1.pdf",
            "../pdf_docs/2502.12134v1.pdf",
            "../../pdf_docs/2502.12134v1.pdf",
            "2502.12134v1.pdf"
        ]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                pdf_path = alt_path
                print(f"✅ PDF bulundu: {pdf_path}")
                break
        else:
            print("❌ PDF dosyası hiçbir yolda bulunamadı!")
            return
    
    print("✅ PDF RAG Sistemi Başlatılıyor...\n")
    
    # PDF'yi yükle
    chunk_count = rag.load_pdf_document(pdf_path)
    
    if chunk_count == 0:
        print("❌ PDF yüklenemedi!")
        return
    
    # Graph bilgilerini göster
    graph_info = rag.get_graph_info()
    print(f"\n📊 Graph İstatistikleri:")
    print(f"   - Toplam Node: {graph_info['total_nodes']}")
    print(f"   - Toplam Edge: {graph_info['total_edges']}")
    print(f"   - Belge Sayısı: {graph_info['documents']}")
    print(f"   - Entity Sayısı: {graph_info['entities']}")
    
    print("\n" + "="*60)
    print("💬 İNTERAKTİF SORU-CEVAP SİSTEMİ")
    print("="*60)
    print("Komutlar:")
    print("  - Soru sorun")
    print("  - 'eval' - Sistemi değerlendir")
    print("  - 'test' - Test case ekle")
    print("  - 'quit' - Çık")
    print()
    
    while True:
        try:
            # Kullanıcıdan komut al
            command = input("❓ Komut/Soru: ").strip()
            
            if command.lower() in ['quit', 'exit', 'çık', 'q']:
                print("👋 Görüşürüz!")
                break
                
            elif command.lower() == 'eval':
                if DEEPEVAL_AVAILABLE:
                    rag.evaluate_rag_system()
                else:
                    print("❌ DeepEval yüklü değil! 'pip install deepeval' ile yükleyin.")
                continue
                
            elif command.lower() == 'test':
                if DEEPEVAL_AVAILABLE:
                    query = input("Test sorusu: ").strip()
                    expected = input("Beklenen yanıt (opsiyonel): ").strip()
                    rag.add_test_case(query, expected if expected else None)
                else:
                    print("❌ DeepEval yüklü değil!")
                continue
            
            if not command:
                print("⚠️ Lütfen bir komut veya soru girin.")
                continue
            
            print("\n🤖 Yanıt aranıyor...")
            start_time = time.time()
            
            # Yanıt oluştur
            response = rag.generate_response(command)
            
            end_time = time.time()
            print(f"\n💡 Yanıt: {response}")
            print(f"⏱️ Süre: {end_time - start_time:.2f} saniye")
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\n👋 Görüşürüz!")
            break
        except Exception as e:
            print(f"❌ Hata: {e}")

if __name__ == "__main__":
    interactive_qa()