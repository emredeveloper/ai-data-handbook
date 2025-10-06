import psycopg2
import os
import requests
import json
from dotenv import load_dotenv

# .env dosyasından ortam değişkenlerini yükle
load_dotenv()

# Bağlantı bilgilerini .env dosyasından al
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST")

# Ollama ayarları
OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "embeddinggemma:latest"

# Türkçe metin örnekleri (10 satır)
turkish_texts = [
    "Merhaba, bugün hava çok güzel ve güneşli.",
    "Türkiye'nin başkenti Ankara'dır ve çok güzel bir şehirdir.",
    "Kahve içmeyi çok seviyorum, özellikle Türk kahvesini.",
    "İstanbul Boğazı'nın manzarası gerçekten muhteşemdir.",
    "Yazın deniz kenarında tatil yapmak çok keyifli oluyor.",
    "Türk mutfağı dünyaca ünlü ve çok lezzetlidir.",
    "Kitap okumak zihni geliştirir ve hayal gücünü artırır.",
    "Müzik dinlemek ruh halini iyileştirir ve stresi azaltır.",
    "Spor yapmak sağlık için çok önemlidir ve vücudu güçlendirir.",
    "Arkadaşlarla vakit geçirmek çok eğlenceli ve mutluluk verir."
]

def get_embedding(text, model=MODEL_NAME):
    """Ollama kullanarak metni embed et"""
    try:
        url = f"{OLLAMA_URL}/api/embeddings"
        data = {
            "model": model,
            "prompt": text
        }
        
        response = requests.post(url, json=data, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        return result.get("embedding", [])
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Ollama API hatası: {e}")
        return None
    except Exception as e:
        print(f"❌ Embedding hatası: {e}")
        return None

def create_embedding_table():
    """Embedding tablosunu oluştur"""
    conn_string = f"dbname={DB_NAME} user={DB_USER} password={DB_PASS} host={DB_HOST}"
    
    try:
        with psycopg2.connect(conn_string) as conn:
            with conn.cursor() as cur:
                # pgvector uzantısını etkinleştir
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                print("✅ pgvector uzantısı etkinleştirildi!")
                
                # Embedding tablosunu oluştur (pgvector ile)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS embeddings (
                        id SERIAL PRIMARY KEY,
                        text_content TEXT NOT NULL,
                        embedding VECTOR(768),  -- Embedding Gemma boyutu 768
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # Hnsw index oluştur (semantic search için)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS embeddings_embedding_idx 
                    ON embeddings USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64);
                """)
                
                conn.commit()
                print("✅ Embedding tablosu ve index başarıyla oluşturuldu!")
                
    except psycopg2.Error as e:
        print(f"❌ Tablo oluşturma hatası: {e}")
        print("💡 pgvector kurulumu için: pgvector-kurulum.md dosyasını kontrol edin!")

def insert_embeddings():
    """Türkçe metinleri embed et ve veritabanına kaydet"""
    conn_string = f"dbname={DB_NAME} user={DB_USER} password={DB_PASS} host={DB_HOST}"
    
    try:
        with psycopg2.connect(conn_string) as conn:
            with conn.cursor() as cur:
                print(f"\n🚀 {len(turkish_texts)} adet Türkçe metin embed ediliyor...")
                print("-" * 80)
                
                for i, text in enumerate(turkish_texts, 1):
                    print(f"📝 Metin {i}: {text[:50]}...")
                    
                    # Metni embed et
                    embedding = get_embedding(text)
                    
                    if embedding:
                        # Embedding'i array olarak veritabanına kaydet
                        cur.execute("""
                            INSERT INTO embeddings (text_content, embedding)
                            VALUES (%s, %s)
                        """, (text, embedding))
                        
                        print(f"   ✅ Başarıyla embed edildi ve kaydedildi!")
                        print(f"   📊 Embedding boyutu: {len(embedding)}")
                    else:
                        print(f"   ❌ Embedding oluşturulamadı!")
                    print()
                
                conn.commit()
                print("🎉 Tüm embeddingler başarıyla veritabanına kaydedildi!")
                
    except psycopg2.Error as e:
        print(f"❌ Veritabanı hatası: {e}")

def show_embeddings():
    """Kaydedilen embeddingleri göster"""
    conn_string = f"dbname={DB_NAME} user={DB_USER} password={DB_PASS} host={DB_HOST}"
    
    try:
        with psycopg2.connect(conn_string) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM embeddings;")
                count = cur.fetchone()[0]
                
                print(f"\n📊 Veritabanında toplam {count} adet embedding bulunuyor.")
                
                if count > 0:
                    cur.execute("""
                        SELECT id, text_content, 768 as embedding_size, created_at
                        FROM embeddings
                        ORDER BY id;
                    """)
                    
                    results = cur.fetchall()
                    print("\n📋 Kaydedilen Embeddingler:")
                    print("-" * 80)
                    
                    for row in results:
                        id_val, text, emb_size, created_at = row
                        print(f"🔹 ID: {id_val}")
                        print(f"   📝 Metin: {text[:60]}{'...' if len(text) > 60 else ''}")
                        print(f"   📊 Embedding boyutu: {emb_size}")
                        print(f"   🕐 Oluşturulma: {created_at}")
                        print()
                
    except psycopg2.Error as e:
        print(f"❌ Veritabanı okuma hatası: {e}")

def cosine_similarity(vec1, vec2):
    """Cosine similarity hesapla"""
    import math
    
    # Dot product
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    
    # Magnitudes
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    
    return dot_product / (magnitude1 * magnitude2)

def word_similarity(query_text, target_text):
    """Kelime bazlı benzerlik hesapla"""
    import re
    
    # Türkçe karakterleri normalize et
    def normalize_text(text):
        replacements = {
            'ç': 'c', 'ğ': 'g', 'ı': 'i', 'ö': 'o', 'ş': 's', 'ü': 'u',
            'Ç': 'C', 'Ğ': 'G', 'İ': 'I', 'Ö': 'O', 'Ş': 'S', 'Ü': 'U'
        }
        for tr_char, en_char in replacements.items():
            text = text.replace(tr_char, en_char)
        return text.lower()
    
    # Metinleri temizle ve kelimelere böl
    query_words = set(re.findall(r'\w+', normalize_text(query_text)))
    target_words = set(re.findall(r'\w+', normalize_text(target_text)))
    
    # Stop words (Türkçe)
    stop_words = {'ve', 'ile', 'bir', 'bu', 'da', 'de', 'den', 'dan', 'için', 'olan', 'oldu', 'var', 'yok', 'çok', 'daha', 'en', 'gibi', 'kadar', 'sonra', 'önce', 'üzerine', 'altında', 'arasında', 'içinde', 'dışında'}
    
    # Stop words'leri çıkar
    query_words = query_words - stop_words
    target_words = target_words - stop_words
    
    if not query_words:
        return 0
    
    # Jaccard similarity
    intersection = query_words.intersection(target_words)
    union = query_words.union(target_words)
    
    jaccard = len(intersection) / len(union) if union else 0
    
    # Exact word match bonus
    exact_matches = len(intersection)
    word_bonus = exact_matches * 0.1
    
    return min(jaccard + word_bonus, 1.0)

def hybrid_search(query_text, top_k=3, semantic_weight=0.7, word_weight=0.3):
    """Hibrit search - semantic + kelime bazlı benzerlik"""
    conn_string = f"dbname={DB_NAME} user={DB_USER} password={DB_PASS} host={DB_HOST}"
    
    try:
        # Query metnini embed et
        query_embedding = get_embedding(query_text)
        if not query_embedding:
            print("❌ Query embedding oluşturulamadı!")
            return
        
        with psycopg2.connect(conn_string) as conn:
            with conn.cursor() as cur:
                # Tüm verileri al
                cur.execute("SELECT id, text_content, embedding FROM embeddings;")
                all_results = cur.fetchall()
                
                # Her metin için hibrit skor hesapla
                hybrid_scores = []
                
                for id_val, text, embedding in all_results:
                    # Embedding'i float array'e çevir
                    try:
                        if isinstance(embedding, str):
                            # String'den float array'e çevir ([] veya {} formatı)
                            clean_embedding = embedding.strip('[]{}')
                            embedding_float = [float(x.strip()) for x in clean_embedding.split(',') if x.strip()]
                        else:
                            embedding_float = embedding
                        
                        # Semantic similarity
                        semantic_sim = cosine_similarity(query_embedding, embedding_float)
                    except Exception as e:
                        print(f"❌ Embedding dönüştürme hatası (ID: {id_val}): {e}")
                        continue
                    
                    # Word similarity
                    word_sim = word_similarity(query_text, text)
                    
                    # Hibrit skor
                    hybrid_score = (semantic_weight * semantic_sim) + (word_weight * word_sim)
                    
                    hybrid_scores.append((id_val, text, semantic_sim, word_sim, hybrid_score))
                
                # Hibrit skora göre sırala
                hybrid_scores.sort(key=lambda x: x[4], reverse=True)
                
                print(f"\n🔍 '{query_text}' için hibrit search sonuçları:")
                print(f"   📊 Ağırlıklar: Semantic {semantic_weight*100}% + Kelime {word_weight*100}%")
                print("-" * 80)
                
                for i, (id_val, text, semantic_sim, word_sim, hybrid_score) in enumerate(hybrid_scores[:top_k], 1):
                    print(f"🏆 #{i} - Hibrit Skor: {hybrid_score:.4f}")
                    print(f"   🧠 Semantic: {semantic_sim:.4f} | 📝 Kelime: {word_sim:.4f}")
                    print(f"   📄 Metin: {text}")
                    print(f"   🆔 ID: {id_val}")
                    print()
                
    except psycopg2.Error as e:
        print(f"❌ Hibrit search hatası: {e}")

def semantic_search(query_text, top_k=3):
    """Geliştirilmiş semantic search"""
    return hybrid_search(query_text, top_k, semantic_weight=0.8, word_weight=0.2)

def drop_and_recreate_table():
    """Mevcut tabloyu sil ve yeniden oluştur"""
    conn_string = f"dbname={DB_NAME} user={DB_USER} password={DB_PASS} host={DB_HOST}"
    
    try:
        with psycopg2.connect(conn_string) as conn:
            with conn.cursor() as cur:
                # Mevcut tabloyu sil
                cur.execute("DROP TABLE IF EXISTS embeddings CASCADE;")
                print("🗑️ Eski embedding tablosu silindi!")
                
                conn.commit()
                
    except psycopg2.Error as e:
        print(f"❌ Tablo silme hatası: {e}")

if __name__ == "__main__":
    print("🤖 Ollama Embedding Gemma ile Türkçe Metin Embedding İşlemi")
    print("=" * 80)
    
    # 0. Eski tabloyu temizle (boyut uyumsuzluğu için)
    drop_and_recreate_table()
    
    # 1. Tabloyu oluştur
    create_embedding_table()
    
    # 2. Embeddingleri oluştur ve kaydet
    insert_embeddings()
    
    # 3. Sonuçları göster
    show_embeddings()
    
    # 4. Geliştirilmiş hibrit search örnekleri
    print("\n" + "="*80)
    print("🔍 HİBRİT SEARCH ÖRNEKLERİ")
    print("="*80)
    
    test_queries = [
        ("güzel hava", "Hava durumu ile ilgili metinleri bul"),
        ("kahve", "Kahve ile ilgili metinleri bul"),
        ("Türkiye Ankara", "Türkiye ve Ankara ile ilgili metinleri bul"),
        ("spor sağlık", "Spor ve sağlık ile ilgili metinleri bul")
    ]
    
    for query, description in test_queries:
        print(f"\n🎯 {description}")
        hybrid_search(query, top_k=2, semantic_weight=0.6, word_weight=0.4)
    
    print("\n✅ İşlem tamamlandı!")
