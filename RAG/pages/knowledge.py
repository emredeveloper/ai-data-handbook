import streamlit as st

st.title("📖 Proje Açıklamaları")

st.header("Projenin Amacı ve Akışı")
st.markdown("""
Bu proje, uzun bağlamlı RAG (Retrieval-Augmented Generation) senaryolarında **REFRAG** fikrini prototip düzeyde uygulamayı amaçlar.

- **Retriever**: İlk adımda TF-IDF kullanılır; sorguya en yakın belgeler seçilir.  
- **Reranker**: Embedding ile yeniden sıralama yapılır, böylece daha alakalı pasajlar seçilir.  
- **REFRAG – sıkıştırma + seçici açma**: Belgeler chunk’lanır, her chunk soft token’a dönüştürülür. *Seçici* modda en alakalı `e` chunk tam açılır, geri kalanları soft token biçiminde tutulur.  
- **Bandit Politikası**: `e` değeri “ödül = kalite − λ·gecikme” ile güncellenir (softmax bandit).  
- **Metrik İzleme**: Baseline / Compact / Selective yolları için proxy metrikler (L, TTFT, KV) hesaplanır ve karşılaştırılır.  
""")

st.header("Kullanılan Parametreler ve Ne Anlam Taşıyor")
st.markdown("""
| Parametre | Anlamı |
|-----------|---------|
| `k` | Her chunk’ın token uzunluğu |
| `stride` | Chunk’lar arasındaki kayma (overlap) |
| `qlen` | Maksimum sorgu token sayısı |
| `budget` | Token bütçesi; seçici modda üst sınırı belirler |
| `M` | İlk seçilecek belge sayısı (retriever) |
| `λ (bandit_lam)` | Gecikme cezası katsayısı |
| `bandit_lr` | Bandit karar güncelleme hız katsayısı |
| `max_e` | En fazla açılabilecek chunk sayısı (e üst limiti) |
""")

st.header("Metrikler ve Sonuç Tablolarındaki Açıklama")
st.markdown("""
| Metrik | Açıklama |
|--------|-----------|
| **L** | Girdi toplam token uzunluğu (soft + tam açılan chunk + sorgu) |
| **TTFT (ms)** | Model üretim süresi (milisaniye) — proxy olarak ölçülür |
| **KV (~L²D)** | Key-Value cache elektroniği için tahmini maliyet (L² · D) |
| **Qual** | Seçilen chunk’ların ortalama ilişki skoru |
| **Reward** | `qual − λ · TTFT` ile hesaplanan toplam ödül |
| **e** | Seçici modda açılan chunk sayısı |
""")
