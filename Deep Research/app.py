import os
import json
import requests
from openai import OpenAI
from datetime import datetime
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from bs4 import BeautifulSoup
import wikipedia
from googlesearch import search
import time
import re
from urllib.parse import urljoin, urlparse

class DeepResearchApp:
    def __init__(self):
        self.client = None
        self.research_history = []
        self.sources = []
        self.setup_client()
    
    def setup_client(self):
        """OpenRouter API istemcisini kurar"""
        try:
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key="sk-or-v1...",
            )
        except Exception as e:
            st.error(f"API istemcisi kurulumunda hata: {e}")
    
    def web_search(self, query, num_results=5):
        """Akıllı Google araması yapar - dinamik sorgu optimizasyonu"""
        try:
            # Sorguyu optimize et
            optimized_queries = self.optimize_search_query(query)
            
            all_results = []
            for search_query in optimized_queries:
                try:
                    search_results = []
                    for url in search(search_query, num_results=2, sleep_interval=1):
                        if url not in all_results:  # Duplikasyon önleme
                            search_results.append(url)
                            all_results.append(url)
                        if len(all_results) >= num_results:
                            break
                    
                    if len(all_results) >= num_results:
                        break
                        
                except Exception as e:
                    st.warning(f"Arama hatası ({search_query}): {e}")
                    continue
            
            return all_results[:num_results]
            
        except Exception as e:
            st.warning(f"Web araması hatası: {e}")
            return self.get_fallback_urls(query)
    
    def optimize_search_query(self, original_query):
        """Arama sorgusunu optimize eder ve çeşitlendirir"""
        queries = [original_query]  # Orijinal sorgu
        
        # Tarih bilgisi varsa farklı formatlar dene
        if any(month in original_query.lower() for month in ['ocak', 'şubat', 'mart', 'nisan', 'mayıs', 'haziran', 
                                                           'temmuz', 'ağustos', 'eylül', 'ekim', 'kasım', 'aralık']):
            # Tarihli aramalar için farklı formatlar
            queries.extend([
                f"{original_query} site:arxiv.org",
                f"{original_query} site:medium.com",
                f"{original_query} site:towardsdatascience.com",
                f"{original_query} 2025"
            ])
        
        # AI/ML konuları için özel aramalar
        ai_keywords = ['yapay zeka', 'ai', 'artificial intelligence', 'machine learning', 'deep learning', 'makine öğrenmesi']
        if any(keyword in original_query.lower() for keyword in ai_keywords):
            queries.extend([
                f"{original_query} arxiv",
                f"{original_query} research paper",
                f"{original_query} academic",
                f"{original_query} latest news"
            ])
        
        # Teknik konular için
        tech_keywords = ['programlama', 'coding', 'software', 'teknoloji', 'tech']
        if any(keyword in original_query.lower() for keyword in tech_keywords):
            queries.extend([
                f"{original_query} github",
                f"{original_query} stackoverflow",
                f"{original_query} documentation"
            ])
        
        # Haber konuları için
        news_keywords = ['haber', 'news', 'güncel', 'son dakika']
        if any(keyword in original_query.lower() for keyword in news_keywords):
            queries.extend([
                f"{original_query} site:bbc.com",
                f"{original_query} site:cnn.com",
                f"{original_query} site:reuters.com"
            ])
        
        return queries[:4]  # Maksimum 4 farklı sorgu
    
    def get_fallback_urls(self, query):
        """Web araması başarısız olursa alternatif URL'ler döndürür"""
        # AI konuları için özel URL'ler
        ai_keywords = ["ai", "yapay zeka", "artificial intelligence", "machine learning", "deep learning"]
        if any(keyword in query.lower() for keyword in ai_keywords):
            return [
                "https://tr.wikipedia.org/wiki/Yapay_zeka",
                "https://tr.wikipedia.org/wiki/Makine_öğrenmesi",
                "https://www.bbc.com/turkce/haberler-teknoloji"
            ]
        
        # Genel konular için alternatif kaynaklar
        fallback_urls = [
            "https://tr.wikipedia.org/wiki/" + query.replace(" ", "_"),
            "https://www.bbc.com/turkce",
            "https://www.aa.com.tr/tr"
        ]
        return fallback_urls[:3]  # İlk 3'ünü döndür
    
    def extract_article_content(self, url):
        """Web sayfasından makale içeriğini çıkarır"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Başlık
            title = soup.find('title')
            title_text = title.get_text().strip() if title else "Başlık bulunamadı"
            
            # Ana içerik
            content_selectors = [
                'article', 'main', '.content', '.post-content', 
                '.article-content', '.entry-content', 'p'
            ]
            
            content = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    content = ' '.join([elem.get_text().strip() for elem in elements])
                    break
            
            if not content:
                content = soup.get_text()
            
            # İlk 2000 karakteri al
            content = content[:2000] + "..." if len(content) > 2000 else content
            
            return {
                'title': title_text,
                'content': content,
                'url': url,
                'domain': urlparse(url).netloc
            }
            
        except Exception as e:
            return {
                'title': f"Hata: {str(e)}",
                'content': f"Bu sayfa okunamadı: {url}",
                'url': url,
                'domain': urlparse(url).netloc
            }
    
    def wikipedia_search(self, query):
        """Akıllı Wikipedia araması yapar"""
        try:
            wikipedia.set_lang("tr")
            
            # Sorguyu optimize et
            optimized_queries = self.optimize_wikipedia_query(query)
            
            all_articles = []
            seen_titles = set()
            
            for search_query in optimized_queries:
                try:
                    search_results = wikipedia.search(search_query, results=2)
                    
                    for title in search_results:
                        if title not in seen_titles:
                            try:
                                page = wikipedia.page(title)
                                all_articles.append({
                                    'title': page.title,
                                    'content': page.content[:1500] + "..." if len(page.content) > 1500 else page.content,
                                    'url': page.url,
                                    'domain': 'wikipedia.org',
                                    'search_query': search_query
                                })
                                seen_titles.add(title)
                                
                                if len(all_articles) >= 3:
                                    break
                            except:
                                continue
                    
                    if len(all_articles) >= 3:
                        break
                        
                except Exception as e:
                    continue
            
            return all_articles
            
        except Exception as e:
            st.warning(f"Wikipedia araması hatası: {e}")
            return []
    
    def optimize_wikipedia_query(self, original_query):
        """Wikipedia araması için sorguyu optimize eder"""
        queries = [original_query]
        
        # Tarih bilgisini çıkar
        import re
        clean_query = re.sub(r'\d{4}|\b(ocak|şubat|mart|nisan|mayıs|haziran|temmuz|ağustos|eylül|ekim|kasım|aralık)\b', '', original_query, flags=re.IGNORECASE)
        clean_query = clean_query.strip()
        
        if clean_query != original_query:
            queries.append(clean_query)
        
        # Anahtar kelimeleri çıkar
        keywords = original_query.split()
        if len(keywords) > 2:
            # İlk 2 kelimeyi al
            queries.append(' '.join(keywords[:2]))
            # Son 2 kelimeyi al
            queries.append(' '.join(keywords[-2:]))
        
        # AI/ML konuları için özel aramalar
        if any(keyword in original_query.lower() for keyword in ['yapay zeka', 'ai', 'artificial intelligence']):
            queries.extend(['Yapay zeka', 'Makine öğrenmesi', 'Derin öğrenme'])
        
        return list(set(queries))[:3]  # Duplikasyon önleme ve maksimum 3 sorgu
    
    def analyze_with_llm(self, query, sources_data, research_mode="standard"):
        """LLM ile kaynakları analiz eder"""
        if not self.client:
            return "API istemcisi kurulmamış."
        
        try:
            # Kaynakları birleştir
            combined_sources = ""
            for i, source in enumerate(sources_data, 1):
                combined_sources += f"\n--- Kaynak {i} ---\n"
                combined_sources += f"Başlık: {source['title']}\n"
                combined_sources += f"URL: {source['url']}\n"
                combined_sources += f"İçerik: {source['content']}\n"
            
            system_messages = {
                "standard": "Sen bir araştırma uzmanısın. Verilen kaynakları analiz ederek kullanıcının sorusunu kapsamlı şekilde yanıtla. Kaynakları referans göster.",
                "detailed": "Sen bir derin araştırma uzmanısın. Verilen kaynakları detaylı analiz et, farklı perspektifleri ele al ve kapsamlı bir rapor hazırla. Her kaynağı referans göster.",
                "creative": "Sen yaratıcı bir araştırmacısın. Verilen kaynakları kullanarak özgün analizler yap, yeni bağlantılar kur ve yaratıcı öneriler sun."
            }
            
            prompt = f"""
            Araştırma Sorusu: {query}
            
            Bulunan Kaynaklar:
            {combined_sources}
            
            Lütfen bu kaynakları analiz ederek soruyu yanıtla. Her kaynağı referans göster ve güvenilir bilgiler sun.
            """
            
            messages = [
                {"role": "system", "content": system_messages.get(research_mode, system_messages["standard"])},
                {"role": "user", "content": prompt}
            ]
            
            completion = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://deep-research-app.com",
                    "X-Title": "Deep Research App",
                },
                model="alibaba/tongyi-deepresearch-30b-a3b:free",
                messages=messages,
                temperature=0.7,
                max_tokens=3000
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            return f"Analiz sırasında hata oluştu: {e}"
    
    def deep_research(self, query, research_mode="standard", include_web=True, include_wikipedia=True):
        """Gerçek deep research yapar - web arama + LLM analizi"""
        research_results = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'sources': [],
            'analysis': '',
            'research_mode': research_mode,
            'confidence_score': 0,
            'source_diversity': 0
        }
        
        all_sources = []
        
        # Web araması
        if include_web:
            with st.spinner("🌐 Akıllı web araması yapılıyor..."):
                try:
                    # Optimize edilmiş sorguları göster
                    optimized_queries = self.optimize_search_query(query)
                    if len(optimized_queries) > 1:
                        st.info(f"🔍 {len(optimized_queries)} farklı arama sorgusu kullanılıyor")
                    
                    web_urls = self.web_search(query, num_results=3)
                    st.info(f"🌐 {len(web_urls)} web sayfası bulundu")
                    
                    for i, url in enumerate(web_urls):
                        with st.spinner(f"📄 Sayfa okunuyor ({i+1}/{len(web_urls)}): {url[:50]}..."):
                            article_data = self.extract_article_content(url)
                            if article_data['content'] and len(article_data['content']) > 100:
                                # Kaynak kalitesi skorlaması
                                article_data['quality_score'] = self.calculate_source_quality(article_data)
                                all_sources.append(article_data)
                                research_results['sources'].append(article_data)
                except Exception as e:
                    st.warning(f"Web araması hatası: {e}")
        
        # Wikipedia araması
        if include_wikipedia:
            with st.spinner("📚 Akıllı Wikipedia araması yapılıyor..."):
                try:
                    # Optimize edilmiş Wikipedia sorgularını göster
                    wiki_queries = self.optimize_wikipedia_query(query)
                    if len(wiki_queries) > 1:
                        st.info(f"🔍 {len(wiki_queries)} farklı Wikipedia sorgusu kullanılıyor")
                    
                    wiki_articles = self.wikipedia_search(query)
                    st.info(f"📚 {len(wiki_articles)} Wikipedia makalesi bulundu")
                    
                    for article in wiki_articles:
                        article['quality_score'] = 0.9  # Wikipedia yüksek kalite
                        all_sources.append(article)
                        research_results['sources'].append(article)
                except Exception as e:
                    st.warning(f"Wikipedia araması hatası: {e}")
        
        # Kaynak çeşitliliği hesapla
        research_results['source_diversity'] = self.calculate_source_diversity(all_sources)
        
        # Eğer hiç kaynak yoksa, LLM'den genel bilgi al
        if not all_sources:
            st.warning("⚠️ Hiç kaynak bulunamadı. LLM'den genel bilgi alınıyor...")
            research_results['analysis'] = self.get_general_analysis(query, research_mode)
            research_results['confidence_score'] = 0.3  # Düşük güven
        else:
            # LLM ile analiz
            with st.spinner("🧠 Tongyi DeepResearch-30B-A3B ile analiz yapılıyor..."):
                analysis = self.analyze_with_llm(query, all_sources, research_mode)
                research_results['analysis'] = analysis
                
                # Güven skoru hesapla
                research_results['confidence_score'] = self.calculate_confidence_score(all_sources, analysis)
        
        # Araştırma geçmişine ekle
        self.research_history.append(research_results)
        
        return research_results
    
    def calculate_source_quality(self, source):
        """Kaynak kalitesini hesaplar"""
        score = 0.5  # Base score
        
        # İçerik uzunluğu
        if len(source['content']) > 1000:
            score += 0.2
        elif len(source['content']) > 500:
            score += 0.1
        
        # Domain güvenilirliği
        trusted_domains = ['wikipedia.org', 'bbc.com', 'aa.com.tr', 'haberturk.com', 'ntv.com.tr']
        if any(domain in source['domain'] for domain in trusted_domains):
            score += 0.2
        
        # Başlık kalitesi
        if len(source['title']) > 20 and not source['title'].startswith('Hata'):
            score += 0.1
        
        return min(score, 1.0)
    
    def calculate_source_diversity(self, sources):
        """Kaynak çeşitliliğini hesaplar"""
        if not sources:
            return 0
        
        domains = set(source['domain'] for source in sources)
        return len(domains) / len(sources) if sources else 0
    
    def calculate_confidence_score(self, sources, analysis):
        """Genel güven skorunu hesaplar"""
        if not sources:
            return 0.3
        
        # Kaynak sayısı
        source_score = min(len(sources) / 5, 1.0) * 0.3
        
        # Kaynak kalitesi
        avg_quality = sum(source.get('quality_score', 0.5) for source in sources) / len(sources)
        quality_score = avg_quality * 0.4
        
        # Çeşitlilik
        diversity_score = self.calculate_source_diversity(sources) * 0.3
        
        return source_score + quality_score + diversity_score
    
    def get_general_analysis(self, query, research_mode="standard"):
        """Kaynak bulunamazsa LLM'den genel analiz alır"""
        if not self.client:
            return "API istemcisi kurulmamış."
        
        try:
            system_messages = {
                "standard": "Sen Tongyi DeepResearch-30B-A3B modelisin. Kullanıcının sorusunu mevcut bilgilerinle yanıtla.",
                "detailed": "Sen Tongyi DeepResearch-30B-A3B modelisin. Kullanıcının sorusunu detaylı şekilde analiz et ve kapsamlı bilgi ver.",
                "creative": "Sen Tongyi DeepResearch-30B-A3B modelisin. Kullanıcının sorusuna yaratıcı ve özgün bir yaklaşımla yanıt ver."
            }
            
            messages = [
                {"role": "system", "content": system_messages.get(research_mode, system_messages["standard"])},
                {"role": "user", "content": f"Lütfen şu konu hakkında bilgi ver: {query}"}
            ]
            
            completion = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://deep-research-app.com",
                    "X-Title": "Deep Research App",
                },
                model="alibaba/tongyi-deepresearch-30b-a3b:free",
                messages=messages,
                temperature=0.7,
                max_tokens=2000
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            return f"Analiz sırasında hata oluştu: {e}"
    
    def create_research_visualization(self, research_results):
        """Araştırma sonuçlarını görselleştirir"""
        if not research_results['sources']:
            return None
        
        # Kaynak türleri dağılımı
        source_types = {}
        for source in research_results['sources']:
            domain = source['domain']
            if 'wikipedia' in domain:
                source_types['Wikipedia'] = source_types.get('Wikipedia', 0) + 1
            else:
                source_types['Web'] = source_types.get('Web', 0) + 1
        
        if source_types:
            fig = px.pie(
                values=list(source_types.values()),
                names=list(source_types.keys()),
                title="Kaynak Türleri Dağılımı"
            )
            return fig
        
        return None
    
    def multi_step_research(self, main_query, sub_queries, research_mode="standard"):
        """Çok adımlı deep research yapar"""
        results = {
            "main_query": main_query,
            "timestamp": datetime.now().isoformat(),
            "sub_research": {},
            "summary": "",
            "all_sources": []
        }
        
        # Ana araştırma
        main_result = self.deep_research(main_query, research_mode)
        results["main_research"] = main_result
        results["all_sources"].extend(main_result['sources'])
        
        # Alt araştırmalar
        for i, sub_query in enumerate(sub_queries, 1):
            sub_result = self.deep_research(f"{main_query} - {sub_query}", research_mode)
            results["sub_research"][f"step_{i}"] = {
                "query": sub_query,
                "result": sub_result,
                "sources": sub_result['sources']
            }
            results["all_sources"].extend(sub_result['sources'])
        
        # Özet oluştur
        summary_query = f"""
        Ana araştırma konusu: {main_query}
        
        Ana bulgular: {main_result['analysis']}
        
        Alt araştırma sonuçları:
        {json.dumps({k: v['result']['analysis'] for k, v in results['sub_research'].items()}, ensure_ascii=False, indent=2)}
        
        Bu araştırmanın kapsamlı bir özetini çıkar ve ana bulguları özetle.
        """
        
        # Özet için LLM kullan
        if self.client:
            try:
                messages = [
                    {"role": "system", "content": "Sen bir araştırma özet uzmanısın. Verilen araştırma sonuçlarını analiz ederek kapsamlı bir özet hazırla."},
                    {"role": "user", "content": summary_query}
                ]
                
                completion = self.client.chat.completions.create(
                    extra_headers={
                        "HTTP-Referer": "https://deep-research-app.com",
                        "X-Title": "Deep Research App",
                    },
                    model="alibaba/tongyi-deepresearch-30b-a3b:free",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=2000
                )
                
                results["summary"] = completion.choices[0].message.content
            except:
                results["summary"] = "Özet oluşturulamadı."
        else:
            results["summary"] = "API bağlantısı yok, özet oluşturulamadı."
        
        return results

def main():
    st.set_page_config(
        page_title="Deep Research App",
        page_icon="🔬",
        layout="wide"
    )
    
    st.title("🔬 Deep Research App")
    st.markdown("**Tongyi DeepResearch-30B-A3B** + Web Arama + Wikipedia ile gerçek araştırma yapın")
    
    # Sidebar
    st.sidebar.header("⚙️ Ayarlar")
    
    # API Key input
    api_key = st.sidebar.text_input(
        "OpenRouter API Key",
        type="password",
        help="OpenRouter API anahtarınızı girin"
    )
    
    if api_key:
        os.environ["OPENROUTER_API_KEY"] = api_key
    
    # Research mode selection
    research_mode = st.sidebar.selectbox(
        "Araştırma Stili",
        ["standard", "detailed", "creative"],
        help="Standard: Genel analiz, Detailed: Derin analiz, Creative: Yaratıcı yaklaşım"
    )
    
    mode_descriptions = {
        "standard": "📊 Genel araştırma ve analiz",
        "detailed": "🔍 Derin analiz ve detaylı rapor",
        "creative": "💡 Yaratıcı araştırma ve özgün yaklaşım"
    }
    
    # Kaynak seçenekleri
    st.sidebar.subheader("📚 Kaynak Seçenekleri")
    include_web = st.sidebar.checkbox("Web Arama", value=True, help="Google'da web araması yap")
    include_wikipedia = st.sidebar.checkbox("Wikipedia", value=True, help="Wikipedia'da arama yap")
    
    # Test modu
    test_mode = st.sidebar.checkbox("Test Modu", value=False, help="Hızlı test için sadece LLM kullan")
    
    st.sidebar.info(f"**Seçilen Mod:** {mode_descriptions[research_mode]}")
    
    # Main interface
    app = DeepResearchApp()
    
    # Research type selection
    research_type = st.radio(
        "Araştırma Türü",
        ["Tek Konu Araştırması", "Çok Adımlı Araştırma"],
        horizontal=True
    )
    
    if research_type == "Tek Konu Araştırması":
        st.subheader("🔍 Tek Konu Deep Research")
        
        query = st.text_area(
            "Araştırma Konunuz",
            placeholder="Örnek: yapay zeka güncel makaleler 2025 eylül ayı",
            height=100
        )
        
        # Akıllı arama örnekleri
        with st.expander("💡 Akıllı Arama Örnekleri"):
            st.markdown("""
            **Tarihli Aramalar:**
            - `yapay zeka güncel makaleler 2025 eylül ayı`
            - `machine learning son gelişmeler 2025`
            - `blockchain teknolojisi 2025 haberler`
            
            **Akademik Aramalar:**
            - `deep learning arxiv makaleleri 2025`
            - `computer vision research papers`
            - `natural language processing güncel çalışmalar`
            
            **Teknik Aramalar:**
            - `python programlama 2025 yenilikler`
            - `react.js güncel özellikler`
            - `docker containerization best practices`
            
            **Haber Aramaları:**
            - `teknoloji haberleri son dakika`
            - `startup funding 2025`
            - `cryptocurrency market analysis`
            """)
        
        if st.button("🚀 Deep Research Başlat", type="primary"):
            if query:
                # Debug bilgileri
                st.info(f"🔍 Araştırma konusu: {query}")
                st.info(f"📊 Araştırma stili: {research_mode}")
                st.info(f"🌐 Web arama: {'Açık' if include_web else 'Kapalı'}")
                st.info(f"📚 Wikipedia: {'Açık' if include_wikipedia else 'Kapalı'}")
                st.info(f"🧪 Test modu: {'Açık' if test_mode else 'Kapalı'}")
                
                # Test modu kontrolü
                if test_mode:
                    st.warning("🧪 Test modu aktif - Sadece LLM kullanılıyor")
                    research_results = {
                        'query': query,
                        'timestamp': datetime.now().isoformat(),
                        'sources': [],
                        'analysis': app.get_general_analysis(query, research_mode),
                        'research_mode': research_mode
                    }
                else:
                    # Deep research yap
                    research_results = app.deep_research(
                        query, 
                        research_mode, 
                        include_web=include_web, 
                        include_wikipedia=include_wikipedia
                    )
                
                # Sonuçları göster
                st.subheader("📊 Araştırma Sonuçları")
                
                # Kalite metrikleri
                col1, col2, col3 = st.columns(3)
                with col1:
                    confidence = research_results.get('confidence_score', 0)
                    st.metric(
                        "🎯 Güven Skoru", 
                        f"{confidence:.1%}",
                        help="Kaynak kalitesi ve çeşitliliğine göre hesaplanan güven skoru"
                    )
                
                with col2:
                    diversity = research_results.get('source_diversity', 0)
                    st.metric(
                        "🌐 Kaynak Çeşitliliği", 
                        f"{diversity:.1%}",
                        help="Farklı domain'lerden gelen kaynak oranı"
                    )
                
                with col3:
                    source_count = len(research_results.get('sources', []))
                    st.metric(
                        "📚 Toplam Kaynak", 
                        f"{source_count}",
                        help="Analiz edilen toplam kaynak sayısı"
                    )
                
                # Analiz
                st.markdown("### 🧠 Tongyi DeepResearch-30B-A3B Analizi")
                st.markdown(research_results['analysis'])
                
                # Kaynaklar
                if research_results['sources']:
                    st.markdown("### 📚 Bulunan Kaynaklar")
                    
                    for i, source in enumerate(research_results['sources'], 1):
                        quality_score = source.get('quality_score', 0.5)
                        quality_color = "🟢" if quality_score > 0.7 else "🟡" if quality_score > 0.5 else "🔴"
                        
                        # Arama sorgusu bilgisi
                        search_info = ""
                        if 'search_query' in source:
                            search_info = f" (Arama: {source['search_query']})"
                        
                        with st.expander(f"**{i}. {source['title']}** ({source['domain']}) {quality_color}{search_info}"):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"**URL:** {source['url']}")
                                st.markdown(f"**İçerik:** {source['content'][:500]}...")
                            with col2:
                                st.metric("Kalite", f"{quality_score:.1%}")
                    
                    # Görselleştirme
                    viz = app.create_research_visualization(research_results)
                    if viz:
                        st.plotly_chart(viz, use_container_width=True)
                
                # İndirme seçenekleri
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📥 Analizi İndir (TXT)",
                        data=research_results['analysis'],
                        file_name=f"research_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                
                with col2:
                    json_data = json.dumps(research_results, ensure_ascii=False, indent=2)
                    st.download_button(
                        label="📥 Tüm Verileri İndir (JSON)",
                        data=json_data,
                        file_name=f"research_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            else:
                st.warning("Lütfen bir araştırma konusu girin.")
    
    else:  # Multi-step research
        st.subheader("🔄 Çok Adımlı Deep Research")
        
        main_query = st.text_input(
            "Ana Araştırma Konusu",
            placeholder="Örnek: İklim değişikliğinin ekonomik etkileri"
        )
        
        st.subheader("📝 Alt Araştırma Konuları")
        
        # Dynamic sub-queries
        if 'sub_queries' not in st.session_state:
            st.session_state.sub_queries = [""]
        
        for i, sub_query in enumerate(st.session_state.sub_queries):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.session_state.sub_queries[i] = st.text_input(
                    f"Alt Konu {i+1}",
                    value=sub_query,
                    key=f"sub_query_{i}",
                    placeholder=f"Alt araştırma konusu {i+1}"
                )
            with col2:
                if st.button("❌", key=f"remove_{i}", disabled=len(st.session_state.sub_queries) <= 1):
                    st.session_state.sub_queries.pop(i)
                    st.rerun()
        
        if st.button("➕ Alt Konu Ekle"):
            st.session_state.sub_queries.append("")
            st.rerun()
        
        if st.button("🚀 Çok Adımlı Deep Research Başlat", type="primary"):
            if main_query and any(st.session_state.sub_queries):
                valid_sub_queries = [q for q in st.session_state.sub_queries if q.strip()]
                
                if valid_sub_queries:
                    with st.spinner("Çok adımlı deep research yapılıyor..."):
                        results = app.multi_step_research(main_query, valid_sub_queries, research_mode)
                    
                    # Display results
                    st.subheader("📊 Çok Adımlı Araştırma Sonuçları")
                    
                    # Main research
                    st.markdown("### 🎯 Ana Araştırma")
                    st.markdown(results["main_research"]['analysis'])
                    
                    # Show main research sources
                    if results["main_research"]['sources']:
                        with st.expander("📚 Ana Araştırma Kaynakları"):
                            for i, source in enumerate(results["main_research"]['sources'], 1):
                                st.markdown(f"**{i}. {source['title']}** - {source['domain']}")
                                st.markdown(f"URL: {source['url']}")
                    
                    # Sub research
                    st.markdown("### 📋 Alt Araştırmalar")
                    for step, data in results["sub_research"].items():
                        with st.expander(f"**{data['query']}**"):
                            st.markdown(data["result"]['analysis'])
                            
                            # Show sources for this sub-research
                            if data['sources']:
                                st.markdown("**Kaynaklar:**")
                                for i, source in enumerate(data['sources'], 1):
                                    st.markdown(f"- {source['title']} ({source['domain']})")
                    
                    # Summary
                    st.markdown("### 📝 Kapsamlı Özet")
                    st.markdown(results["summary"])
                    
                    # All sources summary
                    if results["all_sources"]:
                        st.markdown("### 📚 Tüm Kaynaklar Özeti")
                        source_df = pd.DataFrame(results["all_sources"])
                        source_df = source_df[['title', 'domain', 'url']]
                        st.dataframe(source_df, use_container_width=True)
                    
                    # Download results
                    json_results = json.dumps(results, ensure_ascii=False, indent=2)
                    st.download_button(
                        label="📥 Tüm Sonuçları İndir (JSON)",
                        data=json_results,
                        file_name=f"multi_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                else:
                    st.warning("Lütfen en az bir geçerli alt konu girin.")
            else:
                st.warning("Lütfen ana konu ve en az bir alt konu girin.")
    
    # Model info
    with st.expander("ℹ️ Deep Research App Hakkında"):
        st.markdown("""
        ### 🔬 Deep Research App
        
        **Özellikler:**
        - 🧠 **Tongyi DeepResearch-30B-A3B** - 30 milyar parametreli güçlü dil modeli
        - 🌐 **Web Arama** - Google'da gerçek zamanlı arama
        - 📚 **Wikipedia Entegrasyonu** - Türkçe Wikipedia desteği
        - 📊 **Veri Görselleştirme** - Plotly ile interaktif grafikler
        - 📥 **Rapor İndirme** - TXT ve JSON formatında sonuçlar
        
        **Araştırma Süreci:**
        1. **Web Arama** - Google'da konuyla ilgili sayfalar bulunur
        2. **İçerik Çıkarma** - Web sayfalarından makale içerikleri çıkarılır
        3. **Wikipedia Arama** - Türkçe Wikipedia'da ilgili makaleler bulunur
        4. **LLM Analizi** - Tongyi DeepResearch-30B-A3B tüm kaynakları analiz eder
        5. **Rapor Oluşturma** - Kapsamlı araştırma raporu hazırlanır
        
        **Kullanım Alanları:**
        - Akademik araştırma
        - Pazar analizi
        - Teknik dokümantasyon
        - Haber analizi
        - Stratejik planlama
        - Rekabet analizi
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "🔬 **Deep Research App** - Tongyi DeepResearch-30B-A3B ile güçlendirilmiştir | "
        f"Son güncelleme: {datetime.now().strftime('%d.%m.%Y %H:%M')}"
    )

if __name__ == "__main__":
    main()
