import networkx as nx
import requests
import json
import time
from typing import Dict, List, Tuple

class SosyalAgOllamaAnalizor:
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.model_name = "llama3.2:1b"  # Varsayılan model
        self.G = None
        self.merkezilikler = None
        self.metrikler = None
        self.etki_skorlari = None
        
    def ollama_available(self) -> bool:
        """Ollama servisinin çalışıp çalışmadığını kontrol eder"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_available_models(self) -> List[str]:
        """Kullanılabilir modelleri listeler"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model['name'] for model in models]
            return []
        except:
            return []
    
    def set_model(self, model_name: str):
        """Analiz için kullanılacak modeli ayarlar"""
        self.model_name = model_name
    
    def query_ollama(self, prompt: str) -> str:
        """Ollama modeline sorgu gönderir"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.5,
                    "top_p": 0.9,
                    "num_predict": 500
                }
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json().get('response', '')
            else:
                return f"Hata: {response.status_code}"
                
        except Exception as e:
            return f"Bağlantı hatası: {str(e)}"
    
    def sosyal_ag_olustur(self, kisiler: List[str] = None, arkadasliklar: List[Tuple] = None):
        """Sosyal ağ oluşturur"""
        self.G = nx.Graph()
        
        if kisiler is None:
            kisiler = [
                "Ahmet Yılmaz", "Ayşe Demir", "Mehmet Kaya", "Fatma Özkan", "Ali Çelik",
                "Zeynep Arslan", "Mustafa Yıldız", "Elif Şahin", "Hasan Özkan", "Selin Korkmaz",
                "Burak Aydın", "Merve Çetin", "Emre Yılmaz", "Deniz Özkan", "Gamze Demir",
                "Can Arslan", "Burcu Yıldız", "Ece Şahin", "Hakan Korkmaz", "Seda Aydın",
                "Kemal Çetin", "Aslı Yılmaz", "Tolga Özkan", "Merve Demir", "Serkan Arslan"
            ]
        
        if arkadasliklar is None:
            arkadasliklar = [
                # Merkez grup - Çok bağlantılı
                ("Ahmet Yılmaz", "Ayşe Demir"), ("Ahmet Yılmaz", "Mehmet Kaya"), ("Ahmet Yılmaz", "Fatma Özkan"),
                ("Ahmet Yılmaz", "Ali Çelik"), ("Ahmet Yılmaz", "Zeynep Arslan"), ("Ahmet Yılmaz", "Mustafa Yıldız"),
                ("Ayşe Demir", "Mehmet Kaya"), ("Ayşe Demir", "Fatma Özkan"), ("Ayşe Demir", "Ali Çelik"),
                ("Ayşe Demir", "Zeynep Arslan"), ("Ayşe Demir", "Mustafa Yıldız"), ("Ayşe Demir", "Elif Şahin"),
                ("Mehmet Kaya", "Fatma Özkan"), ("Mehmet Kaya", "Ali Çelik"), ("Mehmet Kaya", "Zeynep Arslan"),
                ("Mehmet Kaya", "Mustafa Yıldız"), ("Mehmet Kaya", "Elif Şahin"), ("Mehmet Kaya", "Hasan Özkan"),
                
                # İkinci grup - Orta bağlantılı
                ("Fatma Özkan", "Ali Çelik"), ("Fatma Özkan", "Zeynep Arslan"), ("Fatma Özkan", "Mustafa Yıldız"),
                ("Fatma Özkan", "Elif Şahin"), ("Fatma Özkan", "Hasan Özkan"), ("Fatma Özkan", "Selin Korkmaz"),
                ("Ali Çelik", "Zeynep Arslan"), ("Ali Çelik", "Mustafa Yıldız"), ("Ali Çelik", "Elif Şahin"),
                ("Ali Çelik", "Hasan Özkan"), ("Ali Çelik", "Selin Korkmaz"), ("Ali Çelik", "Burak Aydın"),
                ("Zeynep Arslan", "Mustafa Yıldız"), ("Zeynep Arslan", "Elif Şahin"), ("Zeynep Arslan", "Hasan Özkan"),
                ("Zeynep Arslan", "Selin Korkmaz"), ("Zeynep Arslan", "Burak Aydın"), ("Zeynep Arslan", "Merve Çetin"),
                
                # Üçüncü grup - Köprü kişiler
                ("Mustafa Yıldız", "Elif Şahin"), ("Mustafa Yıldız", "Hasan Özkan"), ("Mustafa Yıldız", "Selin Korkmaz"),
                ("Mustafa Yıldız", "Burak Aydın"), ("Mustafa Yıldız", "Merve Çetin"), ("Mustafa Yıldız", "Emre Yılmaz"),
                ("Elif Şahin", "Hasan Özkan"), ("Elif Şahin", "Selin Korkmaz"), ("Elif Şahin", "Burak Aydın"),
                ("Elif Şahin", "Merve Çetin"), ("Elif Şahin", "Emre Yılmaz"), ("Elif Şahin", "Deniz Özkan"),
                ("Hasan Özkan", "Selin Korkmaz"), ("Hasan Özkan", "Burak Aydın"), ("Hasan Özkan", "Merve Çetin"),
                ("Hasan Özkan", "Emre Yılmaz"), ("Hasan Özkan", "Deniz Özkan"), ("Hasan Özkan", "Gamze Demir"),
                
                # Dördüncü grup - Daha az bağlantılı
                ("Selin Korkmaz", "Burak Aydın"), ("Selin Korkmaz", "Merve Çetin"), ("Selin Korkmaz", "Emre Yılmaz"),
                ("Selin Korkmaz", "Deniz Özkan"), ("Selin Korkmaz", "Gamze Demir"), ("Selin Korkmaz", "Can Arslan"),
                ("Burak Aydın", "Merve Çetin"), ("Burak Aydın", "Emre Yılmaz"), ("Burak Aydın", "Deniz Özkan"),
                ("Burak Aydın", "Gamze Demir"), ("Burak Aydın", "Can Arslan"), ("Burak Aydın", "Burcu Yıldız"),
                ("Merve Çetin", "Emre Yılmaz"), ("Merve Çetin", "Deniz Özkan"), ("Merve Çetin", "Gamze Demir"),
                ("Merve Çetin", "Can Arslan"), ("Merve Çetin", "Burcu Yıldız"), ("Merve Çetin", "Ece Şahin"),
                
                # Beşinci grup - Kenar kişiler
                ("Emre Yılmaz", "Deniz Özkan"), ("Emre Yılmaz", "Gamze Demir"), ("Emre Yılmaz", "Can Arslan"),
                ("Emre Yılmaz", "Burcu Yıldız"), ("Emre Yılmaz", "Ece Şahin"), ("Emre Yılmaz", "Hakan Korkmaz"),
                ("Deniz Özkan", "Gamze Demir"), ("Deniz Özkan", "Can Arslan"), ("Deniz Özkan", "Burcu Yıldız"),
                ("Deniz Özkan", "Ece Şahin"), ("Deniz Özkan", "Hakan Korkmaz"), ("Deniz Özkan", "Seda Aydın"),
                ("Gamze Demir", "Can Arslan"), ("Gamze Demir", "Burcu Yıldız"), ("Gamze Demir", "Ece Şahin"),
                ("Gamze Demir", "Hakan Korkmaz"), ("Gamze Demir", "Seda Aydın"), ("Gamze Demir", "Kemal Çetin"),
                
                # Altıncı grup - En az bağlantılı
                ("Can Arslan", "Burcu Yıldız"), ("Can Arslan", "Ece Şahin"), ("Can Arslan", "Hakan Korkmaz"),
                ("Can Arslan", "Seda Aydın"), ("Can Arslan", "Kemal Çetin"), ("Can Arslan", "Aslı Yılmaz"),
                ("Burcu Yıldız", "Ece Şahin"), ("Burcu Yıldız", "Hakan Korkmaz"), ("Burcu Yıldız", "Seda Aydın"),
                ("Burcu Yıldız", "Kemal Çetin"), ("Burcu Yıldız", "Aslı Yılmaz"), ("Burcu Yıldız", "Tolga Özkan"),
                ("Ece Şahin", "Hakan Korkmaz"), ("Ece Şahin", "Seda Aydın"), ("Ece Şahin", "Kemal Çetin"),
                ("Ece Şahin", "Aslı Yılmaz"), ("Ece Şahin", "Tolga Özkan"), ("Ece Şahin", "Merve Demir"),
                
                # Yedinci grup - İzole kişiler
                ("Hakan Korkmaz", "Seda Aydın"), ("Hakan Korkmaz", "Kemal Çetin"), ("Hakan Korkmaz", "Aslı Yılmaz"),
                ("Hakan Korkmaz", "Tolga Özkan"), ("Hakan Korkmaz", "Merve Demir"), ("Hakan Korkmaz", "Serkan Arslan"),
                ("Seda Aydın", "Kemal Çetin"), ("Seda Aydın", "Aslı Yılmaz"), ("Seda Aydın", "Tolga Özkan"),
                ("Seda Aydın", "Merve Demir"), ("Seda Aydın", "Serkan Arslan"),
                ("Kemal Çetin", "Aslı Yılmaz"), ("Kemal Çetin", "Tolga Özkan"), ("Kemal Çetin", "Merve Demir"),
                ("Kemal Çetin", "Serkan Arslan"),
                ("Aslı Yılmaz", "Tolga Özkan"), ("Aslı Yılmaz", "Merve Demir"), ("Aslı Yılmaz", "Serkan Arslan"),
                ("Tolga Özkan", "Merve Demir"), ("Tolga Özkan", "Serkan Arslan"),
                ("Merve Demir", "Serkan Arslan"),
                
                # Köprü bağlantıları - Farklı grupları birleştiren
                ("Ahmet Yılmaz", "Emre Yılmaz"), ("Ayşe Demir", "Deniz Özkan"), ("Mehmet Kaya", "Gamze Demir"),
                ("Fatma Özkan", "Can Arslan"), ("Ali Çelik", "Burcu Yıldız"), ("Zeynep Arslan", "Ece Şahin"),
                ("Mustafa Yıldız", "Hakan Korkmaz"), ("Elif Şahin", "Seda Aydın"), ("Hasan Özkan", "Kemal Çetin"),
                ("Selin Korkmaz", "Aslı Yılmaz"), ("Burak Aydın", "Tolga Özkan"), ("Merve Çetin", "Merve Demir"),
                ("Emre Yılmaz", "Serkan Arslan"), ("Deniz Özkan", "Ahmet Yılmaz"), ("Gamze Demir", "Ayşe Demir"),
                ("Can Arslan", "Mehmet Kaya"), ("Burcu Yıldız", "Fatma Özkan"), ("Ece Şahin", "Ali Çelik"),
                ("Hakan Korkmaz", "Zeynep Arslan"), ("Seda Aydın", "Mustafa Yıldız"), ("Kemal Çetin", "Elif Şahin"),
                ("Aslı Yılmaz", "Hasan Özkan"), ("Tolga Özkan", "Selin Korkmaz"), ("Merve Demir", "Burak Aydın"),
                ("Serkan Arslan", "Merve Çetin")
            ]
        
        self.G.add_nodes_from(kisiler)
        self.G.add_edges_from(arkadasliklar)
        
        # Analizleri yap
        self.analiz_yap()
        
        return self.G
    
    def analiz_yap(self):
        """Tüm analizleri gerçekleştirir"""
        if self.G is None:
            return
            
        # Merkezilik analizleri
        derece = nx.degree_centrality(self.G)
        betweenness = nx.betweenness_centrality(self.G)
        closeness = nx.closeness_centrality(self.G)
        eigenvector = nx.eigenvector_centrality(self.G, max_iter=1000)
        
        self.merkezilikler = (derece, betweenness, closeness, eigenvector)
        
        # Ağ metrikleri
        self.metrikler = self.ag_metrikleri()
        
        # Etki skorları
        self.etki_skorlari = self.en_etkili_kisiler()
    
    def ag_metrikleri(self) -> Dict:
        """Ağ metriklerini hesaplar"""
        if self.G is None:
            return {}
            
        metrikler = {}
        metrikler['node_sayisi'] = self.G.number_of_nodes()
        metrikler['edge_sayisi'] = self.G.number_of_edges()
        metrikler['density'] = nx.density(self.G)
        metrikler['average_clustering'] = nx.average_clustering(self.G)
        metrikler['connected_components'] = nx.number_connected_components(self.G)
        metrikler['is_connected'] = nx.is_connected(self.G)
        
        if metrikler['is_connected']:
            metrikler['average_shortest_path'] = nx.average_shortest_path_length(self.G)
            metrikler['diameter'] = nx.diameter(self.G)
            metrikler['radius'] = nx.radius(self.G)
        
        return metrikler
    
    def en_etkili_kisiler(self) -> Dict:
        """En etkili kişileri hesaplar"""
        if self.merkezilikler is None:
            return {}
            
        derece, betweenness, closeness, eigenvector = self.merkezilikler
        
        etki_skorlari = {}
        for node in self.G.nodes():
            etki_skorlari[node] = (
                derece[node] * 0.3 +
                betweenness[node] * 0.3 +
                closeness[node] * 0.2 +
                eigenvector[node] * 0.2
            )
        
        return etki_skorlari
    
    def ag_bilgilerini_goster(self):
        """Ağ bilgilerini metin olarak gösterir"""
        if self.G is None:
            print("Önce ağ oluşturulmalı!")
            return
            
        print("\n📊 SOSYAL AĞ BİLGİLERİ:")
        print("="*60)
        print(f"Toplam Kişi: {self.G.number_of_nodes()}")
        print(f"Toplam Bağlantı: {self.G.number_of_edges()}")
        print(f"Ağ Yoğunluğu: {nx.density(self.G):.3f}")
        
        # Kişileri gruplara ayır
        derece = nx.degree_centrality(self.G)
        sorted_kisiler = sorted(derece.items(), key=lambda x: x[1], reverse=True)
        
        print("\n🏆 EN BAĞLANTILI 10 KİŞİ:")
        for i, (kisi, deger) in enumerate(sorted_kisiler[:10], 1):
            print(f"  {i}. {kisi}: {deger:.3f}")
        
        print("\n🔗 BAĞLANTI ÖRNEKLERİ (İlk 20):")
        for i, edge in enumerate(list(self.G.edges())[:20], 1):
            print(f"  {i}. {edge[0]} ↔ {edge[1]}")
        
        if len(self.G.edges()) > 20:
            print(f"  ... ve {len(self.G.edges()) - 20} bağlantı daha")
        
        # Topluluk analizi
        try:
            communities = list(nx.community.greedy_modularity_communities(self.G))
            print(f"\n🏘️ TAHMİNİ TOPLULUK SAYISI: {len(communities)}")
            for i, community in enumerate(communities[:5], 1):
                members = list(community)[:5]  # İlk 5 üye
                print(f"  Topluluk {i}: {', '.join(members)}")
                if len(community) > 5:
                    print(f"    ... ve {len(community) - 5} kişi daha")
        except:
            print("\n🏘️ Topluluk analizi yapılamadı")
    
    def analiz_raporu_olustur(self) -> str:
        """Analiz raporu oluşturur"""
        if self.G is None or self.merkezilikler is None:
            return "Ağ analizi yapılmamış!"
        
        derece, betweenness, closeness, eigenvector = self.merkezilikler
        
        rapor = f"""
SOSYAL AĞ ANALİZ RAPORU
{'='*60}

AĞ GENEL BİLGİLERİ:
- Toplam Kişi Sayısı: {self.metrikler['node_sayisi']}
- Toplam Bağlantı Sayısı: {self.metrikler['edge_sayisi']}
- Ağ Yoğunluğu: {self.metrikler['density']:.3f}
- Ortalama Kümeleme Katsayısı: {self.metrikler['average_clustering']:.3f}
- Bağlantılı Bileşen Sayısı: {self.metrikler['connected_components']}
- Ağ Bağlantılı mı?: {'Evet' if self.metrikler['is_connected'] else 'Hayır'}

EN ETKİLİ KİŞİLER (Toplam Skor - İlk 15):
"""
        
        # En etkili kişileri sırala
        sorted_etki = sorted(self.etki_skorlari.items(), key=lambda x: x[1], reverse=True)
        for i, (kisi, skor) in enumerate(sorted_etki[:15], 1):
            rapor += f"{i:2d}. {kisi}: {skor:.3f}\n"
        
        rapor += f"""
MERKEZİLİK ANALİZİ:

En Çok Bağlantısı Olanlar (Derece - İlk 10):
"""
        sorted_derece = sorted(derece.items(), key=lambda x: x[1], reverse=True)
        for i, (kisi, deger) in enumerate(sorted_derece[:10], 1):
            rapor += f"{i:2d}. {kisi}: {deger:.3f}\n"
        
        rapor += f"""
En Önemli Köprü Kişiler (Betweenness - İlk 10):
"""
        sorted_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
        for i, (kisi, deger) in enumerate(sorted_betweenness[:10], 1):
            rapor += f"{i:2d}. {kisi}: {deger:.3f}\n"
        
        rapor += f"""
Tüm Ağa En Hızlı Erişenler (Closeness - İlk 10):
"""
        sorted_closeness = sorted(closeness.items(), key=lambda x: x[1], reverse=True)
        for i, (kisi, deger) in enumerate(sorted_closeness[:10], 1):
            rapor += f"{i:2d}. {kisi}: {deger:.3f}\n"
        
        rapor += f"""
Diğer Etkililerle Bağlantılı Olanlar (Eigenvector - İlk 10):
"""
        sorted_eigenvector = sorted(eigenvector.items(), key=lambda x: x[1], reverse=True)
        for i, (kisi, deger) in enumerate(sorted_eigenvector[:10], 1):
            rapor += f"{i:2d}. {kisi}: {deger:.3f}\n"
        
        return rapor
    
    def llm_analiz_sorgusu(self, soru: str) -> str:
        """LLM ile analiz sorgusu yapar"""
        if self.G is None:
            return "Önce ağ oluşturulmalı!"
        
        # Analiz verilerini hazırla
        analiz_verisi = self.analiz_raporu_olustur()
        
        prompt = f"""Sen bir sosyal ağ analiz uzmanısın. Aşağıdaki verileri kullanarak soruyu yanıtla.

VERİLER:
{analiz_verisi}

SORU: {soru}

Lütfen:
1. Sadece verilen analiz verilerini kullan
2. Kısa ve öz Türkçe yanıt ver
3. Sayısal değerleri belirt
4. En doğru kişiyi/kişileri isim olarak söyle"""
        
        return self.query_ollama(prompt)
    
    def interaktif_analiz(self):
        """İnteraktif analiz modu"""
        print("🤖 Ollama Sosyal Ağ Analiz Sistemine Hoş Geldiniz!")
        print("="*60)
        
        # Ollama kontrolü
        if not self.ollama_available():
            print("❌ Ollama servisi çalışmıyor! Lütfen Ollama'yı başlatın.")
            return
        
        # Model seçimi
        models = self.get_available_models()
        if models:
            print(f"📋 Kullanılabilir modeller: {', '.join(models)}")
            if self.model_name not in models:
                self.model_name = models[0]
                print(f"✅ Model otomatik olarak {self.model_name} olarak ayarlandı.")
        else:
            print("⚠️ Kullanılabilir model bulunamadı!")
            return
        
        # Ağ oluştur
        print("\n🔗 Sosyal ağ oluşturuluyor...")
        self.sosyal_ag_olustur()
        print("✅ Ağ oluşturuldu ve analiz edildi!")
        
        # Ağ bilgilerini göster
        self.ag_bilgilerini_goster()
        
        # İnteraktif sorgu döngüsü
        print("\n💬 Analiz sorgularınızı sorun (çıkmak için 'quit' yazın):")
        print("📋 Komutlar: 'veri' = analiz verilerini göster, 'dogrula' = son cevabı doğrula")
        print("-"*60)
        
        son_cevap = ""
        while True:
            soru = input("\n🤔 Sorunuz: ").strip()
            
            if soru.lower() in ['quit', 'exit', 'çık', 'q']:
                print("👋 Görüşürüz!")
                break
            
            if soru.lower() == 'veri':
                print("\n📊 DETAYLI ANALİZ VERİLERİ:")
                print("="*60)
                print(self.analiz_raporu_olustur())
                continue
            
            if soru.lower() == 'dogrula':
                if son_cevap:
                    print("\n🔍 SON CEVAP DOĞRULAMA:")
                    print("="*60)
                    print(f"Son soru: {son_soru}")
                    print(f"AI cevabı: {son_cevap}")
                    print("\n📊 Gerçek verilerle karşılaştırma:")
                    print(self.analiz_raporu_olustur())
                else:
                    print("❌ Henüz soru sorulmamış!")
                continue
            
            if not soru:
                continue
            
            print("\n🤖 AI analizi yapılıyor... (lütfen bekleyin)")
            try:
                cevap = self.llm_analiz_sorgusu(soru)
                print(f"\n💡 AI Cevabı:\n{cevap}")
                son_cevap = cevap
                son_soru = soru
            except Exception as e:
                print(f"\n❌ Hata: {str(e)}")
            print("-"*60)

# Örnek kullanım
if __name__ == "__main__":
    analyzer = SosyalAgOllamaAnalizor()
    analyzer.interaktif_analiz() 