import networkx as nx
import pandas as pd
import requests
import json
import time
from typing import Dict, List, Tuple, Optional
import os
from pathlib import Path

class TitanicAgAnalizor:
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.model_name = "llama3.2:1b"
        self.G = None
        self.df = None
        self.merkezilikler = None
        self.metrikler = None
        self.etki_skorlari = None
        self.yolcu_gruplari = {}
        
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
    
    def titanic_verisi_yukle(self, dosya_yolu: str) -> bool:
        """Titanic verisini yükler ve ağ oluşturur"""
        try:
            # Titanic verisini yükle
            self.df = pd.read_csv(dosya_yolu)
            print(f"✅ Titanic verisi yüklendi: {len(self.df)} yolcu")
            
            # Veri temizleme ve hazırlama
            self.veri_temizle()
            
            # Ağı oluştur
            self.titanic_agi_olustur()
            return True
            
        except Exception as e:
            print(f"❌ Titanic verisi yükleme hatası: {e}")
            return False
    
    def veri_temizle(self):
        """Titanic verisini temizler ve hazırlar"""
        # Eksik değerleri doldur
        if 'Age' in self.df.columns:
            self.df['Age'] = self.df['Age'].fillna(self.df['Age'].median())
        if 'Fare' in self.df.columns:
            self.df['Fare'] = self.df['Fare'].fillna(self.df['Fare'].median())
        if 'Embarked' in self.df.columns:
            self.df['Embarked'] = self.df['Embarked'].fillna('S')
        if 'Cabin' in self.df.columns:
            self.df['Cabin'] = self.df['Cabin'].fillna('Bilinmeyen')
        
        # Yeni özellikler oluştur
        if 'SibSp' in self.df.columns and 'Parch' in self.df.columns:
            self.df['FamilySize'] = self.df['SibSp'] + self.df['Parch'] + 1
            self.df['IsAlone'] = (self.df['FamilySize'] == 1).astype(int)
        
        # Yaş grupları
        if 'Age' in self.df.columns:
            self.df['AgeGroup'] = pd.cut(self.df['Age'], 
                                        bins=[0, 12, 18, 35, 50, 100], 
                                        labels=['Çocuk', 'Genç', 'Yetişkin', 'Orta Yaş', 'Yaşlı'])
        
        # Bilet sınıfı grupları
        if 'Pclass' in self.df.columns:
            self.df['ClassGroup'] = self.df['Pclass'].map({
                1: 'Birinci Sınıf',
                2: 'İkinci Sınıf', 
                3: 'Üçüncü Sınıf'
            })
        
        # Bilet türü (Ticket'ın ilk karakteri)
        if 'Ticket' in self.df.columns:
            self.df['TicketType'] = self.df['Ticket'].str[0].fillna('N')
        
        # Cabin sınıfı (Cabin'in ilk harfi)
        if 'Cabin' in self.df.columns:
            self.df['CabinClass'] = self.df['Cabin'].str[0].fillna('N')
        
        print("✅ Veri temizleme tamamlandı")
    
    def titanic_agi_olustur(self):
        """Titanic verisinden sosyal ağ oluşturur"""
        self.G = nx.Graph()
        
        # Yolcu gruplarını oluştur
        self.yolcu_gruplari = {
            'Aile Grupları': self.df[self.df['FamilySize'] > 1],
            'Yalnız Yolcular': self.df[self.df['IsAlone'] == 1],
            'Birinci Sınıf': self.df[self.df['Pclass'] == 1],
            'İkinci Sınıf': self.df[self.df['Pclass'] == 2],
            'Üçüncü Sınıf': self.df[self.df['Pclass'] == 3],
            'Kadın Yolcular': self.df[self.df['Sex'] == 'female'],
            'Erkek Yolcular': self.df[self.df['Sex'] == 'male'],
            'Çocuklar (0-12)': self.df[self.df['Age'] <= 12],
            'Gençler (13-18)': self.df[(self.df['Age'] > 12) & (self.df['Age'] <= 18)],
            'Yetişkinler (19-50)': self.df[(self.df['Age'] > 18) & (self.df['Age'] <= 50)],
            'Yaşlılar (50+)': self.df[self.df['Age'] > 50],
            'Hayatta Kalanlar': self.df[self.df['Survived'] == 1],
            'Hayatta Kalmayanlar': self.df[self.df['Survived'] == 0],
            'Yüksek Ücretli': self.df[self.df['Fare'] > self.df['Fare'].median()],
            'Düşük Ücretli': self.df[self.df['Fare'] <= self.df['Fare'].median()],
            'Cabin Sahipleri': self.df[self.df['Cabin'] != 'Bilinmeyen'],
            'Cabin Sahibi Olmayanlar': self.df[self.df['Cabin'] == 'Bilinmeyen'],
            'Southampton\'dan Binenler': self.df[self.df['Embarked'] == 'S'],
            'Cherbourg\'dan Binenler': self.df[self.df['Embarked'] == 'C'],
            'Queenstown\'dan Binenler': self.df[self.df['Embarked'] == 'Q']
        }
        
        # Her yolcuyu node olarak ekle
        for idx, row in self.df.iterrows():
            yolcu_id = f"Yolcu_{idx}"
            self.G.add_node(yolcu_id, **row.to_dict())
        
        # Aile bağlantıları ekle
        for idx, row in self.df.iterrows():
            if row['FamilySize'] > 1:
                # Aynı aile üyelerini bul
                aile_uyeleri = self.df[
                    (self.df['SibSp'] == row['SibSp']) & 
                    (self.df['Parch'] == row['Parch']) &
                    (self.df['Ticket'] == row['Ticket'])
                ]
                
                for _, aile_uyesi in aile_uyeleri.iterrows():
                    if aile_uyesi.name != idx:
                        self.G.add_edge(f"Yolcu_{idx}", f"Yolcu_{aile_uyesi.name}", 
                                      weight=5, type='Aile')
        
        # Aynı sınıf bağlantıları
        for idx, row in self.df.iterrows():
            ayni_sinif = self.df[self.df['Pclass'] == row['Pclass']]
            for _, sinif_arkadasi in ayni_sinif.iterrows():
                if sinif_arkadasi.name != idx:
                    self.G.add_edge(f"Yolcu_{idx}", f"Yolcu_{sinif_arkadasi.name}", 
                                  weight=2, type='Sınıf')
        
        # Aynı yaş grubu bağlantıları
        for idx, row in self.df.iterrows():
            ayni_yas_grubu = self.df[self.df['AgeGroup'] == row['AgeGroup']]
            for _, yas_arkadasi in ayni_yas_grubu.iterrows():
                if yas_arkadasi.name != idx:
                    self.G.add_edge(f"Yolcu_{idx}", f"Yolcu_{yas_arkadasi.name}", 
                                  weight=1, type='Yaş Grubu')
        
        # Aynı bilet türü bağlantıları
        for idx, row in self.df.iterrows():
            if 'TicketType' in self.df.columns:
                ayni_bilet_turu = self.df[self.df['TicketType'] == row['TicketType']]
                for _, bilet_arkadasi in ayni_bilet_turu.iterrows():
                    if bilet_arkadasi.name != idx:
                        self.G.add_edge(f"Yolcu_{idx}", f"Yolcu_{bilet_arkadasi.name}", 
                                      weight=1, type='Bilet Türü')
        
        # Aynı cabin sınıfı bağlantıları
        for idx, row in self.df.iterrows():
            if 'CabinClass' in self.df.columns and row['CabinClass'] != 'N':
                ayni_cabin_sinifi = self.df[self.df['CabinClass'] == row['CabinClass']]
                for _, cabin_arkadasi in ayni_cabin_sinifi.iterrows():
                    if cabin_arkadasi.name != idx:
                        self.G.add_edge(f"Yolcu_{idx}", f"Yolcu_{cabin_arkadasi.name}", 
                                      weight=3, type='Cabin Sınıfı')
        
        # Aynı biniş limanı bağlantıları
        for idx, row in self.df.iterrows():
            if 'Embarked' in self.df.columns and pd.notna(row['Embarked']):
                ayni_liman = self.df[self.df['Embarked'] == row['Embarked']]
                for _, liman_arkadasi in ayni_liman.iterrows():
                    if liman_arkadasi.name != idx:
                        self.G.add_edge(f"Yolcu_{idx}", f"Yolcu_{liman_arkadasi.name}", 
                                      weight=1, type='Biniş Limanı')
        
        print(f"✅ Titanic ağı oluşturuldu: {self.G.number_of_nodes()} yolcu, {self.G.number_of_edges()} bağlantı")
        
        # Analizleri yap
        self.analiz_yap()
    
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
        self.etki_skorlari = self.en_etkili_yolcular()
    
    def ag_metrikleri(self) -> Dict:
        """Ağ metriklerini hesaplar"""
        if self.G is None:
            return {}
            
        metrikler = {}
        metrikler['yolcu_sayisi'] = self.G.number_of_nodes()
        metrikler['baglanti_sayisi'] = self.G.number_of_edges()
        metrikler['density'] = nx.density(self.G)
        metrikler['connected_components'] = nx.number_connected_components(self.G)
        metrikler['is_connected'] = nx.is_connected(self.G)
        
        if metrikler['is_connected']:
            metrikler['average_clustering'] = nx.average_clustering(self.G)
            metrikler['average_shortest_path'] = nx.average_shortest_path_length(self.G)
            metrikler['diameter'] = nx.diameter(self.G)
            metrikler['radius'] = nx.radius(self.G)
        
        return metrikler
    
    def en_etkili_yolcular(self) -> Dict:
        """En etkili yolcuları hesaplar"""
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
    
    def titanic_bilgilerini_goster(self):
        """Titanic ağ bilgilerini gösterir"""
        if self.G is None:
            print("Önce Titanic verisi yüklenmelidir!")
            return
            
        print("\n🚢 TİTANIC SOSYAL AĞ BİLGİLERİ:")
        print("="*60)
        print(f"Toplam Yolcu: {self.G.number_of_nodes()}")
        print(f"Toplam Bağlantı: {self.G.number_of_edges()}")
        print(f"Ağ Yoğunluğu: {nx.density(self.G):.3f}")
        
        # Yolcu grupları istatistikleri
        print("\n📊 YOLCU GRUPLARI:")
        for grup_adi, grup_df in self.yolcu_gruplari.items():
            hayatta_kalan = len(grup_df[grup_df['Survived'] == 1])
            toplam = len(grup_df)
            hayatta_kalma_orani = (hayatta_kalan / toplam * 100) if toplam > 0 else 0
            print(f"  {grup_adi}: {toplam} kişi ({hayatta_kalma_orani:.1f}% hayatta kaldı)")
        
        # Genel istatistikler
        toplam_yolcu = len(self.df)
        hayatta_kalan = len(self.df[self.df['Survived'] == 1])
        genel_hayatta_kalma = (hayatta_kalan / toplam_yolcu * 100)
        
        print(f"\n📈 GENEL İSTATİSTİKLER:")
        print(f"  Toplam Yolcu: {toplam_yolcu}")
        print(f"  Hayatta Kalan: {hayatta_kalan}")
        print(f"  Genel Hayatta Kalma Oranı: {genel_hayatta_kalma:.1f}%")
        
        # Cinsiyet bazlı analiz
        kadinlar = self.df[self.df['Sex'] == 'female']
        erkekler = self.df[self.df['Sex'] == 'male']
        
        if len(kadinlar) > 0:
            kadin_hayatta_kalma = (len(kadinlar[kadinlar['Survived'] == 1]) / len(kadinlar) * 100)
            print(f"  Kadınlar: {len(kadinlar)} kişi ({kadin_hayatta_kalma:.1f}% hayatta kaldı)")
        
        if len(erkekler) > 0:
            erkek_hayatta_kalma = (len(erkekler[erkekler['Survived'] == 1]) / len(erkekler) * 100)
            print(f"  Erkekler: {len(erkekler)} kişi ({erkek_hayatta_kalma:.1f}% hayatta kaldı)")
        
        # En bağlantılı yolcular
        derece = nx.degree_centrality(self.G)
        sorted_yolcular = sorted(derece.items(), key=lambda x: x[1], reverse=True)
        
        print("\n🏆 EN BAĞLANTILI 10 YOLCU:")
        for i, (yolcu_id, deger) in enumerate(sorted_yolcular[:10], 1):
            yolcu_bilgi = self.G.nodes[yolcu_id]
            isim = yolcu_bilgi.get('Name', 'Bilinmeyen')
            sinif = yolcu_bilgi.get('Pclass', '?')
            hayatta_kaldi = "✅" if yolcu_bilgi.get('Survived', 0) == 1 else "❌"
            print(f"  {i:2d}. {isim[:30]}... (Sınıf {sinif}) {hayatta_kaldi} - {deger:.3f}")
        
        # Bağlantı türleri
        baglanti_turleri = {}
        for edge in self.G.edges(data=True):
            baglanti_tipi = edge[2].get('type', 'Diğer')
            baglanti_turleri[baglanti_tipi] = baglanti_turleri.get(baglanti_tipi, 0) + 1
        
        print("\n🔗 BAĞLANTI TÜRLERİ:")
        for tip, sayi in baglanti_turleri.items():
            print(f"  {tip}: {sayi} bağlantı")
        
        # Bağlantı yoğunluğu analizi
        toplam_baglanti = sum(baglanti_turleri.values())
        if toplam_baglanti > 0:
            print(f"\n📊 BAĞLANTI ANALİZİ:")
            print(f"  Toplam Bağlantı: {toplam_baglanti}")
            print(f"  Ortalama Bağlantı/Yolcu: {toplam_baglanti/len(self.df):.1f}")
            
            # En yaygın bağlantı türü
            en_yaygin_tip = max(baglanti_turleri.items(), key=lambda x: x[1])
            print(f"  En Yaygın Bağlantı Türü: {en_yaygin_tip[0]} ({en_yaygin_tip[1]} bağlantı)")
        
        print("\n🔗 BAĞLANTI ÖRNEKLERİ (İlk 15):")
        for i, edge in enumerate(list(self.G.edges())[:15], 1):
            weight_info = ""
            edge_type = ""
            if 'weight' in self.G.edges[edge]:
                weight_info = f" (ağırlık: {self.G.edges[edge]['weight']:.2f})"
            if 'type' in self.G.edges[edge]:
                edge_type = f" [{self.G.edges[edge]['type']}]"
            print(f"  {i:2d}. {edge[0]} ↔ {edge[1]}{weight_info}{edge_type}")
        
        if len(self.G.edges()) > 15:
            print(f"  ... ve {len(self.G.edges()) - 15} bağlantı daha")
    
    def analiz_raporu_olustur(self) -> str:
        """Titanic analiz raporu oluşturur"""
        if self.G is None or self.merkezilikler is None:
            return "Titanic ağ analizi yapılmamış!"
        
        derece, betweenness, closeness, eigenvector = self.merkezilikler
        
        # Genel istatistikler
        hayatta_kalan = len(self.df[self.df['Survived'] == 1])
        toplam_yolcu = len(self.df)
        hayatta_kalma_orani = (hayatta_kalan / toplam_yolcu * 100)
        
        rapor = f"""
TİTANIC SOSYAL AĞ ANALİZ RAPORU
{'='*60}

GENEL BİLGİLER:
- Toplam Yolcu: {self.metrikler['yolcu_sayisi']}
- Toplam Bağlantı: {self.metrikler['baglanti_sayisi']}
- Ağ Yoğunluğu: {self.metrikler['density']:.3f}
- Hayatta Kalma Oranı: {hayatta_kalma_orani:.1f}% ({hayatta_kalan}/{toplam_yolcu})

YOLCU GRUPLARI:
"""
        
        # Yolcu grupları
        for grup_adi, grup_df in self.yolcu_gruplari.items():
            hayatta_kalan = len(grup_df[grup_df['Survived'] == 1])
            toplam = len(grup_df)
            hayatta_kalma_orani = (hayatta_kalan / toplam * 100) if toplam > 0 else 0
            rapor += f"- {grup_adi}: {toplam} kişi ({hayatta_kalma_orani:.1f}% hayatta kaldı)\n"
        
        rapor += f"""
EN ETKİLİ YOLCULAR (Toplam Skor - İlk 10):
"""
        
        # En etkili yolcular
        sorted_etki = sorted(self.etki_skorlari.items(), key=lambda x: x[1], reverse=True)
        for i, (yolcu_id, skor) in enumerate(sorted_etki[:10], 1):
            yolcu_bilgi = self.G.nodes[yolcu_id]
            isim = yolcu_bilgi.get('Name', 'Bilinmeyen')
            sinif = yolcu_bilgi.get('Pclass', '?')
            hayatta_kaldi = "✅" if yolcu_bilgi.get('Survived', 0) == 1 else "❌"
            rapor += f"{i:2d}. {isim[:25]}... (Sınıf {sinif}) {hayatta_kaldi} - {skor:.3f}\n"
        
        rapor += f"""
MERKEZİLİK ANALİZİ:

En Çok Bağlantısı Olanlar (Derece - İlk 10):
"""
        sorted_derece = sorted(derece.items(), key=lambda x: x[1], reverse=True)
        for i, (yolcu_id, deger) in enumerate(sorted_derece[:10], 1):
            yolcu_bilgi = self.G.nodes[yolcu_id]
            isim = yolcu_bilgi.get('Name', 'Bilinmeyen')
            rapor += f"{i:2d}. {isim[:25]}... - {deger:.3f}\n"
        
        return rapor
    
    def llm_analiz_sorgusu(self, soru: str) -> str:
        """LLM ile Titanic analiz sorgusu yapar"""
        if self.G is None:
            return "Önce Titanic verisi yüklenmelidir!"
        
        # Analiz verilerini hazırla
        analiz_verisi = self.analiz_raporu_olustur()
        
        prompt = f"""Sen bir Titanic gemisi sosyal ağ analiz uzmanısın. Aşağıdaki Titanic verilerinden oluşturulan ağ analizini kullanarak soruyu yanıtla.

VERİLER:
{analiz_verisi}

SORU: {soru}

Lütfen:
1. Sadece verilen analiz verilerini kullan
2. Detaylı ve açıklayıcı Türkçe yanıt ver
3. Sayısal değerleri ve yüzdeleri belirt
4. Tarihsel bağlamda yanıtla (1912 Titanic faciası)
5. Sosyal sınıf, cinsiyet ve yaş faktörlerini değerlendir
6. "Kadınlar ve çocuklar önce" politikasının etkisini analiz et"""
        
        return self.query_ollama(prompt)
    
    def ornek_titanic_verisi_olustur(self, dosya_adi: str = "titanic_ornek.csv"):
        """Örnek Titanic verisi oluşturur"""
        ornek_veriler = [
            [1, 0, 3, "Braund, Mr. Owen Harris", "male", 22, 1, 0, "A/5 21171", 7.25, "S"],
            [2, 1, 1, "Cumings, Mrs. John Bradley", "female", 38, 1, 0, "PC 17599", 71.2833, "C"],
            [3, 1, 3, "Heikkinen, Miss. Laina", "female", 26, 0, 0, "STON/O2. 3101282", 7.925, "S"],
            [4, 1, 1, "Futrelle, Mrs. Jacques Heath", "female", 35, 1, 0, "113803", 53.1, "S"],
            [5, 0, 3, "Allen, Mr. William Henry", "male", 35, 0, 0, "373450", 8.05, "S"],
            [6, 0, 3, "Moran, Mr. James", "male", 27, 0, 0, "330877", 8.4583, "Q"],
            [7, 0, 1, "McCarthy, Mr. Timothy J", "male", 54, 0, 0, "17463", 51.8625, "S"],
            [8, 0, 3, "Palsson, Master. Gosta Leonard", "male", 2, 3, 1, "349909", 21.075, "S"],
            [9, 1, 3, "Johnson, Mrs. Oscar W", "female", 27, 0, 2, "347742", 11.1333, "S"],
            [10, 1, 2, "Nasser, Mrs. Nicholas", "female", 14, 1, 0, "237736", 30.0708, "C"]
        ]
        
        df = pd.DataFrame(ornek_veriler, columns=[
            'PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 
            'SibSp', 'Parch', 'Ticket', 'Fare', 'Embarked'
        ])
        df.to_csv(dosya_adi, index=False)
        print(f"✅ Örnek Titanic verisi oluşturuldu: {dosya_adi}")
        print("📋 CSV Formatı: PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Embarked")
    
    def interaktif_analiz(self):
        """İnteraktif Titanic analiz modu"""
        print("🚢 Titanic Sosyal Ağ Analiz Sistemine Hoş Geldiniz!")
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
        
        # Titanic verisi yükleme
        print("\n📁 Titanic verisi yükleyin veya örnek veri oluşturun:")
        print("1. 'ornek' yazın - Örnek Titanic verisi oluşturulur")
        print("2. Dosya yolunu girin - Titanic CSV dosyası yüklenir")
        
        while True:
            secim = input("\n🤔 Seçiminiz: ").strip()
            
            if secim.lower() == 'ornek':
                self.ornek_titanic_verisi_olustur()
                if self.titanic_verisi_yukle("titanic_ornek.csv"):
                    break
            elif os.path.exists(secim):
                if self.titanic_verisi_yukle(secim):
                    break
            else:
                print("❌ Geçersiz seçim! Lütfen 'ornek' yazın veya geçerli dosya yolu girin.")
        
        # Titanic bilgilerini göster
        self.titanic_bilgilerini_goster()
        
        # İnteraktif sorgu döngüsü
        print("\n💬 Titanic analiz sorgularınızı sorun (çıkmak için 'quit' yazın):")
        print("📋 Komutlar: 'veri' = analiz verilerini göster, 'dogrula' = son cevabı doğrula")
        print("-"*60)
        
        son_cevap = ""
        while True:
            soru = input("\n🤔 Sorunuz: ").strip()
            
            if soru.lower() in ['quit', 'exit', 'çık', 'q']:
                print("👋 Görüşürüz!")
                break
            
            if soru.lower() == 'veri':
                print("\n📊 DETAYLI TİTANIC ANALİZ VERİLERİ:")
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
    analyzer = TitanicAgAnalizor()
    analyzer.interaktif_analiz() 