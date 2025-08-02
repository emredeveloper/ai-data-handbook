import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import seaborn as sns

# Basit bir sosyal ağ oluşturalım (gerçek veri yerine örnek veri)
def sosyal_ag_olustur():
    G = nx.Graph()
    
    # Kişiler ve arkadaşlıklar
    kisiler = ["Ali", "Burcu", "Can", "Deniz", "Ece", "Fatma", "Gamze", "Hakan"]
    arkadasliklar = [
        ("Ali", "Burcu"), ("Ali", "Can"), ("Ali", "Deniz"),
        ("Burcu", "Can"), ("Burcu", "Ece"), ("Burcu", "Fatma"),
        ("Can", "Deniz"), ("Deniz", "Ece"), ("Deniz", "Gamze"),
        ("Ece", "Fatma"), ("Ece", "Gamze"), ("Fatma", "Gamze"),
        ("Gamze", "Hakan"), ("Hakan", "Fatma")
    ]
    
    G.add_nodes_from(kisiler)
    G.add_edges_from(arkadasliklar)
    return G

# Ağı görselleştirme
def agi_goster(G):
    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    
    # Merkezilik değerlerine göre renklendirme
    derece = nx.degree_centrality(G)
    node_colors = [derece[node] for node in G.nodes()]
    
    # Node boyutlarını merkeziliğe göre ayarlama
    node_sizes = [3000 * derece[node] + 500 for node in G.nodes()]
    
    nx.draw(G, pos, with_labels=True, 
            node_color=node_colors, 
            node_size=node_sizes,
            font_weight='bold',
            cmap=plt.cm.Reds,
            edge_color='gray',
            width=2,
            ax=ax)
    
    ax.set_title("Sosyal Ağ Yapısı (Node boyutu merkeziliği gösterir)")
    
    # Colorbar ekleme
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Derece Merkeziliği')
    
    plt.show()

# Merkezilik analizi yapma
def merkezilik_analizi(G):
    # Derece merkeziliği (en çok bağlantısı olan)
    derece = nx.degree_centrality(G)
    
    # Betweenness (köprü görevi görenler)
    betweenness = nx.betweenness_centrality(G)
    
    # Closeness (tüm ağa hızlı erişim)
    closeness = nx.closeness_centrality(G)
    
    # Eigenvector (diğer etkililerle bağlantı)
    eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
    
    return derece, betweenness, closeness, eigenvector

# Topluluk tespiti
def topluluk_tespiti(G):
    # Louvain algoritması ile topluluk tespiti
    try:
        from community import best_partition
        partition = best_partition(G)
        return partition
    except ImportError:
        # Girvan-Newman algoritması (daha yavaş ama built-in)
        communities = list(nx.community.girvan_newman(G))
        if communities:
            return communities[0]
        return None

# Ağ metrikleri hesaplama
def ag_metrikleri(G):
    metrikler = {}
    
    # Temel metrikler
    metrikler['node_sayisi'] = G.number_of_nodes()
    metrikler['edge_sayisi'] = G.number_of_edges()
    metrikler['density'] = nx.density(G)
    metrikler['average_clustering'] = nx.average_clustering(G)
    metrikler['average_shortest_path'] = nx.average_shortest_path_length(G)
    
    # Bağlantılılık
    metrikler['connected_components'] = nx.number_connected_components(G)
    metrikler['is_connected'] = nx.is_connected(G)
    
    # Çap ve yarıçap
    if nx.is_connected(G):
        metrikler['diameter'] = nx.diameter(G)
        metrikler['radius'] = nx.radius(G)
    
    return metrikler

# En etkili kişileri bulma
def en_etkili_kisiler(G, merkezilikler):
    derece, betweenness, closeness, eigenvector = merkezilikler
    
    # Toplam etki skoru hesaplama
    etki_skorlari = {}
    for node in G.nodes():
        etki_skorlari[node] = (
            derece[node] * 0.3 +
            betweenness[node] * 0.3 +
            closeness[node] * 0.2 +
            eigenvector[node] * 0.2
        )
    
    return etki_skorlari

# Sonuçları gösterme
def sonuclari_goster(kisiler, merkezilikler):
    derece, betweenness, closeness, eigenvector = merkezilikler
    
    print("En Çok Bağlantısı Olanlar (Derece Merkeziliği):")
    for kisi in sorted(kisiler, key=lambda x: -derece[x]):
        print(f"{kisi}: {derece[kisi]:.2f}")
    
    print("\nEn Önemli Köprü Kişiler (Betweenness Merkeziliği):")
    for kisi in sorted(kisiler, key=lambda x: -betweenness[x]):
        print(f"{kisi}: {betweenness[kisi]:.2f}")
    
    print("\nTüm Ağa En Hızlı Erişenler (Closeness Merkeziliği):")
    for kisi in sorted(kisiler, key=lambda x: -closeness[x]):
        print(f"{kisi}: {closeness[kisi]:.2f}")
    
    print("\nDiğer Etkililerle Bağlantılı Olanlar (Eigenvector Merkeziliği):")
    for kisi in sorted(kisiler, key=lambda x: -eigenvector[x]):
        print(f"{kisi}: {eigenvector[kisi]:.2f}")

# Metrikleri görselleştirme
def metrikleri_gorsellestir(G, merkezilikler, etki_skorlari):
    derece, betweenness, closeness, eigenvector = merkezilikler
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Derece merkeziliği
    axes[0,0].bar(derece.keys(), derece.values(), color='skyblue')
    axes[0,0].set_title('Derece Merkeziliği')
    axes[0,0].set_ylabel('Merkezilik Değeri')
    
    # Betweenness merkeziliği
    axes[0,1].bar(betweenness.keys(), betweenness.values(), color='lightgreen')
    axes[0,1].set_title('Betweenness Merkeziliği')
    axes[0,1].set_ylabel('Merkezilik Değeri')
    
    # Closeness merkeziliği
    axes[1,0].bar(closeness.keys(), closeness.values(), color='lightcoral')
    axes[1,0].set_title('Closeness Merkeziliği')
    axes[1,0].set_ylabel('Merkezilik Değeri')
    
    # Etki skorları
    axes[1,1].bar(etki_skorlari.keys(), etki_skorlari.values(), color='gold')
    axes[1,1].set_title('Toplam Etki Skoru')
    axes[1,1].set_ylabel('Etki Skoru')
    
    plt.tight_layout()
    plt.show()

# Ağ istatistiklerini gösterme
def ag_istatistikleri_goster(G, metrikler):
    print("\n" + "="*50)
    print("AĞ İSTATİSTİKLERİ")
    print("="*50)
    print(f"Toplam Kişi Sayısı: {metrikler['node_sayisi']}")
    print(f"Toplam Bağlantı Sayısı: {metrikler['edge_sayisi']}")
    print(f"Ağ Yoğunluğu: {metrikler['density']:.3f}")
    print(f"Ortalama Kümeleme Katsayısı: {metrikler['average_clustering']:.3f}")
    print(f"Ortalama En Kısa Yol Uzunluğu: {metrikler['average_shortest_path']:.3f}")
    print(f"Bağlantılı Bileşen Sayısı: {metrikler['connected_components']}")
    print(f"Ağ Bağlantılı mı?: {'Evet' if metrikler['is_connected'] else 'Hayır'}")
    
    if metrikler['is_connected']:
        print(f"Ağ Çapı: {metrikler['diameter']}")
        print(f"Ağ Yarıçapı: {metrikler['radius']}")

# Ana program
if __name__ == "__main__":
    print("Gelişmiş Sosyal Ağ Analizine Hoş Geldiniz!")
    print("="*50)
    
    # Ağı oluştur ve göster
    G = sosyal_ag_olustur()
    agi_goster(G)
    
    # Analiz yap
    kisiler = list(G.nodes())
    merkezilikler = merkezilik_analizi(G)
    metrikler = ag_metrikleri(G)
    etki_skorlari = en_etkili_kisiler(G, merkezilikler)
    
    # Sonuçları göster
    sonuclari_goster(kisiler, merkezilikler)
    
    # Ağ istatistiklerini göster
    ag_istatistikleri_goster(G, metrikler)
    
    # En etkili kişileri göster
    print("\n" + "="*50)
    print("EN ETKİLİ KİŞİLER (Toplam Skor)")
    print("="*50)
    for kisi in sorted(etki_skorlari.keys(), key=lambda x: -etki_skorlari[x]):
        print(f"{kisi}: {etki_skorlari[kisi]:.3f}")
    
    # Metrikleri görselleştir
    metrikleri_gorsellestir(G, merkezilikler, etki_skorlari)