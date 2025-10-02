import requests
from bs4 import BeautifulSoup

def scrape_wikipedia_article(url):
    """
    Wikipedia makalesinden sadece article kısmının text içeriğini çeker.
    
    Args:
        url (str): Wikipedia makale URL'si
    
    Returns:
        str: Makalenin text içeriği
    """
    try:
        # Wikipedia sayfasını çek (User-Agent header ile)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # HTML'i parse et
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Article içeriğini bul (Wikipedia'da content div'i içinde)
        article_content = soup.find('div', {'id': 'mw-content-text'})
        
        if not article_content:
            return "Article içeriği bulunamadı."
        
        # Sadece paragrafları al (article text'i)
        paragraphs = article_content.find_all('p')
        
        # Text'leri birleştir
        article_text = '\n\n'.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        
        return article_text
    
    except requests.exceptions.RequestException as e:
        return f"Hata oluştu: {e}"

if __name__ == "__main__":
    # Mustafa Kemal Atatürk Wikipedia sayfası
    url = "https://en.wikipedia.org/wiki/Mustafa_Kemal_Atat%C3%BCrk"
    
    print("Wikipedia makalesinden text çekiliyor...\n")
    print("=" * 80)
    
    # Makale text'ini çek
    article_text = scrape_wikipedia_article(url)
    
    # Sonucu yazdır
    print(article_text)
    
    print("\n" + "=" * 80)
    print(f"\nToplam karakter sayısı: {len(article_text)}")
    
    # İsterseniz dosyaya da kaydedebilirsiniz
    with open('ataturk_article.txt', 'w', encoding='utf-8') as f:
        f.write(article_text)
    print("Makale 'ataturk_article.txt' dosyasına kaydedildi.")
