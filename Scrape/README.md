# 🎓 English Learning Platform

Wikipedia makaleleriyle İngilizce öğrenme platformu. Grok AI destekli!

## 🚀 Özellikler

- **📰 Wikipedia Scraping**: Herhangi bir İngilizce Wikipedia makalesini otomatik çeker
- **📊 Seviye Tespiti**: İngilizce seviyenizi A1-C2 arasında ölçer
- **📚 Kelime Pratiği**: Makaleden önemli kelimeleri öğrenin
- **✍️ Gramer Pratiği**: Makaledeki gramer yapılarını pratikte kullanın
- **🤔 Anlama Pratiği**: Okuma anlama sorularıyla kendinizi test edin
- **🎮 Quiz**: İnteraktif quiz'lerle öğrenin
- **💬 Metin Açıklama**: Anlamadığınız bölümleri seçip açıklama alın

## 📦 Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

2. OpenRouter API Key alın:
   - https://openrouter.ai/ adresine gidin
   - Ücretsiz API key alın

3. API Key'i ayarlayın:
```bash
# Windows
set OPENROUTER_API_KEY=your_api_key_here

# Linux/Mac
export OPENROUTER_API_KEY=your_api_key_here
```

Veya `app.py` dosyasındaki `OPENROUTER_API_KEY` değişkenini düzenleyin.

## 🎯 Kullanım

1. Uygulamayı başlatın:
```bash
python app.py
```

2. Tarayıcınızda açın:
```
http://localhost:5000
```

3. Wikipedia URL'si girin (varsayılan: Atatürk makalesi)

4. Öğrenmeye başlayın! 🎉

## 🛠️ Teknik Detaylar

- **Backend**: Flask (Python)
- **AI Model**: Grok-2-1212 (OpenRouter API)
- **Scraping**: BeautifulSoup4
- **Frontend**: Vanilla JavaScript + Modern CSS

## 📝 Örnek Workflow

1. Wikipedia makalesini yükle
2. "Seviyemi Ölç" ile İngilizce seviyeni öğren
3. Önerilen kelime ve gramer yapılarını gör
4. Pratik sorularını çöz
5. Anlamadığın kısımları seç ve açıkla

## 🌟 Özellikler

### AI Destekli Özellikler:
- ✅ Seviye analizi (A1-C2 CEFR)
- ✅ Kelime listesi (Türkçe çeviriyle)
- ✅ Gramer yapıları
- ✅ Anlama soruları
- ✅ Quiz oluşturma
- ✅ Seçili metin açıklama

### Kullanıcı Arayüzü:
- ✅ Modern gradient tasarım
- ✅ Responsive layout
- ✅ Metin seçimi desteği
- ✅ İstatistikler
- ✅ Okuma süresi hesaplama

## 📚 Desteklenen Wikipedia Dilleri

- İngilizce (en.wikipedia.org) ✅

## 🤝 Katkıda Bulunma

Pull request'ler her zaman kabul edilir!

## 📄 Lisans

MIT License

## 👨‍💻 Geliştirici

Emre - AI & Data Science Enthusiast

---

**Not**: OpenRouter API ücretsiz tier'da sınırlı kullanım sunar. Yoğun kullanım için ücretli planlara bakabilirsiniz.
