# YouTube Yorumlarında Spam Tespiti (TF-IDF + MultinomialNB + GridSearchCV)

Bu klasör, YouTube yorumlarında spam tespiti için bir metin sınıflandırma pipeline’ı içerir.

## Genel Açıklama

- Temizleme: URL/etiket/sayı/noktalama temizliği, küçük harfe dönüştürme
- Özellik: TF-IDF (1–2 n-gram, `stop_words='english'`)
- Model: Multinomial Naive Bayes
- Arama: GridSearchCV ile temel hiperparametre taraması
- Bölme: `train_test_split(..., stratify=y)` ile dengeli ayrım
- Değerlendirme: accuracy, `classification_report`, confusion matrix, ROC-AUC
- Demo: 10 örnek metin üzerinde toplu tahmin çıktısı

## Kurulum

```bash
python -m pip install -r "Classification/requirements.txt"
```

## Çalıştırma

`Classification/spam-or-not.ipynb` notebook’unu açıp hücreleri sırayla çalıştırın.

## Notlar

- CSV dosyaları `.gitignore` ile versiyon kontrolünden hariç tutulur (`Classification/*.csv`).
- İstenirse Türkçe stopword veya ek temizleme adımları kolayca eklenebilir.


