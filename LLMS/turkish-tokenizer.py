from datasets import load_dataset
from tokenizers import Tokenizer, normalizers, pre_tokenizers, decoders
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.processors import TemplateProcessing
import os
import re
from collections import Counter

print("="*70)
print("TÜRKÇE WORDPIECE TOKENIZER - Production Ready")
print("="*70)

# 1. DATASET YÜKLEME
print("\n📚 DATASET YÜKLEME...")
print("-"*70)

datasets_config = [
    ("winvoker/turkish-sentiment-analysis-dataset", "train", "text"),
    ("kmkarakaya/turkishReviews-ds-mini", "train", "review"),
    ("savasy/ttc4900", "train", "text"),
    ("merve/turkish_instructions", "train", "talimat"),
]

corpus = []
dataset_stats = {}

for config in datasets_config:
    dataset_name, split, text_field = config
    
    try:
        print(f"⏳ {dataset_name[:40]}...", end=" ")
        ds = load_dataset(dataset_name, split=split, streaming=True)
        
        count = 0
        for item in ds:
            if text_field in item and item[text_field]:
                text = str(item[text_field]).strip()
                if len(text) > 50:
                    corpus.append(text)
                    count += 1
            
            if count >= 100000:
                break
        
        dataset_stats[dataset_name] = count
        print(f"✓ {count:,}")
        
    except Exception as e:
        print(f"✗ {str(e)[:30]}")

# Fallback
if len(corpus) < 1000:
    print("\n⚠️  Manuel örnekler ekleniyor...")
    manual = [
        "Merhaba, nasılsın? Bugün hava çok güzel.",
        "Türkiye'nin başkenti Ankara'dır.",
        "Bu ürün çok kaliteli, tavsiye ederim.",
        "Yapay zeka teknolojileri gelişiyor.",
        "İstanbul'da çok güzel yerler var.",
    ] * 300
    corpus.extend(manual)
    dataset_stats["manual"] = len(manual)

# 2. İSTATİSTİKLER
print("\n" + "="*70)
print("📊 CORPUS İSTATİSTİKLERİ")
print("="*70)

total_texts = len(corpus)
total_chars = sum(len(t) for t in corpus)
total_words = sum(len(t.split()) for t in corpus)

print(f"📝 Metin: {total_texts:,}")
print(f"📝 Karakter: {total_chars:,}")
print(f"📝 Kelime: {total_words:,}")
print(f"📝 Ort metin: {total_chars/total_texts:.1f} char")
print(f"📝 Ort kelime: {total_chars/total_words:.1f} char")

all_chars = Counter(''.join(corpus))
print(f"\n🔤 Unique karakter: {len(all_chars)}")

# 3. DİNAMİK PARAMETRE - WordPiece için optimize
print("\n" + "="*70)
print("⚙️  PARAMETRE OPTİMİZASYONU (WordPiece)")
print("="*70)

def calculate_wordpiece_params(total_chars, total_words):
    """WordPiece için optimal parametreler - Min freq düşürüldü"""
    
    # WordPiece için küçük vocab + DÜŞÜK min_freq (daha fazla merge için)
    if total_chars < 100_000:
        vocab = 2_000
        min_freq = 2
    elif total_chars < 500_000:
        vocab = 3_500
        min_freq = 2
    elif total_chars < 2_000_000:
        vocab = 5_000
        min_freq = 2
    elif total_chars < 10_000_000:
        vocab = 8_000
        min_freq = 3  # 10M+ için min 3
    else:
        vocab = 10_000
        min_freq = 3  # Büyük corpus için bile min 3 yeterli
    
    return {'vocab_size': vocab, 'min_frequency': min_freq}

params = calculate_wordpiece_params(total_chars, total_words)

print(f"✓ Vocab boyutu: {params['vocab_size']:,}")
print(f"✓ Min frekans: {params['min_frequency']}")
print(f"  ℹ️  Min freq düşük tutuldu → Daha fazla merge, daha uzun eğitim")

# Efficiency check - Daha agresif
estimated_tokens = total_words * 1.15
efficiency = (estimated_tokens / params['vocab_size']) * 100
print(f"✓ Tahmini kullanım: {efficiency:.1f}%")

if efficiency > 700:  # Çok yüksek threshold
    params['vocab_size'] = int(params['vocab_size'] * 1.2)
    print(f"  → Vocab artırıldı: {params['vocab_size']:,}")
elif efficiency < 50:  # Düşük kullanım
    params['vocab_size'] = int(params['vocab_size'] * 0.8)
    print(f"  → Vocab azaltıldı: {params['vocab_size']:,}")

# 4. CORPUS TEMİZLEME
print("\n" + "="*70)
print("🧹 TEMİZLEME")
print("="*70)

def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

cleaned = [clean_text(t) for t in corpus if len(clean_text(t)) >= 30]
print(f"✓ Temiz: {len(cleaned):,}")
print(f"✓ Filtre: {len(corpus) - len(cleaned):,}")

corpus_file = "turkish_corpus.txt"
with open(corpus_file, "w", encoding="utf-8") as f:
    for line in cleaned:
        f.write(line + "\n")
print(f"✓ Dosya: {corpus_file}")

# 5. WORDPIECE TOKENIZER OLUŞTURMA
print("\n" + "="*70)
print("🔧 WORDPIECE TOKENIZER")
print("="*70)

# WordPiece modeli
tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))

# Normalizasyon - Türkçe karakterleri KORU (lowercase YOK)
tokenizer.normalizer = normalizers.Sequence([
    normalizers.NFC(),  # Unicode normalize
])

# Pre-tokenizer - Basit ve etkili
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Decoder - WordPiece için
tokenizer.decoder = decoders.WordPiece(prefix="##")

# Special tokens
special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

# Trainer
trainer = WordPieceTrainer(
    vocab_size=params['vocab_size'],
    min_frequency=params['min_frequency'],
    special_tokens=special_tokens,
    show_progress=True,
    continuing_subword_prefix="##",
)

# Post-processor
tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", special_tokens.index("[CLS]")),
        ("[SEP]", special_tokens.index("[SEP]")),
    ],
)

# Padding & truncation
tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
tokenizer.enable_truncation(max_length=512)

print(f"✓ Model: WordPiece")
print(f"✓ Vocab: {params['vocab_size']:,}")
print(f"✓ Min freq: {params['min_frequency']}")
print(f"✓ Prefix: ##")

# 6. EĞİTİM
print("\n" + "="*70)
print("🚀 EĞİTİM")
print("="*70)
print("ℹ️  Eğitim süresi vocab boyutu ve min_freq'e bağlıdır")
print("ℹ️  Daha fazla merge = Daha uzun eğitim = Daha iyi tokenizer")
print()

import time
start_time = time.time()
tokenizer.train([corpus_file], trainer)
elapsed = time.time() - start_time

output_file = "turkish_wordpiece.json"
tokenizer.save(output_file)
print(f"\n✓ Kaydedildi: {output_file}")
print(f"✓ Final vocab: {tokenizer.get_vocab_size():,}")
print(f"✓ Eğitim süresi: {elapsed:.1f} saniye")

# Eğitim analizi
actual_vocab = tokenizer.get_vocab_size()
target_vocab = params['vocab_size']
vocab_fill_rate = (actual_vocab / target_vocab) * 100

print(f"\n📊 Eğitim Analizi:")
print(f"  Hedef vocab: {target_vocab:,}")
print(f"  Gerçekleşen vocab: {actual_vocab:,}")
print(f"  Doluluk oranı: {vocab_fill_rate:.1f}%")

if vocab_fill_rate < 95:
    print(f"  ⚠️  Vocab hedefine ulaşılamadı!")
    print(f"  💡 Çözüm: min_frequency'i {max(1, params['min_frequency']-1)} yap")
elif vocab_fill_rate > 99:
    print(f"  ✅ Vocab hedefine ulaşıldı!")
else:
    print(f"  ✓ Vocab hedefine yakın")

# 7. TEST
print("\n" + "="*70)
print("🧪 TEST")
print("="*70)

test_texts = [
    "Merhaba, nasılsınız? Bugün hava çok güzel!",
    "Türkçe karakterler: ç ğ ı ö ş ü Ç Ğ İ Ö Ş Ü",
    "Morfologi: yapmışsınız, gidiyorum, kalkmış, verdikten",
    "Yapay zeka, makine öğrenmesi, derin öğrenme",
    "İstanbul'da Boğaz'da balık ekmek yedik.",
    "Cumhurbaşkanı bugün önemli açıklama yaptı.",
    "Bu ürünü aldım ve çok memnun kaldım!",
]

test_results = []
for i, text in enumerate(test_texts, 1):
    output = tokenizer.encode(text)
    decoded = tokenizer.decode(output.ids, skip_special_tokens=True)
    
    test_results.append({
        'text': text,
        'tokens': output.tokens,
        'token_count': len(output.tokens),
        'char_count': len(text),
        'decoded': decoded,
    })
    
    print(f"\n{i}. {text}")
    print(f"   Tokens ({len(output.tokens)}): {output.tokens[:12]}...")
    print(f"   Decode: {decoded}")
    
    # Türkçe karakter kaybı kontrolü
    original_turkish = set('çğıöşüÇĞİÖŞÜ')
    text_turkish = original_turkish & set(text)
    decoded_turkish = original_turkish & set(decoded)
    
    if text_turkish and not decoded_turkish:
        print(f"   ⚠️  UYARI: Türkçe karakterler kayboldu!")
        print(f"   Orijinal: {text_turkish}")
        print(f"   Decode: {decoded_turkish if decoded_turkish else 'YOK'}")

# 8. METRİKLER
print("\n" + "="*70)
print("📈 PERFORMANS")
print("="*70)

total_test_tokens = sum(r['token_count'] for r in test_results)
total_test_chars = sum(r['char_count'] for r in test_results)
total_test_words = sum(len(r['text'].split()) for r in test_results)

char_per_token = total_test_chars / total_test_tokens
token_per_word = total_test_tokens / total_test_words

print(f"✓ Char/Token: {char_per_token:.2f}")
print(f"✓ Token/Word: {token_per_word:.2f}")
print(f"✓ Vocab: {tokenizer.get_vocab_size():,}")

# Değerlendirme
print(f"\n🎯 Değerlendirme:")
if 3.0 <= char_per_token <= 5.0:
    print(f"  ✓ Char/Token optimal")
else:
    print(f"  ⚠️  Char/Token: {char_per_token:.2f}")

if token_per_word <= 1.4:
    print(f"  ✓ Token/Word optimal")
elif token_per_word <= 1.6:
    print(f"  ⚠️  Token/Word kabul edilebilir: {token_per_word:.2f}")
else:
    print(f"  ❌ Token/Word yüksek: {token_per_word:.2f}")

# 9. MORFOLOJİ
print("\n" + "="*70)
print("🔬 MORFOLOJİK ANALİZ")
print("="*70)

morph_words = [
    "yapmışsınız",
    "gidiyorum", 
    "kalkmış",
    "verdikten",
    "okuyorum",
    "geliyorlar",
]

for word in morph_words:
    output = tokenizer.encode(word)
    tokens = [t for t in output.tokens if not t.startswith('[')]
    print(f"  {word:15} -> {tokens}")

# 10. ÖZET
print("\n" + "="*70)
print("✅ TAMAMLANDI")
print("="*70)
print(f"📁 Tokenizer: {output_file}")
print(f"📁 Corpus: {corpus_file}")
print(f"📊 {len(cleaned):,} metin, {total_chars:,} karakter")

# Türkçe karakter kontrolü
turkish_char_lost = any('⚠️  UYARI: Türkçe karakterler kayboldu!' in str(r) for r in test_results)

if token_per_word <= 1.3 and not turkish_char_lost:
    print(f"\n🎉 MÜKEMMEL! Production-ready")
    print(f"   ✓ Token/Word: {token_per_word:.2f}")
    print(f"   ✓ Türkçe karakterler: Korundu")
elif token_per_word <= 1.5 and not turkish_char_lost:
    print(f"\n✅ İYİ! Kullanılabilir")
    print(f"   ✓ Token/Word: {token_per_word:.2f}")
    print(f"   ✓ Türkçe karakterler: Korundu")
    print(f"   💡 Vocab'u {int(params['vocab_size']*0.7):,} yap")
elif turkish_char_lost:
    print(f"\n❌ KRİTİK HATA: Türkçe karakterler kayboldu!")
    print(f"   Normalizasyon StripAccents kullanılmamalı!")
else:
    print(f"\n⚠️  İyileştirme gerekli")
    print(f"   Token/Word: {token_per_word:.2f} (hedef: <1.5)")
    print(f"   💡 Vocab'u {int(params['vocab_size']*0.5):,} yap")