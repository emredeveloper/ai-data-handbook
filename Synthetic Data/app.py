"""
Sentetik metin sınıflandırma veri üretim aracı.

Örnek kullandef build_user_prompt(num_items: int, labels: List[str], topics: List[str], language: str) -> str:
    labels_str = ", ".join(labels)
    topics_str = ", ".join(topics) if topics else "çeşitli günlük yaşam konuları"
    return (
        f"{language} dilinde kısa ve doğal metinler üret. "
        f"Görev: verilen sınıflara yönelik metin sınıflandırma veri seti.\n"
        f"Her örnek bir cümle veya kısa paragraf olsun.\n"
        f"Sınıflar: [{labels_str}]\n"
        f"Kapsam/konular: {topics_str}\n"
        f"Çeşitli üslup, kelime dağarcığı ve bağlam kullan. Abartılı veya toksik içerik üretme. "
        f"Kişisel veri (PII) ve marka/ad kullanımını uydurme. Gerçekçi ama anonim kal.\n"
        f"Tam olarak {num_items} örnek üret ve SADECE geçerli JSON DİZİSİ döndür.\n"
        f"Şema örneği: [{{\"text\": \"metin\", \"label\": \"{labels[0]}\"}}].\n"
        f"JSON dışında hiçbir açıklama, kod bloğu veya etiket ekleme."
    )


def build_diverse_user_prompt(num_items: int, labels: List[str], topics: List[str], language: str, style_variety: int = 1) -> str:
    labels_str = ", ".join(labels)
    topics_str = ", ".join(topics) if topics else "çeşitli günlük yaşam konuları"
    
    # Farklı tarzlar için varyasyonlar
    style_variants = [
        # Tarz 1: Günlük konuşma dili, kişisel deneyimler
        f"{language} dilinde günlük konuşma tarzında, kişisel deneyim ifade eden metinler üret. "
        f"Birinci şahıs anlatım kullan ('ben', 'benim', 'bana' gibi). "
        f"Duygusal ifadeler ve kişisel yorumlar içersin. "
        f"Örnekler: 'Bu sabah kahvemi içerken çok keyif aldım', 'Patronumla yaptığım toplantı beni gerdi'.",
        
        # Tarz 2: Soru cümleleri ve diyaloglar
        f"{language} dilinde soru cümleleri ve diyalog tarzında metinler üret. "
        f"'mı/mi', 'nasıl', 'neden', 'kim' gibi soru kelimeleri kullan. "
        f"Doğrudan hitap içeren ifadeler ekle. "
        f"Örnekler: 'Bu filmi sen de beğendin mi?', 'Neden bu kadar geciktin?'.",
        
        # Tarz 3: Uzun açıklayıcı cümleler
        f"{language} dilinde uzun ve açıklayıcı cümleler üret (25-40 kelime arası). "
        f"Bağlı cümleler ve detaylı anlatımlar kullan. "
        f"Sebep-sonuç ilişkileri kur. "
        f"Örnekler: 'Geçen hafta başlayan yağmurlar yüzünden şehir trafiği tamamen felce uğradı ve herkesin işe gecikmesine neden oldu'.",
        
        # Tarz 4: Argo ve günlük ifadeler
        f"{language} dilinde günlük argo ve konuşma ifadeleri içeren metinler üret. "
        f"'ya', 'işte', 'falan', 'zaten', 'hani' gibi dolgu kelimeler kullan. "
        f"Samimi ve rahat bir ton benimse. "
        f"Örnekler: 'Ya bu işler falan hiç bitmez ki', 'Hani şu dediğin restoran, orası süperdi işte'.",
        
        # Tarz 5: Çok kısa ifadeler ve ünlemler
        f"{language} dilinde çok kısa ifadeler (3-8 kelime) ve ünlem cümleleri üret. "
        f"'Vay be!', 'Ne güzel!', 'Berbat!', 'Süper!' gibi ifadeler kullan. "
        f"Keskin ve net duygusal tepkiler. "
        f"Örnekler: 'Harika bir gün!', 'Berbat hava durumu.', 'Mükemmel performans!'."
    ]
    
    selected_style = style_variants[style_variety % len(style_variants)]
    
    return (
        f"{selected_style}\n"
        f"Sınıflar: [{labels_str}]\n"
        f"Kapsam/konular: {topics_str}\n"
        f"Mevcut veri setindeki örneklerden farklı tarz ve yapıda ol. "
        f"Çeşitli kelime seçimleri ve cümle yapıları kullan. "
        f"Abartılı veya toksik içerik üretme. Gerçekçi ama anonim kal.\n"
        f"Tam olarak {num_items} örnek üret ve SADECE geçerli JSON DİZİSİ döndür.\n"
        f"Şema örneği: [{{\"text\": \"metin\", \"label\": \"{labels[0]}\"}}].\n"
        f"JSON dışında hiçbir açıklama, kod bloğu veya etiket ekleme."
    )Syntetic Data/app.py" \
    --num-samples 200 \
    --labels "olumlu,olumsuz,nötr" \
    --model "x-ai/grok-4-fast:free"

Gerekli ortam değişkenleri:
  - OPENROUTER_API_KEY: OpenRouter API anahtarı
  - (opsiyonel) OPENROUTER_REFERER: HTTP-Referer
  - (opsiyonel) OPENROUTER_TITLE: X-Title
Push işlemi artık ayrı bir dosyada: `push_hf.py`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass
class GenerationConfig:
    model: str
    language: str
    batch_size: int
    topics: List[str]
    labels: List[str]
    temperature: float = 0.8
    max_retries: int = 5
    retry_base_delay: float = 2.0


def get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name, default)
    return value


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_system_prompt(language: str) -> str:
    return (
        "You are a careful data generator for text classification datasets. "
        "Return ONLY valid JSON with no extra text."
    )


def build_user_prompt(num_items: int, labels: List[str], topics: List[str], language: str) -> str:
    labels_str = ", ".join(labels)
    topics_str = ", ".join(topics) if topics else "çeşitli günlük yaşam konuları"
    return (
        f"{language} dilinde kısa ve doğal metinler üret. "
        f"Görev: verilen sınıflara yönelik metin sınıflandırma veri seti.\n"
        f"Her örnek bir cümle veya kısa paragraf olsun.\n"
        f"Sınıflar: [{labels_str}]\n"
        f"Kapsam/konular: {topics_str}\n"
        f"Çeşitli üslup, kelime dağarcığı ve bağlam kullan. Abartılı veya toksik içerik üretme. "
        f"Kişisel veri (PII) ve marka/ad kullanımını uydurma. Gerçekçi ama anonim kal.\n"
        f"Tam olarak {num_items} örnek üret ve SADECE geçerli JSON DİZİSİ döndür.\n"
        f"Şema örneği: [{{\"text\": \"metin\", \"label\": \"{labels[0]}\"}}].\n"
        f"JSON dışında hiçbir açıklama, kod bloğu veya etiket ekleme."
    )


def build_diverse_user_prompt(num_items: int, labels: List[str], topics: List[str], language: str, style_variety: int = 1) -> str:
    labels_str = ", ".join(labels)
    topics_str = ", ".join(topics) if topics else "çeşitli günlük yaşam konuları"
    
    # Farklı tarzlar için varyasyonlar
    style_variants = [
        # Tarz 1: Günlük konuşma dili, kişisel deneyimler
        f"{language} dilinde günlük konuşma tarzında, kişisel deneyim ifade eden metinler üret. "
        f"Birinci şahıs anlatım kullan ('ben', 'benim', 'bana' gibi). "
        f"Duygusal ifadeler ve kişisel yorumlar içersin. "
        f"Örnekler: 'Bu sabah kahvemi içerken çok keyif aldım', 'Patronumla yaptığım toplantı beni gerdi'.",
        
        # Tarz 2: Soru cümleleri ve diyaloglar
        f"{language} dilinde soru cümleleri ve diyalog tarzında metinler üret. "
        f"'mı/mi', 'nasıl', 'neden', 'kim' gibi soru kelimeleri kullan. "
        f"Doğrudan hitap içeren ifadeler ekle. "
        f"Örnekler: 'Bu filmi sen de beğendin mi?', 'Neden bu kadar geciktin?'.",
        
        # Tarz 3: Uzun açıklayıcı cümleler
        f"{language} dilinde uzun ve açıklayıcı cümleler üret (25-40 kelime arası). "
        f"Bağlı cümleler ve detaylı anlatımlar kullan. "
        f"Sebep-sonuç ilişkileri kur. "
        f"Örnekler: 'Geçen hafta başlayan yağmurlar yüzünden şehir trafiği tamamen felce uğradı ve herkesin işe gecikmesine neden oldu'.",
        
        # Tarz 4: Argo ve günlük ifadeler
        f"{language} dilinde günlük argo ve konuşma ifadeleri içeren metinler üret. "
        f"'ya', 'işte', 'falan', 'zaten', 'hani' gibi dolgu kelimeler kullan. "
        f"Samimi ve rahat bir ton benimse. "
        f"Örnekler: 'Ya bu işler falan hiç bitmez ki', 'Hani şu dediğin restoran, orası süperdi işte'.",
        
        # Tarz 5: Çok kısa ifadeler ve ünlemler
        f"{language} dilinde çok kısa ifadeler (3-8 kelime) ve ünlem cümleleri üret. "
        f"'Vay be!', 'Ne güzel!', 'Berbat!', 'Süper!' gibi ifadeler kullan. "
        f"Keskin ve net duygusal tepkiler. "
        f"Örnekler: 'Harika bir gün!', 'Berbat hava durumu.', 'Mükemmel performans!'."
    ]
    
    selected_style = style_variants[style_variety % len(style_variants)]
    
    return (
        f"{selected_style}\n"
        f"Sınıflar: [{labels_str}]\n"
        f"Kapsam/konular: {topics_str}\n"
        f"Mevcut veri setindeki örneklerden farklı tarz ve yapıda ol. "
        f"Çeşitli kelime seçimleri ve cümle yapıları kullan. "
        f"Abartılı veya toksik içerik üretme. Gerçekçi ama anonim kal.\n"
        f"Tam olarak {num_items} örnek üret ve SADECE geçerli JSON DİZİSİ döndür.\n"
        f"Şema örneği: [{{\"text\": \"metin\", \"label\": \"{labels[0]}\"}}].\n"
        f"JSON dışında hiçbir açıklama, kod bloğu veya etiket ekleme."
    )


def call_openrouter(api_key: str, model: str, system_prompt: str, user_prompt: str, temperature: float = 0.8) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    referer = get_env("OPENROUTER_REFERER")
    title = get_env("OPENROUTER_TITLE")
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }

    resp = requests.post(OPENROUTER_URL, headers=headers, data=json.dumps(payload), timeout=120)
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    return content


def extract_json_array(text: str) -> List[Dict[str, Any]]:
    # Doğrudan JSON dizi pars etmeyi dene
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed  # type: ignore
    except Exception:
        pass

    # Köşeli parantez aralığını bulup dener
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                return parsed  # type: ignore
        except Exception:
            pass

    # JSONL benzeri satırları dene
    items: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if not line:
            continue
        if not line.startswith("{") or not line.endswith("}"):
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                items.append(obj)
        except Exception:
            continue
    if items:
        return items

    raise ValueError("Geçerli JSON dizi bulunamadı")


def normalize_and_validate(items: List[Dict[str, Any]], labels: List[str]) -> List[Dict[str, str]]:
    label_set = {l.strip().lower() for l in labels}
    normalized: List[Dict[str, str]] = []
    for it in items:
        text_val = str(it.get("text", "")).strip()
        label_val = str(it.get("label", "")).strip()
        if not text_val or not label_val:
            continue
        label_norm = label_val.lower()
        if label_norm in label_set:
            normalized.append({"text": text_val, "label": label_norm})
        else:
            # Basit eşleştirme: boşluk/aksan farklarına tolerans
            simplified = re.sub(r"[^a-zA-Z0-9çğıöşüÇĞİÖŞÜ]+", "", label_norm)
            match = None
            for l in label_set:
                if simplified == re.sub(r"[^a-zA-Z0-9çğıöşüÇĞİÖŞÜ]+", "", l):
                    match = l
                    break
            if match:
                normalized.append({"text": text_val, "label": match})
    return normalized


def generate_dataset(api_key: str, gen: GenerationConfig, total_items: int, use_diverse_styles: bool = False) -> List[Dict[str, str]]:
    all_items: List[Dict[str, str]] = []
    system_prompt = build_system_prompt(gen.language)

    remaining = total_items
    batch_index = 0
    while remaining > 0:
        batch_index += 1
        this_batch = min(remaining, gen.batch_size)
        
        # Farklı tarzlar için prompt seçimi
        if use_diverse_styles:
            # Her batch'te farklı bir tarz kullan
            style_index = (batch_index - 1) % 5  # 5 farklı tarz var
            user_prompt = build_diverse_user_prompt(this_batch, gen.labels, gen.topics, gen.language, style_index)
            print(f"📝 Parti {batch_index}: Tarz {style_index + 1} kullanılıyor...")
        else:
            user_prompt = build_user_prompt(this_batch, gen.labels, gen.topics, gen.language)

        for attempt in range(gen.max_retries):
            try:
                content = call_openrouter(
                    api_key=api_key,
                    model=gen.model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=gen.temperature,
                )
                raw_items = extract_json_array(content)
                cleaned = normalize_and_validate(raw_items, gen.labels)
                if not cleaned:
                    raise ValueError("Boş veya geçersiz çıktı alındı")
                all_items.extend(cleaned)
                remaining -= len(cleaned)
                print(f"✔️ Parti {batch_index}: {len(cleaned)} örnek eklendi (kalan hedef ~{max(0, remaining)}).")
                break
            except Exception as e:  # noqa: BLE001
                wait = (gen.retry_base_delay ** attempt) + random.random()
                print(f"Uyarı: Parti {batch_index} deneme {attempt+1} başarısız: {e}. {wait:.1f}s bekleniyor…")
                time.sleep(min(wait, 30))
        else:
            print("Hata: Maksimum deneme aşıldı, ilerlemeye devam ediliyor.")
            break

    # Fazla üretim olduysa kırp
    return all_items[: total_items]


def save_as_jsonl(items: List[Dict[str, str]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def save_as_csv(items: List[Dict[str, str]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"]) 
        writer.writeheader()
        for it in items:
            writer.writerow(it)


def train_test_split(items: List[Dict[str, str]], test_size: float = 0.1, seed: int = 42):
    rnd = random.Random(seed)
    shuffled = items[:]
    rnd.shuffle(shuffled)
    n_test = max(1, int(len(shuffled) * test_size))
    test = shuffled[:n_test]
    train = shuffled[n_test:]
    return train, test


def build_dataset_card(path: Path, labels: List[str], language: str, total: int, model: str) -> None:
    content = f"""
---
tags:
  - synthetic
  - text-classification
language:
  - {language}
task_categories:
  - text-classification
---

# Sentetik Metin Sınıflandırma Veri Seti ({language})

Bu veri seti, OpenRouter üzerinden "{model}" modeli kullanılarak otomatik üretilmiş sentetik metin sınıflandırma örneklerini içerir.

- Sınıflar: {", ".join(labels)}
- Örnek sayısı: {total}
- Farklı tarzlarda üretilmiş sentetik metinler (kişisel deneyim, soru cümleleri, argo ifadeler, vs.)
- Not: Metinler tamamen sentetiktir; gerçek kişi/kurum adları kullanılmamaya çalışılmıştır.

"""
    (path / "README.md").write_text(content.strip() + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Sentetik metin sınıflandırma veri üretimi ve HF push")
    parser.add_argument("--num-samples", type=int, default=500, help="Toplam üretilecek örnek sayısı")
    parser.add_argument("--labels", type=str, default="olumlu,olumsuz,nötr", help="Virgülle ayrılmış sınıflar")
    parser.add_argument("--topics", type=str, default="teknoloji,spor,ekonomi,sağlık,seyahat,kültür,sanat,eğitim", help="Virgülle ayrılmış konu/bağlamlar")
    parser.add_argument("--language", type=str, default="tr", help="Dil (ör. tr, en)")
    parser.add_argument("--model", type=str, default="x-ai/grok-4-fast:free", help="OpenRouter model kimliği")
    parser.add_argument("--batch-size", type=int, default=50, help="Tek seferde istenen örnek sayısı")
    parser.add_argument("--temperature", type=float, default=0.8, help="Yaratıcılık sıcaklığı")
    parser.add_argument("--output-dir", type=str, default=None, help="Çıktı klasörü (varsayılan: timestamp ile)")
    parser.add_argument("--seed", type=int, default=42, help="Rastgelelik için seed")
    parser.add_argument("--test-size", type=float, default=0.1, help="Test seti oranı (0.0-1.0 arası, örneğin 0.15 = %15)")
    parser.add_argument("--diverse-styles", action="store_true", help="Farklı tarzlarda veri üret (mevcut dataset'ten farklılaşmak için)")

    # Push işlemi ayrı dosyaya taşındı (push_hf.py)

    args = parser.parse_args(argv)

    random.seed(args.seed)

    labels = [l.strip() for l in args.labels.split(",") if l.strip()]
    if len(labels) < 2:
        print("En az iki sınıf belirtmelisiniz, örn: --labels 'olumlu,olumsuz'")
        return 2

    topics = [t.strip() for t in args.topics.split(",") if t.strip()]

    # Çıktı dizini
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = Path("synthetic_outputs") / f"text_classification_{ts}"
    ensure_dir(out_dir)

    # OpenRouter API key
    api_key = "sk-or-v1-...."
    if not api_key:
        print("Hata: OPENROUTER_API_KEY ortam değişkeni tanımlı değil.")
        return 3

    gen = GenerationConfig(
        model=args.model,
        language=args.language,
        batch_size=max(1, min(200, args.batch_size)),
        topics=topics,
        labels=labels,
        temperature=args.temperature,
    )

    print(f"Model: {gen.model} | Dil: {gen.language} | Sınıflar: {', '.join(labels)}")
    print(f"Toplam hedef örnek: {args.num_samples} (parti boyutu: {gen.batch_size})")
    if args.diverse_styles:
        print("🎨 Farklı tarzlarda veri üretimi aktif!")

    items = generate_dataset(api_key, gen, args.num_samples, use_diverse_styles=args.diverse_styles)
    if not items:
        print("Üretim başarısız veya boş çıktı alındı.")
        return 4

    # Kaydet
    train, test = train_test_split(items, test_size=args.test_size, seed=args.seed)
    print(f"📈 Veri dağılımı: Train={len(train)}, Test={len(test)} (örneklerin %{args.test_size*100:.1f}'i test)")
    jsonl_path = out_dir / "data.jsonl"
    csv_path = out_dir / "data.csv"
    train_jsonl = out_dir / "train.jsonl"
    test_jsonl = out_dir / "test.jsonl"
    save_as_jsonl(items, jsonl_path)
    save_as_csv(items, csv_path)
    save_as_jsonl(train, train_jsonl)
    save_as_jsonl(test, test_jsonl)
    print(f"✔️ Kaydedildi: {jsonl_path}")
    print(f"✔️ Kaydedildi: {csv_path}")
    print(f"✔️ Kaydedildi: {train_jsonl} | {test_jsonl}")

    # Dataset kartı
    build_dataset_card(out_dir, labels, args.language, len(items), gen.model)

    return 0


if __name__ == "__main__":
    sys.exit(main())


