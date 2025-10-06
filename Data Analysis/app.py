from pathlib import Path
import json
import os
import sys
from typing import Any, Dict, List
import logging
import time

import pandas as pd


def resolve_data_path(base_dir: Path) -> Path:
    # Oncelik: CLI (python app.py cli <path>) > ENV DATA_PATH > default Titanic
    if len(sys.argv) > 1:
        first = sys.argv[1].lower()
        if first == "cli":
            if len(sys.argv) > 2:
                user_path = Path(sys.argv[2]).expanduser().resolve()
                if user_path.exists():
                    return user_path
                raise FileNotFoundError(f"Veri yolu bulunamadı: {user_path}")
        elif first not in {"ui"}:
            # Geriye donuk uyumluluk: python app.py <path>
            user_path = Path(sys.argv[1]).expanduser().resolve()
            if user_path.exists():
                return user_path
            raise FileNotFoundError(f"Veri yolu bulunamadı: {user_path}")

    env_path = os.getenv("DATA_PATH")
    if env_path:
        user_path = Path(env_path).expanduser().resolve()
        if user_path.exists():
            return user_path
        raise FileNotFoundError(f"Veri yolu (ENV) bulunamadı: {user_path}")

    default_path = (base_dir / ".." / "Graphs" / "Titanic-Dataset.csv").resolve()
    if default_path.exists():
        return default_path
    raise FileNotFoundError("Veri bulunamadı. CLI ile dosya yolu verin veya ENV DATA_PATH ayarlayın.")


def profile_dataframe(df: pd.DataFrame, sample_rows: int = 10) -> Dict[str, Any]:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

    summary: Dict[str, Any] = {
        "boyut": {"satir": int(df.shape[0]), "sutun": int(df.shape[1])},
        "sutunlar": [
            {
                "ad": str(c),
                "tip": str(df[c].dtype),
                "eksik": int(df[c].isna().sum()),
                "ornek_deger": None if df[c].dropna().empty else df[c].dropna().iloc[0].item() if hasattr(df[c].dropna().iloc[0], "item") else df[c].dropna().iloc[0],
            }
            for c in df.columns
        ],
        "sayisal_istatistikler": df[numeric_cols].describe().transpose().to_dict(orient="index") if numeric_cols else {},
        "kategorik_ozet": {
            col: df[col].astype(str).value_counts(dropna=False).head(10).to_dict() for col in categorical_cols
        },
        "ornek_kayitlar": df.head(sample_rows).to_dict(orient="records"),
    }
    return summary


def build_ollama_prompt(profile: Dict[str, Any], image_names: List[str] = None) -> str:
    # Modelden yalnizca JSON istemek icin net yonerge
    template = (
        "Sen bir veri analistisın. Verilen veri profiline ve (varsa) gönderilen görsellere göre TÜRKÇE ve SADE bir analiz üret.\n"
        "YALNIZCA GEÇERLİ JSON döndür. \n\n"
        "Beklenen JSON şeması:\n"
        "{\n"
        "  \"veri_ozeti\": {\"satir\": int, \"sutun\": int, \"onemli_sutunlar\": [str]},\n"
        "  \"aciklama\": str,\n"
        "  \"bulgular\": [str],\n"
        "  \"veri_kalitesi\": {\"eksik_degerler\": [str], \"aykiri_gozlemler\": [str]},\n"
        "  \"detaylar\": {\"sayisal_ozet\": [str], \"kategorik_ozet\": [str]},\n"
        "  \"oneriler\": [str],\n"
        "  \"olasi_hedef_degiskenler\": [str],\n"
        "  \"grafik_ongoruleri\": [str],\n"
        "  \"gorsel_analizi\": [{\"dosya\": str, \"yorumlar\": [str]}],\n"
        "  \"ml_onerileri\": {\n"
        "    \"model_adaylari\": [str],\n"
        "    \"ozellik_muhendisligi\": [str],\n"
        "    \"on_isleme\": [str],\n"
        "    \"degisken_secimi\": [str],\n"
        "    \"degerlendirme_stratejisi\": [str],\n"
        "    \"hiperparametre_ipuclar\": [str]\n"
        "  }\n"
        "}\n\n"
        "Veri Profili (JSON):\n"
        f"{json.dumps(profile, ensure_ascii=False)}\n\n"
        f"Görseller (dosya adları):\n{json.dumps(image_names or [], ensure_ascii=False)}\n\n"
        "Kurallar:\n"
        "- Sadece JSON döndür. Serbest metin ekleme.\n"
        "- Kısa ve net cümleler kullan.\n"
        "- Sayısal gerekçeleri mümkünse belirt.\n"
        "- Grafik önerilerini EN FAZLA 2 madde ile sınırla.\n"
        "- Eğer görseller gönderilmişse, bulgular ve önerilerde görsellerden yararlan.\n"
    )
    return template


def call_ollama(prompt: str, model: str, image_paths: List[Path] = None) -> Dict[str, Any]:
    import requests
    import base64
    import re

    # Ortak seçenekler: daha deterministik ve kısa cevaplar için
    options = {
        "temperature": float(os.getenv("OLLAMA_TEMPERATURE", "0.2")),
        "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "4096")),
        "top_p": float(os.getenv("OLLAMA_TOP_P", "0.9")),
    }

    use_chat = bool(image_paths) and os.getenv("OLLAMA_USE_CHAT", "true").lower() == "true"
    images_b64: List[str] = []
    if image_paths:
        for p in image_paths:
            try:
                with open(p, "rb") as f:
                    images_b64.append(base64.b64encode(f.read()).decode("utf-8"))
            except Exception:
                continue

    if use_chat and images_b64:
        # Multimodal chat payload (Ollama chat: images alanı user mesajında ayrı dizi olarak)
        url = f"{get_ollama_base_url()}/api/chat"
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Sadece geçerli JSON döndür."},
                {"role": "user", "content": prompt, "images": images_b64},
            ],
            "stream": False,
            "format": "json",
            "options": options,
        }
    else:
        # Text-only generate payload (veya images generate desteği için)
        url = get_ollama_generate_url()
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": options,
        }
        if images_b64:
            payload["images"] = images_b64
    logger = logging.getLogger(__name__)
    timeout_s = float(os.getenv("OLLAMA_TIMEOUT", "120"))
    logger.info(
        "ollama_request_start model=%s timeout_s=%s prompt_chars=%d images=%d use_chat=%s",
        model,
        timeout_s,
        len(prompt),
        len(images_b64),
        use_chat,
    )
    t0 = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, timeout=timeout_s)
        resp.raise_for_status()
    except requests.HTTPError as http_err:
        # Fallback: bazı Ollama sürümleri chat+images'ta 400 döndürebilir; generate+images ile yeniden dene
        if use_chat and getattr(http_err.response, "status_code", None) == 400:
            logger.warning("ollama_chat_400_fallback -> generate endpoint ile tekrar denenecek")
            url = get_ollama_generate_url()
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": options,
                "images": images_b64,
            }
            resp = requests.post(url, json=payload, timeout=timeout_s)
            resp.raise_for_status()
        else:
            raise
    data = resp.json()
    # /api/generate -> { response: "..." }
    # /api/chat -> { message: { content: "..." } }
    text = ""
    if isinstance(data, dict):
        if "response" in data:
            text = str(data.get("response", "")).strip()
        elif "message" in data and isinstance(data["message"], dict):
            text = str(data["message"].get("content", "")).strip()
    if not text:
        text = str(data).strip()
    dt = time.perf_counter() - t0
    logger.info("ollama_request_end duration_s=%.3f response_chars=%d", dt, len(text))

    def _strip_code_fences(s: str) -> str:
        s = s.strip()
        if s.startswith("```") and s.endswith("```"):
            s = s.strip("`")
            # remove optional language tag like json\n
            s = re.sub(r"^json\n", "", s, flags=re.IGNORECASE)
        # Also handle fenced block in the middle
        m = re.search(r"```(?:json)?\n([\s\S]*?)\n```", s, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return s

    def _extract_first_json(s: str) -> str:
        # Try to find first balanced {} or [] block
        s = s.strip()
        # Fast path
        try:
            return json.dumps(json.loads(s), ensure_ascii=False)
        except Exception:
            pass
        stack = []
        start_idx = None
        for i, ch in enumerate(s):
            if ch in '{[':
                if not stack:
                    start_idx = i
                stack.append('}' if ch == '{' else ']')
            elif ch in '}]' and stack:
                expected = stack.pop()
                if (ch == '}' and expected == '}') or (ch == ']' and expected == ']'):
                    if not stack and start_idx is not None:
                        candidate = s[start_idx:i+1]
                        try:
                            json.loads(candidate)
                            return candidate
                        except Exception:
                            # continue scanning
                            start_idx = None
                            continue
        return ""

    cleaned = _strip_code_fences(text)
    try:
        return json.loads(cleaned)
    except Exception as e1:
        candidate = _extract_first_json(cleaned)
        if candidate:
            return json.loads(candidate)
        logger.warning("json_parse_failed chars=%d sample=%s error=%s", len(text), text[:200], e1)
        raise ValueError("Model JSON döndürmedi veya parse edilemedi.")


def suggest_charts(df: pd.DataFrame) -> List[str]:
    suggestions: List[str] = []
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

    if numeric_cols:
        suggestions.append("Sayısal sütunlar için histogram ve yoğunluk grafiği")
        if len(numeric_cols) >= 2:
            suggestions.append("Sayısal sütun çiftleri için saçılım diyagramı")
        suggestions.append("Aykırı değerleri görmek için kutu grafiği (boxplot)")
    if categorical_cols and numeric_cols:
        suggestions.append("Kategorik x Sayısal için kutu/violin grafikleri")
        suggestions.append("Kategori ortalamaları için çubuk grafiği")
    if {"Survived", "Sex"}.issubset(df.columns):
        suggestions.append("Cinsiyet bazında hayatta kalma oranı bar chart")
    if {"score"}.issubset(df.columns):
        suggestions.append("Skor dağılım histogramı ve versiyon bazlı boxplot")
    if {"reviewCreatedVersion", "score"}.issubset(df.columns):
        suggestions.append("Sürüm bazlı ortalama skor çubuk grafiği")

    # Eşsiz grafik önerilerini döndür
    uniq = []
    for s in suggestions:
        if s not in uniq:
            uniq.append(s)
    return uniq[:2]


def build_fallback_analysis(profile: Dict[str, Any], df: pd.DataFrame, error_msg: str = None) -> Dict[str, Any]:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

    aciklama = (
        f"Veri {profile['boyut']['satir']} satır ve {profile['boyut']['sutun']} sütundan oluşuyor. "
        f"Sayısal sütunlar: {len(numeric_cols)}, kategorik sütunlar: {len(categorical_cols)}."
    )

    sayisal_ozet = []
    if numeric_cols:
        desc = df[numeric_cols].describe().transpose()
        for col in numeric_cols[:5]:
            if col in desc.index:
                vals = desc.loc[col]
                parts = []
                for k in ["mean", "std", "min", "25%", "50%", "75%", "max"]:
                    if k in desc.columns and pd.notna(vals.get(k)):
                        parts.append(f"{k}={vals.get(k):.3f}")
                sayisal_ozet.append(f"{col}: " + ", ".join(parts))

    kategorik_ozet = []
    for col in categorical_cols[:5]:
        vc = df[col].astype(str).value_counts(dropna=False).head(3).to_dict()
        kategorik_ozet.append(f"{col}: en sık {list(vc.items())}")

    eksikler = [f"{c['ad']}: {c['eksik']} eksik" for c in profile["sutunlar"] if c["eksik"] > 0][:5]
    grafikler = suggest_charts(df)

    bulgular: List[str] = []
    if "score" in df.columns and pd.api.types.is_numeric_dtype(df["score"]):
        bulgular.append(f"Skor ortalaması {df['score'].mean():.2f}")
    if "Survived" in df.columns and set(df["Survived"].dropna().unique()).issubset({0, 1}):
        bulgular.append(f"Hayatta kalma oranı {df['Survived'].mean():.3f}")
    if error_msg:
        bulgular.append(f"Not: Ollama hatası/erişimi: {error_msg}")

    return {
        "veri_ozeti": {
            "satir": profile["boyut"]["satir"],
            "sutun": profile["boyut"]["sutun"],
            "onemli_sutunlar": [c["ad"] for c in profile["sutunlar"][:5]],
        },
        "aciklama": aciklama,
        "bulgular": bulgular or ["Örnek bulgular: sayısal dağılımlar ve kategorik yoğunluklar incelenebilir."],
        "veri_kalitesi": {
            "eksik_degerler": eksikler,
            "aykiri_gozlemler": [],
        },
        "detaylar": {
            "sayisal_ozet": sayisal_ozet,
            "kategorik_ozet": kategorik_ozet,
        },
        "oneriler": [
            "Eksik değerler için uygun imputasyon stratejileri uygulayın",
            "Sayısal sütunlar için ölçekleme/standardizasyonu değerlendirin",
            "Modelleme için hedef değişkeni netleştirin",
        ],
        "olasi_hedef_degiskenler": [c for c in df.columns if c.lower() in {"score", "survived", "label", "target"}][:3],
        "grafik_ongoruleri": grafikler or ["Temel histogram ve boxplot görselleştirmeleri"],
    }


def _df_to_markdown(df: pd.DataFrame, index: bool = False, max_rows: int = 20) -> str:
    try:
        import tabulate  # noqa: F401
    except Exception:
        pass
    df_show = df.head(max_rows)
    try:
        return df_show.to_markdown(index=index)
    except Exception:
        return "```\n" + df_show.to_string(index=index) + "\n```"


def analysis_to_markdown(profile: Dict[str, Any], analysis: Dict[str, Any], df: pd.DataFrame, images: List[Path]) -> str:
    # Tolerans: model bazı durumlarda liste döndürebilir
    if not isinstance(analysis, dict):
        if isinstance(analysis, list) and len(analysis) > 0:
            if isinstance(analysis[0], dict):
                analysis = analysis[0]
            else:
                # Listeyi bulgulara çevir
                analysis = {"bulgular": [str(x) for x in analysis]}
        else:
            analysis = {"bulgular": [str(analysis)]}

    lines: List[str] = []
    lines.append(f"**Satır/Sütun**: {profile['boyut']['satir']} / {profile['boyut']['sutun']}")
    if analysis.get("aciklama"):
        lines.append(f"\n**Açıklama**\n\n{analysis['aciklama']}")

    # Bulgular
    if analysis.get("bulgular"):
        lines.append("\n**Bulgular**")
        for b in analysis["bulgular"][:10]:
            lines.append(f"- {b}")

    # Veri kalitesi
    vk = analysis.get("veri_kalitesi", {})
    if vk:
        lines.append("\n**Veri Kalitesi**")
        if vk.get("eksik_degerler"):
            lines.append("- Eksik değerler:")
            for e in vk["eksik_degerler"][:10]:
                lines.append(f"  - {e}")
        if vk.get("aykiri_gozlemler"):
            lines.append("- Aykırı gözlemler:")
            for e in vk["aykiri_gozlemler"][:10]:
                lines.append(f"  - {e}")

    # Detaylar
    dt = analysis.get("detaylar", {})
    if dt:
        if dt.get("sayisal_ozet"):
            lines.append("\n**Sayısal Özet (kısa)**")
            for s in dt["sayisal_ozet"][:10]:
                lines.append(f"- {s}")
        if dt.get("kategorik_ozet"):
            lines.append("\n**Kategorik Özet (en sık değerler)**")
            try:
                for col, freq in dt["kategorik_ozet"].items():
                    lines.append(f"- {col}:")
                    for k, v in list(freq.items())[:10]:
                        lines.append(f"  - {k}: {v}")
            except Exception:
                # Bazı modeller liste döndürebilir
                for s in dt["kategorik_ozet"] if isinstance(dt["kategorik_ozet"], list) else []:
                    lines.append(f"- {s}")

    # Tablolar: ilk 10 kayıt ve sayısal describe
    try:
        lines.append("\n**İlk 10 Kayıt**\n")
        lines.append(_df_to_markdown(df.head(10)))
    except Exception:
        pass
    try:
        num_cols = df.select_dtypes(include=["number"]).columns
        if len(num_cols) > 0:
            lines.append("\n**Sayısal İstatistikler**\n")
            lines.append(_df_to_markdown(df[num_cols].describe().transpose(), index=True))
    except Exception:
        pass

    # DF-tabanlı türetilmiş içgörüler
    extra = _derive_insights_from_df(df)
    if extra.get("bullets"):
        lines.append("\n**Model Dışı Otomatik İçgörüler**")
        for b in extra["bullets"]:
            lines.append(f"- {b}")
    if extra.get("tables"):
        for name, tdf in extra["tables"].items():
            lines.append(f"\n**Tablo: {name}**\n")
            lines.append(_df_to_markdown(tdf))

    # Grafik bağlantıları
    if images:
        lines.append("\n**Grafik Dosyaları**")
        for p in images[:2]:
            lines.append(f"- {p.name}")

    # Öneriler
    if analysis.get("oneriler"):
        lines.append("\n**Öneriler**")
        for o in analysis["oneriler"][:10]:
            lines.append(f"- {o}")

    # Olası hedef değişkenler
    if analysis.get("olasi_hedef_degiskenler"):
        lines.append("\n**Olası Hedef Değişkenler**")
        for t in analysis["olasi_hedef_degiskenler"][:10]:
            lines.append(f"- {t}")

    # ML önerileri
    ml = analysis.get("ml_onerileri")
    if isinstance(ml, dict):
        if ml.get("model_adaylari"):
            lines.append("\n**ML - Model Adayları**")
            for m in ml["model_adaylari"][:10]:
                lines.append(f"- {m}")
        if ml.get("ozellik_muhendisligi"):
            lines.append("\n**ML - Özellik Mühendisliği**")
            for m in ml["ozellik_muhendisligi"][:10]:
                lines.append(f"- {m}")
        if ml.get("on_isleme"):
            lines.append("\n**ML - Ön İşleme**")
            for m in ml["on_isleme"][:10]:
                lines.append(f"- {m}")
        if ml.get("degisken_secimi"):
            lines.append("\n**ML - Değişken Seçimi**")
            for m in ml["degisken_secimi"][:10]:
                lines.append(f"- {m}")
        if ml.get("degerlendirme_stratejisi"):
            lines.append("\n**ML - Değerlendirme Stratejisi**")
            for m in ml["degerlendirme_stratejisi"][:10]:
                lines.append(f"- {m}")
        if ml.get("hiperparametre_ipuclar"):
            lines.append("\n**ML - Hiperparametre İpuçları**")
            for m in ml["hiperparametre_ipuclar"][:10]:
                lines.append(f"- {m}")

    return "\n".join(lines)


def _derive_insights_from_df(df: pd.DataFrame) -> Dict[str, Any]:
    insights: Dict[str, Any] = {"bullets": [], "tables": {}}
    try:
        date_col = None
        for c in df.columns:
            if str(c).lower() in {"date", "month", "time", "timestamp"}:
                date_col = c
                break
        sales_col = None
        for c in df.select_dtypes(include=["number"]).columns:
            if str(c).lower() in {"sales", "sale", "amount", "value", "close"}:
                sales_col = c
                break
        if sales_col is None and len(df.select_dtypes(include=["number"]).columns) > 0:
            sales_col = df.select_dtypes(include=["number"]).columns[0]

        if date_col is not None:
            d = pd.to_datetime(df[date_col], errors="coerce")
        else:
            d = pd.Series(pd.NaT, index=df.index)

        s = pd.to_numeric(df[sales_col], errors="coerce") if sales_col else pd.Series(dtype=float)
        s_clean = s.dropna()
        if not s_clean.empty:
            min_val = float(s_clean.min())
            max_val = float(s_clean.max())
            min_idx = int(s_clean.idxmin())
            max_idx = int(s_clean.idxmax())
            min_when = str(df.loc[min_idx, date_col]) if date_col is not None else str(min_idx)
            max_when = str(df.loc[max_idx, date_col]) if date_col is not None else str(max_idx)
            insights["bullets"].append(f"En düşük {sales_col}: {min_val:.2f} (tarih/satır: {min_when})")
            insights["bullets"].append(f"En yüksek {sales_col}: {max_val:.2f} (tarih/satır: {max_when})")

            try:
                import numpy as np
                x = np.arange(len(s_clean))
                y = s_clean.values
                slope = float(np.polyfit(x, y, 1)[0])
                insights["bullets"].append(f"Genel eğilim (doğrusal eğim): {slope:+.2f} / adım")
            except Exception:
                pass

            try:
                start = float(s_clean.iloc[0])
                end = float(s_clean.iloc[-1])
                if start != 0:
                    growth = (end - start) / abs(start) * 100.0
                    insights["bullets"].append(f"Başlangıçtan bugüne toplam değişim: {growth:+.2f}%")
            except Exception:
                pass

            # Sezonsallık özeti (aylık veri varsayımı)
            try:
                if d.notna().any():
                    month_means = pd.DataFrame({"month": d.dt.month, sales_col: s}).dropna().groupby("month")[sales_col].mean().sort_values(ascending=False)
                    top = month_means.head(3).round(2).to_dict()
                    if top:
                        insights["bullets"].append(f"En yüksek ortalama {sales_col} ayları: {list(top.keys())} (ortalama değerler: {list(top.values())})")
            except Exception:
                pass

            # İlk ve son 5 satır özet tablo
            try:
                head_tail = pd.concat([df.head(5), df.tail(5)])
                insights["tables"]["ornek_kayitlar_5_5"] = head_tail
            except Exception:
                pass
    except Exception:
        pass
    return insights


def _get_filepath(file: Any) -> str:
    try:
        if isinstance(file, str):
            return file
        if hasattr(file, "name"):
            return file.name
    except Exception:
        pass
    raise ValueError("Geçersiz dosya girdisi. Lütfen bir CSV dosyası seçin.")


def create_charts(df: pd.DataFrame, outputs_dir: Path) -> List[Path]:
    paths: List[Path] = []
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception:
        return paths

    plt.style.use("seaborn-v0_8")
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

    # 1) İlk sayısal sütun için histogram + kde
    if numeric_cols:
        col = numeric_cols[0]
        t0 = time.perf_counter()
        ax = sns.histplot(df[col].dropna(), kde=True, bins=30)
        ax.set_title(f"{col} dağılımı")
        ax.figure.tight_layout()
        path = outputs_dir / f"hist_{col}.png"
        ax.figure.savefig(path, dpi=150)
        plt.close(ax.figure)
        paths.append(path)
        logging.getLogger(__name__).info("chart_saved path=%s duration_s=%.3f", path, time.perf_counter() - t0)

    # 2) İlk iki sayısal sütun için scatter
    if len(numeric_cols) >= 2:
        x, y = numeric_cols[0], numeric_cols[1]
        t0 = time.perf_counter()
        ax = sns.scatterplot(data=df, x=x, y=y)
        ax.set_title(f"{x} vs {y}")
        ax.figure.tight_layout()
        path = outputs_dir / f"scatter_{x}_vs_{y}.png"
        ax.figure.savefig(path, dpi=150)
        plt.close(ax.figure)
        paths.append(path)
        logging.getLogger(__name__).info("chart_saved path=%s duration_s=%.3f", path, time.perf_counter() - t0)

    # 3) Kategorik x Sayısal için boxplot
    if categorical_cols and numeric_cols:
        cx, ny = categorical_cols[0], numeric_cols[0]
        t0 = time.perf_counter()
        ax = sns.boxplot(data=df, x=cx, y=ny)
        ax.set_title(f"{cx} bazında {ny} dağılımı")
        ax.figure.tight_layout()
        path = outputs_dir / f"box_{cx}_{ny}.png"
        ax.figure.savefig(path, dpi=150)
        plt.close(ax.figure)
        paths.append(path)
        logging.getLogger(__name__).info("chart_saved path=%s duration_s=%.3f", path, time.perf_counter() - t0)

    # 4) Kategori ortalamaları için barplot
    if categorical_cols and numeric_cols:
        cx, ny = categorical_cols[0], numeric_cols[0]
        t0 = time.perf_counter()
        means = df.groupby(cx, dropna=False)[ny].mean().reset_index()
        ax = sns.barplot(data=means, x=cx, y=ny)
        ax.set_title(f"{cx} ortalama {ny}")
        ax.figure.tight_layout()
        path = outputs_dir / f"bar_{cx}_{ny}.png"
        ax.figure.savefig(path, dpi=150)
        plt.close(ax.figure)
        paths.append(path)
        logging.getLogger(__name__).info("chart_saved path=%s duration_s=%.3f", path, time.perf_counter() - t0)

    # 5) score özel: histogram
    if "score" in df.columns and pd.api.types.is_numeric_dtype(df["score"]):
        t0 = time.perf_counter()
        ax = sns.histplot(df["score"].dropna(), kde=True, bins=20)
        ax.set_title("score dağılımı")
        ax.figure.tight_layout()
        path = outputs_dir / "hist_score.png"
        ax.figure.savefig(path, dpi=150)
        plt.close(ax.figure)
        paths.append(path)
        logging.getLogger(__name__).info("chart_saved path=%s duration_s=%.3f", path, time.perf_counter() - t0)

    # 6) version bazlı boxplot ve ortalama çubuk
    if {"reviewCreatedVersion", "score"}.issubset(df.columns):
        t0 = time.perf_counter()
        ax = sns.boxplot(data=df, x="reviewCreatedVersion", y="score")
        ax.set_title("Versiyon bazlı score boxplot")
        ax.figure.tight_layout()
        path = outputs_dir / "box_version_score.png"
        ax.figure.savefig(path, dpi=150)
        plt.close(ax.figure)
        paths.append(path)
        logging.getLogger(__name__).info("chart_saved path=%s duration_s=%.3f", path, time.perf_counter() - t0)

        t0 = time.perf_counter()
        means = df.groupby("reviewCreatedVersion")["score"].mean().reset_index()
        ax = sns.barplot(data=means, x="reviewCreatedVersion", y="score")
        ax.set_title("Versiyon bazlı ortalama score")
        ax.figure.tight_layout()
        path = outputs_dir / "bar_version_score.png"
        ax.figure.savefig(path, dpi=150)
        plt.close(ax.figure)
        paths.append(path)
        logging.getLogger(__name__).info("chart_saved path=%s duration_s=%.3f", path, time.perf_counter() - t0)

    return paths


def get_ollama_base_url() -> str:
    base = os.getenv("OLLAMA_API", "http://localhost:11434")
    # Eğer generate/embeddings gibi bir path verilmişse sadece kökü al
    for suffix in ("/api/generate", "/api/chat", "/api/embeddings", "/api/tags", "/api"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base.rstrip("/")


def get_ollama_generate_url() -> str:
    return f"{get_ollama_base_url()}/api/generate"


def ensure_ollama_ready(model: str) -> None:
    import requests
    base = get_ollama_base_url()
    health = f"{base}/api/tags"
    logger = logging.getLogger(__name__)
    try:
        r = requests.get(health, timeout=5)
        r.raise_for_status()
        tags = r.json().get("models", []) or r.json()
        names = set()
        for item in tags:
            name = item.get("name") if isinstance(item, dict) else None
            if name:
                names.add(name)
        if model not in names:
            logger.warning("ollama_model_missing model=%s available=%s", model, sorted(list(names))[:5])
    except Exception as e:
        logger.warning("ollama_health_failed url=%s error=%s", health, e)


def setup_logging(outputs_dir: Path) -> None:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    fmt = "%(asctime)s %(levelname)s %(name)s - %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handlers = [logging.StreamHandler()]
    log_file = outputs_dir / "run.log"
    try:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    except Exception:
        pass
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format=fmt, datefmt=datefmt, handlers=handlers)
def main() -> None:
    # Dizinler
    base_dir = Path(__file__).resolve().parent
    outputs_dir = (base_dir / "outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(outputs_dir)
    logger = logging.getLogger(__name__)

    # Veri yükleme
    t0 = time.perf_counter()
    data_path = resolve_data_path(base_dir)
    logger.info("load_csv_start path=%s size_bytes=%s", data_path, os.path.getsize(data_path) if data_path.exists() else "?")
    df = pd.read_csv(data_path)
    logger.info("load_csv_end duration_s=%.3f rows=%d cols=%d", time.perf_counter() - t0, df.shape[0], df.shape[1])

    # Profil çıkar
    t0 = time.perf_counter()
    profile = profile_dataframe(df)
    logger.info("profile_done duration_s=%.3f", time.perf_counter() - t0)

    # Ollama model seçimi (text vs vision)
    text_model = os.getenv("OLLAMA_TEXT_MODEL", "granite4:tiny-h")
    vision_model = os.getenv("OLLAMA_VISION_MODEL", os.getenv("OLLAMA_MODEL", "qwen2.5vl:3b"))

    # Görselleri üret (opsiyonel) ve modele gönder
    t0 = time.perf_counter()
    image_paths: List[Path] = create_charts(df, outputs_dir)[:2]
    logger.info("charts_created duration_s=%.3f count=%d", time.perf_counter() - t0, len(image_paths))

    # Prompt'u, üretilen görsellerin adlarıyla birlikte oluştur
    prompt = build_ollama_prompt(profile, [p.name for p in image_paths])

    analysis: Dict[str, Any]
    try:
        t0 = time.perf_counter()
        selected_model = vision_model if len(image_paths) > 0 else text_model
        ensure_ollama_ready(selected_model)
        logger.info("ollama_model_selected model=%s images=%d", selected_model, len(image_paths))
        analysis = call_ollama(prompt, selected_model, image_paths=image_paths)
        logger.info("ollama_total duration_s=%.3f", time.perf_counter() - t0)
    except Exception as e:
        analysis = build_fallback_analysis(profile, df, error_msg=str(e))
        logger.warning("ollama_failed error=%s", e)

    # Sonucu kaydet ve yazdır
    output_file = outputs_dir / "analysis.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    logger.info("analysis_saved path=%s", output_file)
    # Konsol ve ayrıca ayrı bir markdown çıktısı
    md = analysis_to_markdown(profile, analysis, df, image_paths)
    md_file = outputs_dir / "analysis.md"
    with md_file.open("w", encoding="utf-8") as f:
        f.write(md)
    print(md)


def launch_ui() -> None:
    import gradio as gr

    base_dir = Path(__file__).resolve().parent

    def analyze_file(file: Any, model: str, include_charts: bool) -> Dict[str, Any]:
        if file is None:
            return {"hata": "Lütfen bir CSV dosyası yükleyin."}
        try:
            t_start = time.perf_counter()
            df_local = pd.read_csv(file.name)
            logging.getLogger(__name__).info("ui_load_csv path=%s rows=%d cols=%d", file.name, df_local.shape[0], df_local.shape[1])
            profile = profile_dataframe(df_local)
            outputs_dir = (base_dir / "outputs")
            outputs_dir.mkdir(parents=True, exist_ok=True)
            img_paths: List[Path] = create_charts(df_local, outputs_dir) if include_charts else []
            prompt = build_ollama_prompt(profile, [p.name for p in img_paths])

            # Model seçimi: auto ise görsel varsa vision, yoksa text; alan doluysa kullanıcıya saygı
            text_model = os.getenv("OLLAMA_TEXT_MODEL", "granite4:tiny-h")
            vision_model = os.getenv("OLLAMA_VISION_MODEL", os.getenv("OLLAMA_MODEL", "qwen2.5vl:3b"))
            if model and model.strip().lower() != "auto":
                selected_model = model.strip()
            else:
                selected_model = vision_model if (include_charts and len(img_paths) > 0) else text_model
            ensure_ollama_ready(selected_model)
            logging.getLogger(__name__).info("ui_model_selected model=%s images=%d", selected_model, len(img_paths))
            try:
                t0 = time.perf_counter()
                result = call_ollama(prompt, selected_model, image_paths=img_paths[:2])
                logging.getLogger(__name__).info("ui_ollama_total duration_s=%.3f", time.perf_counter() - t0)
            except Exception as e:
                result = build_fallback_analysis(profile, df_local, error_msg=str(e))
                logging.getLogger(__name__).warning("ui_ollama_failed error=%s", e)
            # Kaydet
            out_path = outputs_dir / "analysis.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logging.getLogger(__name__).info("ui_total duration_s=%.3f", time.perf_counter() - t_start)
            return result
        except Exception as e:
            return {"hata": str(e)}

    with gr.Blocks(title="Veri Analizi - Ollama") as demo:
        gr.Markdown("**CSV yükleyin, modeli seçin, Türkçe JSON analiz alın.**")
        with gr.Row():
            file_in = gr.File(label="CSV Dosyası", file_types=[".csv"], type="filepath")
            model_in = gr.Textbox(value="auto", label="Model (auto / model-etiketi)")
        include_charts = gr.Checkbox(value=True, label="Grafikleri üret ve modele gönder (önerilir)")
        analyze_btn = gr.Button("Analiz Et")
        json_out = gr.Markdown(label="Analiz (Markdown)")

        def ui_handler(file: Any, model: str, include_charts: bool) -> str:
            # Dosya yolunu güvenle al
            try:
                filepath = _get_filepath(file)
            except Exception as e:
                return f"**Hata**: {e}"

            result = analyze_file(file, model, include_charts)
            if isinstance(result, dict) and "hata" in result:
                return f"**Hata**: {result['hata']}"
            # df'yi yeniden oku markdown üretmek için (küçük dosyalar için kabul edilebilir)
            try:
                df_local = pd.read_csv(filepath)
            except Exception:
                df_local = pd.DataFrame()
            # Profil tekrar hesaplanabilir ya da JSON'dan alınabilir; basitçe profilden üretelim
            profile_local = profile_dataframe(df_local) if not df_local.empty else {"boyut": {"satir": 0, "sutun": 0}}
            # Son üretilen görüntüleri listele
            outs = Path("outputs")
            imgs = []
            try:
                imgs = sorted([p for p in (Path(__file__).resolve().parent / "outputs").glob("*.png")], key=os.path.getmtime, reverse=True)[:2]
            except Exception:
                pass
            return analysis_to_markdown(profile_local, result, df_local, imgs)

        analyze_btn.click(ui_handler, inputs=[file_in, model_in, include_charts], outputs=[json_out])

    demo.launch()


if __name__ == "__main__":
    # Kullanım: python app.py            -> Gradio arayüz (varsayılan)
    #          python app.py cli         -> CLI (varsayılan veri)
    #          python app.py cli <path>  -> CLI (özel dosya)
    if len(sys.argv) > 1 and sys.argv[1].lower() == "cli":
        main()
    else:
        launch_ui()


