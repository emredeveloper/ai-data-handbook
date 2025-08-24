"""GluonTS DeepAR ile KaggleHub (House-hold Energy) zaman serisi tahmini.

Akış:
- KaggleHub ile veri indirilir: "jaganadhg/house-hold-energy-data".
- Uygun CSV/TXT dosyası otomatik bulunur ve saatlik seriye dönüştürülür.
- Veri, GluonTS `PandasDataset` ile hazırlanır (freq="H").
- DeepAR modeli eğitilir ve 7 günlük (168 saat) tahmin üretilir.
- Gerçek ve tahminler görselleştirilir.

Kaynak dokümantasyon: https://ts.gluon.ai/stable/index.html
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional
import warnings
import logging

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import kagglehub
import torch

from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.common import ListDataset
from gluonts.dataset.split import split
from gluonts.torch import DeepAREstimator


DATA_DIR = Path("data_kaggle")
DATA_DIR.mkdir(exist_ok=True)

KAGGLEHUB_DATASET = "jaganadhg/house-hold-energy-data"


def download_with_kagglehub() -> Path:
    dataset_path_str = kagglehub.dataset_download(KAGGLEHUB_DATASET)
    return Path(dataset_path_str)


def find_data_file(dataset_dir: Path) -> Optional[Path]:
    # Öncelik CSV, yoksa TXT ara
    csv_files = list(dataset_dir.rglob("*.csv"))
    if csv_files:
        # Öngörülebilir seçim için isimde power/energy geçen varsa onu seç
        for f in csv_files:
            name = f.name.lower()
            if "power" in name or "energy" in name or "household" in name:
                return f
        return csv_files[0]
    txt_files = list(dataset_dir.rglob("*.txt"))
    if txt_files:
        for f in txt_files:
            name = f.name.lower()
            if "power" in name or "energy" in name or "household" in name:
                return f
        return txt_files[0]
    return None


def load_household_dataframe(file_path: Path) -> pd.DataFrame:
    # Birkaç farklı okuma denemesi: ayraç ve ondalık varyasyonları
    read_attempts = [
        dict(sep=",", na_values=["?", "NA", "NaN", ""], decimal="."),
        dict(sep=";", na_values=["?", "NA", "NaN", ""], decimal="."),
        dict(sep=";", na_values=["?", "NA", "NaN", ""], decimal=","),
    ]
    last_error: Optional[Exception] = None
    df: Optional[pd.DataFrame] = None
    for params in read_attempts:
        try:
            df = pd.read_csv(file_path, **params)
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            df = None
            continue
    if df is None:
        raise RuntimeError(f"Dosya okunamadı: {file_path}: {last_error}")

    # Kolon adlarını normalize et
    lower_map = {c.lower().strip(): c for c in df.columns}

    # Tarih-Zamanı tespit et ve indeks yap (Kaggle açıklamasına göre DATE + START TIME mevcut)
    if "datetime" in lower_map:
        dt_col = lower_map["datetime"]
        df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce", dayfirst=False)
        df = df.set_index(dt_col)
    elif "date" in lower_map and ("start time" in lower_map or "start_time" in lower_map):
        date_col = lower_map["date"]
        time_key = "start time" if "start time" in lower_map else "start_time"
        time_col = lower_map[time_key]
        df["Datetime"] = pd.to_datetime(
            df[date_col].astype(str) + " " + df[time_col].astype(str), errors="coerce", dayfirst=False
        )
        df = df.set_index("Datetime")
    elif "date" in lower_map and "time" in lower_map:
        date_col = lower_map["date"]
        time_col = lower_map["time"]
        df["Datetime"] = pd.to_datetime(
            df[date_col].astype(str) + " " + df[time_col].astype(str), errors="coerce", dayfirst=False
        )
        df = df.set_index("Datetime")
    elif "date" in lower_map:
        date_col = lower_map["date"]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=False)
        df = df.set_index(date_col)
    else:
        # Son çare: ilk sütunu datetime kabul et
        first_col = df.columns[0]
        df[first_col] = pd.to_datetime(df[first_col], errors="coerce", dayfirst=False)
        df = df.set_index(first_col)

    # İndeks temizliği
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    # Hedef kolon: USAGE (kWh, 15 dakikalık)
    usage_col: Optional[str] = None
    for key in ["usage", "kwh", "consumption", "energy", "value"]:
        if key in lower_map:
            usage_col = lower_map[key]
            break
    if usage_col is None:
        # Yedek: sayısal kolonlardan ilki
        numeric_df = df.apply(pd.to_numeric, errors="coerce")
        num_cols = [c for c in numeric_df.columns if pd.api.types.is_numeric_dtype(numeric_df[c])]
        if not num_cols:
            raise ValueError("Sayısal bir ölçüm kolonu bulunamadı (USAGE vb.).")
        usage_col = num_cols[0]

    # Sayısal dönüştürme (virgül ondalık gelebilir)
    values = pd.to_numeric(df[usage_col], errors="coerce")

    # 15 dakikalık değerleri saatlik toplama çevir (kWh)
    hourly = values.resample("h").sum(min_count=1)
    # Seyrek boşluklar için ileri doldurma + interpolasyon
    hourly = hourly.ffill()
    hourly = hourly.interpolate(method="time", limit_direction="both")
    hourly = hourly.to_frame(name="value").dropna()
    if hourly.empty:
        raise ValueError("Saatlik yeniden örnekleme sonrası veri boş kaldı.")
    return hourly


def ensure_data_with_kagglehub() -> pd.DataFrame:
    dataset_dir = download_with_kagglehub()
    data_file = find_data_file(dataset_dir)
    if data_file is None:
        raise FileNotFoundError(
            f"Veri dosyası bulunamadı: {dataset_dir}. CSV/TXT aranıyor."
        )
    df = load_household_dataframe(data_file)
    return df


def build_gluonts_dataset(hourly_df: pd.DataFrame, freq: str, prediction_length: int) -> ListDataset:
    # Son 180 günü al, 'last' depreceated; .loc ile al
    end_ts = hourly_df.index.max()
    start_ts = end_ts - pd.Timedelta(days=180)
    subset = hourly_df.loc[start_ts:]
    # Alt küme boş ise tüm veriyi kullan
    if subset.empty:
        subset = hourly_df.copy()
    # Tamamen NaN ise hata ver
    if subset["value"].dropna().empty:
        raise ValueError("Seçilen zaman aralığında geçerli veri bulunamadı.")
    # ListDataset ile tek seri hazırla
    data = ListDataset(
        [
            {
                "start": subset.index[0],
                "target": subset["value"].astype(float).to_numpy(),
            }
        ],
        freq=freq,
    )
    return data


def train_deepar(training_data, prediction_length: int, freq: str) -> any:
    estimator = DeepAREstimator(
        prediction_length=prediction_length,  # 7 gün x 24 saat
        freq=freq,
        trainer_kwargs={
            "max_epochs": 5,
            "logger": False,
            "enable_progress_bar": False,
            "enable_model_summary": False,
        },
    )
    predictor = estimator.train(training_data)
    return predictor


def plot_forecasts(subset: pd.DataFrame, forecasts: Iterable, days_to_plot: int = 14) -> None:
    fig, ax = plt.subplots(figsize=(14, 6), facecolor="#ffffff")
    ax.set_facecolor("#ffffff")
    # Son 'days_to_plot' gün gerçek değerleri
    plot_start = subset.index.max() - pd.Timedelta(days=days_to_plot)
    subset.loc[plot_start:, "value"].plot(ax=ax, color="#111111", label="Gerçek", linewidth=2.2, zorder=3)

    # Tahmin ortalaması ve belirsizlik bandı (varsa)
    for i, fc in enumerate(forecasts):
        try:
            fc_index = pd.DatetimeIndex(fc.index)
        except Exception:
            # Geri dönüş: basit aralık ekseni
            fc_index = pd.date_range(start=subset.index.max(), periods=len(fc.mean), freq="h")

        mean_vals = fc.mean
        ax.plot(fc_index, mean_vals, color="#d62728", linewidth=2.6, zorder=4,
                label="Tahmin (ortalama)" if i == 0 else None)

        # Belirsizlik bantları
        try:
            q10 = fc.quantile(0.1)
            q90 = fc.quantile(0.9)
            ax.fill_between(fc_index, q10, q90, color="#ff7f0e", alpha=0.25, zorder=2,
                            label="Tahmin %10-%90" if i == 0 else None)
        except Exception:
            pass

    ax.set_title("DeepAR - Household Energy (Son 14 gün ve 7 günlük tahmin)")
    ax.grid(True, linestyle=":", alpha=0.5)
    # Yinelenen etiketleri önlemek için benzersiz birleştir
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
    if uniq:
        leg = ax.legend(*zip(*uniq), loc="upper left", frameon=True, facecolor="#ffffff", framealpha=0.9)
        for txt in leg.get_texts():
            txt.set_color("#111111")
    plt.tight_layout()
    plt.show()


def summarize_forecast(forecast) -> None:
    mean_values = forecast.mean  # numpy array [prediction_length]
    total_kwh = float(mean_values.sum())
    hourly_avg = float(mean_values.mean())
    # Güven aralığı özeti (isteğe bağlı)
    try:
        p10 = float(forecast.quantile(0.1).sum())
        p90 = float(forecast.quantile(0.9).sum())
        print(f"7 günlük toplam tahmin: {total_kwh:.2f} kWh (P10={p10:.2f}, P90={p90:.2f})")
    except Exception:
        print(f"7 günlük toplam tahmin: {total_kwh:.2f} kWh")
    print(f"Saatlik ortalama tahmin: {hourly_avg:.3f} kWh/sa")


def configure_runtime_silence() -> None:
    # Uyarıları ve logları azalt
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message="This axis already has a converter set")
    for name in [
        "lightning.pytorch",
        "pytorch_lightning",
        "gluonts",
        "matplotlib",
    ]:
        logging.getLogger(name).setLevel(logging.ERROR)
    # CUDA matmul uyarısını kapatmak için önerilen ayar
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    # Görsel tema ve arkaplanları beyaza sabitle, renk döngüsünü belirgin yap
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        try:
            plt.style.use("default")
        except Exception:
            pass
    mpl.rcParams["axes.facecolor"] = "#ffffff"
    mpl.rcParams["figure.facecolor"] = "#ffffff"
    mpl.rcParams["savefig.facecolor"] = "#ffffff"
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=[
        "#111111",  # koyu siyah
        "#d62728",  # kırmızı
        "#1f77b4",  # mavi
        "#2ca02c",  # yeşil
        "#ff7f0e",  # turuncu
    ])


def main() -> None:
    # Parametreler
    prediction_length = 168
    freq = "h"  # 'H' depreceated uyarısını engelle

    configure_runtime_silence()
    hourly_df = ensure_data_with_kagglehub()
    hourly_df = hourly_df.sort_index()

    # Eğitim verisi (tek seri) ListDataset ile
    training_data = build_gluonts_dataset(hourly_df, freq=freq, prediction_length=prediction_length)

    predictor = train_deepar(training_data, prediction_length=prediction_length, freq=freq)
    forecasts = list(predictor.predict(training_data))

    # Grafikte son 14 gün göster
    end_ts = hourly_df.index.max()
    start_ts = end_ts - pd.Timedelta(days=180)
    subset = hourly_df.loc[start_ts:]
    plot_forecasts(subset, forecasts, days_to_plot=14)
    # Konsola anlaşılır kısa özet yazdır
    if forecasts:
        summarize_forecast(forecasts[0])


if __name__ == "__main__":
    main()


