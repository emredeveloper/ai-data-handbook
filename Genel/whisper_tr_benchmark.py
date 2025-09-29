import argparse
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import Audio, Dataset, load_dataset
import evaluate
from transformers import pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Whisper small vs medium - Turkish ASR benchmark (ysdede/khanacademy-turkish)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Değerlendirme için kullanılacak örnek sayısı (varsayılan: 200)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="İşlenecek toplu örnek sayısı (varsayılan: 8)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Kullanılacak veri bölümü (mevcut değilse 'train' yedek olarak kullanılır)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Cihaz: cuda, cuda:0, cpu (varsayılan: otomatik)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["openai/whisper-small", "openai/whisper-medium"],
        help="Karşılaştırılacak model kimlikleri",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Uygunsa yarım hassasiyet (float16) ile çalıştır",
    )
    return parser.parse_args()


def select_device(user_device: str | None) -> str:
    if user_device:
        return user_device
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_turkish_dataset(split: str, max_samples: int) -> Dataset:
    # Veri kümesini yükle ve ses sütununu Whisper için 16kHz'e dönüştür
    try:
        ds = load_dataset("ysdede/khanacademy-turkish", split=split)
    except Exception:
        # Bazı veri kümelerinde 'test' olmayabilir
        fallback_split = "train"
        ds = load_dataset("ysdede/khanacademy-turkish", split=fallback_split)

    # Beklenen sütunlar: 'audio', 'transcription'
    if "audio" not in ds.column_names or "transcription" not in ds.column_names:
        raise ValueError(
            f"Beklenen sütunlar bulunamadı. Var olanlar: {ds.column_names}. 'audio' ve 'transcription' gerekli."
        )

    # Whisper 16kHz bekler
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    # Boş/eksik etiketleri filtrele
    ds = ds.filter(lambda x: x["audio"] is not None and isinstance(x["transcription"], str) and len(x["transcription"].strip()) > 0)

    if max_samples and max_samples > 0:
        ds = ds.select(range(min(max_samples, ds.num_rows)))
    return ds


def build_asr_pipeline(model_id: str, device: str, fp16: bool):
    # Pipeline, Whisper için generate ayarlarını doğrudan alabilir
    torch_dtype = torch.float16 if (fp16 and device.startswith("cuda")) else None
    generate_kwargs = {"task": "transcribe", "language": "turkish"}
    asr = pipeline(
        task="automatic-speech-recognition",
        model=model_id,
        torch_dtype=torch_dtype,
        device=device,
        return_timestamps=False,
        generate_kwargs=generate_kwargs,
    )
    return asr


def batched(iterable: List, n: int) -> List[List]:
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def prepare_inputs(batch: List[Dict]) -> List[Dict[str, object]]:
    inputs = []
    for example in batch:
        audio = example["audio"]
        inputs.append({"array": audio["array"], "sampling_rate": audio["sampling_rate"]})
    return inputs


def run_inference(asr, ds: Dataset, batch_size: int) -> Tuple[List[str], float, float]:
    predictions: List[str] = []
    total_audio_sec = 0.0
    start = time.perf_counter()

    for batch_indices in batched(list(range(ds.num_rows)), batch_size):
        batch = [ds[i] for i in batch_indices]
        inputs = prepare_inputs(batch)
        outputs = asr(inputs)
        if isinstance(outputs, dict):  # Tek örnek dönebilir
            outputs = [outputs]
        for out, ex in zip(outputs, batch):
            text = out.get("text") if isinstance(out, dict) else str(out)
            predictions.append(text)
            audio = ex["audio"]
            total_audio_sec += float(len(audio["array"]) / audio["sampling_rate"])

    elapsed = time.perf_counter() - start
    return predictions, elapsed, total_audio_sec


def compute_metrics(preds: List[str], refs: List[str]) -> Dict[str, float]:
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    # JiWER metrikleri küçük harfe indirgeme ve basit normalizasyon ile daha stabil olur
    refs_n = [r.strip().lower() for r in refs]
    preds_n = [p.strip().lower() for p in preds]
    wer = float(wer_metric.compute(references=refs_n, predictions=preds_n))
    cer = float(cer_metric.compute(references=refs_n, predictions=preds_n))
    return {"wer": wer, "cer": cer}


def main():
    args = parse_args()
    device = select_device(args.device)

    print(f"Cihaz: {device}")
    print(f"Maksimum örnek: {args.max_samples} | Toplu boyutu: {args.batch_size}\n")

    ds = load_turkish_dataset(args.split, args.max_samples)
    print(f"Veri kümesi: ysdede/khanacademy-turkish | Satır sayısı: {ds.num_rows}")
    print(f"Sütunlar: {ds.column_names}")

    references = [ds[i]["transcription"] for i in range(ds.num_rows)]

    results = []
    for model_id in args.models:
        print(f"\nModel hazırlanıyor: {model_id}")
        asr = build_asr_pipeline(model_id=model_id, device=device, fp16=args.fp16)

        print("Çıkarım başlıyor...")
        preds, wall_time, total_audio_sec = run_inference(asr, ds, args.batch_size)
        rtf = wall_time / max(total_audio_sec, 1e-8)

        print("Met rikler hesaplanıyor (WER/CER)...")
        metrics = compute_metrics(preds, references)

        result = {
            "model": model_id,
            "num_samples": ds.num_rows,
            "wer": metrics["wer"],
            "cer": metrics["cer"],
            "wall_time_sec": wall_time,
            "audio_duration_sec": total_audio_sec,
            "real_time_factor": rtf,
        }
        results.append(result)

        print(
            (
                f"Model: {model_id}\n"
                f"  WER: {result['wer']:.4f}\n"
                f"  CER: {result['cer']:.4f}\n"
                f"  Süre (sn): {result['wall_time_sec']:.2f}\n"
                f"  Toplam ses süresi (sn): {result['audio_duration_sec']:.2f}\n"
                f"  RTF: {result['real_time_factor']:.2f}"
            )
        )

    print("\nÖzet Karşılaştırma:")
    for r in results:
        print(
            f"- {r['model']}: WER={r['wer']:.4f}, CER={r['cer']:.4f}, RTF={r['real_time_factor']:.2f}"
        )

    print(
        "\nNot: Whisper kullanımı için bkz. Transformers Whisper dokümantasyonu."
        " Kaynak: https://huggingface.co/docs/transformers/model_doc/whisper"
    )


if __name__ == "__main__":
    main()


