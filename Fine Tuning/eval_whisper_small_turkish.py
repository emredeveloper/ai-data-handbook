#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import datasets
from datasets import Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration, GenerationConfig
import evaluate
from tqdm import tqdm

OUTPUT_DIR = "./whisper-small-turkish-khanacademy"  # fine-tune çıktısı
MODEL_ID = OUTPUT_DIR
VAL_SPLIT_SIZE = 200  # hızlı değerlendirme için (None -> tamamı)
TARGET_SR = 16000


def main():
    # Hız/performans
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Model ve processor yükleniyor...")
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID,
        attn_implementation="sdpa"  # Transformers >=4.36 için
    ).to(device)

    # Greedy, hızlı değerlendirme
    model.generation_config = GenerationConfig(
        max_length=192,
        num_beams=1,
        do_sample=False,
        no_repeat_ngram_size=0,
        repetition_penalty=1.0,
    )

    print("Doğrulama verisi yükleniyor...")
    ds = datasets.load_dataset("ysdede/khanacademy-turkish")
    if "test" in ds and "validation" not in ds:
        ds = datasets.DatasetDict({"train": ds["train"], "validation": ds["test"]})

    ds = ds.cast_column("audio", Audio(sampling_rate=TARGET_SR))
    val_ds = ds["validation"]
    if VAL_SPLIT_SIZE:
        val_ds = val_ds.select(range(min(VAL_SPLIT_SIZE, len(val_ds))))
    print(f"Validation örnek sayısı: {len(val_ds)}")

    def preprocess(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
        ).input_features[0]
        text = batch["transcription"] if "transcription" in batch else batch.get("text", "")
        batch["labels"] = processor.tokenizer(text, padding=False, truncation=True, max_length=448).input_ids
        return batch

    val_proc = val_ds.map(preprocess, remove_columns=val_ds.column_names)

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    print("Değerlendirme başlıyor (greedy)...")
    preds = []
    refs = []
    batch_size = 12 if device == "cuda" else 2

    for i in tqdm(range(0, len(val_proc), batch_size)):
        batch = val_proc[i : i + batch_size]
        input_features = torch.stack(batch["input_features"]).to(device)
        with torch.no_grad():
            pred_ids = model.generate(input_features)
        pred_text = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        # labels decode (mask yok)
        ref_ids = batch["labels"]
        ref_text = processor.tokenizer.batch_decode(ref_ids, skip_special_tokens=True)
        preds.extend(pred_text)
        refs.extend(ref_text)

    wer = wer_metric.compute(predictions=preds, references=refs)
    cer = cer_metric.compute(predictions=preds, references=refs)

    print("\n=== EVALUATION (Validation) ===")
    print(f"WER: {wer:.4f}")
    print(f"CER: {cer:.4f}")

    # Örnek çıktı göster
    print("\nÖrnek çıktı:\n-----------------")
    for i in range(min(3, len(preds))):
        print(f"Pred: {preds[i]}")
        print(f"Ref : {refs[i]}")
        print("---")


if __name__ == "__main__":
    main()


