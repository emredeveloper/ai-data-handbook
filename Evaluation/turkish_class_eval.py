"""Turkish Synthetic Text Classification Evaluation with Local Ollama

Loads the Hugging Face dataset `emredeveloper/Turkish-Synthetic-Text-Classification`
and evaluates a local Ollama model (e.g. qwen3:4b) on the sentiment / stance
labels: olumlu, olumsuz, nötr.

Reasoning / thinking tag handling:
Some models emit <think>...</think> sections (internal reasoning). These are
stripped before classification so they do not affect parsing. Raw text is still
stored in exports for audit.

Features:
 - Select split (train/test)
 - Limit number of samples (for quick runs)
 - Simple prompt → model → parse class
 - Accuracy, per-class precision/recall/F1 (macro & weighted F1) computed manually
 - Optional JSONL export of per-sample predictions
 - Lightweight caching of model calls (avoids repeated generations if rerun)
 - Configurable field names (if dataset schema changes) via CLI
    - OPTIONAL: Deepeval metric (--deepeval) for a unified test case interface

Assumptions:
 - Dataset columns are (text, label). If unsure, script attempts to guess.
 - Model outputs may be verbose; first occurrence of any class token in lower
   case output is taken as prediction. If none found → 'nötr' fallback.

Run examples:
  python turkish_class_eval.py --model qwen3:4b --split test --limit 100
  python turkish_class_eval.py --model qwen3:4b --export results.jsonl
  python turkish_class_eval.py --model qwen3:4b --no-cache

Install deps (if not yet):
  pip install -r Evaluation/requirements.txt

Future ideas (not implemented yet):
  - Parallel requests with asyncio
  - Deepeval integration per sample (would be slower)
  - Confusion matrix heatmap export
  - Retry logic for transient Ollama failures
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import re
try:
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import BaseMetric
except ImportError:
    LLMTestCase = None  # type: ignore
    BaseMetric = object  # fallback dummy
import requests
from datasets import load_dataset
from tqdm import tqdm


CLASSES = ["olumlu", "olumsuz", "nötr"]  # order used in metrics


# ----------------------------- Ollama Wrapper ---------------------------------
class OllamaClient:
    def __init__(self, model: str, host: str = "http://localhost:11434", timeout: int = 120):
        self.model = model
        self.host = host.rstrip("/")
        self.url = f"{self.host}/api/generate"
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        try:
            r = requests.post(self.url, json=payload, timeout=self.timeout)
            r.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc
        text_parts = []
        for line in r.text.splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            content = obj.get("response") or obj.get("content") or ""
            text_parts.append(content)
        return "".join(text_parts).strip()


# ---------------------------- Caching Layer -----------------------------------
class SimpleCache:
    def __init__(self, path: str):
        self.path = path
        self.data: Dict[str, str] = {}
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            self.data[obj["key"]] = obj["value"]
                        except Exception:
                            pass
            except Exception:
                pass

    @staticmethod
    def _make_key(model: str, prompt: str) -> str:
        h = hashlib.sha256(f"{model}\n{prompt}".encode("utf-8")).hexdigest()[:32]
        return h

    def get(self, model: str, prompt: str) -> str | None:
        return self.data.get(self._make_key(model, prompt))

    def set(self, model: str, prompt: str, value: str):
        key = self._make_key(model, prompt)
        self.data[key] = value
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")


# ----------------------------- Prediction Logic -------------------------------
def build_prompt(text: str) -> str:
    return (
        "Aşağıdaki Türkçe cümlenin duygu/sentiment sınıfını belirle.\n"
        "Sadece şu etiketlerden birini ver: olumlu, olumsuz, nötr.\n"
        "Ek açıklama yapma.\n\n"
        f"Metin: {text}\n\nCevap:"
    )


def strip_think(raw_output: str) -> str:
    # Remove <think>...</think> blocks and stray tags
    cleaned = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def parse_prediction(raw_output: str) -> str:
    raw_output = strip_think(raw_output)
    low = raw_output.lower()
    for cls in CLASSES:
        if cls in low:
            return cls
    # common ascii fallback for 'nötr'
    if "notr" in low:
        return "nötr"
    return "nötr"  # default fallback


# ----------------------------- Metrics ----------------------------------------
@dataclass
class Metrics:
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class: Dict[str, Dict[str, float]]  # precision, recall, f1, support


def compute_metrics(golds: List[str], preds: List[str]) -> Metrics:
    assert len(golds) == len(preds)
    # confusion counts
    counts: Dict[str, Dict[str, int]] = {g: {p: 0 for p in CLASSES} for g in CLASSES}
    support: Dict[str, int] = {c: 0 for c in CLASSES}
    for g, p in zip(golds, preds):
        if g not in CLASSES:
            continue
        support[g] += 1
        if p not in CLASSES:
            p = "nötr"
        counts[g][p] += 1

    per_class_stats: Dict[str, Dict[str, float]] = {}
    f1s = []
    weighted_f1_numer = 0.0
    total = len(golds) if golds else 1
    correct = sum(counts[c][c] for c in CLASSES)
    for c in CLASSES:
        tp = counts[c][c]
        fp = sum(counts[g][c] for g in CLASSES if g != c)
        fn = sum(counts[c][p] for p in CLASSES if p != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        f1s.append(f1)
        weighted_f1_numer += f1 * support[c]
        per_class_stats[c] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": support[c],
        }

    accuracy = correct / total if total else 0.0
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    weighted_f1 = weighted_f1_numer / total if total else 0.0
    return Metrics(accuracy=accuracy, macro_f1=macro_f1, weighted_f1=weighted_f1, per_class=per_class_stats)


# ----------------------------- Deepeval Metric --------------------------------
class DeepevalAccuracyMetric(BaseMetric):  # type: ignore
    """Simple Deepeval compatible metric for exact class match.

    Score = 1 if predicted label equals expected label else 0. Aggregation is
    handled outside.
    """
    NAME = "deepeval_class_accuracy"

    def __init__(self):
        self.score = 0.0
        self.reason = ""

    def measure(self, test_case: 'LLMTestCase'):  # type: ignore
        exp = (test_case.expected_output or "").strip().lower()
        act = (test_case.actual_output or "").strip().lower()
        if exp == act:
            self.score = 1.0
            self.reason = "match"
        else:
            self.score = 0.0
            self.reason = f"expected={exp} got={act}"

    def is_successful(self) -> bool:  # type: ignore
        return self.score == 1.0

    def to_dict(self):
        return {"name": self.NAME, "score": self.score, "reason": self.reason}


# ----------------------------- Dataset Helpers --------------------------------
def guess_fields(example: Dict[str, object], args) -> Tuple[str, str]:
    cols = list(example.keys())
    text_field = args.text_field
    label_field = args.label_field
    if text_field and label_field:
        return text_field, label_field
    # heuristics
    text_candidates = ["text", "sentence", "metin", "content"]
    label_candidates = ["label", "labels", "kategori", "class"]
    if not text_field:
        for c in text_candidates:
            if c in cols:
                text_field = c
                break
    if not label_field:
        for c in label_candidates:
            if c in cols:
                label_field = c
                break
    # fallback to first two columns
    if not text_field and cols:
        text_field = cols[0]
    if not label_field and len(cols) > 1:
        label_field = cols[1]
    if not (text_field and label_field):
        raise ValueError(f"Could not determine text/label fields. Columns present: {cols}")
    return text_field, label_field


# ----------------------------- Main Evaluation --------------------------------
def evaluate(args):
    print(f"[INFO] Loading dataset: {args.dataset} (split={args.split})")
    ds = load_dataset(args.dataset, split=args.split)
    total_rows = len(ds)
    print(f"[INFO] Total rows in split: {total_rows}")

    if args.shuffle:
        ds = ds.shuffle(seed=args.seed)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"[INFO] Evaluating {len(ds)} samples")

    example0 = ds[0]
    text_field, label_field = guess_fields(example0, args)
    print(f"[INFO] Using text_field='{text_field}' label_field='{label_field}'")

    client = OllamaClient(args.model, host=args.host, timeout=args.timeout)
    cache = None if args.no_cache else SimpleCache(args.cache_file)

    golds: List[str] = []
    preds: List[str] = []
    export_fp = open(args.export, "w", encoding="utf-8") if args.export else None

    deepeval_cases = []
    deepeval_scores = []

    try:
        for row in tqdm(ds, desc="Evaluating"):
            text = str(row[text_field])
            label = str(row[label_field]).lower().strip()
            golds.append(label)
            prompt = build_prompt(text)

            cached = cache.get(args.model, prompt) if cache else None
            if cached is not None:
                raw_output = cached
            else:
                raw_output = client.generate(prompt)
                if cache:
                    cache.set(args.model, prompt, raw_output)

            pred = parse_prediction(raw_output)
            preds.append(pred)

            if args.deepeval and LLMTestCase is not None:
                case = LLMTestCase(
                    input=prompt,
                    expected_output=label,
                    actual_output=pred,
                )
                metric = DeepevalAccuracyMetric()
                metric.measure(case)
                deepeval_cases.append(case)
                deepeval_scores.append(metric.score)

            if export_fp:
                export_fp.write(json.dumps({
                    "text": text,
                    "gold": label,
                    "prediction": pred,
                    "raw_output": raw_output,
                }, ensure_ascii=False) + "\n")
    finally:
        if export_fp:
            export_fp.close()

    metrics = compute_metrics(golds, preds)
    print("\n=== RESULTS ===")
    print(f"Accuracy     : {metrics.accuracy:.4f}")
    print(f"Macro F1     : {metrics.macro_f1:.4f}")
    print(f"Weighted F1  : {metrics.weighted_f1:.4f}")
    print("Per-class:")
    for c in CLASSES:
        pc = metrics.per_class[c]
        print(
            f"  {c:8s} -> P={pc['precision']:.3f} R={pc['recall']:.3f} F1={pc['f1']:.3f} support={pc['support']}"
        )

    if args.metrics_json:
        with open(args.metrics_json, "w", encoding="utf-8") as f:
            json.dump({
                "accuracy": metrics.accuracy,
                "macro_f1": metrics.macro_f1,
                "weighted_f1": metrics.weighted_f1,
                "per_class": metrics.per_class,
                "samples": len(golds),
                "model": args.model,
                "dataset": args.dataset,
                "split": args.split,
                "deepeval_accuracy": (sum(deepeval_scores) / len(deepeval_scores)) if deepeval_scores else None,
            }, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Metrics JSON written to {args.metrics_json}")

    if args.deepeval:
        if LLMTestCase is None:
            print("[WARN] Deepeval not installed; skipping --deepeval metrics.")
        else:
            if deepeval_scores:
                print(f"[DEEPEVAL] Accuracy over {len(deepeval_scores)} cases: {sum(deepeval_scores)/len(deepeval_scores):.4f}")
            else:
                print("[DEEPEVAL] No cases processed.")


def parse_args(argv: List[str]):
    p = argparse.ArgumentParser(description="Evaluate a local Ollama model on a Turkish classification dataset.")
    p.add_argument("--dataset", default="emredeveloper/Turkish-Synthetic-Text-Classification", help="HF dataset path")
    p.add_argument("--split", default="test", help="Dataset split (train/test)")
    p.add_argument("--model", default="qwen3:4b", help="Ollama model tag")
    p.add_argument("--host", default="http://localhost:11434", help="Ollama host URL")
    p.add_argument("--timeout", type=int, default=120, help="HTTP timeout seconds")
    p.add_argument("--limit", type=int, default=0, help="Limit number of samples (0 = all)")
    p.add_argument("--shuffle", action="store_true", help="Shuffle before limiting")
    p.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    p.add_argument("--export", help="Path to write JSONL lines with per-sample predictions")
    p.add_argument("--metrics-json", help="Write aggregate metrics to JSON file")
    p.add_argument("--text-field", help="Explicit text field name")
    p.add_argument("--label-field", help="Explicit label field name")
    p.add_argument("--cache-file", default="ollama_eval_cache.jsonl", help="Cache file path")
    p.add_argument("--no-cache", action="store_true", help="Disable caching of generations")
    p.add_argument("--deepeval", action="store_true", help="Also run Deepeval accuracy metric (requires deepeval installed)")
    return p.parse_args(argv)


def main(argv: List[str]):
    args = parse_args(argv)
    random.seed(args.seed)
    evaluate(args)


if __name__ == "__main__":
    main(sys.argv[1:])
