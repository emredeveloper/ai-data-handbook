from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
from datasets import load_dataset
import torch
import os
import re
from typing import List, Tuple
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()
from collections import Counter
import time
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER = True
except Exception:
    _VADER = False
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
    from rich.rule import Rule
    _RICH = True
except Exception:
    Console = None  # type: ignore
    _RICH = False

# model ve processor'u yükle
model_id_small = os.environ.get("ASR_BASE_MODEL", "openai/whisper-small")
processor = WhisperProcessor.from_pretrained(model_id_small)
model = WhisperForConditionalGeneration.from_pretrained(model_id_small)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Khan Academy Türkçe veri kümesinden tek bir örnek yükle
ds = load_dataset("ysdede/khanacademy-turkish", split="test[:1]")
example = ds[0]
audio = example["audio"]

# input özelliklerini hazırla
inputs = processor(
    audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
)
input_features = inputs.input_features.to(device)
attention_mask = inputs.attention_mask.to(device) if hasattr(inputs, "attention_mask") else None

# token id'lerini üret (beam search + Türkçe transkripsiyon)
generate_kwargs = dict(
    num_beams=5,
    task="transcribe",
    language="tr",
    temperature=0.0,
    return_timestamps=False,
    repetition_penalty=1.0,
)
if attention_mask is not None:
    generate_kwargs["attention_mask"] = attention_mask

with torch.no_grad():
    predicted_ids = model.generate(input_features, **generate_kwargs)

# metne çöz ve çıktı ver
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

def normalize_text(text: str) -> str:
    text = text.lower().strip()
    # Unicode-aware: kaldırma (noktalama/dışı karakterler). \w Türkçe harfleri de kapsar.
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def levenshtein_distance(seq_a: List[str], seq_b: List[str]) -> int:
    len_a, len_b = len(seq_a), len(seq_b)
    if len_a == 0:
        return len_b
    if len_b == 0:
        return len_a
    dp = [[0] * (len_b + 1) for _ in range(len_a + 1)]
    for i in range(len_a + 1):
        dp[i][0] = i
    for j in range(len_b + 1):
        dp[0][j] = j
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )
    return dp[len_a][len_b]

def compute_wer(ref: str, hyp: str) -> float:
    ref_tokens = normalize_text(ref).split()
    hyp_tokens = normalize_text(hyp).split()
    if len(ref_tokens) == 0:
        return 0.0 if len(hyp_tokens) == 0 else 1.0
    dist = levenshtein_distance(ref_tokens, hyp_tokens)
    return dist / max(1, len(ref_tokens))

def compute_cer(ref: str, hyp: str) -> float:
    ref_chars = list(normalize_text(ref))
    hyp_chars = list(normalize_text(hyp))
    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0
    dist = levenshtein_distance(ref_chars, hyp_chars)
    return dist / max(1, len(ref_chars))

def align_tokens(ref_tokens: List[str], hyp_tokens: List[str]) -> List[Tuple[str, str, str]]:
    """Kelime hizalaması döndürür: (ref_word or '', hyp_word or '', op)
    op ∈ {equal, sub, del, ins}
    """
    n, m = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back: List[List[Tuple[int, int, str]]] = [[(0, 0, '')] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = (i - 1, 0, 'del')
    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = (0, j - 1, 'ins')
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            # deletion
            best = (dp[i - 1][j] + 1, (i - 1, j, 'del'))
            # insertion
            cand = (dp[i][j - 1] + 1, (i, j - 1, 'ins'))
            if cand[0] < best[0]:
                best = cand
            # substitution / equal
            op = 'equal' if cost == 0 else 'sub'
            cand = (dp[i - 1][j - 1] + cost, (i - 1, j - 1, op))
            if cand[0] <= best[0]:
                best = cand
            dp[i][j] = best[0]
            back[i][j] = best[1]
    # backtrace
    i, j = n, m
    aligned: List[Tuple[str, str, str]] = []
    while i > 0 or j > 0:
        pi, pj, op = back[i][j]
        ref_word = ref_tokens[i - 1] if i > 0 and (pi == i - 1 and pj == j) or (pi == i - 1 and pj == j - 1) else ''
        hyp_word = hyp_tokens[j - 1] if j > 0 and (pi == i and pj == j - 1) or (pi == i - 1 and pj == j - 1) else ''
        if op == 'equal':
            aligned.append((ref_tokens[i - 1], hyp_tokens[j - 1], 'equal'))
        elif op == 'sub':
            aligned.append((ref_tokens[i - 1], hyp_tokens[j - 1], 'sub'))
        elif op == 'del':
            aligned.append((ref_tokens[i - 1], '', 'del'))
        elif op == 'ins':
            aligned.append(('', hyp_tokens[j - 1], 'ins'))
        i, j = pi, pj
    aligned.reverse()
    return aligned

def analyze_alignment(ref_text: str, hyp_text: str) -> Tuple[Counter, Counter, Counter]:
    """Döndürür: (confusion_pairs, wrong_ref_words, wrong_hyp_words)
    confusion_pairs: (ref->hyp) sadece substitution
    wrong_ref_words: substitution veya deletion’daki ref kelimesi
    wrong_hyp_words: substitution veya insertion’daki hyp kelimesi
    """
    ref_tokens = normalize_text(ref_text).split()
    hyp_tokens = normalize_text(hyp_text).split()
    aligned = align_tokens(ref_tokens, hyp_tokens)
    conf = Counter()
    wrong_ref = Counter()
    wrong_hyp = Counter()
    for r, h, op in aligned:
        if op == 'sub':
            conf[(r, h)] += 1
            if r:
                wrong_ref[r] += 1
            if h:
                wrong_hyp[h] += 1
        elif op == 'del':
            if r:
                wrong_ref[r] += 1
        elif op == 'ins':
            if h:
                wrong_hyp[h] += 1
    return conf, wrong_ref, wrong_hyp

def quality_from_wer(wer: float) -> str:
    if wer < 0.05:
        return "mükemmel"
    if wer < 0.15:
        return "çok iyi"
    if wer < 0.30:
        return "iyi"
    if wer < 0.50:
        return "orta"
    return "zayıf"

def colorize_sentiment(label: str) -> str:
    label = label.lower()
    if label == "positive":
        return "[green]positive[/green]"
    if label == "negative":
        return "[red]negative[/red]"
    if label == "neutral":
        return "[yellow]neutral[/yellow]"
    return label

def analyze_sentiment(text: str) -> Tuple[str, float, str]:
    backend = os.environ.get("SENTIMENT_BACKEND", "vader").lower()
    if backend == "vader" and _VADER:
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(text)
        comp = scores.get("compound", 0.0)
        label = "positive" if comp >= 0.05 else ("negative" if comp <= -0.05 else "neutral")
        return label, float(comp), "vader"
    # HF pipeline (hafif çok dilli model, yıldız dereceli)
    hf_model = os.environ.get("HF_SENTIMENT_MODEL", "nlptown/bert-base-multilingual-uncased-sentiment")
    try:
        clf = pipeline(
            "sentiment-analysis",
            model=hf_model,
            device=0 if device == "cuda" else -1,
        )
        out = clf(text)[0]
        # nlptown çıktısı genelde '1 star'..'5 stars'
        raw_label = str(out.get("label", "")).lower()
        score = float(out.get("score", 0.0))
        if "1" in raw_label or "2" in raw_label:
            label = "negative"
        elif "3" in raw_label:
            label = "neutral"
        else:
            label = "positive"
        return label, score, hf_model
    except Exception:
        return "unavailable", 0.0, "none"

ref_text = example["transcription"]
wer_score = compute_wer(ref_text, transcription)
cer_score = compute_cer(ref_text, transcription)
sentiment_label, sentiment_score, sentiment_backend = analyze_sentiment(transcription)

if _RICH:
    console = Console()
    console.print(Rule("Khan Academy TR - Whisper Değerlendirme"))
    qual = quality_from_wer(wer_score)
    panel_text = (
        f"[bold]Tahmin:[/bold] {transcription}\n"
        f"[bold]Gerçek:[/bold] {ref_text}\n\n"
        f"[bold]Duygu:[/bold] {colorize_sentiment(sentiment_label)} ({sentiment_score:.3f}) [dim]({sentiment_backend})[/dim]\n"
        f"[bold]WER:[/bold] {wer_score:.3f} [dim](kelime hata oranı, {qual})[/dim]    "
        f"[bold]CER:[/bold] {cer_score:.3f} [dim](karakter hata oranı)[/dim]"
    )
    console.print(Panel(panel_text, title="Tek Örnek Sonuç", box=box.ROUNDED))
    explain = (
        "[bold]WER[/bold]: Kelime düzeyinde Levenshtein düzenleme mesafesinin, referans kelime sayısına oranıdır. "
        "0.00 mükemmel; 0.10≈çok iyi; 0.30+ hatalar artar.\n"
        "[bold]CER[/bold]: Karakter düzeyinde aynı orandır. Noktalama ve büyük/küçük harf normalizasyonu uygulanır."
    )
    console.print(Panel(explain, title="Metrik Açıklamaları", box=box.MINIMAL))
else:
    print("Tahmin:", transcription)
    print("Gerçek:", ref_text)
    print("Duygu:", sentiment_label, f"({sentiment_score:.3f})")
    print("WER:", f"{wer_score:.3f}", "(kelime hata oranı – Levenshtein/kelime sayısı)")
    print("CER:", f"{cer_score:.3f}", "(karakter hata oranı – Levenshtein/karakter sayısı)")

# Mini değerlendirme (N örnek) — hızlı çalışması için küçük tutun
eval_count = int(os.environ.get("EVAL_COUNT", "10"))
eval_split = f"test[:{eval_count}]"
eval_ds = load_dataset("ysdede/khanacademy-turkish", split=eval_split)

eval_results: List[Tuple[float, float]] = []
hyps_small: List[str] = []
wers_small: List[float] = []
cers_small: List[float] = []
times_small: List[float] = []  # seconds
confusion_pairs = Counter()
wrong_ref_words = Counter()
wrong_hyp_words = Counter()
if _RICH:
    with Progress(
        TextColumn("[bold blue]Değerlendirme[/bold blue]"),
        BarColumn(bar_width=None),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("", total=len(eval_ds))
        for i, row in enumerate(eval_ds):
            a = row["audio"]
            inp = processor(a["array"], sampling_rate=a["sampling_rate"], return_tensors="pt")
            feats = inp.input_features.to(device)
            attn = inp.attention_mask.to(device) if hasattr(inp, "attention_mask") else None
            kwargs = dict(generate_kwargs)
            if attn is not None:
                kwargs["attention_mask"] = attn
            t0_i = time.time()
            with torch.no_grad():
                pred_ids = model.generate(feats, **kwargs)
            t1_i = time.time()
            hyp = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
            ref = row["transcription"]
            w_i = compute_wer(ref, hyp)
            c_i = compute_cer(ref, hyp)
            eval_results.append((w_i, c_i))
            hyps_small.append(hyp)
            wers_small.append(w_i)
            cers_small.append(c_i)
            times_small.append(t1_i - t0_i)
            conf, wref, whyp = analyze_alignment(ref, hyp)
            confusion_pairs.update(conf)
            wrong_ref_words.update(wref)
            wrong_hyp_words.update(whyp)
            progress.advance(task)
else:
    for i, row in enumerate(eval_ds):
        a = row["audio"]
        inp = processor(a["array"], sampling_rate=a["sampling_rate"], return_tensors="pt")
        feats = inp.input_features.to(device)
        attn = inp.attention_mask.to(device) if hasattr(inp, "attention_mask") else None
        kwargs = dict(generate_kwargs)
        if attn is not None:
            kwargs["attention_mask"] = attn
        t0_i = time.time()
        with torch.no_grad():
            pred_ids = model.generate(feats, **kwargs)
        t1_i = time.time()
        hyp = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
        ref = row["transcription"]
        w_i = compute_wer(ref, hyp)
        c_i = compute_cer(ref, hyp)
        eval_results.append((w_i, c_i))
        hyps_small.append(hyp)
        wers_small.append(w_i)
        cers_small.append(c_i)
        times_small.append(t1_i - t0_i)
        conf, wref, whyp = analyze_alignment(ref, hyp)
        confusion_pairs.update(conf)
        wrong_ref_words.update(wref)
        wrong_hyp_words.update(whyp)

if eval_results:
    avg_wer = sum(w for w, _ in eval_results) / len(eval_results)
    avg_cer = sum(c for _, c in eval_results) / len(eval_results)
    if _RICH:
        table = Table(title=f"Mini Değerlendirme (ilk {len(eval_results)} örnek)", box=box.SIMPLE_HEAVY, show_edge=True)
        table.add_column("Örnek", justify="right", style="bold")
        table.add_column("WER", justify="right")
        table.add_column("CER", justify="right")
        for idx, (w, c) in enumerate(eval_results):
            table.add_row(str(idx), f"{w:.3f}", f"{c:.3f}")
        console.print(table)
        summary = (
            f"[bold]Ortalama WER:[/bold] {avg_wer:.3f} [dim]({quality_from_wer(avg_wer)})[/dim]    "
            f"[bold]Ortalama CER:[/bold] {avg_cer:.3f}"
        )
        console.print(Panel(summary, title="Değerlendirme Özeti", box=box.ROUNDED))
        # Hata analizi tabloları
        top_n = int(os.environ.get("TOP_N", "10"))
        if confusion_pairs:
            tconf = Table(title=f"En Sık Karışan Çiftler (ilk {top_n})", box=box.SIMPLE_HEAVY, show_edge=True)
            tconf.add_column("Ref", style="bold red")
            tconf.add_column("→")
            tconf.add_column("Hyp", style="bold green")
            tconf.add_column("Adet", justify="right")
            for (r, h), cnt in confusion_pairs.most_common(top_n):
                tconf.add_row(r, "→", h, str(cnt))
            console.print(tconf)
        if wrong_ref_words:
            trw = Table(title=f"En Çok Yanlış Tanınan Referans Kelimeler (ilk {top_n})", box=box.SIMPLE_HEAVY, show_edge=True)
            trw.add_column("Kelime", style="bold")
            trw.add_column("Adet", justify="right")
            for w, cnt in wrong_ref_words.most_common(top_n):
                trw.add_row(w, str(cnt))
            console.print(trw)
        if wrong_hyp_words:
            thw = Table(title=f"En Çok Üretilen Hatalı Hyp Kelimeler (ilk {top_n})", box=box.SIMPLE_HEAVY, show_edge=True)
            thw.add_column("Kelime", style="bold")
            thw.add_column("Adet", justify="right")
            for w, cnt in wrong_hyp_words.most_common(top_n):
                thw.add_row(w, str(cnt))
            console.print(thw)
    else:
        print(f"Mini Değerlendirme (ilk {len(eval_results)} örnek)")
        for idx, (w, c) in enumerate(eval_results):
            print(f"#{idx}: WER={w:.3f} CER={c:.3f}")
        print(f"Ortalama WER={avg_wer:.3f}  Ortalama CER={avg_cer:.3f}")
        if confusion_pairs:
            print("\nEn Sık Karışan Çiftler:")
            for (r, h), cnt in confusion_pairs.most_common(10):
                print(f"{r} -> {h}: {cnt}")
        if wrong_ref_words:
            print("\nEn Çok Yanlış Tanınan Referans Kelimeler:")
            for w, cnt in wrong_ref_words.most_common(10):
                print(f"{w}: {cnt}")
        if wrong_hyp_words:
            print("\nEn Çok Üretilen Hatalı Hyp Kelimeler:")
            for w, cnt in wrong_hyp_words.most_common(10):
                print(f"{w}: {cnt}")

# Opsiyonel: Whisper-medium ile kıyaslama
compare_models = os.environ.get("COMPARE_MODELS", "1") == "1"
if compare_models and eval_results:
    model_id_medium = os.environ.get("ASR_COMPARE_MODEL", "openai/whisper-medium")
    try:
        if _RICH:
            console.print(Panel(f"Model Karşılaştırması: [bold]{model_id_small}[/bold] vs [bold]{model_id_medium}[/bold] — model indirilebilir, lütfen bekleyin", box=box.MINIMAL))
        t0 = time.time()
        # Güvenli yükleyici: CUDA OOM durumunda CPU'ya düş
        try:
            model_med = WhisperForConditionalGeneration.from_pretrained(model_id_medium).to(device)
        except RuntimeError as e:
            if "CUDA" in str(e) or "out of memory" in str(e).lower():
                if _RICH:
                    console.print(Panel("CUDA bellek yetersiz; medium modeli CPU'da yüklenecek.", box=box.MINIMAL))
                model_med = WhisperForConditionalGeneration.from_pretrained(model_id_medium).to("cpu")
            else:
                raise
        t1 = time.time()
        eval_results_med: List[Tuple[float, float]] = []
        hyps_med: List[str] = []
        wers_med: List[float] = []
        cers_med: List[float] = []
        times_med: List[float] = []
        for i, row in enumerate(eval_ds):
            a = row["audio"]
            inp = processor(a["array"], sampling_rate=a["sampling_rate"], return_tensors="pt")
            feats = inp.input_features.to(device)
            attn = inp.attention_mask.to(device) if hasattr(inp, "attention_mask") else None
            kwargs = dict(generate_kwargs)
            if attn is not None:
                kwargs["attention_mask"] = attn
            t0_i = time.time()
            with torch.no_grad():
                pred_ids = model_med.generate(feats, **kwargs)
            t1_i = time.time()
            hyp = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
            ref = row["transcription"]
            w_i = compute_wer(ref, hyp)
            c_i = compute_cer(ref, hyp)
            eval_results_med.append((w_i, c_i))
            hyps_med.append(hyp)
            wers_med.append(w_i)
            cers_med.append(c_i)
            times_med.append(t1_i - t0_i)
        load_ms = (t1 - t0) * 1000
        avg_wer_small = sum(w for w, _ in eval_results) / len(eval_results)
        avg_cer_small = sum(c for _, c in eval_results) / len(eval_results)
        avg_wer_med = sum(w for w, _ in eval_results_med) / len(eval_results_med)
        avg_cer_med = sum(c for _, c in eval_results_med) / len(eval_results_med)
        avg_t_small = (sum(times_small) / len(times_small)) * 1000 if times_small else 0.0
        avg_t_med = (sum(times_med) / len(times_med)) * 1000 if times_med else 0.0
        # per-örnek iyileşme/bozulma sayıları
        improved = sum(1 for s, m in zip(wers_small, wers_med) if m < s)
        worse = sum(1 for s, m in zip(wers_small, wers_med) if m > s)
        tie = len(wers_small) - improved - worse
        if _RICH:
            comp = Table(title="Model Karşılaştırma (aynı eval set)", box=box.SIMPLE_HEAVY, show_edge=True)
            comp.add_column("Model", style="bold")
            comp.add_column("Ortalama WER", justify="right")
            comp.add_column("Ortalama CER", justify="right")
            comp.add_column("Ort. Süre (ms)", justify="right")
            comp.add_column("Yükleme (ms)", justify="right")
            comp.add_row(model_id_small, f"{avg_wer_small:.3f}", f"{avg_cer_small:.3f}", f"{avg_t_small:.0f}", "-")
            comp.add_row(model_id_medium, f"{avg_wer_med:.3f}", f"{avg_cer_med:.3f}", f"{avg_t_med:.0f}", f"{load_ms:.0f}")
            console.print(comp)
            # En çok iyileşen ve kötüleşen örnekler (WER delta sırasına göre)
            deltas = [(i, wers_small[i], wers_med[i], wers_med[i] - wers_small[i]) for i in range(len(wers_small))]
            best = sorted(deltas, key=lambda x: x[3])[: min(10, len(deltas))]
            worst = sorted(deltas, key=lambda x: x[3], reverse=True)[: min(10, len(deltas))]
            tb = Table(title="En Çok İyileşen Örnekler (WER)", box=box.SIMPLE_HEAVY, show_edge=True)
            tb.add_column("#", justify="right")
            tb.add_column("Small", justify="right")
            tb.add_column("Medium", justify="right")
            tb.add_column("Δ (Med-Small)", justify="right")
            for i, ws, wm, d in best:
                tb.add_row(str(i), f"{ws:.3f}", f"{wm:.3f}", f"{d:.3f}")
            console.print(tb)
            tw = Table(title="En Çok Kötüleşen Örnekler (WER)", box=box.SIMPLE_HEAVY, show_edge=True)
            tw.add_column("#", justify="right")
            tw.add_column("Small", justify="right")
            tw.add_column("Medium", justify="right")
            tw.add_column("Δ (Med-Small)", justify="right")
            for i, ws, wm, d in worst:
                tw.add_row(str(i), f"{ws:.3f}", f"{wm:.3f}", f"{d:.3f}")
            console.print(tw)
            console.print(Panel(f"İyileşen: {improved}   Kötüleşen: {worse}   Aynı: {tie}", box=box.MINIMAL))
        else:
            print("Model Karşılaştırma:")
            print(f"{model_id_small}: WER={avg_wer_small:.3f} CER={avg_cer_small:.3f} avg_t={avg_t_small:.0f}ms")
            print(f"{model_id_medium}: WER={avg_wer_med:.3f} CER={avg_cer_med:.3f} avg_t={avg_t_med:.0f}ms (load ~{load_ms:.0f} ms)")
            print(f"İyileşen={improved} Kötüleşen={worse} Aynı={tie}")
    except Exception as e:
        if _RICH:
            console.print(Panel(f"Karşılaştırma başarısız: {e}", title="Hata", box=box.MINIMAL))
        else:
            print("Karşılaştırma başarısız:", e)