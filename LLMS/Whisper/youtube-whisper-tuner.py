"""
YouTube tabanlı Whisper Türkçe config tuner (No Prompt)
- Verilen video URL'lerinden sesi indirir
- Altyazı (TR>EN>any) bulursa referans kabul eder
- Küçük bir grid üzerinde decoding ayarlarını dener (beam/penalty)
- Ortalama WER'e göre en iyi config'i whisper_best_config.json olarak kaydeder
"""

import os
import json
import warnings
import subprocess
from typing import List, Tuple, Optional

import torch
import librosa
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.utils import logging as hf_logging

# Rich çıktı
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
    from rich import box
    console = Console()
except Exception:
    console = None

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
hf_logging.set_verbosity_error()

# Çıktılar için klasör
BASE_DIR = os.path.dirname(__file__)
OUTPUTS_RELATIVE = "outputs/"  # @outputs/ yolu
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.environ["WHISPER_OUTPUTS_DIR"] = OUTPUT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    from youtube_transcript_api import YouTubeTranscriptApi
except Exception as e:
    print("youtube-transcript-api gerekli: pip install youtube-transcript-api")
    raise


def extract_video_id(url: str) -> str:
    if "shorts/" in url:
        return url.split("shorts/")[1].split("?")[0]
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    if "youtu.be" in url:
        return url.split("/")[-1].split("?")[0]
    return url


def fetch_transcript_text(video_id: str) -> Optional[str]:
    """Yeni API (>=1.2.x) ve eski fallback ile altyazıyı getirir."""
    try:
        api = YouTubeTranscriptApi()
        try:
            tr = api.fetch(video_id, languages=['tr'])
        except Exception:
            try:
                tr = api.fetch(video_id, languages=['en'])
            except Exception:
                tr = api.fetch(video_id)
        return " ".join([s.text for s in tr])
    except Exception:
        try:
            data = YouTubeTranscriptApi.get_transcript(video_id, languages=['tr'])
            return " ".join([x['text'] for x in data])
        except Exception:
            try:
                data = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
                return " ".join([x['text'] for x in data])
            except Exception:
                return None


def download_audio(url: str, out_name: str) -> Optional[str]:
    """yt-dlp ile indirir; başarıyla oluştursa yol döner."""
    try:
        target_no_ext = os.path.join(OUTPUT_DIR, out_name)
        cmd = [
            'yt-dlp', '-x', '--audio-format', 'mp3', '--audio-quality', '0',
            '-o', target_no_ext, url, '--no-warnings', '--quiet'
        ]
        subprocess.run(cmd, capture_output=True, text=True)
        for p in [f"{target_no_ext}", f"{target_no_ext}.mp3"]:
            if os.path.exists(p):
                return p
    except FileNotFoundError:
        msg = "yt-dlp bulunamadı: pip install yt-dlp"
        console.print(f"[red]{msg}[/red]") if console else print(msg)
    return None


def wer(reference: str, hypothesis: str) -> float:
    ref = reference.split()
    hyp = hypothesis.split()
    d = np.zeros((len(ref) + 1, len(hyp) + 1))
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1].lower() == hyp[j - 1].lower():
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(d[i - 1][j - 1] + 1, d[i][j - 1] + 1, d[i - 1][j] + 1)
    return float(d[len(ref)][len(hyp)]) / max(1, len(ref)) * 100.0


class WhisperNoPrompt:
    def __init__(self, model_name: str = "openai/whisper-small") -> None:
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def transcribe(self, audio_path: str, gen_kwargs: dict) -> str:
        audio, _ = librosa.load(audio_path, sr=16000)
        input_features = self.processor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(self.device)
        generated_ids = self.model.generate(input_features, **gen_kwargs)
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


def tune_on_videos(urls: List[str]) -> Tuple[dict, list]:
    model = WhisperNoPrompt()

    beam_list = [1, 4, 8, 12]
    lp_list = [1.0, 1.2, 1.5]
    ngram_list = [0, 2, 3]

    candidates = []
    for nb in beam_list:
        for lp in lp_list:
            for ng in ngram_list:
                gen_kwargs = {
                    "language": "tr",
                    "task": "transcribe",
                    "max_length": 448,
                    "num_beams": max(1, nb),
                    "early_stopping": True if nb > 1 else False,
                }
                if lp != 1.0:
                    gen_kwargs["length_penalty"] = lp
                if ng > 0:
                    gen_kwargs["no_repeat_ngram_size"] = ng
                candidates.append(gen_kwargs)

    results = []  # [(gen_kwargs, avg_wer, used_count)]

    # Progress
    if console:
        progress = Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), console=console)
        task = progress.add_task("[cyan]Adaylar deneniyor...", total=len(candidates))
        progress.start()

    try:
        for cfg in candidates:
            wers = []
            used = 0
            for url in urls:
                vid = extract_video_id(url)
                ref = fetch_transcript_text(vid)
                if not ref:
                    continue
                audio_path = download_audio(url, out_name=f"tuner_{vid}")
                if not audio_path:
                    continue
                try:
                    hyp = model.transcribe(audio_path, cfg)
                    wers.append(wer(ref, hyp))
                    used += 1
                finally:
                    try:
                        if os.path.exists(audio_path):
                            os.remove(audio_path)
                    except Exception:
                        pass
            if used > 0:
                avg_wer = float(np.mean(wers))
                results.append((cfg, avg_wer, used))
            if console:
                progress.update(task, advance=1, description=f"[cyan]Adaylar deneniyor... ({len(results)} sonuç)")
    finally:
        if console:
            progress.stop()

    if not results:
        raise RuntimeError("Hiçbir videodan referans/hipotez üretilemedi; tuning yapılamadı.")

    best_cfg, best_avg_wer, used = sorted(results, key=lambda x: x[1])[0]
    return best_cfg, results


def save_best_config(gen_kwargs: dict, outfile: str = None) -> None:
    if outfile is None:
        outfile = os.path.join(OUTPUT_DIR, "whisper_best_config.json")
    config = {
        "name": "Tuned - Turkish ASR (No Prompt)",
        "use_prompt": False,
        "prompt_type": None,
        "prompt_text": "",
        "generation_params": gen_kwargs,
    }
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    if console:
        console.print(Panel.fit(f"✓ En iyi config kaydedildi: [green]{outfile}[/green]", border_style="green"))
    else:
        print(f"✓ En iyi config kaydedildi: {outfile}")

    # Raporu da sakla
    report_path = os.path.join(OUTPUT_DIR, "tune_report.json")
    try:
        with open(report_path, "w", encoding="utf-8") as rf:
            json.dump({"best_config": config}, rf, indent=2, ensure_ascii=False)
    except Exception:
        pass


if __name__ == "__main__":
    video_urls = [
        "https://youtu.be/IzaAlIj0uZc?si=-OqBz9Jk6ZMMudr4",
        "https://youtu.be/lV9RyTxb3v4?si=NTXqrgReTERHxuR9",
    ]

    if console:
        console.print(Panel.fit(
            f"[bold cyan]YouTube Tuning Başlıyor (No Prompt)[/bold cyan]\n"
            f"[dim]{len(video_urls)} video • Çıkış klasörü: {OUTPUT_DIR}[/dim]\n"
            f"[dim]Relative: {OUTPUTS_RELATIVE} • Env: WHISPER_OUTPUTS_DIR[/dim]",
            border_style="cyan"
        ))

    best_cfg, all_results = tune_on_videos(video_urls)

    # Özet tablo (ilk 8)
    if console:
        table = Table(title="Aday Sonuçları (En iyi 8)", box=box.MINIMAL_DOUBLE_HEAD)
        table.add_column("num_beams", justify="right")
        table.add_column("len_penalty", justify="right")
        table.add_column("no_rep_ngram", justify="right")
        table.add_column("avg WER %", justify="right")
        table.add_column("n", justify="right")
        for cfg, avg_wer, used in sorted(all_results, key=lambda x: x[1])[:8]:
            nb = cfg.get("num_beams", 1)
            lp = cfg.get("length_penalty", 1.0)
            ng = cfg.get("no_repeat_ngram_size", 0)
            table.add_row(str(nb), f"{lp:.2f}", str(ng), f"{avg_wer:.2f}", str(used))
        console.print(table)

        nb = best_cfg.get("num_beams", 1)
        lp = best_cfg.get("length_penalty", 1.0)
        ng = best_cfg.get("no_repeat_ngram_size", 0)
        console.print(Panel.fit(f"Seçilen En İyi Config -> num_beams={nb}, length_penalty={lp}, no_repeat_ngram_size={ng}", border_style="yellow"))
    else:
        print("\nAdaylar (ilk 5 gösteriliyor):")
        for cfg, avg_wer, used in sorted(all_results, key=lambda x: x[1])[:5]:
            nb = cfg.get("num_beams", 1)
            lp = cfg.get("length_penalty", 1.0)
            ng = cfg.get("no_repeat_ngram_size", 0)
            print(f"  num_beams={nb}, length_penalty={lp}, ngram={ng} -> avg WER={avg_wer:.2f}% (n={used})")
        nb = best_cfg.get("num_beams", 1)
        lp = best_cfg.get("length_penalty", 1.0)
        ng = best_cfg.get("no_repeat_ngram_size", 0)
        print(f"\nSeçilen en iyi config: num_beams={nb}, length_penalty={lp}, ngram={ng}")

    save_best_config(best_cfg)
