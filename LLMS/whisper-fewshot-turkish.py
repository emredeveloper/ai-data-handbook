"""
Whisper Small ile Türkçe Few-Shot Prompting
Khan Academy Türkçe dataset - Pandas ile doğrudan yükleme
"""

import os
import sys
import io
import time
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import numpy as np

# Rich kütüphanesi - güzel çıktılar için
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
    from rich.layout import Layout
    from rich.live import Live
    from rich import box
    from rich.text import Text
    console = Console()
except ImportError:
    print("Rich kütüphanesi kurulu değil. Kuruluyor...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich"])
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
    from rich.layout import Layout
    from rich.live import Live
    from rich import box
    from rich.text import Text
    console = Console()

# Model ve processor yükleme
MODEL_NAME = "openai/whisper-small"

console.print("\n")
console.print(Panel.fit(
    f"[bold cyan]🤖 Whisper Model Yükleniyor[/bold cyan]\n"
    f"[yellow]Model:[/yellow] {MODEL_NAME}\n"
    f"[yellow]Görev:[/yellow] Türkçe ASR + Few-Shot Prompting",
    border_style="cyan"
))

with console.status("[bold green]Model yükleniyor...", spinner="dots"):
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

device_icon = "🚀" if device == "cuda" else "💻"
console.print(f"{device_icon} [bold green]Model hazır![/bold green] Cihaz: [cyan]{device.upper()}[/cyan]\n")

# Few-Shot Prompting Stratejileri:
# 1. Uzun ve detaylı örnekler (220+ karakter öneriliyor)
# 2. Hedef domain'e özgü kelime dağarcığı (eğitim, fizik, matematik)
# 3. Noktalama ve büyük harf kullanımı örnekleri
# 4. Sayı ve ölçü birimi formatları

FEW_SHOT_PROMPTS = {
    "short": """
Yerin yakınında yer çekimi ivmesi 9,8 metre bölü saniye kare ve aşağı yönlüdür. Newton'ın birinci kanununa göre, bir cisim üzerine net kuvvet etki etmiyorsa o cismin hızı sabittir. Kompozisyonda yer alan her bir öğe, kabın formunun yuvarlak olması ile uyum sağlayacak şekilde yerleştirilmiştir.
""",
    
    "long": """
Yerin yakınında yer çekimi ivmesi 9,8 metre bölü saniye kare ve aşağı yönlüdür. Bu, Newton'ın evrensel çekim yasasından türetilmiştir. Newton'ın birinci kanununa göre, bir cisim üzerine net kuvvet etki etmiyorsa o cismin hızı sabittir. Sabit hız derken, hem büyüklük hem de yön olarak sabit olduğunu kastediyoruz. İkinci kanunda ise net kuvvet kütle ile ivmenin çarpımına eşittir. Dolayısıyla, bir cisme etki eden kuvvet arttıkça ivme de artar. Kompozisyonda yer alan her bir öğe, kabın formunun yuvarlak olması ile uyum sağlayacak şekilde yerleştirilmiştir. Leonardo'nun kendisini sanatçı olarak geliştirmeye devam ettiği 1470'lerin Floransa'sında yaşayan pek çok önemli ve başarılı sanatçı bulunuyordu.
""",
    
    "domain_specific": """
Fizik ve matematik derslerinde öğrendiğimiz kavramlar günlük hayatımızda karşımıza çıkar. Örneğin, yer çekimi ivmesi 9,8 metre bölü saniye kare olarak hesaplanır. Bu değer, deniz seviyesinde ve ekvatorda ölçülmüştür. Newton'ın hareket yasalarına göre, bir cisme etki eden net kuvvet sıfırsa cisim ya durur ya da sabit hızla hareket eder. Kuvvet birimi newton, iş birimi joule, güç birimi watt'tır. Kimyada ise elementler periyodik tabloda düzenlenmiştir. Sanat tarihinde Rönesans dönemi çok önemlidir. Leonardo da Vinci, Michelangelo gibi sanatçılar bu dönemde yaşamıştır.
"""
}

# Tüm prompt tipleri test edilecek
PROMPT_TYPES_TO_TEST = ["short", "long", "domain_specific"]

# Dataset yükleme
console.print(Panel.fit(
    "[bold magenta]📚 Dataset Yükleme[/bold magenta]\n"
    "[yellow]Kaynak:[/yellow] ysdede/khanacademy-turkish\n"
    "[yellow]Format:[/yellow] Parquet (Direct Load)",
    border_style="magenta"
))

# Gerekli kütüphaneleri kontrol et ve yükle
try:
    import soundfile as sf
    import pandas as pd
except ImportError as e:
    print(f"Eksik kütüphane: {e}")
    print("Gerekli kütüphaneler kuruluyor...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "soundfile", "pandas", "pyarrow"])
    import soundfile as sf
    import pandas as pd

try:
    # Parquet'i DOĞRUDAN pandas ile oku
    parquet_url = "https://huggingface.co/datasets/ysdede/khanacademy-turkish/resolve/main/data/test-00000-of-00001.parquet"
    
    with console.status("[bold green]Parquet dosyası indiriliyor...", spinner="dots"):
        df = pd.read_parquet(parquet_url)
    
    # Rastgele N örnek seç
    NUM_SAMPLES = 10
    df = df.sample(n=NUM_SAMPLES, random_state=42)
    df = df.reset_index(drop=True)
    
    console.print(f"[green]✓[/green] Rastgele {NUM_SAMPLES} örnek seçildi")
    
    # Her satırı işle - progress bar ile
    samples_list = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Ses dosyaları decode ediliyor...", total=NUM_SAMPLES)
        
        for idx, row in df.iterrows():
            try:
                transcription = row['transcription']
                audio_data = row['audio']
                audio_bytes = audio_data['bytes']
                audio_array, sr = sf.read(io.BytesIO(audio_bytes))
                
                sample = {
                    'transcription': transcription,
                    'audio': {
                        'array': audio_array,
                        'sampling_rate': sr
                    },
                    'duration': len(audio_array) / sr
                }
                
                samples_list.append(sample)
                progress.update(task, advance=1, description=f"[cyan]Örnek {idx+1}/{NUM_SAMPLES} yüklendi ({len(audio_array)/sr:.1f}s)")
                
            except Exception as e:
                console.print(f"[red]✗ Örnek {idx+1} hatası: {e}[/red]")
                progress.update(task, advance=1)
                continue
    
    if len(samples_list) == 0:
        raise Exception("Hiçbir örnek yüklenemedi!")
    
    dataset = samples_list
    
    # Dataset istatistikleri
    total_duration = sum(s['duration'] for s in dataset)
    avg_duration = total_duration / len(dataset)
    
    stats_table = Table(show_header=False, box=box.SIMPLE)
    stats_table.add_row("[cyan]Toplam Örnek:[/cyan]", f"[green]{len(dataset)}[/green]")
    stats_table.add_row("[cyan]Toplam Süre:[/cyan]", f"[green]{total_duration:.1f}s ({total_duration/60:.1f} dk)[/green]")
    stats_table.add_row("[cyan]Ortalama Süre:[/cyan]", f"[green]{avg_duration:.1f}s[/green]")
    
    console.print(Panel(stats_table, title="[bold green]✅ Dataset Hazır", border_style="green"))
    
except Exception as e:
    console.print(f"\n[bold red]❌ Hata: {e}[/bold red]")
    import traceback
    traceback.print_exc()
    sys.exit(1)

def transcribe_audio(audio_array, sampling_rate, use_prompt=True, technique="standard", prompt_type="long"):
    """
    Gelişmiş tekniklerle ses transkribe et
    
    Args:
        audio_array: Ses verisi
        sampling_rate: Örnekleme oranı
        use_prompt: Few-shot prompt kullan
        technique: Kullanılacak teknik
            - "standard": Standart decoding
            - "beam_search": Beam search optimized (num_beams=8)
            - "beam_search_aggressive": Agresif beam search (num_beams=12)
        prompt_type: Prompt tipi (short, long, domain_specific)
    
    Returns:
        Transkripsiyon metni
    """
    # 16kHz'e resample
    if sampling_rate != 16000:
        audio_array = librosa.resample(
            audio_array, 
            orig_sr=sampling_rate, 
            target_sr=16000
        )
    
    # Input features
    input_features = processor(
        audio_array,
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features.to(device)
    
    # Generation parametreleri
    gen_kwargs = {
        "max_length": 448,  # Whisper maksimum
        "language": "tr",
        "task": "transcribe"
    }
    
    # Teknik bazlı parametreler - İyileştirilmiş
    if technique == "beam_search":
        gen_kwargs.update({
            "num_beams": 8,              # 5'ten 8'e yükseltildi
            "early_stopping": True,
            "length_penalty": 1.2,       # 1.0'dan 1.2'ye (daha uzun cümleler için)
            "no_repeat_ngram_size": 3,   # Tekrar önleme
            "num_return_sequences": 1
        })
    elif technique == "beam_search_aggressive":
        gen_kwargs.update({
            "num_beams": 12,             # Agresif mode
            "early_stopping": True,
            "length_penalty": 1.5,
            "no_repeat_ngram_size": 3,
            "num_return_sequences": 1,
            "num_beam_groups": 3,        # Diversity için gerekli
            "diversity_penalty": 0.5     # Çeşitlilik
        })
    
    # Prompt kullan
    if use_prompt:
        # Prompt'u tokenize et
        prompt_text = FEW_SHOT_PROMPTS[prompt_type].strip()
        prompt_ids = processor.get_prompt_ids(prompt_text, return_tensors="pt")
        
        # Prompt'u decoder'a başlangıç olarak ver
        generated_ids = model.generate(
            input_features,
            prompt_ids=prompt_ids.to(device) if prompt_ids is not None else None,
            **gen_kwargs
        )
    else:
        generated_ids = model.generate(
            input_features,
            **gen_kwargs
        )
    
    # Decode
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return transcription

def calculate_wer(reference, hypothesis):
    """
    Word Error Rate (WER) hesaplama
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    # Basit Levenshtein distance hesaplama
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
    
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion = d[i][j-1] + 1
                deletion = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)
    
    wer = d[len(ref_words)][len(hyp_words)] / len(ref_words)
    return wer * 100

# Test konfigürasyonu
TEST_CONFIG = {
    "test_prompt_types": True,      # Üç prompt tipini karşılaştır
    "test_beam_variations": True,   # İki beam search varyasyonunu test et
    "show_details": False,          # Detaylı çıktı (çok fazla olur)
}

console.print("\n")
test_modes_text = "Baseline + 3 Prompt Tipi × 3 Teknik = 10 Kombinasyon"
console.print(Panel.fit(
    f"[bold yellow]🧪 Kapsamlı Test Başlıyor[/bold yellow]\n"
    f"[cyan]Örnek Sayısı:[/cyan] {len(dataset)}\n"
    f"[cyan]Prompt Tipleri:[/cyan] {', '.join(PROMPT_TYPES_TO_TEST)}\n"
    f"[cyan]Teknikler:[/cyan] Standard, Beam (8), Beam Aggressive (12)\n"
    f"[cyan]Test Modları:[/cyan] {test_modes_text}",
    border_style="yellow"
))

# Sonuç takibi - Genişletilmiş
results = {
    "baseline": {"total_wer": 0, "name": "🔷 Baseline", "times": [], "improvements": [], "prompt": None, "technique": "standard"},
}

# Her prompt tipi için sonuç kategorileri
for prompt_type in PROMPT_TYPES_TO_TEST:
    # Standard
    key = f"{prompt_type}_standard"
    results[key] = {
        "total_wer": 0,
        "name": f"📝 {prompt_type.title()}",
        "times": [],
        "improvements": [],
        "prompt": prompt_type,
        "technique": "standard"
    }
    
    # Beam Search
    key = f"{prompt_type}_beam"
    results[key] = {
        "total_wer": 0,
        "name": f"⚡ {prompt_type.title()} + Beam",
        "times": [],
        "improvements": [],
        "prompt": prompt_type,
        "technique": "beam_search"
    }
    
    # Beam Aggressive
    if TEST_CONFIG["test_beam_variations"]:
        key = f"{prompt_type}_beam_agg"
        results[key] = {
            "total_wer": 0,
            "name": f"🚀 {prompt_type.title()} + Beam Agg",
            "times": [],
            "improvements": [],
            "prompt": prompt_type,
            "technique": "beam_search_aggressive"
        }

sample_results = []  # Her örneğin detaylı sonuçları

console.print(f"\n[dim]Toplam {len(results)} farklı kombinasyon test edilecek...[/dim]\n")

# Progress bar ile test
total_tests = len(dataset) * len(results)

with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    TextColumn("•"),
    TimeRemainingColumn(),
    console=console
) as progress:
    
    task = progress.add_task("[cyan]Tüm kombinasyonlar test ediliyor...", total=len(dataset))
    
    for idx, sample in enumerate(dataset):
        audio_array = sample['audio']['array']
        sampling_rate = sample['audio']['sampling_rate']
        reference_text = sample['transcription']
        duration = sample['duration']
        
        sample_result = {
            'idx': idx + 1,
            'reference': reference_text,
            'duration': duration,
            'results': {}
        }
        
        # 1. Baseline (prompt olmadan)
        start_time = time.time()
        trans_baseline = transcribe_audio(audio_array, sampling_rate, use_prompt=False, technique="standard")
        baseline_time = time.time() - start_time
        wer_baseline = calculate_wer(reference_text, trans_baseline)
        
        results["baseline"]["total_wer"] += wer_baseline
        results["baseline"]["times"].append(baseline_time)
        sample_result['results']['baseline'] = {'text': trans_baseline, 'wer': wer_baseline, 'time': baseline_time}
        
        # 2. Tüm prompt tipi + teknik kombinasyonları
        for result_key, result_data in results.items():
            if result_key == "baseline":
                continue
            
            prompt_type = result_data["prompt"]
            technique = result_data["technique"]
            
            start_time = time.time()
            trans = transcribe_audio(
                audio_array, 
                sampling_rate, 
                use_prompt=True, 
                technique=technique,
                prompt_type=prompt_type
            )
            trans_time = time.time() - start_time
            wer = calculate_wer(reference_text, trans)
            improvement = wer_baseline - wer
            
            result_data["total_wer"] += wer
            result_data["times"].append(trans_time)
            result_data["improvements"].append(improvement)
            
            sample_result['results'][result_key] = {
                'text': trans,
                'wer': wer,
                'time': trans_time,
                'improvement': improvement
            }
        
        sample_results.append(sample_result)
        progress.update(task, advance=1, description=f"[cyan]Örnek {idx+1}/{len(dataset)} → {len(results)} kombinasyon test edildi")
        
        # Detaylı çıktı (sadece ilk örnek için)
        if TEST_CONFIG["show_details"] and idx == 0:
            console.print(f"\n[bold cyan]═══ Örnek {idx+1} - İlk Sonuçlar ═══[/bold cyan]")
            console.print(f"[dim]Referans:[/dim] {reference_text[:80]}...")
            
            mini_table = Table(show_header=True, box=box.SIMPLE)
            mini_table.add_column("Kombinasyon", style="cyan")
            mini_table.add_column("WER", justify="right")
            mini_table.add_column("İyileşme", justify="right")
            
            mini_table.add_row("Baseline", f"{wer_baseline:.1f}%", "-")
            
            for rkey in list(results.keys())[:3]:  # İlk 3 sonuç
                if rkey == "baseline":
                    continue
                r = sample_result['results'][rkey]
                color = "green" if r['improvement'] > 0 else "red"
                mini_table.add_row(
                    results[rkey]['name'],
                    f"{r['wer']:.1f}%",
                    f"[{color}]{r['improvement']:+.1f}%[/{color}]"
                )
            
            console.print(mini_table)
            console.print("[dim]... diğer kombinasyonlar da test ediliyor[/dim]\n")

console.print("\n[bold green]✅ Tüm testler tamamlandı![/bold green]")
console.print(f"[dim]{len(dataset)} örnek × {len(results)} kombinasyon = {len(dataset) * len(results)} test yapıldı[/dim]\n")

# Özet sonuçları hesapla
num_samples = len(dataset)
baseline_avg = results["baseline"]["total_wer"] / num_samples

summary_data = {}
for key, data in results.items():
    avg_wer = data["total_wer"] / num_samples
    avg_time = np.mean(data["times"]) if data["times"] else 0
    improvement = baseline_avg - avg_wer
    
    summary_data[key] = {
        "name": data["name"],
        "avg_wer": avg_wer,
        "avg_time": avg_time,
        "improvement": improvement,
        "improvements": data["improvements"]
    }

# Ana Özet Tablosu
console.print(Panel.fit(
    "[bold yellow]📊 PERFORMANS ANALİZİ[/bold yellow]",
    border_style="yellow"
))

summary_table = Table(show_header=True, box=box.DOUBLE_EDGE, title="[bold]Genel Sonuçlar[/bold]", title_style="bold magenta")
summary_table.add_column("Metod", style="cyan", no_wrap=True)
summary_table.add_column("Ort. WER", justify="right", style="yellow")
summary_table.add_column("İyileşme", justify="right")
summary_table.add_column("Ort. Süre", justify="right", style="dim")
summary_table.add_column("Durum", justify="center")

for key, data in summary_data.items():
    improvement_text = "-" if key == "baseline" else f"{data['improvement']:+.2f}%"
    improvement_style = ""
    
    if key != "baseline":
        if data['improvement'] > 2:
            improvement_style = "bold green"
            status = "🎯 İyi"
        elif data['improvement'] > 0:
            improvement_style = "green"
            status = "✅ Hafif+"
        elif data['improvement'] == 0:
            improvement_style = "yellow"
            status = "⚠️  Eşit"
        else:
            improvement_style = "red"
            status = "❌ Kötü"
    else:
        status = "📍 Baz"
    
    summary_table.add_row(
        data['name'],
        f"{data['avg_wer']:.2f}%",
        f"[{improvement_style}]{improvement_text}[/{improvement_style}]" if improvement_style else improvement_text,
        f"{data['avg_time']:.2f}s",
        status
    )

console.print(summary_table)

# Prompt Tipi Karşılaştırması
console.print("\n")
prompt_comparison = Table(
    show_header=True, 
    box=box.HEAVY_EDGE, 
    title="[bold magenta]🎯 Prompt Tipi Karşılaştırması[/bold magenta]",
    title_style="bold magenta"
)
prompt_comparison.add_column("Prompt Tipi", style="cyan", width=20)
prompt_comparison.add_column("Standard", justify="right")
prompt_comparison.add_column("Beam (8)", justify="right")
if TEST_CONFIG["test_beam_variations"]:
    prompt_comparison.add_column("Beam Agg (12)", justify="right")
prompt_comparison.add_column("En İyi", justify="right", style="bold")

for prompt_type in PROMPT_TYPES_TO_TEST:
    wers = []
    
    # Standard
    std_key = f"{prompt_type}_standard"
    std_wer = summary_data[std_key]['avg_wer']
    std_imp = summary_data[std_key]['improvement']
    std_color = "green" if std_imp > 0 else "red"
    wers.append(std_wer)
    
    # Beam
    beam_key = f"{prompt_type}_beam"
    beam_wer = summary_data[beam_key]['avg_wer']
    beam_imp = summary_data[beam_key]['improvement']
    beam_color = "green" if beam_imp > 0 else "red"
    wers.append(beam_wer)
    
    row_data = [
        prompt_type.upper(),
        f"[{std_color}]{std_wer:.2f}%[/{std_color}]\n({std_imp:+.2f}%)",
        f"[{beam_color}]{beam_wer:.2f}%[/{beam_color}]\n({beam_imp:+.2f}%)"
    ]
    
    # Beam Aggressive
    if TEST_CONFIG["test_beam_variations"]:
        beam_agg_key = f"{prompt_type}_beam_agg"
        beam_agg_wer = summary_data[beam_agg_key]['avg_wer']
        beam_agg_imp = summary_data[beam_agg_key]['improvement']
        beam_agg_color = "green" if beam_agg_imp > 0 else "red"
        wers.append(beam_agg_wer)
        row_data.append(f"[{beam_agg_color}]{beam_agg_wer:.2f}%[/{beam_agg_color}]\n({beam_agg_imp:+.2f}%)")
    
    # En iyi
    best_wer = min(wers)
    row_data.append(f"[bold green]{best_wer:.2f}%[/bold green]")
    
    prompt_comparison.add_row(*row_data)

console.print(prompt_comparison)

# Teknik Karşılaştırması
console.print("\n")
technique_comparison = Table(
    show_header=True,
    box=box.HEAVY_EDGE,
    title="[bold yellow]⚡ Teknik Karşılaştırması[/bold yellow]",
    title_style="bold yellow"
)
technique_comparison.add_column("Teknik", style="cyan", width=20)
technique_comparison.add_column("Ort. WER", justify="right")
technique_comparison.add_column("Ort. İyileşme", justify="right")
technique_comparison.add_column("Başarı Oranı", justify="right")
technique_comparison.add_column("Ort. Süre", justify="right")

# Tekniklere göre grupla
technique_stats = {}
for key, data in summary_data.items():
    if key == "baseline":
        continue
    
    technique = results[key]['technique']
    if technique not in technique_stats:
        technique_stats[technique] = {
            'wers': [],
            'improvements': [],
            'times': [],
            'success_counts': []
        }
    
    technique_stats[technique]['wers'].append(data['avg_wer'])
    technique_stats[technique]['improvements'].extend(data['improvements'])
    technique_stats[technique]['times'].append(data['avg_time'])

# Tabloyu doldur
technique_names = {
    'standard': '📝 Standard',
    'beam_search': '⚡ Beam Search (8)',
    'beam_search_aggressive': '🚀 Beam Aggressive (12)'
}

for tech, stats in technique_stats.items():
    avg_wer = np.mean(stats['wers'])
    avg_imp = np.mean(stats['improvements'])
    success_rate = sum(1 for imp in stats['improvements'] if imp > 0) / len(stats['improvements']) * 100
    avg_time = np.mean(stats['times'])
    
    imp_color = "green" if avg_imp > 0 else "red"
    success_color = "green" if success_rate > 50 else "yellow" if success_rate > 30 else "red"
    
    technique_comparison.add_row(
        technique_names.get(tech, tech),
        f"{avg_wer:.2f}%",
        f"[{imp_color}]{avg_imp:+.2f}%[/{imp_color}]",
        f"[{success_color}]{success_rate:.1f}%[/{success_color}]",
        f"{avg_time:.2f}s"
    )

console.print(technique_comparison)

# En iyi metodu belirle
best_method = min(summary_data.items(), key=lambda x: x[1]['avg_wer'])

# Akıllı Analiz ve Öneriler
console.print("\n")
recommendations = []

# En iyi prompt tipini bul
prompt_type_scores = {}
for prompt_type in PROMPT_TYPES_TO_TEST:
    scores = []
    for key, data in summary_data.items():
        if results[key]['prompt'] == prompt_type:
            scores.append(data['avg_wer'])
    prompt_type_scores[prompt_type] = np.mean(scores)

best_prompt_type = min(prompt_type_scores.items(), key=lambda x: x[1])
worst_prompt_type = max(prompt_type_scores.items(), key=lambda x: x[1])

# En iyi tekniği bul
best_technique = min(technique_stats.items(), key=lambda x: np.mean(x[1]['wers']))

recommendations.append(f"🏆 [bold]GENEL KAZANAN[/bold]")
recommendations.append(f"   {best_method[1]['name']}: {best_method[1]['avg_wer']:.2f}% WER")
recommendations.append(f"   Baseline'dan {best_method[1]['improvement']:+.2f}% daha iyi!\n")

recommendations.append(f"🎯 [bold]EN İYİ PROMPT TİPİ[/bold]")
recommendations.append(f"   {best_prompt_type[0].upper()}: {best_prompt_type[1]:.2f}% ortalama WER")
recommendations.append(f"   {worst_prompt_type[0].upper()}'dan {worst_prompt_type[1] - best_prompt_type[1]:.2f}% daha iyi\n")

recommendations.append(f"⚡ [bold]EN İYİ TEKNİK[/bold]")
best_tech_name = technique_names[best_technique[0]]
best_tech_wer = np.mean(best_technique[1]['wers'])
best_tech_success = sum(1 for imp in best_technique[1]['improvements'] if imp > 0) / len(best_technique[1]['improvements']) * 100
recommendations.append(f"   {best_tech_name}: {best_tech_wer:.2f}% WER")
recommendations.append(f"   Başarı oranı: {best_tech_success:.1f}%\n")

# Prompt tipi önerileri
if best_prompt_type[0] == "long":
    recommendations.append("📝 [bold]PROMPT ANALİZİ[/bold]")
    recommendations.append("   ✅ Uzun prompt en iyi sonucu verdi")
    recommendations.append("   💡 Öneri: Prompt'u daha da uzatabilirsiniz (500+ karakter)")
elif best_prompt_type[0] == "domain_specific":
    recommendations.append("📝 [bold]PROMPT ANALİZİ[/bold]")
    recommendations.append("   ✅ Domain-specific prompt kazandı")
    recommendations.append("   💡 Öneri: Daha fazla domain terminolojisi ekleyin")
else:
    recommendations.append("📝 [bold]PROMPT ANALİZİ[/bold]")
    recommendations.append("   ⚠️  Kısa prompt yeterli oldu")
    recommendations.append("   💡 Öneri: Daha uzun prompt'lar test edin\n")

# Beam search analizi
if best_technique[0] == "beam_search_aggressive":
    recommendations.append("\n⚡ [bold]BEAM SEARCH ANALİZİ[/bold]")
    recommendations.append("   ✅ Agresif beam search (12) daha iyi")
    recommendations.append("   💡 Öneri: num_beams=15-20 deneyin")
elif best_technique[0] == "beam_search":
    recommendations.append("\n⚡ [bold]BEAM SEARCH ANALİZİ[/bold]")
    recommendations.append("   ✅ Orta seviye beam (8) optimal")
    recommendations.append("   ⚖️  Hız/kalite dengesi iyi")
else:
    recommendations.append("\n⚡ [bold]BEAM SEARCH ANALİZİ[/bold]")
    recommendations.append("   ⚠️  Standard decoding yeterli")
    recommendations.append("   💰 Beam search ekstra maliyet getirmiyor")

# Genel öneriler
avg_improvement = best_method[1]['improvement']
if avg_improvement > 3:
    recommendations.append("\n\n🎉 [bold green]BAŞARILI![/bold green]")
    recommendations.append("   Few-shot prompting güçlü etki gösterdi")
    recommendations.append("   Bu ayarlarla production'a geçebilirsiniz")
elif avg_improvement > 0:
    recommendations.append("\n\n⚠️  [bold yellow]ORTA PERFORMANS[/bold yellow]")
    recommendations.append("   Few-shot prompting hafif iyileştirme sağladı")
    recommendations.append("   💡 Öneri: Daha fazla örnek veya fine-tuning deneyin")
else:
    recommendations.append("\n\n❌ [bold red]DÜŞÜK PERFORMANS[/bold red]")
    recommendations.append("   Few-shot prompting etkili olmadı")
    recommendations.append("   💡 Öneri: Model fine-tuning şart!")

recommendations.append("\n\n🚀 [bold]İLERİ ADIMLAR[/bold]")
recommendations.append("   1. Whisper Medium/Large modelleri test edin")
recommendations.append("   2. Bu dataset ile fine-tuning yapın")
recommendations.append(f"   3. En iyi ayarları kullanın: {best_prompt_type[0]} + {best_technique[0]}")
recommendations.append("   4. Daha fazla örnek (50+) ile doğrulayın")

console.print(Panel(
    "\n".join(recommendations),
    title="[bold green]💡 Detaylı Analiz ve Öneriler[/bold green]",
    border_style="green",
    padding=(1, 2)
))

# Production config oluştur ve kaydet
console.print("\n")
save_config = console.input("[cyan]En iyi ayarları production config olarak kaydetmek ister misiniz? (e/h): [/cyan]").strip().lower()

if save_config == 'e':
    # En iyi kombinasyonun ayarlarını al
    best_prompt_type = results[best_method[0]]['prompt']
    best_technique = results[best_method[0]]['technique']
    
    # Prompt metni
    prompt_text = FEW_SHOT_PROMPTS[best_prompt_type] if best_prompt_type else ""
    
    # Generation params
    gen_params = {
        "max_length": 448,
        "language": "tr",
        "task": "transcribe"
    }
    
    if best_technique == "beam_search":
        gen_params.update({
            "num_beams": 8,
            "early_stopping": True,
            "length_penalty": 1.2,
            "no_repeat_ngram_size": 3,
            "num_return_sequences": 1
        })
    elif best_technique == "beam_search_aggressive":
        gen_params.update({
            "num_beams": 12,
            "early_stopping": True,
            "length_penalty": 1.5,
            "no_repeat_ngram_size": 3,
            "num_return_sequences": 1,
            "num_beam_groups": 3,
            "diversity_penalty": 0.5
        })
    
    # Config dict
    production_config = {
        "name": f"Best Config: {best_method[1]['name']}",
        "test_results": {
            "average_wer": round(best_method[1]['avg_wer'], 2),
            "improvement_over_baseline": round(best_method[1]['improvement'], 2),
            "tested_on": len(dataset),
            "test_date": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "use_prompt": best_prompt_type is not None,
        "prompt_type": best_prompt_type,
        "prompt_text": prompt_text,
        "generation_params": gen_params
    }
    
    # JSON olarak kaydet
    import json
    config_filename = "whisper_best_config.json"
    with open(config_filename, 'w', encoding='utf-8') as f:
        json.dump(production_config, f, indent=2, ensure_ascii=False)
    
    console.print(f"\n[bold green]✓ Config kaydedildi: {config_filename}[/bold green]")
    console.print("\n[cyan]Kullanım:[/cyan]")
    console.print(f"""
    [dim]from whisper_production import WhisperTurkishASR
    
    # Kaydedilmiş config ile yükle
    asr = WhisperTurkishASR(config_file="{config_filename}")
    
    # Transkribe et
    text = asr.transcribe("ses_dosyasi.wav")
    print(text)[/dim]
    """)

# Final mesaj
console.print("\n[dim]Test tamamlandı. Sonuçlar yukarıda özetlenmiştir.[/dim]\n")

