"""
Whisper Turkish Fine-tuned Model'i Hugging Face Hub'a Push Etme Scripti
Bu script, fine-tuning tamamlandıktan sonra modeli Hugging Face Hub'a yükler.
"""

import os
import torch
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperFeatureExtractor
)
from huggingface_hub import HfApi, Repository, create_repo
import json
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
import warnings
warnings.filterwarnings("ignore")

def main():
    # Initialize Rich console
    console = Console()
    
    # Welcome panel
    console.print(Panel.fit(
        "[bold cyan]🚀 Whisper Turkish Model Push to Hugging Face[/bold cyan]\n"
        "[dim]Fine-tuned model'i Hugging Face Hub'a yükleme[/dim]",
        border_style="cyan"
    ))
    
    # Model ve output dizinleri
    model_path = "./whisper-small-turkish"
    model_name = "whisper-small-turkish"  # Hugging Face'deki model adı
    
    # Model dizininin var olup olmadığını kontrol et
    if not os.path.exists(model_path):
        console.print(f"[red]❌ Model dizini bulunamadı: {model_path}[/red]")
        console.print("[yellow]💡 Önce fine-tuning scriptini çalıştırın![/yellow]")
        return
    
    # Gerekli dosyaların varlığını kontrol et
    required_files = ["config.json", "pytorch_model.bin", "tokenizer.json", "preprocessor_config.json"]
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(os.path.join(model_path, file)):
            missing_files.append(file)
    
    if missing_files:
        console.print(f"[red]❌ Eksik dosyalar: {', '.join(missing_files)}[/red]")
        console.print("[yellow]💡 Model tam olarak fine-tune edilmemiş olabilir.[/yellow]")
        return
    
    console.print("[green]✅ Model dosyaları bulundu![/green]")
    
    # Checkpoint bilgilerini göster
    console.print("\n[bold blue]📊 Checkpoint Bilgileri:[/bold blue]")
    try:
        import json
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            
            checkpoint_info = Table(title="Model Checkpoint")
            checkpoint_info.add_column("Özellik", style="cyan")
            checkpoint_info.add_column("Değer", style="green")
            checkpoint_info.add_row("Model Type", config.get("model_type", "N/A"))
            checkpoint_info.add_row("Architecture", config.get("architectures", ["N/A"])[0])
            checkpoint_info.add_row("Vocab Size", str(config.get("vocab_size", "N/A")))
            checkpoint_info.add_row("Max Length", str(config.get("max_length", "N/A")))
            checkpoint_info.add_row("Language", config.get("forced_decoder_ids", {}).get("language", "Turkish"))
            console.print(checkpoint_info)
    except Exception as e:
        console.print(f"[yellow]⚠️ Config bilgileri alınamadı: {e}[/yellow]")
    
    # Hugging Face Hub'a giriş
    console.print("\n[bold blue]🔐 Hugging Face Hub'a giriş yapılıyor...[/bold blue]")
    try:
        from huggingface_hub import login
        login()
        console.print("[green]✅ Giriş başarılı![/green]")
    except Exception as e:
        console.print(f"[red]❌ Giriş hatası: {e}[/red]")
        console.print("[yellow]💡 huggingface_login.py scriptini çalıştırın![/yellow]")
        return
    
    # Model bilgilerini oluştur
    console.print("\n[bold blue]📝 Model bilgileri hazırlanıyor...[/bold blue]")
    
    # Model card oluştur
    model_card_content = f"""---
language: tr
license: mit
tags:
- automatic-speech-recognition
- whisper
- turkish
- speech-to-text
- audio
datasets:
- ysdede/khanacademy-turkish
metrics:
- wer
- cer
model-index:
- name: {model_name}
  results:
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: Khan Academy Turkish
      type: ysdede/khanacademy-turkish
    metrics:
    - name: WER
      type: wer
      value: "TBD"
    - name: CER
      type: cer
      value: "TBD"
---

# Whisper Small Turkish Fine-tuned Model

Bu model, OpenAI'nin Whisper Small modelinin Türkçe konuşma tanıma için fine-tune edilmiş versiyonudur.

## Model Detayları

- **Base Model**: openai/whisper-small
- **Language**: Turkish (tr)
- **Task**: Automatic Speech Recognition
- **Dataset**: Khan Academy Turkish Dataset
- **Fine-tuning**: 500 steps, 5000 samples, GPU optimized with SpecAugment
- **Training Steps**: 500 (checkpoint-400 selected for best WER)
- **Dataset Size**: 5000 train + 100 test samples

## Kullanım

```python
from transformers import pipeline

# Model'i yükle
pipe = pipeline('automatic-speech-recognition', model='{model_name}')

# Ses dosyasını işle
result = pipe('path/to/your/audio.wav')
print(result['text'])
```

## Gereksinimler

```bash
pip install torch torchaudio transformers datasets evaluate librosa soundfile
```

## Model Performansı

Model, Türkçe konuşma tanıma görevlerinde optimize edilmiştir. Performans metrikleri:

- **WER (Word Error Rate)**: TBD
- **CER (Character Error Rate)**: TBD

## Lisans

MIT License

## Katkıda Bulunanlar

Bu model, Türkçe konuşma tanıma için fine-tune edilmiştir.
"""
    
    # README.md dosyasını kaydet
    readme_path = os.path.join(model_path, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(model_card_content)
    
    console.print("[green]✅ Model card oluşturuldu![/green]")
    
    # Repository oluştur
    console.print(f"\n[bold blue]📦 Repository oluşturuluyor: {model_name}[/bold blue]")
    try:
        # Repository oluştur
        create_repo(
            repo_id=model_name,
            exist_ok=True,
            private=False,  # Public repository
            repo_type="model"
        )
        console.print("[green]✅ Repository oluşturuldu![/green]")
    except Exception as e:
        console.print(f"[yellow]⚠️ Repository zaten mevcut veya oluşturulamadı: {e}[/yellow]")
    
    # Windows dosya kilitleme sorunlarını önlemek için geçici kopyalama
    console.print("\n[bold blue]📁 Windows dosya kilitleme sorunları için model kopyalanıyor...[/bold blue]")
    import shutil
    import tempfile
    import time
    
    temp_model_path = None
    try:
        # Geçici dizin oluştur
        temp_model_path = tempfile.mkdtemp(prefix="whisper_upload_")
        console.print(f"[blue]📂 Geçici dizin: {temp_model_path}[/blue]")
        
        # Gerekli dosyaları kopyala (checkpoint dosyalarını hariç tut)
        essential_files = [
            "config.json", "generation_config.json", "model.safetensors", 
            "tokenizer.json", "tokenizer_config.json", "vocab.json", 
            "merges.txt", "normalizer.json", "added_tokens.json",
            "special_tokens_map.json", "preprocessor_config.json", "README.md"
        ]
        
        for file in essential_files:
            src = os.path.join(model_path, file)
            dst = os.path.join(temp_model_path, file)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                console.print(f"[green]✅ Copied: {file}[/green]")
            else:
                console.print(f"[yellow]⚠️ Missing: {file}[/yellow]")
        
        # PyTorch model dosyasını kontrol et ve gerekirse SafeTensors'dan oluştur
        pytorch_model_src = os.path.join(model_path, "pytorch_model.bin")
        safetensors_src = os.path.join(model_path, "model.safetensors")
        pytorch_model_dst = os.path.join(temp_model_path, "pytorch_model.bin")
        
        if os.path.exists(pytorch_model_src):
            console.print("[blue]📦 PyTorch model dosyası mevcut, kopyalanıyor...[/blue]")
            shutil.copy2(pytorch_model_src, pytorch_model_dst)
            console.print("[green]✅ PyTorch model kopyalandı[/green]")
        elif os.path.exists(safetensors_src):
            console.print("[blue]🔄 SafeTensors'dan PyTorch formatına çeviriliyor...[/blue]")
            try:
                from safetensors.torch import load_file
                import torch
                
                # SafeTensors yükle ve PyTorch formatında kaydet
                state_dict = load_file(safetensors_src)
                torch.save(state_dict, pytorch_model_dst)
                console.print("[green]✅ SafeTensors → PyTorch dönüşümü tamamlandı[/green]")
            except Exception as conv_e:
                console.print(f"[yellow]⚠️ Dönüşüm hatası: {conv_e}[/yellow]")
                console.print("[blue]SafeTensors dosyasını kullanmaya devam ediliyor...[/blue]")
                shutil.copy2(safetensors_src, os.path.join(temp_model_path, "model.safetensors"))
        
        console.print("[green]✅ Model dosyaları geçici dizine kopyalandı![/green]")
        
        # Kısa bekleme - dosya işlemlerinin tamamlanması için
        time.sleep(2)
        
        # Model'i geçici dizinden yükle
        console.print("\n[bold blue]🤖 Model geçici dizinden yükleniyor...[/bold blue]")
        model = WhisperForConditionalGeneration.from_pretrained(temp_model_path)
        processor = WhisperProcessor.from_pretrained(temp_model_path)
        
        console.print("[green]✅ Model yüklendi![/green]")
        
        # Model'i push et
        console.print("\n[bold blue]🚀 Model Hugging Face Hub'a yükleniyor...[/bold blue]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Model yükleniyor...", total=None)
            
            try:
                # Model'i push et
                model.push_to_hub(
                    model_name,
                    commit_message="Add fine-tuned Turkish Whisper model (500 steps, optimized for Turkish)",
                    private=False,
                    safe_serialization=True,  # SafeTensors kullan
                    max_shard_size="1GB"  # Büyük dosyalar için parçalama
                )
                
                # Processor'ı push et
                processor.push_to_hub(
                    model_name,
                    commit_message="Add processor for Turkish Whisper model",
                    private=False
                )
                
                progress.update(task, description="[green]✅ Model başarıyla yüklendi!")
                
            except Exception as upload_e:
                console.print(f"[red]❌ Upload hatası: {upload_e}[/red]")
                
                # Alternatif yöntem: Dosyaları tek tek yükle
                console.print("[yellow]💡 Alternatif yöntem deneniyor...[/yellow]")
                from huggingface_hub import HfApi
                
                api = HfApi()
                
                # Dosyaları tek tek yükle
                for file in essential_files:
                    file_path = os.path.join(temp_model_path, file)
                    if os.path.exists(file_path):
                        try:
                            api.upload_file(
                                path_or_fileobj=file_path,
                                path_in_repo=file,
                                repo_id=model_name,
                                repo_type="model"
                            )
                            console.print(f"[green]✅ Uploaded: {file}[/green]")
                            time.sleep(1)  # Rate limiting için bekleme
                        except Exception as file_e:
                            console.print(f"[red]❌ {file} upload failed: {file_e}[/red]")
                
                raise upload_e
        
        console.print("[green]✅ Model başarıyla Hugging Face Hub'a yüklendi![/green]")
        
    except Exception as e:
        console.print(f"[red]❌ Model yükleme hatası: {e}[/red]")
        console.print("[yellow]💡 Windows dosya kilitleme sorunu olabilir. Lütfen tüm Python süreçlerini kapatıp tekrar deneyin.[/yellow]")
        return
        
    finally:
        # Geçici dizini temizle
        if temp_model_path and os.path.exists(temp_model_path):
            try:
                shutil.rmtree(temp_model_path)
                console.print(f"[blue]🧹 Geçici dizin temizlendi: {temp_model_path}[/blue]")
            except Exception as cleanup_e:
                console.print(f"[yellow]⚠️ Geçici dizin temizleme hatası: {cleanup_e}[/yellow]")
    
    # Başarı mesajı
    success_table = Table(title="Model Başarıyla Yüklendi!")
    success_table.add_column("Özellik", style="cyan")
    success_table.add_column("Değer", style="green")
    success_table.add_row("Model Adı", model_name)
    success_table.add_row("Repository", f"https://huggingface.co/{model_name}")
    success_table.add_row("Dil", "Turkish")
    success_table.add_row("Görev", "Speech-to-Text")
    success_table.add_row("Base Model", "openai/whisper-small")
    
    console.print(success_table)
    
    # Kullanım örneği
    console.print("\n[bold blue]📝 Kullanım Örneği:[/bold blue]")
    console.print(Panel(
        f"""from transformers import pipeline

# Model'i yükle
pipe = pipeline('automatic-speech-recognition', model='{model_name}')

# Ses dosyasını işle
result = pipe('path/to/your/audio.wav')
print(result['text'])

# Veya doğrudan model ile
from transformers import WhisperProcessor, WhisperForConditionalGeneration

processor = WhisperProcessor.from_pretrained('{model_name}')
model = WhisperForConditionalGeneration.from_pretrained('{model_name}')

# Ses işleme
inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    generated_ids = model.generate(inputs["input_features"])
transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]""",
        title="Python Kullanım Örneği",
        border_style="green"
    ))
    
    console.print(f"\n[bold green]🎉 Model başarıyla yüklendi![/bold green]")
    console.print(f"[blue]🔗 Model linki: https://huggingface.co/{model_name}[/blue]")
    console.print("[dim]Model artık Hugging Face Hub'da kullanıma hazır![/dim]")

if __name__ == "__main__":
    main()
