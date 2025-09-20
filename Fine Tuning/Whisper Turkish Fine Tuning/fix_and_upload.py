"""
Whisper Turkish Model - Problem Çözücü ve Upload Yardımcısı
Windows dosya kilitleme sorunlarını çözer ve modeli Hugging Face'e yükler
"""

import os
import sys
import subprocess
import time
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

def main():
    console = Console()
    
    # Başlık
    console.print(Panel.fit(
        "[bold cyan]🛠️ Whisper Turkish Model - Problem Çözücü[/bold cyan]\n"
        "[dim]Windows dosya kilitleme sorunlarını çözer ve modeli Hugging Face'e yükler[/dim]",
        border_style="cyan"
    ))
    
    # Model dizinini kontrol et
    model_dir = "./whisper-small-turkish"
    if not os.path.exists(model_dir):
        console.print(f"[red]❌ Model dizini bulunamadı: {model_dir}[/red]")
        console.print("[yellow]💡 Önce fine-tuning scriptini çalıştırın![/yellow]")
        return
    
    # Mevcut durumu kontrol et
    console.print("\n[bold blue]📊 Model Durumu Kontrolü[/bold blue]")
    
    status_table = Table()
    status_table.add_column("Dosya", style="cyan")
    status_table.add_column("Durum", style="green")
    status_table.add_column("Boyut", style="yellow")
    
    essential_files = [
        "config.json", "model.safetensors", "pytorch_model.bin",
        "tokenizer.json", "vocab.json", "merges.txt"
    ]
    
    missing_files = []
    for file in essential_files:
        file_path = os.path.join(model_dir, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / (1024*1024)
            status_table.add_row(file, "✅ Mevcut", f"{size:.1f} MB")
        else:
            status_table.add_row(file, "❌ Eksik", "0 MB")
            missing_files.append(file)
    
    console.print(status_table)
    
    # PyTorch model dosyası eksikse dönüştür
    pytorch_path = os.path.join(model_dir, "pytorch_model.bin")
    safetensors_path = os.path.join(model_dir, "model.safetensors")
    
    if not os.path.exists(pytorch_path) and os.path.exists(safetensors_path):
        console.print("\n[yellow]⚠️ PyTorch model dosyası eksik, SafeTensors'dan dönüştürülüyor...[/yellow]")
        
        try:
            result = subprocess.run([
                sys.executable, "convert_to_pytorch.py"
            ], capture_output=True, text=True, cwd=".")
            
            if result.returncode == 0:
                console.print("[green]✅ PyTorch dönüşümü başarılı![/green]")
            else:
                console.print(f"[red]❌ Dönüşüm hatası: {result.stderr}[/red]")
                return
                
        except Exception as e:
            console.print(f"[red]❌ Dönüşüm scripti çalıştırılamadı: {e}[/red]")
            return
    
    # Kritik dosyalar eksikse uyar
    if missing_files:
        console.print(f"\n[red]❌ Kritik dosyalar eksik: {', '.join(missing_files)}[/red]")
        console.print("[yellow]💡 Fine-tuning işlemini tekrar çalıştırmanız gerekebilir.[/yellow]")
        return
    
    # Kullanıcıya seçenek sun
    console.print("\n[bold blue]🚀 Ne yapmak istiyorsunuz?[/bold blue]")
    console.print("1. [green]Sadece PyTorch dönüşümü yap[/green]")
    console.print("2. [cyan]Modeli Hugging Face'e yükle[/cyan]")
    console.print("3. [yellow]Her ikisini de yap[/yellow]")
    console.print("4. [red]İptal et[/red]")
    
    choice = input("\nSeçiminizi yapın (1-4): ").strip()
    
    if choice == "1" or choice == "3":
        console.print("\n[blue]🔄 PyTorch dönüşümü başlatılıyor...[/blue]")
        try:
            result = subprocess.run([
                sys.executable, "convert_to_pytorch.py"
            ], cwd=".")
            
            if result.returncode == 0:
                console.print("[green]✅ PyTorch dönüşümü tamamlandı![/green]")
            else:
                console.print("[red]❌ PyTorch dönüşümü başarısız![/red]")
                if choice == "3":
                    return
        except Exception as e:
            console.print(f"[red]❌ Dönüşüm hatası: {e}[/red]")
            if choice == "3":
                return
    
    if choice == "2" or choice == "3":
        console.print("\n[blue]🚀 Hugging Face upload başlatılıyor...[/blue]")
        console.print("[yellow]💡 Bu işlem birkaç dakika sürebilir...[/yellow]")
        
        try:
            # Upload scriptini çalıştır
            result = subprocess.run([
                sys.executable, "huggingface_push_model.py"
            ], cwd=".")
            
            if result.returncode == 0:
                console.print("[green]✅ Model başarıyla Hugging Face'e yüklendi![/green]")
            else:
                console.print("[red]❌ Upload işlemi başarısız![/red]")
                console.print("[yellow]💡 Lütfen internet bağlantınızı ve Hugging Face token'ınızı kontrol edin.[/yellow]")
                
        except Exception as e:
            console.print(f"[red]❌ Upload hatası: {e}[/red]")
    
    elif choice == "4":
        console.print("[blue]İşlem iptal edildi.[/blue]")
        return
    
    else:
        console.print("[red]❌ Geçersiz seçim![/red]")
        return
    
    # Son durum
    console.print("\n[bold green]🎉 İşlemler tamamlandı![/bold green]")
    
    # Kullanım örnekleri
    console.print("\n[bold blue]📝 Model Kullanım Örnekleri:[/bold blue]")
    console.print(Panel(
        """[bold]1. Pipeline ile kullanım:[/bold]
from transformers import pipeline

pipe = pipeline('automatic-speech-recognition', model='whisper-small-turkish')
result = pipe('path/to/audio.wav')
print(result['text'])

[bold]2. Doğrudan model ile kullanım:[/bold]
from transformers import WhisperProcessor, WhisperForConditionalGeneration

processor = WhisperProcessor.from_pretrained('./whisper-small-turkish')
model = WhisperForConditionalGeneration.from_pretrained('./whisper-small-turkish')

# Ses işleme
inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    generated_ids = model.generate(inputs["input_features"])
transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]""",
        title="Python Kullanım Örnekleri",
        border_style="green"
    ))
    
    console.print("\n[dim]Model artık kullanıma hazır! 🎯[/dim]")

if __name__ == "__main__":
    main()
