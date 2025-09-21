"""
SafeTensors modelini PyTorch formatına çeviren script
Windows dosya kilitleme sorunlarını çözen gelişmiş versiyon
"""

import torch
from transformers import WhisperForConditionalGeneration
from safetensors.torch import load_file
import os
import shutil
import tempfile
import time
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

def convert_safetensors_to_pytorch():
    console = Console()
    
    model_dir = "./whisper-small-turkish"
    safetensors_path = os.path.join(model_dir, "model.safetensors")
    pytorch_path = os.path.join(model_dir, "pytorch_model.bin")
    
    # Dosya varlığını kontrol et
    if not os.path.exists(safetensors_path):
        console.print(f"[red]❌ SafeTensors dosyası bulunamadı: {safetensors_path}[/red]")
        return False
    
    if os.path.exists(pytorch_path):
        console.print("[yellow]⚠️ PyTorch model dosyası zaten mevcut![/yellow]")
        response = input("Üzerine yazılsın mı? (y/N): ").lower().strip()
        if response != 'y' and response != 'yes':
            console.print("[blue]İşlem iptal edildi.[/blue]")
            return True
    
    console.print("[bold blue]🔄 SafeTensors'dan PyTorch formatına çeviriliyor...[/bold blue]")
    
    # Windows dosya kilitleme sorunlarını önlemek için geçici dizin kullan
    temp_dir = None
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("SafeTensors yükleniyor...", total=None)
            
            # Geçici dizin oluştur
            temp_dir = tempfile.mkdtemp(prefix="safetensors_convert_")
            temp_safetensors = os.path.join(temp_dir, "model.safetensors")
            temp_pytorch = os.path.join(temp_dir, "pytorch_model.bin")
            
            # SafeTensors dosyasını geçici dizine kopyala
            progress.update(task, description="SafeTensors kopyalanıyor...")
            shutil.copy2(safetensors_path, temp_safetensors)
            
            # Kısa bekleme
            time.sleep(1)
            
            # SafeTensors dosyasını yükle
            progress.update(task, description="SafeTensors yükleniyor...")
            state_dict = load_file(temp_safetensors)
            console.print("[green]✅ SafeTensors dosyası yüklendi![/green]")
            
            # PyTorch formatında geçici dizine kaydet
            progress.update(task, description="PyTorch formatına çeviriliyor...")
            torch.save(state_dict, temp_pytorch)
            
            # Dosya boyutlarını kontrol et
            safetensors_size = os.path.getsize(temp_safetensors) / (1024*1024)
            pytorch_size = os.path.getsize(temp_pytorch) / (1024*1024)
            
            console.print(f"[blue]📊 SafeTensors: {safetensors_size:.1f} MB[/blue]")
            console.print(f"[blue]📊 PyTorch: {pytorch_size:.1f} MB[/blue]")
            
            # Kısa bekleme
            time.sleep(1)
            
            # Ana dizine kopyala
            progress.update(task, description="Ana dizine kopyalanıyor...")
            shutil.copy2(temp_pytorch, pytorch_path)
            
            progress.update(task, description="[green]✅ Dönüşüm tamamlandı!")
        
        console.print("[green]✅ PyTorch formatında kaydedildi![/green]")
        
        # Model yükleme testi
        console.print("[blue]🧪 Model yükleme testi yapılıyor...[/blue]")
        try:
            model = WhisperForConditionalGeneration.from_pretrained(model_dir)
            console.print("[green]✅ Model başarıyla yüklendi ve doğrulandı![/green]")
            del model  # Belleği temizle
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception as test_e:
            console.print(f"[yellow]⚠️ Model yükleme testi başarısız: {test_e}[/yellow]")
            console.print("[yellow]Model dosyası oluşturuldu ama doğrulama yapılamadı.[/yellow]")
        
        return True
        
    except Exception as e:
        console.print(f"[red]❌ Hata: {e}[/red]")
        console.print("[yellow]💡 Lütfen tüm Python süreçlerini kapatıp tekrar deneyin.[/yellow]")
        return False
        
    finally:
        # Geçici dizini temizle
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                console.print(f"[dim]Gecici dizin temizlendi: {temp_dir}[/dim]")
            except Exception as cleanup_e:
                console.print(f"[yellow]Gecici dizin temizleme hatasi: {cleanup_e}[/yellow]")

def main():
    # Windows UTF-8 encoding sorunu için console ayarları
    console = Console(force_terminal=True, legacy_windows=False)
    
    console.print("[bold cyan]SafeTensors -> PyTorch Donusturucu[/bold cyan]")
    console.print("[dim]Windows dosya kilitleme sorunlarini cozen gelismis versiyon[/dim]\n")
    
    success = convert_safetensors_to_pytorch()
    
    if success:
        console.print("\n[bold green]Donusum basariyla tamamlandi![/bold green]")
        console.print("[blue]Model artik PyTorch formatinda kullanima hazir.[/blue]")
    else:
        console.print("\n[bold red]Donusum basarisiz![/bold red]")
        console.print("[yellow]Lutfen hata mesajlarini kontrol edin ve tekrar deneyin.[/yellow]")

if __name__ == "__main__":
    main()
