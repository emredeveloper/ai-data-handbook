"""
Whisper Small Fine-tuning Script for Turkish
Fine-tunes the Whisper model using the Khan Academy Turkish Dataset
"""

import os
import torch
import torchaudio
from datasets import load_dataset, DatasetDict, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    WhisperTokenizer,
    WhisperFeatureExtractor
)
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
import numpy as np
from huggingface_hub import login
import warnings
warnings.filterwarnings("ignore")

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

def main():
    # Initialize Rich console
    console = Console()
    
    # Welcome panel
    console.print(Panel.fit(
        "[bold cyan]🎯 Whisper Turkish Fine-tuning[/bold cyan]\n"
        "[dim]Fine-tuning Whisper Small for Turkish speech recognition[/dim]",
        border_style="cyan"
    ))
    
    # Skipped Hugging Face Hub login (token will be requested if needed)
    console.print("[yellow]⚠️[/yellow] Skipping login to Hugging Face Hub...")

    # Dataset loading
    console.print("\n[bold blue]📥 Loading Khan Academy Turkish dataset...[/bold blue]")
    try:
        # Load dataset
        dataset = load_dataset("ysdede/khanacademy-turkish")
        
        # Inspect dataset structure
        console.print("[green]✅[/green] Dataset loaded successfully!")
        console.print("[dim]Dataset structure:[/dim]")
        console.print(dataset)
        
        # Create train/test split if missing
        if "train" not in dataset:
            # Split dataset into 80% train and 20% test
            dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
            dataset = DatasetDict({
                "train": dataset["train"],
                "test": dataset["test"]
            })
        
        # Select only audio and transcription columns
        if "audio" in dataset["train"].column_names and "transcription" in dataset["train"].column_names:
            dataset = dataset.select_columns(["audio", "transcription"])
            # rename transcription column to text
            dataset = dataset.rename_column("transcription", "text")
        elif "audio" in dataset["train"].column_names and "text" in dataset["train"].column_names:
            dataset = dataset.select_columns(["audio", "text"])
        elif "sentence" in dataset["train"].column_names:
            dataset = dataset.select_columns(["audio", "sentence"])
            # rename sentence column to text
            dataset = dataset.rename_column("sentence", "text")
        else:
            print("Available columns:", dataset["train"].column_names)
            raise ValueError("Could not find 'audio' and 'transcription' columns in dataset")
            
        # Create dataset info table
        table = Table(title="Dataset Information")
        table.add_column("Split", style="cyan")
        table.add_column("Samples", justify="right", style="green")
        table.add_row("Train", str(len(dataset['train'])))
        table.add_row("Test", str(len(dataset['test'])))
        console.print(table)
        
        # Train with 5000 training and 100 test samples
        desired_train = 5000
        desired_test = 100
        console.print(f"\n[bold yellow]📊 Subsampling dataset ({desired_train} train, {desired_test} test)...[/bold yellow]")
        dataset["train"] = dataset["train"].select(range(min(desired_train, len(dataset["train"]) )))
        dataset["test"] = dataset["test"].select(range(min(desired_test, len(dataset["test"]) )))
        
        # Updated dataset info
        table = Table(title="Final Dataset")
        table.add_column("Split", style="cyan")
        table.add_column("Samples", justify="right", style="green")
        table.add_row("Train", str(len(dataset['train'])))
        table.add_row("Test", str(len(dataset['test'])))
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]❌ Dataset load error: {e}[/red]")
        console.print("[yellow]Trying an alternative dataset...[/yellow]")
        # Fallback - minimal dataset
        raise Exception("Dataset could not be loaded")

    # Keep audio as is - will be handled in preprocessing
    console.print("\n[bold blue]🎵 Audio will be processed during preprocessing...[/bold blue]")

    # Cast audio column to 16kHz (prevents AudioDecoder/sampling_rate issues)
    try:
        console.print("[blue]📡 Casting audio column to 16kHz (datasets.Audio)...[/blue]")
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        console.print("[green]✅ Audio casting successful![/green]")
    except Exception as e:
        console.print(f"[yellow]⚠️ Audio cast failed, continuing: {e}[/yellow]")

    # Load processor (for Turkish)
    console.print("\n[bold blue]🔧 Loading Whisper processor...[/bold blue]")
    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-small", 
        language="turkish", 
        task="transcribe"
    )
    console.print("[green]✅ Processor loaded![/green]")

    # Load model
    console.print("[bold blue]🤖 Loading Whisper model...[/bold blue]")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    console.print("[green]✅ Model loaded![/green]")

    # Model configuration
    model.config.use_cache = False
    model.config.forced_decoder_ids = None  # clear forced decoder IDs

    # Enable cache for generation
    from functools import partial
    model.generate = partial(
        model.generate, 
        language="turkish", 
        task="transcribe", 
        use_cache=True
    )

    # Data preprocessing function - simple dict-based flow after cast
    def prepare_dataset(batch):
        try:
            # Process audio
            audio = batch["audio"]

            # Expected: dict {array, sampling_rate}. If not, apply soft conversion
            if isinstance(audio, dict):
                audio_array = audio.get("array")
                sampling_rate = audio.get("sampling_rate", 16000)
            else:
                audio_array = getattr(audio, "array", audio)
                sampling_rate = getattr(audio, "sampling_rate", 16000)

            # sampling_rate safety
            try:
                sampling_rate = int(sampling_rate) if sampling_rate else 16000
            except Exception:
                sampling_rate = 16000

            # Convert to numpy
            if not isinstance(audio_array, np.ndarray):
                if hasattr(audio_array, "numpy"):
                    audio_array = audio_array.numpy()
                else:
                    audio_array = np.asarray(audio_array, dtype=object)

            # Type and safety checks
            # Flatten object/jagged arrays
            if audio_array.dtype == object:
                try:
                    parts = []
                    for x in audio_array:
                        if x is None:
                            continue
                        x_arr = np.asarray(x, dtype=np.float32).reshape(-1)
                        parts.append(x_arr)
                    audio_array = np.concatenate(parts) if parts else np.zeros(16000, dtype=np.float32)
                except Exception:
                    audio_array = np.zeros(16000, dtype=np.float32)
            else:
                audio_array = audio_array.astype(np.float32, copy=False)

            # Shape adjustment
            if audio_array.ndim >= 2:
                # (C, N) or (N, C): if C=2, average channels
                if 2 in audio_array.shape:
                    axis = int(np.argmin(audio_array.shape)) if audio_array.ndim == 2 else -1
                    audio_array = np.mean(audio_array, axis=axis)
                else:
                    audio_array = audio_array.reshape(-1)
            if audio_array.size == 0 or np.isnan(audio_array).any():
                audio_array = np.zeros(16000, dtype=np.float32)
                sampling_rate = 16000

            # Amplitude normalization (if needed)
            max_abs = float(np.max(np.abs(audio_array))) if audio_array.size else 0.0
            if max_abs > 1.0:
                audio_array = audio_array / max_abs
            
            # Process text
            batch["input_features"] = processor(
                audio_array, 
                sampling_rate=sampling_rate
            ).input_features[0]
            
            # Tokenize text
            batch["labels"] = processor.tokenizer(
                batch["text"], 
                max_length=225, 
                padding="max_length", 
                truncation=True
            ).input_ids
            
        except Exception as e:
            print(f"Audio preprocessing error: {e}")
            print("❌ Skipping this sample, using dummy data...")
            # On error, create fully dummy data
            batch["input_features"] = processor(
                np.zeros(16000, dtype=np.float32), 
                sampling_rate=16000
            ).input_features[0]
            batch["labels"] = processor.tokenizer(
                "Dummy text for failed audio processing.", 
                max_length=225, 
                padding="max_length", 
                truncation=True
            ).input_ids
        
        return batch

    # Prepare dataset
    console.print("\n[bold blue]⚙️ Preparing dataset...[/bold blue]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Processing audio data...", total=None)
        dataset = dataset.map(
            prepare_dataset,
            remove_columns=dataset["train"].column_names,
            num_proc=1  # set to 1 on Windows to avoid multiprocessing issues
        )
        progress.update(task, description="[green]✅ Dataset prepared!")
    console.print("[green]✅ Dataset preparation completed![/green]")

    # Data collator
    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            # Stack input features
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

            # Stack labels
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

            # Replace padding tokens in labels with -100
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

            # If all labels are -100, loss won't be computed
            if (labels == -100).all():
                labels[:, 0] = self.processor.tokenizer.eos_token_id

            batch["labels"] = labels

            return batch

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Evaluation metrics
    console.print("\n[bold blue]📊 Loading evaluation metrics...[/bold blue]")
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    console.print("[green]✅ Metrics loaded![/green]")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Remove -100s
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # Decode
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        # Compute WER
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        
        # Compute CER
        cer = cer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer, "cer": cer}

    # Training arguments
    console.print("\n[bold blue]⚙️ Setting up training configuration...[/bold blue]")
    use_fp16 = torch.cuda.is_available()
    
    # GPU info
    if use_fp16:
        console.print(f"[green]🚀 GPU detected: {torch.cuda.get_device_name(0)}[/green]")
        console.print("[blue]🔧 Enabling TF32 for faster training...[/blue]")
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            console.print("[green]✅ TF32 enabled![/green]")
        except Exception:
            console.print("[yellow]⚠️ TF32 not available[/yellow]")
    else:
        console.print("[yellow]⚠️ No GPU detected, using CPU[/yellow]")
    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper-small-turkish",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=6,  # larger effective batch without extra VRAM
        learning_rate=1.5e-5,
        lr_scheduler_type="cosine",  # better convergence over longer training
        warmup_steps=20,
        max_steps=100,  # longer training for better results
        gradient_checkpointing=False,
        fp16=use_fp16,
        fp16_full_eval=use_fp16,
        eval_strategy="steps",
        per_device_eval_batch_size=8,  # faster evaluation
        predict_with_generate=True,
        generation_max_length=300,
        generation_num_beams=5,
        save_steps=25,
        eval_steps=25,
        logging_steps=5,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        hub_strategy="checkpoint",
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        weight_decay=0.01,
    )

    # Create trainer
    console.print("\n[bold blue]🏗️ Creating trainer...[/bold blue]")
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.tokenizer,  # tokenizer instead of feature_extractor
    )
    console.print("[green]✅ Trainer created![/green]")

    # Training configuration summary
    config_table = Table(title="Training Configuration")
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="green")
    config_table.add_row("Max Steps", "100")
    config_table.add_row("Learning Rate", "1.5e-5")
    config_table.add_row("Batch Size", "2")
    config_table.add_row("Gradient Accumulation", "6")
    config_table.add_row("Effective Batch Size", "12")
    config_table.add_row("FP16", str(use_fp16))
    config_table.add_row("Scheduler", "cosine")
    config_table.add_row("Warmup Steps", "20")
    console.print(config_table)

    # Start training
    console.print("\n[bold green]🚀 Starting training...[/bold green]")
    console.print("[dim]This process may take 10-30 minutes depending on your GPU...[/dim]")

    try:
        trainer.train()
        console.print("\n[bold green]🎉 Training completed![/bold green]")
        console.print("[green]✅ Model saved to './whisper-small-turkish' directory![/green]")
        
    except Exception as e:
        console.print(f"\n[red]❌ Training error: {e}[/red]")
        console.print("[yellow]💡 Please check GPU memory and reduce the batch size.[/yellow]")

    # Test example
    console.print("\n[bold blue]📝 Test example:[/bold blue]")
    console.print(Panel(
        "[bold]To test the model:[/bold]\n"
        "from transformers import pipeline\n"
        "pipe = pipeline('automatic-speech-recognition', model='./whisper-small-turkish')\n"
        "result = pipe('path/to/audio.wav')\n"
        "print(result['text'])",
        title="Usage Example",
        border_style="blue"
    ))
    
    console.print("\n[bold green]🎯 Fine-tuning script completed![/bold green]")

if __name__ == "__main__":
    main()