"""
Whisper Fine-tuning Script for Turkish
Fine-tunes the Whisper model using various Turkish datasets with flexible configuration
"""

import os
import torch
import argparse
import evaluate
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import DatasetDict, Audio, load_dataset, concatenate_datasets
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers import (
    WhisperFeatureExtractor, 
    WhisperTokenizer, 
    WhisperProcessor, 
    WhisperForConditionalGeneration, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    EarlyStoppingCallback
)
import warnings
warnings.filterwarnings("ignore")

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

###############################     DATA COLLATOR DEFINITION     ########################

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

#######################     ARGUMENT PARSING        #########################

def parse_arguments():
    parser = argparse.ArgumentParser(description='Fine-tuning script for Whisper Models of various sizes.')
    parser.add_argument(
        '--model_name', 
        type=str, 
        required=False, 
        default='openai/whisper-small', 
        help='Huggingface model name to fine-tune. Eg: openai/whisper-small'
    )
    parser.add_argument(
        '--language', 
        type=str, 
        required=False, 
        default='turkish', 
        help='Language the model is being adapted to in lowercase.'
    )
    parser.add_argument(
        '--sampling_rate', 
        type=int, 
        required=False, 
        default=16000, 
        help='Sampling rate of audios.'
    )
    parser.add_argument(
        '--num_proc', 
        type=int, 
        required=False, 
        default=1, 
        help='Number of parallel jobs to run. Helps parallelize the dataset prep stage.'
    )
    parser.add_argument(
        '--train_strategy', 
        type=str, 
        required=False, 
        default='steps', 
        help='Training strategy. Choose between steps and epoch.'
    )
    parser.add_argument(
        '--learning_rate', 
        type=float, 
        required=False, 
        default=5e-6, 
        help='Learning rate for the fine-tuning process. Lower rate to prevent overfitting.'
    )
    parser.add_argument(
        '--warmup', 
        type=int, 
        required=False, 
        default=50, 
        help='Number of warmup steps. Increased for gradual learning rate increase.'
    )
    parser.add_argument(
        '--train_batchsize', 
        type=int, 
        required=False, 
        default=1, 
        help='Batch size during the training phase.'
    )
    parser.add_argument(
        '--eval_batchsize', 
        type=int, 
        required=False, 
        default=2, 
        help='Batch size during the evaluation phase.'
    )
    parser.add_argument(
        '--num_epochs', 
        type=int, 
        required=False, 
        default=3, 
        help='Number of epochs to train for.'
    )
    parser.add_argument(
        '--num_steps', 
        type=int, 
        required=False, 
        default=500, 
        help='Number of steps to train for. Increased for better convergence.'
    )
    parser.add_argument(
        '--resume_from_ckpt', 
        type=str, 
        required=False, 
        default=None, 
        help='Path to a trained checkpoint to resume training from.'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        required=False, 
        default='./whisper-small-turkish', 
        help='Output directory for the checkpoints generated.'
    )
    parser.add_argument(
        '--train_datasets', 
        type=str, 
        nargs='+', 
        required=False, 
        default=['ysdede/khanacademy-turkish'], 
        help='List of datasets to be used for training.'
    )
    parser.add_argument(
        '--train_dataset_configs', 
        type=str, 
        nargs='+', 
        required=False, 
        default=['default'], 
        help="List of training dataset configs. Eg. 'hi' for the Hindi part of Common Voice",
    )
    parser.add_argument(
        '--train_dataset_splits', 
        type=str, 
        nargs='+', 
        required=False, 
        default=['train'], 
        help="List of training dataset splits. Eg. 'train' for the train split of Common Voice",
    )
    parser.add_argument(
        '--train_dataset_text_columns', 
        type=str, 
        nargs='+', 
        required=False, 
        default=['transcription'], 
        help="Text column name of each training dataset. Eg. 'sentence' for Common Voice",
    )
    parser.add_argument(
        '--eval_datasets', 
        type=str, 
        nargs='+', 
        required=False, 
        default=['ysdede/khanacademy-turkish'], 
        help='List of datasets to be used for evaluation.'
    )
    parser.add_argument(
        '--eval_dataset_configs', 
        type=str, 
        nargs='+', 
        required=False, 
        default=['default'], 
        help="List of evaluation dataset configs. Eg. 'hi_in' for the Hindi part of Google Fleurs",
    )
    parser.add_argument(
        '--eval_dataset_splits', 
        type=str, 
        nargs='+', 
        required=False, 
        default=['test'], 
        help="List of evaluation dataset splits. Using 'test' for unseen data evaluation.",
    )
    parser.add_argument(
        '--eval_dataset_text_columns', 
        type=str, 
        nargs='+', 
        required=False, 
        default=['transcription'], 
        help="Text column name of each evaluation dataset. Eg. 'transcription' for Google Fleurs",
    )
    parser.add_argument(
        '--max_train_samples', 
        type=int, 
        required=False, 
        default=5000, 
        help='Maximum number of training samples to use.'
    )
    parser.add_argument(
        '--max_eval_samples', 
        type=int, 
        required=False, 
        default=100, 
        help='Maximum number of evaluation samples to use.'
    )

    return parser.parse_args()

def validate_arguments(args):
    if args.train_strategy not in ['steps', 'epoch']:
        raise ValueError('The train strategy should be either steps and epoch.')

    if len(args.train_datasets) == 0:
        raise ValueError('No train dataset has been passed')
    if len(args.eval_datasets) == 0:
        raise ValueError('No evaluation dataset has been passed')

    # Validation for list arguments
    list_args = [
        ('train_datasets', 'train_dataset_configs'),
        ('train_datasets', 'train_dataset_splits'),
        ('train_datasets', 'train_dataset_text_columns'),
        ('eval_datasets', 'eval_dataset_configs'),
        ('eval_datasets', 'eval_dataset_splits'),
        ('eval_datasets', 'eval_dataset_text_columns')
    ]
    
    for arg1, arg2 in list_args:
        if len(getattr(args, arg1)) != len(getattr(args, arg2)):
            raise ValueError(f"Ensure that the number of entries in {arg1} equals {arg2}. "
                           f"Received {len(getattr(args, arg1))} for {arg1} and {len(getattr(args, arg2))} for {arg2}.")

def main():
    # Parse arguments
    args = parse_arguments()
    validate_arguments(args)
    
    # Initialize Rich console
    console = Console()
    
    # Welcome panel
    console.print(Panel.fit(
        "[bold cyan]🎯 Whisper Turkish Fine-tuning[/bold cyan]\n"
        "[dim]Fine-tuning Whisper model for Turkish speech recognition[/dim]",
        border_style="cyan"
    ))
    
    # Print arguments
    console.print('\n[bold blue]📋 Configuration:[/bold blue]')
    config_table = Table()
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="green")
    for key, value in vars(args).items():
        config_table.add_row(str(key), str(value))
    console.print(config_table)
    
    # Global configuration
    gradient_checkpointing = False  # GPU'da gradient checkpointing kapatıldı
    freeze_feature_encoder = False
    freeze_encoder = False
    do_normalize_eval = True
    do_lower_case = False
    do_remove_punctuation = False
    normalizer = BasicTextNormalizer()

    #############################       MODEL LOADING       #####################################

    console.print("\n[bold blue]🔧 Loading model components...[/bold blue]")
    
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_name)
    tokenizer = WhisperTokenizer.from_pretrained(args.model_name, language=args.language, task="transcribe")
    processor = WhisperProcessor.from_pretrained(args.model_name, language=args.language, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that config.decoder_start_token_id is correctly defined")

    if freeze_feature_encoder:
        model.freeze_feature_encoder()

    if freeze_encoder:
        model.freeze_encoder()
        model.model.encoder.gradient_checkpointing = False

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False  # Training için cache kapatıldı
    
    # Overfitting önleme için dropout artır
    model.config.dropout = 0.15  # Default 0.1'den artırıldı
    model.config.attention_dropout = 0.15  # Default 0.1'den artırıldı
    model.config.activation_dropout = 0.15  # Default 0.1'den artırıldı

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_disable()
    
    # Device optimization
    if not torch.cuda.is_available():
        torch.set_num_threads(os.cpu_count())
        console.print(f"[blue]💻 CPU kullanılıyor - {os.cpu_count()} cores[/blue]")
    else:
        console.print("[green]🚀 GPU kullanılıyor[/green]")
    
    # Training mode
    model.train()

    console.print("[green]✅ Model components loaded![/green]")

    ############################        DATASET LOADING AND PREP        ##########################

    def load_all_datasets(split):    
        combined_dataset = []
        if split == 'train':
            for i, ds in enumerate(args.train_datasets):
                console.print(f"[blue]📥 Loading train dataset: {ds}[/blue]")
                dataset = load_dataset(ds, args.train_dataset_configs[i], split=args.train_dataset_splits[i])
                dataset = dataset.cast_column("audio", Audio(args.sampling_rate))
                if args.train_dataset_text_columns[i] != "sentence":
                    dataset = dataset.rename_column(args.train_dataset_text_columns[i], "sentence")
                dataset = dataset.remove_columns(set(dataset.features.keys()) - set(["audio", "sentence"]))
                combined_dataset.append(dataset)
        elif split == 'eval':
            for i, ds in enumerate(args.eval_datasets):
                console.print(f"[blue]📥 Loading eval dataset: {ds}[/blue]")
                try:
                    # İlk olarak test split'ini dene
                    dataset = load_dataset(ds, args.eval_dataset_configs[i], split=args.eval_dataset_splits[i])
                    console.print(f"[green]✅ Test split loaded for evaluation[/green]")
                except:
                    # Test split yoksa train split'in son %10'unu kullan
                    console.print(f"[yellow]⚠️ Test split not found, using validation subset from train[/yellow]")
                    full_dataset = load_dataset(ds, args.eval_dataset_configs[i], split="train")
                    # Train'in son %10'unu validation olarak kullan
                    val_size = max(100, len(full_dataset) // 10)
                    start_idx = len(full_dataset) - val_size
                    dataset = full_dataset.select(range(start_idx, len(full_dataset)))
                
                dataset = dataset.cast_column("audio", Audio(args.sampling_rate))
                if args.eval_dataset_text_columns[i] != "sentence":
                    dataset = dataset.rename_column(args.eval_dataset_text_columns[i], "sentence")
                dataset = dataset.remove_columns(set(dataset.features.keys()) - set(["audio", "sentence"]))
                combined_dataset.append(dataset)
        
        ds_to_return = concatenate_datasets(combined_dataset)
        ds_to_return = ds_to_return.shuffle(seed=22)
        return ds_to_return

    console.print('\n[bold blue]📊 DATASET PREPARATION IN PROGRESS...[/bold blue]')
    raw_dataset = DatasetDict()
    raw_dataset["train"] = load_all_datasets('train')
    raw_dataset["eval"] = load_all_datasets('eval')
    
    # Limit dataset sizes
    console.print(f"[yellow]📊 Limiting train to {args.max_train_samples} and eval to {args.max_eval_samples} samples...[/yellow]")
    raw_dataset["train"] = raw_dataset["train"].select(range(min(args.max_train_samples, len(raw_dataset["train"]))))
    raw_dataset["eval"] = raw_dataset["eval"].select(range(min(args.max_eval_samples, len(raw_dataset["eval"]))))
    
    # Dataset info
    table = Table(title="Dataset Information")
    table.add_column("Split", style="cyan")
    table.add_column("Samples", justify="right", style="green")
    table.add_row("Train", str(len(raw_dataset['train'])))
    table.add_row("Eval", str(len(raw_dataset['eval'])))
    console.print(table)

    def prepare_dataset(batch):
        # load and (possibly) resample audio data to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array 
        batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        # compute input length of audio sample in seconds
        batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]
        
        # optional pre-processing steps
        transcription = batch["sentence"]
        if do_lower_case:
            transcription = transcription.lower()
        if do_remove_punctuation:
            transcription = normalizer(transcription).strip()
        
        # encode target text to label ids
        batch["labels"] = processor.tokenizer(transcription).input_ids
        return batch

    max_label_length = model.config.max_length
    min_input_length = 0.0
    max_input_length = 30.0
    def is_in_length_range(length, labels):
        return min_input_length < length < max_input_length and 0 < len(labels) < max_label_length

    console.print("\n[bold blue]⚙️ Preparing dataset...[/bold blue]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Processing audio data...", total=None)
        raw_dataset = raw_dataset.map(prepare_dataset, num_proc=args.num_proc)
        progress.update(task, description="Filtering by length...")
        raw_dataset = raw_dataset.filter(
            is_in_length_range,
            input_columns=["input_length", "labels"],
            num_proc=args.num_proc,
        ) 
        progress.update(task, description="[green]✅ Dataset prepared!")
    console.print("[green]✅ Dataset preparation completed![/green]")

    ###############################     DATA COLLATOR AND METRIC DEFINITION     ########################

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    console.print("[green]✅ Data collator created![/green]")

    console.print("\n[bold blue]📊 Loading evaluation metrics...[/bold blue]")
    metric = evaluate.load("wer")
    console.print("[green]✅ Metrics loaded![/green]")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        if do_normalize_eval:
            pred_str = [normalizer(pred) for pred in pred_str]
            label_str = [normalizer(label) for label in label_str]

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    ###############################     TRAINING ARGS AND TRAINING      ############################

    console.print("\n[bold blue]⚙️ Setting up training configuration...[/bold blue]")
    
    use_fp16 = torch.cuda.is_available()  # Use FP16 only if GPU is available
    use_cpu = not torch.cuda.is_available()
    
    if use_cpu:
        console.print("[blue]💻 CPU kullanılıyor[/blue]")
        console.print(f"[green]🔧 CPU Cores: {os.cpu_count()}[/green]")
    else:
        console.print("[green]🚀 GPU kullanılıyor[/green]")

    if args.train_strategy == 'epoch':
        training_args = Seq2SeqTrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.train_batchsize,
            gradient_accumulation_steps=1,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup,
            gradient_checkpointing=gradient_checkpointing,
            fp16=use_fp16,
            eval_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=args.num_epochs,
            save_total_limit=5,  # Daha az checkpoint sakla
            per_device_eval_batch_size=args.eval_batchsize,
            predict_with_generate=True,
            generation_max_length=225,
            logging_steps=25,
            report_to=["tensorboard"],
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            # Overfitting önleme parametreleri
            weight_decay=0.01,  # L2 regularization
            lr_scheduler_type="cosine",  # Cosine learning rate decay
            dataloader_num_workers=0,  # Windows uyumluluğu için 0
            dataloader_pin_memory=False,  # Windows uyumluluğu için False
            resume_from_checkpoint=args.resume_from_ckpt,
        )

    elif args.train_strategy == 'steps':
        training_args = Seq2SeqTrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.train_batchsize,
            gradient_accumulation_steps=8 if use_cpu else 1,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup,
            gradient_checkpointing=gradient_checkpointing,
            fp16=use_fp16,
            eval_strategy="steps",
            eval_steps=max(args.num_steps // 10, 50),  # Daha sık evaluation
            save_strategy="steps",
            save_steps=max(args.num_steps // 10, 50),  # Daha sık save
            max_steps=args.num_steps,
            save_total_limit=5,  # Daha az checkpoint sakla
            per_device_eval_batch_size=args.eval_batchsize,
            predict_with_generate=True,
            generation_max_length=225,
            logging_steps=max(args.num_steps // 20, 10),
            report_to=["tensorboard"],
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            # Overfitting önleme parametreleri
            weight_decay=0.01,  # L2 regularization
            lr_scheduler_type="cosine",  # Cosine learning rate decay
            dataloader_num_workers=0,  # Windows uyumluluğu için 0
            dataloader_pin_memory=False,  # Windows uyumluluğu için False
            resume_from_checkpoint=args.resume_from_ckpt,
        )

    # Early stopping callback ekle
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3,  # 3 evaluation boyunca iyileşme yoksa dur
        early_stopping_threshold=0.001  # Minimum iyileşme threshold
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=raw_dataset["train"],
        eval_dataset=raw_dataset["eval"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
        callbacks=[early_stopping_callback],  # Early stopping callback ekle
    )

    processor.save_pretrained(training_args.output_dir)
    console.print("[green]✅ Trainer created![/green]")

    # Training configuration summary
    config_table = Table(title="Final Training Configuration")
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="green")
    config_table.add_row("Model", args.model_name)
    config_table.add_row("Language", args.language)
    config_table.add_row("Strategy", args.train_strategy)
    if args.train_strategy == 'steps':
        config_table.add_row("Max Steps", str(args.num_steps))
    else:
        config_table.add_row("Epochs", str(args.num_epochs))
    config_table.add_row("Learning Rate", str(args.learning_rate))
    config_table.add_row("Train Batch Size", str(args.train_batchsize))
    config_table.add_row("Eval Batch Size", str(args.eval_batchsize))
    config_table.add_row("Gradient Accumulation", str(training_args.gradient_accumulation_steps))
    config_table.add_row("FP16", str(use_fp16))
    config_table.add_row("Device", "GPU" if not use_cpu else "CPU")
    config_table.add_row("Train Samples", str(len(raw_dataset["train"])))
    config_table.add_row("Eval Samples", str(len(raw_dataset["eval"])))
    console.print(config_table)

    # Start training
    console.print("\n[bold green]🚀 Starting training...[/bold green]")
    if use_cpu:
        console.print(f"[dim]{args.num_steps} adım CPU eğitimi zaman alabilir...[/dim]")
    else:
        console.print(f"[dim]GPU ile {args.num_steps} adım eğitim başlıyor...[/dim]")

    try:
        console.print('\n[bold blue]TRAINING IN PROGRESS...[/bold blue]')
        trainer.train()
        console.print('\n[bold green]DONE TRAINING[/bold green]')
        console.print(f"[green]✅ Model saved to '{args.output_dir}' directory![/green]")
        
    except Exception as e:
        console.print(f"\n[red]❌ Training error: {e}[/red]")
        console.print("[yellow]💡 Eğitim sırasında hata oluştu. Lütfen sistem kaynaklarını kontrol edin.[/yellow]")

    # Test example
    console.print("\n[bold blue]📝 Test example:[/bold blue]")
    console.print(Panel(
        "[bold]To test the model:[/bold]\n"
        "from transformers import pipeline\n"
        f"pipe = pipeline('automatic-speech-recognition', model='{args.output_dir}')\n"
        "result = pipe('path/to/audio.wav')\n"
        "print(result['text'])",
        title="Usage Example",
        border_style="blue"
    ))
    
    console.print("\n[bold green]🎯 Fine-tuning script completed![/bold green]")

if __name__ == "__main__":
    main()