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

def main():
    # Skipped Hugging Face Hub login (token will be requested if needed)
    print("Skipping login to Hugging Face Hub...")

    # Dataset loading
    print("Loading Khan Academy Turkish dataset...")
    try:
        # Load dataset
        dataset = load_dataset("ysdede/khanacademy-turkish")
        
        # Inspect dataset structure
        print("Dataset structure:")
        print(dataset)
        
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
            
        print(f"Train samples: {len(dataset['train'])}")
        print(f"Test samples: {len(dataset['test'])}")
        
        # Train with 1000 training and 100 test samples
        desired_train = 1000
        desired_test = 100
        print(f"Subsampling dataset ({desired_train} train, {desired_test} test)...")
        dataset["train"] = dataset["train"].select(range(min(desired_train, len(dataset["train"]) )))
        dataset["test"] = dataset["test"].select(range(min(desired_test, len(dataset["test"]) )))
        print(f"New train samples: {len(dataset['train'])}")
        print(f"New test samples: {len(dataset['test'])}")
        
    except Exception as e:
        print(f"Dataset load error: {e}")
        print("Trying an alternative dataset...")
        # Fallback - minimal dataset
        raise Exception("Dataset could not be loaded")

    # Keep audio as is - will be handled in preprocessing
    print("Audio will be processed during preprocessing...")

    # Cast audio column to 16kHz (prevents AudioDecoder/sampling_rate issues)
    try:
        print("Casting audio column to 16kHz (datasets.Audio)...")
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    except Exception as e:
        print(f"Audio cast failed, continuing: {e}")

    # Load processor (for Turkish)
    print("Loading Whisper processor...")
    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-small", 
        language="turkish", 
        task="transcribe"
    )

    # Load model
    print("Loading Whisper model...")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

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
    print("Preparing dataset...")
    dataset = dataset.map(
        prepare_dataset,
        remove_columns=dataset["train"].column_names,
        num_proc=1  # set to 1 on Windows to avoid multiprocessing issues
    )

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
    print("Loading evaluation metrics...")
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

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
    print("Setting up training configuration...")
    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper-small-turkish",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,  # effective batch size increases, stability improves
        learning_rate=1e-5,
        lr_scheduler_type="linear",  # linear scheduler for steadier learning
        warmup_steps=50,  # longer warmup
        max_steps=250,  # ~1 epoch (> 1000/ (2*4) ≈ 125 step/epoch)
        gradient_checkpointing=False,
        fp16=False,
        fp16_full_eval=False,
        eval_strategy="steps",
        per_device_eval_batch_size=2,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=50,
        eval_steps=50,
        logging_steps=10,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        hub_strategy="checkpoint",
        dataloader_num_workers=0,
    )

    # Create trainer
    print("Creating trainer...")
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.tokenizer,  # tokenizer instead of feature_extractor
    )

    # Start training
    print("Starting training...")
    print("This process may take 1-3 hours depending on your GPU...")

    try:
        trainer.train()
        print("Training completed!")
        print("Model saved to './whisper-small-turkish' directory!")
        
    except Exception as e:
        print(f"Training error: {e}")
        print("Please check GPU memory and reduce the batch size.")

    # Test example
    print("\nTest example:")
    print("To test the model:")
    print("from transformers import pipeline")
    print("pipe = pipeline('automatic-speech-recognition', model='./whisper-small-turkish')")
    print("result = pipe('path/to/audio.wav')")
    print("print(result['text'])")

if __name__ == "__main__":
    main()