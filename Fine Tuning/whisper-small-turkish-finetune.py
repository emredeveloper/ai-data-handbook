import torch
from datasets import load_dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
import evaluate
import numpy as np

# 1. Dataset yükle (5k örnek)
dataset = load_dataset("erenfazlioglu/turkishvoicedataset", split="train[:5000]")

# 2. Validation split (%10)
split = dataset.train_test_split(test_size=0.1, seed=42)
train_ds = split["train"]
val_ds = split["test"]

# 3. Audio sütununu 16kHz'e cast et
train_ds = train_ds.cast_column("audio", Audio(sampling_rate=16000))
val_ds = val_ds.cast_column("audio", Audio(sampling_rate=16000))

# 4. Processor ve Model
model_name = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(model_name, language="turkish", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# 5. Preprocess fonksiyonu (padding + truncation dahil)
def prepare_batch(batch):
    audio = batch["audio"]["array"]

    # Features
    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding="longest",  # tüm batch aynı uzunlukta olacak
        truncation=True
    )

    batch["input_features"] = inputs["input_features"][0]

    if "attention_mask" in inputs:
        batch["attention_mask"] = inputs["attention_mask"][0]

    # Labels
    labels = processor.tokenizer(
        batch["transcription"],
        return_tensors="pt",
        padding="longest",
        truncation=True
    )
    batch["labels"] = labels.input_ids[0]

    return batch

# 6. Dataset'i map et
train_ds = train_ds.map(prepare_batch, remove_columns=train_ds.column_names, num_proc=1)
val_ds = val_ds.map(prepare_batch, remove_columns=val_ds.column_names, num_proc=1)

# 7. WER Metric
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # -100 olanları pad token ile değiştir
    label_ids = np.where(label_ids != -100, label_ids, processor.tokenizer.pad_token_id)

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# 8. Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-turkish-5k",
    eval_strategy="steps",
    eval_steps=200,
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    learning_rate=1e-5,
    predict_with_generate=True,
    fp16=True,
    report_to="none",
)

# 9. Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor  # artık tokenizer yerine bu kullanılıyor
)

# 10. Eğitim başlat
trainer.train()

# 11. Inference örneği
test_example = val_ds[0]
audio = test_example["input_features"].unsqueeze(0)

attention_mask = test_example.get("attention_mask", None)
if attention_mask is not None:
    attention_mask = attention_mask.unsqueeze(0)

generated_ids = model.generate(input_features=audio, attention_mask=attention_mask)
pred = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("Gerçek:", processor.tokenizer.decode(test_example["labels"], skip_special_tokens=True))
print("Tahmin:", pred)