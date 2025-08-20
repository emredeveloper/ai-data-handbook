import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import re

# 1. Harmony Chat Template ve Tokenler
HARMONY_CHAT_TEMPLATE = """
{%- for message in messages -%}
    {%- if message['role'] == 'system' -%}
<|start|>system<|message|>{{ message['content'] }}<|end|>
    {%- elif message['role'] == 'user' -%}
<|start|>user<|message|>{{ message['content'] }}<|end|>
    {%- elif message['role'] == 'assistant' -%}
        {%- if message.get('channel') == 'analysis' -%}
<|start|>assistant<|channel|>analysis<|message|>{{ message['content'] }}<|end|>
        {%- else -%}
<|start|>assistant<|channel|>final<|message|>{{ message['content'] }}<|end|>
        {%- endif -%}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
<|start|>assistant<|channel|>
{%- endif -%}
"""

SPECIAL_TOKENS = {
    "additional_special_tokens": [
        "<|start|>", "<|end|>", "<|message|>", "<|channel|>",
        "analysis", "final", "system", "user", "assistant"
    ]
}

# 2. Model ve Tokenizer Hazırlama
model_name = "meta-llama/Llama-3.2-1B"  # Küçük model

# Try to use bitsandbytes 4-bit quantization if available. On some systems
# (especially plain Windows installs) the `bitsandbytes` package may not
# be installed or available. In that case we'll fall back to a non-quantized
# load to avoid the importlib.metadata.PackageNotFoundError seen earlier.
try:
    # Quick import check for bitsandbytes
    import bitsandbytes  # type: ignore
    # Choose a compute dtype that exists on the current torch build
    compute_dtype = getattr(torch, "bfloat16", None) or getattr(torch, "float16", torch.float32)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
except Exception:
    # If bitsandbytes is not available, warn and load without quantization.
    print("[warning] bitsandbytes not found or failed to initialize — loading model without 4-bit quantization.\n"
          "To enable 4-bit loading, install bitsandbytes and ensure CUDA toolchain is compatible.")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto"
    )

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens(SPECIAL_TOKENS)
model.resize_token_embeddings(len(tokenizer))
tokenizer.chat_template = HARMONY_CHAT_TEMPLATE
tokenizer.pad_token = tokenizer.eos_token

# 3. Veri Seti Oluşturma
def create_dataset():
    examples = [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": "Write a Python function to calculate factorial"},
                {"role": "assistant", "channel": "analysis", 
                 "content": "I need to create a factorial function. Factorial of n is n! = n × (n-1) × ... × 1"},
                {"role": "assistant", "channel": "final",
                 "content": "```python\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)\n```"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Explain recursion"},
                {"role": "assistant", "channel": "analysis",
                 "content": "Recursion is when a function calls itself. Need to explain base case and recursive case."},
                {"role": "assistant", "channel": "final",
                 "content": "Recursion is a programming technique where a function calls itself to solve smaller instances of the same problem. It requires:\n1. Base case: stops recursion\n2. Recursive case: function calls itself with modified parameters"}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is a bubble sort?"},
                {"role": "assistant", "channel": "analysis",
                 "content": "Bubble sort is a simple sorting algorithm. I should explain how it works and provide code."},
                {"role": "assistant", "channel": "final",
                 "content": "Bubble sort repeatedly compares adjacent elements and swaps them if they're in wrong order:\n```python\ndef bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr\n```"}
            ]
        }
    ]
    
    formatted = []
    for ex in examples:
        text = tokenizer.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)
        formatted.append({"text": text})
    
    return Dataset.from_list(formatted)

train_dataset = create_dataset()

# 4. LoRA Configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# 5. Training
training_args = TrainingArguments(
    output_dir="./harmony-model",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    logging_steps=10,
    learning_rate=2e-4,
    fp16=True,
    save_strategy="epoch",
    optim="paged_adamw_8bit",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    args=training_args,
    dataset_text_field="text",
    max_seq_length=512,
)

# 6. Fine-tuning Başlat
trainer.train()
trainer.save_model("./harmony-final")

# 7. Test Fonksiyonu
def generate_harmony(prompt, show_analysis=True):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Parse channels
    analysis = re.search(r'<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>', response, re.DOTALL)
    final = re.search(r'<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|$)', response, re.DOTALL)
    
    if show_analysis and analysis:
        print("🧠 Analysis:", analysis.group(1).strip())
        print("-" * 50)
    
    if final:
        print("💬 Response:", final.group(1).strip())
    
    return {
        "analysis": analysis.group(1).strip() if analysis else None,
        "final": final.group(1).strip() if final else None
    }

# 8. Test Et
print("Model eğitildi! Test ediliyor...\n")
generate_harmony("Write a function to reverse a string")