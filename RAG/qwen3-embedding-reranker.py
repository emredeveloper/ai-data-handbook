import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import numpy as np

# 1. Qwen3-Embedding-0.6B Modelini Yükle
embedding_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
embedding_model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B")

# 2. Qwen3-Reranker-0.6B Modelini Yükle
reranker_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B", padding_side='left')
reranker_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B").eval()

# 3. Belgeleri Yükle

def load_documents(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        docs = [line.strip() for line in f if line.strip()]
    return docs

documents = load_documents("belgeler.txt")

# 4. Belgelerin Embeddinglerini Hesapla

def get_embeddings(texts):
    inputs = embedding_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = embedding_model(**inputs)
        # Qwen3 embedding çıkışı: last_hidden_state[:, 0, :] (CLS token)
        embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings

doc_embeddings = get_embeddings(documents)

# 5. Sorgu Al
query = input("Sorgunuzu girin: ")

# 6. Sorgu Embeddingini Hesapla
query_embedding = get_embeddings([query])[0]

# 7. En Yakın Belgeleri Bul (ilk 5)
cos_scores = torch.nn.functional.cosine_similarity(query_embedding.unsqueeze(0), doc_embeddings)
top_k = 5
values, indices = torch.topk(cos_scores, k=top_k)
top_docs = [documents[i] for i in indices.tolist()]

# 8. Qwen3-Reranker ile Sıralama

def format_instruction(instruction, query, doc):
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

def process_inputs(pairs, tokenizer, max_length, prefix_tokens, suffix_tokens):
    inputs = tokenizer(
        pairs, padding=False, truncation='longest_first',
        return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
    )
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
    for key in inputs:
        inputs[key] = inputs[key].to(reranker_model.device)
    return inputs

@torch.no_grad()
def compute_logits(inputs, model, token_true_id, token_false_id):
    batch_scores = model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    scores = batch_scores[:, 1].exp().tolist()
    return scores

# Reranker ayarları
prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
max_length = 8192
prefix_tokens = reranker_tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = reranker_tokenizer.encode(suffix, add_special_tokens=False)
token_true_id = reranker_tokenizer.convert_tokens_to_ids("yes")
token_false_id = reranker_tokenizer.convert_tokens_to_ids("no")

instruction = 'Given a web search query, retrieve relevant passages that answer the query'
pairs = [format_instruction(instruction, query, doc) for doc in top_docs]
inputs = process_inputs(pairs, reranker_tokenizer, max_length, prefix_tokens, suffix_tokens)
scores = compute_logits(inputs, reranker_model, token_true_id, token_false_id)

# Sonuçları skor ile sırala
top_reranked = sorted(zip(top_docs, scores), key=lambda x: x[1], reverse=True)

print("\nReranker Sonuçları:")
for i, (doc, score) in enumerate(top_reranked, 1):
    print(f"{i}. Skor: {score:.4f} - {doc}")
