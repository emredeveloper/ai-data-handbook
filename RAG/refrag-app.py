# -*- coding: utf-8 -*-
import os
import io
import time
import math
from typing import List

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

# --- PDF çıkarım yardımcıları ---
@st.cache_data(show_spinner=False)
def extract_text_pypdf2(file_bytes: bytes) -> List[str]:
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        return []
    pages = []
    reader = PdfReader(io.BytesIO(file_bytes))
    for pg in reader.pages:
        try:
            txt = pg.extract_text() or ""
        except Exception:
            txt = ""
        pages.append(txt)
    return pages

@st.cache_data(show_spinner=False)
def extract_text_pdfminer(file_bytes: bytes) -> str:
    try:
        from pdfminer.high_level import extract_text
    except ImportError:
        return ""
    bio = io.BytesIO(file_bytes)
    try:
        text = extract_text(bio) or ""
    except Exception:
        text = ""
    return text

# --- Model / embedding yükleme ---
@st.cache_resource(show_spinner=False)
def load_qwen_generator(model_name: str, dtype: str, device: str):
    torch_dtype = torch.float16 if (dtype == "float16" and device.startswith("cuda")) else torch.float32
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto" if device.startswith("cuda") else None,
    )
    if not device.startswith("cuda"):
        model.to(device)
    model.eval()
    return tok, model

@st.cache_resource(show_spinner=False)
def load_qwen_embedding(model_name: str, device: str):
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    mdl = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    if device:
        mdl.to(device)
    mdl.eval()
    return tok, mdl

@st.cache_data(show_spinner=False)
def embed_texts(texts: List[str], emb_tok_name: str, emb_model_name: str, device: str, batch_size: int = 16, normalize: bool = True):
    tok = AutoTokenizer.from_pretrained(emb_tok_name, trust_remote_code=True)
    mdl = AutoModel.from_pretrained(emb_model_name, trust_remote_code=True)
    if device:
        mdl.to(device)
    mdl.eval()

    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_tok = tok(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = mdl(**batch_tok)
            h = out.last_hidden_state
            mask = batch_tok.attention_mask.unsqueeze(-1)
            mean = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            if normalize:
                mean = F.normalize(mean, dim=-1)
        vecs.append(mean.detach().cpu().numpy())
    return np.concatenate(vecs, axis=0)

@st.cache_data(show_spinner=False)
def build_tfidf(texts: List[str]):
    vec = TfidfVectorizer(max_features=50000, ngram_range=(1,2))
    mat = vec.fit_transform(texts)
    return vec, mat

# --- Reranker ---
class LightCrossEncoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim*4, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
        )
    def forward(self, q: torch.Tensor, p: torch.Tensor):
        z = torch.cat([q, p, torch.abs(q-p), q*p], dim=-1)
        return self.net(z).squeeze(-1)

@st.cache_resource(show_spinner=False)
def get_reranker(emb_dim: int):
    m = LightCrossEncoder(emb_dim)
    m.eval()
    return m

# --- Chunk + Projector + Decoder body ---
class ChunkEncoder(nn.Module):
    def __init__(self, d_model: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, token_embs: torch.Tensor):
        pooled = token_embs.mean(dim=1)
        z = self.net(pooled)
        return F.normalize(z, dim=-1)

class Projector(nn.Module):
    def __init__(self, in_dim: int, d_model: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, d_model)
    def forward(self, z):
        return self.lin(z)

def build_positional(d_model: int, max_len: int=65536):
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)

def apply_decoder_body(emb: torch.Tensor, d_model: int):
    enc_layer = nn.TransformerEncoderLayer(d_model, 8, 4*d_model, batch_first=True)
    tr = nn.TransformerEncoder(enc_layer, num_layers=2).to(emb.device)
    L = emb.size(1)
    mask = torch.full((L, L), float('-inf'), device=emb.device)
    mask = torch.triu(mask, diagonal=1)
    pos = build_positional(d_model).to(emb.device)
    x = emb + pos[:, :L, :]
    return tr(x, mask=mask)

# --- Yardımcı fonksiyonlar ---
def sliding_chunks(ids: List[int], k: int, stride: int):
    out = []
    i = 0
    n = len(ids)
    while i < n:
        out.append(ids[i:i+k])
        if i + k >= n:
            break
        i += stride
    return out

def kv_proxy(L: int, D: int):
    return L * L * D

def ttft_proxy(fn, *args, reps: int = 2):
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        _ = fn(*args)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return float(sum(times) / len(times))

# --- UI / Ayarlar ---
st.sidebar.title("⚙️ Ayarlar")
device = st.sidebar.selectbox("Cihaz", ["cuda" if torch.cuda.is_available() else "cpu", "cpu"])
dtype = st.sidebar.selectbox("dtype", ["float16", "float32"], index=0 if device == "cuda" else 1)
gen_model_name = st.sidebar.text_input("Üretici Model (Qwen)", "Qwen/Qwen3-0.6B")
emb_model_name = st.sidebar.text_input("Embedding Model", "Qwen/Qwen3-Embedding-0.6B")
M = st.sidebar.number_input("Top-M (doküman)", 1, 64, 16, step=1)
k = st.sidebar.number_input("Chunk token uzunluğu (k)", 8, 2048, 64, step=8)
stride = st.sidebar.number_input("Stride", 4, 2048, 64, step=4)
qlen = st.sidebar.number_input("Query token sınırı", 8, 4096, 128, step=8)
budget = st.sidebar.number_input("Token bütçesi (Selective)", 64, 8192, 512, step=64)
bandit_lam = st.sidebar.slider("Bandit λ (gecikme cezası)", 0.0, 2.0, 0.5, 0.05)
bandit_lr = st.sidebar.slider("Bandit öğrenme hızı", 0.01, 1.0, 0.1, 0.01)
max_e = st.sidebar.number_input("Maks e", 0, 32, 6, step=1)

pdf_mode = st.sidebar.selectbox("PDF yükleme modu", ["Tek", "Çoklu"])
pdf_as_pages = st.sidebar.checkbox("PDF’i sayfa sayfa chunk’la", value=True)
pdf_extractor = st.sidebar.selectbox("PDF metin çıkarıcı", ["PyPDF2", "pdfminer.six"], index=0)

st.title("REFRAG UI — Qwen3 + Embedding + Reranker + Bandit")
# (Altındaki açıklama yazısı kaldırıldı)

colL, colR = st.columns([2,1])
with colL:
    st.subheader("📄 Belgeler / Yükleme")
    accept_multi = (pdf_mode == "Çoklu")
    uploads = st.file_uploader("TXT veya PDF yükle", type=["txt", "pdf"], accept_multiple_files=accept_multi)

    texts: List[str] = []
    if uploads:
        files = uploads if isinstance(uploads, list) else [uploads]
        for f in files:
            name = f.name.lower()
            data = f.read()
            if name.endswith(".txt"):
                try:
                    texts.append(data.decode("utf-8", errors="ignore"))
                except Exception:
                    texts.append("")
            elif name.endswith(".pdf"):
                if pdf_extractor == "PyPDF2":
                    pages = extract_text_pypdf2(data)
                    if pdf_as_pages:
                        texts.extend(pages)
                    else:
                        texts.append("\n".join(pages))
                else:
                    txt = extract_text_pdfminer(data)
                    texts.append(txt)
    else:
        st.info("PDF / TXT dosyalarını yükleyin.")
        texts = [
            "REFRAG fikri, RAG bağlamını sıkıştırıp gerektiğinde genişleterek verim artırır.",
            "Prefix-tuning, yumuşak vektörlerle modelin davranışını yönlendirir.",
            "KV cache, üretken modellerde token token üretimini hızlandırmak için kullanılır.",
            "Reranking, retrieval sonrası aday sıralamasını iyileştirir."
        ]

with colR:
    st.subheader("ℹ️ İpuçları")
    st.markdown(
        "- transformers ≥ 4.51 olmalı.  \n"
        "- PDF çıkarım için PyPDF2 veya pdfminer seçilebilir."
    )

with st.form("query_form", clear_on_submit=False):
    query = st.text_input("🔎 Sorgu", "RAG performansını uzun bağlamda nasıl hızlandırabilirim?")
    submitted = st.form_submit_button("🚀 Çalıştır")

if submitted:
    # Modeller yükleniyor
    with st.spinner("Modeller yükleniyor..."):
        gen_tok, gen_model = load_qwen_generator(gen_model_name, dtype, device)
        emb_tok, emb_model = load_qwen_embedding(emb_model_name, device)

    # TF-IDF index
    with st.spinner("TF-IDF index hazırlanıyor..."):
        vec, mat = build_tfidf(texts)
        qv = vec.transform([query])
        sims = cosine_similarity(qv, mat)[0]
        top_idx = np.argsort(-sims)[:M]
        cand_docs = [texts[i] for i in top_idx]

    # Reranking
    with st.spinner("Reranking yapılıyor..."):
        q_emb = embed_texts([query], emb_model_name, emb_model_name, device)
        p_emb = embed_texts(cand_docs, emb_model_name, emb_model_name, device)
        emb_dim = q_emb.shape[-1]
        reranker = get_reranker(emb_dim)
        q_t = torch.tensor(q_emb[0], dtype=torch.float32).to(device).unsqueeze(0).repeat(len(cand_docs), 1)
        p_t = torch.tensor(p_emb, dtype=torch.float32).to(device)
        with torch.no_grad():
            scores = reranker(q_t, p_t).detach().cpu().numpy()
        order = np.argsort(-scores)
        cand_docs = [cand_docs[i] for i in order]

    # Chunk’lama
    chunks: List[List[int]] = []
    for d in cand_docs[:M]:
        ids = gen_tok.encode(d, add_special_tokens=False)
        parts = sliding_chunks(ids, k, stride)
        if parts:
            chunks.append(parts[0])

    def embed_ids(ids_2d: torch.Tensor):
        if ids_2d.dim() == 1:
            ids_2d = ids_2d.unsqueeze(0)
        return gen_model.get_input_embeddings()(ids_2d)

    q_ids = gen_tok.encode(query, add_special_tokens=False)[:qlen]
    q_ids_t = torch.tensor(q_ids, dtype=torch.long, device=device).unsqueeze(0)
    q_emb_tok = embed_ids(q_ids_t)

    d_model = int(gen_model.get_input_embeddings().weight.shape[-1])
    chunk_encoder = ChunkEncoder(d_model, hidden=4*d_model, out_dim=d_model//2).to(device)
    projector = Projector(d_model//2, d_model).to(device)

    # Baseline yolu
    base_seq = []
    for ch in chunks:
        base_seq.extend(ch)
    base_seq.extend(q_ids)
    base_ids = torch.tensor(base_seq, dtype=torch.long, device=device).unsqueeze(0)
    base_emb = embed_ids(base_ids)
    base_ttft = ttft_proxy(lambda args_emb: apply_decoder_body(args_emb, d_model), base_emb)
    base_kv = kv_proxy(base_emb.size(1), d_model)

    # Compact yolu
    if chunks:
        ch_stack = torch.tensor(chunks, dtype=torch.long, device=device)
        ch_emb_tok = embed_ids(ch_stack)
        z = chunk_encoder(ch_emb_tok)
        soft = projector(z).unsqueeze(0)
        comp_x = torch.cat([soft, q_emb_tok], dim=1)
        comp_ttft = ttft_proxy(lambda args_emb: apply_decoder_body(args_emb, d_model), comp_x)
        comp_kv = kv_proxy(comp_x.size(1), d_model)
    else:
        comp_x, comp_ttft, comp_kv = base_emb, base_ttft, base_kv

    # Seçici + bandit
    theoretic_max_e = max(0, min(len(chunks), int((budget - qlen - len(chunks)) // (k - 1)) if k>1 else 0))
    Emax = max(0, min(theoretic_max_e, max_e))
    if "bandit_prefs" not in st.session_state or len(st.session_state.bandit_prefs) != (Emax+1):
        st.session_state.bandit_prefs = torch.zeros(Emax+1, dtype=torch.float32)
    probs = torch.softmax(st.session_state.bandit_prefs, dim=0).cpu().numpy()
    e = int(np.random.choice(np.arange(Emax+1), p=probs)) if Emax>0 else 0

    ch_emb_tok = embed_ids(torch.tensor(chunks, dtype=torch.long, device=device)) if chunks else torch.empty(0,0,d_model, device=device)
    z = chunk_encoder(ch_emb_tok) if chunks else torch.empty(0, d_model//2, device=device)
    soft = projector(z) if chunks else torch.empty(0, d_model, device=device)

    q_repr = q_emb_tok[:, -1, :].squeeze(0)
    c = F.normalize(soft, dim=-1) if soft.numel()>0 else torch.empty_like(soft)
    qn = F.normalize(q_repr, dim=-1)
    rel = (c * qn).sum(-1).detach() if soft.numel()>0 else torch.tensor([])

    if e>0 and len(chunks)>0:
        top_idx = torch.topk(rel, k=min(e, len(chunks))).indices.tolist()
    else:
        top_idx = []

    M_now = len(chunks)
    mask_soft = torch.ones(M_now, dtype=torch.bool, device=device)
    for i in top_idx:
        mask_soft[i] = False

    soft_part = soft[mask_soft].unsqueeze(0) if M_now>0 else torch.empty(1,0,d_model, device=device)
    if e>0:
        full_tok = torch.tensor([chunks[i] for i in range(M_now) if not mask_soft[i]], dtype=torch.long, device=device).reshape(1, -1)
        full_emb = embed_ids(full_tok)
    else:
        full_emb = torch.empty(1,0,d_model, device=device)
    sel_x = torch.cat([soft_part, full_emb, q_emb_tok], dim=1)
    sel_ttft = ttft_proxy(lambda args_emb: apply_decoder_body(args_emb, d_model), sel_x)
    sel_kv = kv_proxy(sel_x.size(1), d_model)
    qual = float(rel[top_idx].mean().item()) if top_idx else float(rel.mean().item()) if rel.numel()>0 else 0.0
    reward = qual - bandit_lam * sel_ttft

    base_val = float(st.session_state.bandit_prefs.mean().item()) if len(st.session_state.bandit_prefs)>0 else 0.0
    adv = reward - base_val
    if len(st.session_state.bandit_prefs) > e:
        st.session_state.bandit_prefs[e] += bandit_lr * adv

    st.subheader("🔬 Sonuçlar (proxy metrikler)")
    st.write(f"Seçilen e = {e} | Emax = {Emax} | Reward ≈ {reward:.4f} | Qual ≈ {qual:.4f}")

    import pandas as pd
    rows = [
        {"Yol":"Baseline", "L": int(base_emb.size(1)), "TTFT (ms)": round(base_ttft*1000,2), "KV": f"{base_kv:.2e}"},
        {"Yol":"Compact", "L": int(comp_x.size(1)), "TTFT (ms)": round(comp_ttft*1000,2), "KV": f"{comp_kv:.2e}"},
        {"Yol":f"Selective (e={e})", "L": int(sel_x.size(1)), "TTFT (ms)": round(sel_ttft*1000,2), "KV": f"{sel_kv:.2e}"},
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    with st.expander("🔎 Reranker ilk 10"):
        show = min(10, len(cand_docs))
        for i in range(show):
            st.markdown(f"**{i+1}.** {cand_docs[i][:300]}{'...' if len(cand_docs[i])>300 else ''}")