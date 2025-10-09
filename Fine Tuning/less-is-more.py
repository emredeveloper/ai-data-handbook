import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import copy
import os
try:
    from datasets import load_dataset
except ImportError as e:
    raise ImportError(
        "datasets paketi gerekli. Kurulum: pip install datasets"
    )


def print_trm_hyperparam_guidelines():
    """Eğitim başlangıcında TRM hiperparametre önerilerini yazdırır."""
    guidelines = {
        "n_recursions (n)": "3–8",
        "n_deep_recursions (T)": "2–6",
        "n_supervision_steps": "4–16",
        "halt_lambda": "0.3–1.0",
        "step_mu (ponder)": "0.003–0.03",
        "grad_clip_norm": "1.0",
        "ema_decay": "0.999–0.9999",
    }
    print("\n🔧 TRM Hiperparametre Önerileri (aralıklar):")
    for k, v in guidelines.items():
        print(f"  - {k}: {v}")
    print()
    return guidelines

# ============================================
# TRM (Tiny Recursive Model) Implementasyonu
# ============================================

class TinyRecursiveNetwork(nn.Module):
    """2 katmanlı küçük ağ - hem z hem de y güncellemelerini yapar"""
    def __init__(self, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # SwiGLU MLP Katmanları (makaledeki gibi)
        # SwiGLU: hidden_dim * 4 intermediate size kullanır
        self.layer1_gate = nn.Linear(hidden_dim, hidden_dim * 4)
        self.layer1_up = nn.Linear(hidden_dim, hidden_dim * 4)
        self.layer2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_features):
        # input_features: [B, L, D]
        residual = input_features

        # Pre-Norm
        x_norm = self.norm1(input_features)

        # Katman 1 (SwiGLU)
        gate = self.layer1_gate(x_norm)
        up = self.layer1_up(x_norm)
        x = F.silu(gate) * up
        x = self.dropout(x)
        
        # Katman 2 (Down projeksiyon)
        x = self.layer2(x)
        x = self.dropout(x)
        
        # Residual bağlantı (post-norm yok, pre-norm kullanıldı)
        x = x + residual
        
        return x


class TRM(nn.Module):
    """Tiny Recursive Model - Makaledeki ana model"""
    def __init__(self, vocab_size=10, seq_len=81, hidden_dim=128, 
                 n_recursions=6, n_deep_recursions=3):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.n = n_recursions  # Her supervision step'te kaç recursion
        self.T = n_deep_recursions  # Gradient olmadan kaç kez recurse
        
        # Embedding katmanları
        self.input_embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # Tek bir tiny network (makaleye göre)
        self.net = TinyRecursiveNetwork(hidden_dim)

        # x,y,z birleştirme projeksiyonları
        self.combine_proj = nn.Linear(hidden_dim * 3, hidden_dim)
        self.yz_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Output head
        self.output_head = nn.Linear(hidden_dim, vocab_size)
        
        # Q-learning için halting head
        self.q_head = nn.Linear(hidden_dim, 1)
        
        # Learnable initial embeddings
        self.z_init = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.02)
        self.y_init = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))
        
    def latent_recursion(self, x, y, z, n):
        """z'yi n kez recursive olarak güncelle, sonra y'yi güncelle"""
        # n kez z'yi güncelle
        for i in range(n):
            # x, y, z'yi birleştir
            combined = torch.cat([x, y, z], dim=-1)
            combined = self.combine_proj(combined)
            z = self.net(combined)
        
        # y'yi güncelle (sadece y ve z kullanarak)
        y_input = torch.cat([y, z], dim=-1)
        y_input = self.yz_proj(y_input)
        y = self.net(y_input)
        
        return y, z
    
    def deep_recursion(self, x, y, z, n, T):
        """
        T-1 kez gradient olmadan recurse et,
        son kez gradient ile recurse et
        """
        # T-1 kez gradient olmadan
        with torch.no_grad():
            for j in range(T - 1):
                y, z = self.latent_recursion(x, y, z, n)
        
        # 1 kez gradient ile
        y, z = self.latent_recursion(x, y, z, n)
        
        return y, z
    
    def forward(self, x_input, y_prev=None, z_prev=None, training=False):
        """
        x_input: [B, L] - Input token'lar
        y_prev: [B, L, D] - Önceki y embedding (None ise init kullan)
        z_prev: [B, L, D] - Önceki z latent (None ise init kullan)
        """
        batch_size = x_input.shape[0]
        
        # Input embedding
        x = self.input_embedding(x_input)  # [B, L, D]
        
        # Initialize y ve z
        if y_prev is None:
            y = self.y_init.expand(batch_size, -1, -1).to(x.device)
        else:
            y = y_prev
            
        if z_prev is None:
            z = self.z_init.expand(batch_size, -1, -1)
        else:
            z = z_prev
        
        # Deep recursion
        y, z = self.deep_recursion(x, y, z, self.n, self.T)
        
        # Output prediction
        y_pred = self.output_head(y)  # [B, L, vocab_size]
        
        # Q-value for halting (board-level)
        q = torch.sigmoid(self.q_head(y.mean(dim=1)))  # [B, 1]
        
        return y_pred, y, z, q


# ============================================
# Sudoku Dataset (Basitleştirilmiş 4x4)
# ============================================

# 4x4 basit dataset kaldırıldı (makale ile hizalama için yalnızca 9x9 HF seti kullanıyoruz)


class SudokuHFDataset(Dataset):
    """Hugging Face 6elphegor/Sudoku veri seti sarmalayıcısı.
    Her örnek: 'partial' (boşlar None) ve 'complete' (çözüm) 9x9 listeler.
    Çıkış: input (81 uzunluk, 0..9; 0 boş), target (81 uzunluk, 1..9).
    """
    def __init__(self, hf_ds):
        self.ds = hf_ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[int(idx)]
        # Alan isimleri: 'partial' ve 'complete' varsayımı ile ilerliyoruz
        complete = ex.get('complete')
        partial = ex.get('partial')

        if complete is None or partial is None:
            # Alternatif alan adlarını deneyelim (bazı kopyalarda 'solution'/'puzzle')
            complete = ex.get('solution')
            partial = ex.get('puzzle')

        # Çözüm: 9x9 int -> [81]
        target = np.array(complete, dtype=np.int64).reshape(81)

        # Partial: None -> 0, int korunur
        partial_arr = []
        for row in partial:
            row_fixed = [0 if (v is None) else int(v) for v in row]
            partial_arr.append(row_fixed)
        input_flat = np.array(partial_arr, dtype=np.int64).reshape(81)

        return torch.LongTensor(input_flat), torch.LongTensor(target)


# ============================================
# Training Loop
# ============================================

def train_trm(model, train_loader, val_loader, n_epochs=50, 
              n_supervision_steps=4, device='cpu',
              halt_lambda=0.5, step_mu=0.01, warmup_iters=2000,
              gradient_accumulation_steps=1):
    """TRM modelini deep supervision ile eğit"""
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, 
                                  weight_decay=1.0, betas=(0.9, 0.95))
    
    # EMA için
    ema_model = copy.deepcopy(model)
    ema_decay = 0.999
    
    model.to(device)
    ema_model.to(device)
    
    train_losses = []
    val_accuracies = []
    global_step = 0
    
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0
        
        # Learning rate warmup (2K iterasyon)
        if global_step < warmup_iters:
            lr_scale = max(0.01, global_step / warmup_iters)  # En az %1 lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-4 * lr_scale
        
        # Warmup: ilk 2K iterasyonda halting ve ponder devre dışı
        hl = 0.0 if global_step < warmup_iters else halt_lambda
        sm = 0.0 if global_step < warmup_iters else step_mu
        
        for x_input, y_true in train_loader:
            x_input = x_input.to(device)
            y_true = y_true.to(device)
            
            # Deep supervision loop
            y_embedding = None
            z_latent = None
            total_loss = 0
            
            for step in range(n_supervision_steps):
                # Forward pass
                y_pred, y_embedding, z_latent, q = model(
                    x_input, y_embedding, z_latent, training=True
                )
                # Sadece 1-9 sınıflarını kullan (0 yasaklı)
                y_pred_valid = y_pred[..., 1:]  # [B, L, 9]
                
                # Masked prediction loss (sadece başlangıçta boş olan hücreler)
                mask = (x_input == 0).float()  # [B,L]
                ce_flat = F.cross_entropy(
                    y_pred_valid.reshape(-1, 9),
                    (y_true - 1).clamp(0, 8).reshape(-1),  # 1-9 -> 0-8
                    reduction='none'
                )  # [B*L]
                ce_masked = (ce_flat.view_as(mask) * mask).sum() / (mask.sum() + 1e-8)
                
                # Halting loss (doğru tahmin yapıldı mı?)
                y_pred_tokens = y_pred_valid.argmax(dim=-1) + 1  # 0-8 -> 1-9
                target_halt = ((y_pred_tokens == y_true).all(dim=1)).float().unsqueeze(-1)  # [B,1]
                halt_loss = F.binary_cross_entropy(q, target_halt)

                # Ponder/step cost: daha erken durmayı teşvik etmek için (1-q)
                step_cost = (1.0 - q).mean()
                
                # Total loss (gradient accumulation için normalize)
                loss = (ce_masked + hl * halt_loss + sm * step_cost) / gradient_accumulation_steps
                total_loss += loss.item() * gradient_accumulation_steps
                
                # Backward
                loss.backward()
                
                # Gradient accumulation: her N step'te optimizer adım at
                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                
                # EMA update
                with torch.no_grad():
                    for ema_param, param in zip(ema_model.parameters(), 
                                               model.parameters()):
                        ema_param.data.mul_(ema_decay).add_(
                            param.data, alpha=1 - ema_decay
                        )
                
                # Detach for next iteration
                y_embedding = y_embedding.detach()
                z_latent = z_latent.detach()
                # Eğitimde erken durdurma yok (makale uyumu için kaldırıldı)
            
            batch_loss = total_loss / (step + 1)
            epoch_loss += batch_loss
            n_batches += 1
            global_step += 1

            # Hafif ilerleme çıktısı (her 100 batch)
            if n_batches % 100 == 0:
                print(f"  batch {n_batches} | loss: {batch_loss:.4f} | lr: {optimizer.param_groups[0]['lr']:.2e}")
        
        avg_loss = epoch_loss / n_batches
        train_losses.append(avg_loss)
        
        # Validation
        val_acc = evaluate_trm(ema_model, val_loader, n_supervision_steps, device)
        val_accuracies.append(val_acc)
        
        print(f"Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2%}")
    
    return ema_model, train_losses, val_accuracies


def evaluate_trm(model, data_loader, n_supervision_steps=16, device='cpu', halt_threshold=0.5):
    """TRM modelini değerlendir (halting ile dinamik sonlandırma). Ayrıca hücre bazlı metrikleri toplar ve yazdırır."""
    model.eval()
    correct = 0
    total = 0
    cell_correct = 0
    cell_total = 0
    masked_cell_correct = 0
    masked_cell_total = 0
    
    with torch.no_grad():
        for x_input, y_true in data_loader:
            x_input = x_input.to(device)
            y_true = y_true.to(device)
            
            # Deep supervision + halting
            y_embedding = None
            z_latent = None
            halted = torch.zeros(x_input.size(0), 1, device=device, dtype=torch.bool)
            final_pred_tokens = torch.full_like(x_input, fill_value=0)
            
            for step in range(n_supervision_steps):
                y_pred, y_embedding, z_latent, q = model(
                    x_input, y_embedding, z_latent, training=False
                )
                # Sadece 1-9 sınıflarını kullan
                y_pred_valid = y_pred[..., 1:]
                
                step_pred_tokens = y_pred_valid.argmax(dim=-1) + 1  # 0-8 -> 1-9
                # Yeni haltingler
                new_halt = (q > halt_threshold)  # [B,1]
                halted = halted | new_halt
                
                # İlk kez halting olan örneklerde step tahminini sabitle
                just_halted = new_halt & (~halted ^ new_halt)
                final_pred_tokens = torch.where(just_halted, step_pred_tokens, final_pred_tokens)

                y_embedding = y_embedding.detach()
                z_latent = z_latent.detach()

                # Tüm hücreler halt ettiyse erken çık
                if halted.all():
                    break
            
            # Eğer bazı hücreler hiç halt etmediyse, son adımın tahminini kullan
            still_not_halted = (~halted)
            if still_not_halted.any():
                final_pred_tokens = torch.where(still_not_halted, step_pred_tokens, final_pred_tokens)
            
            # Hücre ve tahta metrikleri
            eq = (final_pred_tokens == y_true)
            correct += eq.all(dim=1).sum().item()  # tüm 81 doğru olan tahtalar
            total += y_true.shape[0]
            cell_correct += eq.sum().item()
            cell_total += eq.numel()
            masked = (x_input == 0)
            masked_cell_correct += (eq & masked).sum().item()
            masked_cell_total += masked.sum().item()
    
    board_acc = correct / max(1, total)
    cell_acc = cell_correct / max(1, cell_total)
    masked_cell_acc = masked_cell_correct / max(1, masked_cell_total)
    print(f"Val Board Acc: {board_acc:.2%} | Val Cell Acc: {cell_acc:.2%} | Val Masked Cell Acc: {masked_cell_acc:.2%}")
    return board_acc


# ============================================
# Main Execution
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("TRM (Tiny Recursive Model) - 9x9 Sudoku (Hugging Face) ")
    print("=" * 60)
    print_trm_hyperparam_guidelines()
    
    # Hyperparameters (Makale uyumlu)
    VOCAB_SIZE = 10  # 0 (boş) + 1..9
    SEQ_LEN = 81  # 9x9 sudoku
    HIDDEN_DIM = 640  # ~7M parametre için
    N_RECURSIONS = 6  # Makale n=6
    N_DEEP_RECURSIONS = 3  # Makale T=3 (depth=42)
    BATCH_SIZE = 32  # 8GB VRAM için (gradient accumulation ile efektif 768)
    GRADIENT_ACCUMULATION_STEPS = 24  # 32*24=768 efektif batch
    N_EPOCHS = 50  # Makale 60K epoch; başlangıç için 50
    N_SUPERVISION_STEPS_TRAIN = 8  # Eğitimde 8, değerlendirmede 16
    N_SUPERVISION_STEPS_EVAL = 16
    VAL_FRAC = 0.1
    TEST_FRAC = 0.1
    MAX_TRAIN = 20000   # Veri augmentasyonu sonra eklenecek
    MAX_VAL = 2000
    MAX_TEST = 2000

    # Hızlı başlangıç modu (CPU'da ilk deneme için önerilir)
    FAST_START = False
    if FAST_START:
        BATCH_SIZE = 32
        N_EPOCHS = 3
        N_SUPERVISION_STEPS = 4
        MAX_TRAIN = 5000
        MAX_VAL = 1000
        MAX_TEST = 1000

    # Cihaz seçimi ve CPU iş parçacığı sınırı
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        try:
            torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
        except Exception:
            pass
    print(f"\nDevice: {device}")
    
    # Hugging Face Sudoku veri seti (yalnızca ihtiyaç kadar örnek indir)
    print("\n📊 Hugging Face Sudoku veri seti yükleniyor...")
    total_needed = (MAX_TRAIN or 0) + (MAX_VAL or 0) + (MAX_TEST or 0)
    if total_needed > 0:
        full = load_dataset("6elphegor/Sudoku", split=f"train[:{total_needed}]").shuffle(seed=42)
        # FAST_START modunda daha az boş hücreli örnekler ile başla (hedef: kolaylaştırma)
        if FAST_START:
            def count_zeros(ex):
                partial = ex.get('partial') or ex.get('puzzle')
                zeros = 0
                for row in partial:
                    for v in row:
                        zeros += 1 if (v is None or int(v) == 0) else 0
                return {"zeros": zeros}
            full = full.map(count_zeros)
            full = full.filter(lambda ex: ex["zeros"] <= 60)
        n_full = len(full)
        n_train = min(MAX_TRAIN, n_full)
        n_val = min(MAX_VAL, max(0, n_full - n_train))
        n_test = min(MAX_TEST, max(0, n_full - n_train - n_val))
        train_ds = full.select(range(0, n_train))
        val_ds = full.select(range(n_train, n_train + n_val))
        test_ds = full.select(range(n_train + n_val, n_train + n_val + n_test))
    else:
        raw = load_dataset("6elphegor/Sudoku")
        full = raw["train"].shuffle(seed=42)
        splits = full.train_test_split(test_size=TEST_FRAC, seed=42)
        train_val_ds, test_ds = splits["train"], splits["test"]
        splits2 = train_val_ds.train_test_split(
            test_size=VAL_FRAC / (1.0 - TEST_FRAC), seed=42
        )
        train_ds, val_ds = splits2["train"], splits2["test"]

    train_dataset = SudokuHFDataset(train_ds)
    val_dataset = SudokuHFDataset(val_ds)
    test_dataset = SudokuHFDataset(test_ds)

    pin = (device.type == 'cuda')
    num_workers = 0
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=num_workers, pin_memory=pin, persistent_workers=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        num_workers=num_workers, pin_memory=pin, persistent_workers=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        num_workers=num_workers, pin_memory=pin, persistent_workers=False
    )
    
    # Model oluştur
    print(f"\n🧠 Model oluşturuluyor...")
    model = TRM(
        vocab_size=VOCAB_SIZE,
        seq_len=SEQ_LEN,
        hidden_dim=HIDDEN_DIM,
        n_recursions=N_RECURSIONS,
        n_deep_recursions=N_DEEP_RECURSIONS
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parametre sayısı: {n_params:,}")
    print(f"Recursion depth: {N_RECURSIONS * N_DEEP_RECURSIONS * 2}")
    
    # Training
    print(f"\n🚀 Eğitim başlıyor ({N_EPOCHS} epoch)...")
    print(f"Efektif batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"Recursion depth: {N_RECURSIONS * N_DEEP_RECURSIONS * 2}")
    trained_model, losses, val_accs = train_trm(
        model, train_loader, val_loader,
        n_epochs=N_EPOCHS,
        n_supervision_steps=N_SUPERVISION_STEPS_TRAIN,
        device=device,
        halt_lambda=0.5,
        step_mu=0.01,
        warmup_iters=2000,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS
    )
    
    # Test
    print("\n" + "=" * 60)
    print("📈 TEST SONUÇLARI")
    print("=" * 60)
    
    test_acc = evaluate_trm(trained_model, test_loader, 
                           n_supervision_steps=N_SUPERVISION_STEPS_EVAL, device=device)
    print(f"\nTest Accuracy: {test_acc:.2%}")
    print(f"En iyi Validation Accuracy: {max(val_accs):.2%}")
    
    # Örnek tahmin göster
    print("\n" + "=" * 60)
    print("🔍 ÖRNEK TAHMİN")
    print("=" * 60)
    
    trained_model.eval()
    with torch.no_grad():
        sample_input, sample_target = test_dataset[0]
        sample_input = sample_input.unsqueeze(0).to(device)
        sample_target = sample_target.unsqueeze(0).to(device)
        
        y_embedding = None
        z_latent = None
        
        for step in range(N_SUPERVISION_STEPS_EVAL):
            y_pred, y_embedding, z_latent, q = trained_model(
                sample_input, y_embedding, z_latent, training=False
            )
            y_embedding = y_embedding.detach()
            z_latent = z_latent.detach()
        
        pred_tokens = y_pred.argmax(dim=-1)[0].cpu().numpy()
        input_tokens = sample_input[0].cpu().numpy()
        target_tokens = sample_target[0].cpu().numpy()
        
        print("\nInput (0 = boş):")
        print(input_tokens.reshape(9, 9))
        print("\nTarget (Doğru Çözüm):")
        print(target_tokens.reshape(9, 9))
        print("\nPrediction (Model Tahmini):")
        print(pred_tokens.reshape(9, 9))
        print(f"\nDoğru mu? {np.array_equal(pred_tokens, target_tokens)}")
    
    print("\n" + "=" * 60)
    print("✅ Tamamlandı!")
    print("=" * 60)