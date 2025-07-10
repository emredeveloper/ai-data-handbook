import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Veri seti oluştur
np.random.seed(42)
X, y = make_regression(n_samples=100, n_features=10, n_informative=5, 
                       noise=0.1, random_state=42)

# Veriyi eğitim ve test setlerine böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Veriyi standartlaştır
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Farklı alpha değerleri için LASSO modeli
alpha_values = [0.001, 0.01, 0.1, 1, 10, 100]
lasso_coefficients = []

print("LASSO Katsayıları (Alpha Değerlerine Göre)")
print("=" * 50)

for alpha in alpha_values:
    lasso = Lasso(alpha=alpha, random_state=42)
    lasso.fit(X_train_scaled, y_train)
    lasso_coefficients.append(lasso.coef_)
    
    # Sıfır olmayan katsayıları say
    non_zero_features = np.sum(np.abs(lasso.coef_) > 1e-6)
    
    print(f"Alpha = {alpha:6.3f} -> Aktif Özellik Sayısı: {non_zero_features:2d}")

# Normal Linear Regresyon ile karşılaştırma
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

print(f"\nNormal Regresyon -> Aktif Özellik Sayısı: {len(lr.coef_):2d}")

# Görselleştirme
plt.figure(figsize=(15, 5))

# 1. Alt grafik: Katsayıların Alpha'ya göre değişimi
plt.subplot(1, 3, 1)
lasso_coefficients = np.array(lasso_coefficients)
for i in range(lasso_coefficients.shape[1]):
    plt.plot(alpha_values, lasso_coefficients[:, i], 'o-', label=f'Özellik {i+1}')
plt.xscale('log')
plt.xlabel('Alpha (log scale)')
plt.ylabel('Katsayı Değeri')
plt.title('LASSO Katsayılarının Alpha ile Değişimi')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

# 2. Alt grafik: Özellik Seçimi
plt.subplot(1, 3, 2)
feature_counts = [np.sum(np.abs(coef) > 1e-6) for coef in lasso_coefficients]
plt.plot(alpha_values, feature_counts, 'ro-', linewidth=2, markersize=8)
plt.xscale('log')
plt.xlabel('Alpha (log scale)')
plt.ylabel('Seçilen Özellik Sayısı')
plt.title('Özellik Seçimi (Alpha arttıkça azalır)')
plt.grid(True, alpha=0.3)

# 3. Alt grafik: Katsayılar Isı Haritası
plt.subplot(1, 3, 3)
plt.imshow(lasso_coefficients.T, aspect='auto', cmap='RdBu_r')
plt.colorbar(label='Katsayı Değeri')
plt.xlabel('Alpha İndeksi')
plt.ylabel('Özellik İndeksi')
plt.title('LASSO Katsayıları Isı Haritası')
plt.xticks(range(len(alpha_values)), [f'{a:.3f}' for a in alpha_values], rotation=45)

plt.tight_layout()
plt.show()

# Model performansı karşılaştırması
print("\n" + "="*60)
print("MODEL PERFORMANSI KARŞILAŞTIRMASI")
print("="*60)

best_alpha = 0.1  # En iyi alpha değeri (örnekte)
lasso_best = Lasso(alpha=best_alpha, random_state=42)
lasso_best.fit(X_train_scaled, y_train)

# Test skorları
lr_score = lr.score(X_test_scaled, y_test)
lasso_score = lasso_best.score(X_test_scaled, y_test)

print(f"Normal Regresyon R² Skoru: {lr_score:.4f}")
print(f"LASSO R² Skoru (α={best_alpha}): {lasso_score:.4f}")

# Seçilen özellikler
selected_features = np.where(np.abs(lasso_best.coef_) > 1e-6)[0]
print(f"\nSeçilen Özellikler: {selected_features}")
print(f"Seçilen Özellik Sayısı: {len(selected_features)}")

# Regularization path gösterimi
print("\n" + "="*60)
print("REGULARİZASYON YOLU ANALİZİ")
print("="*60)

# Daha detaylı alpha aralığı
alphas = np.logspace(-4, 2, 50)
coef_path = []

for alpha in alphas:
    lasso = Lasso(alpha=alpha, random_state=42)
    lasso.fit(X_train_scaled, y_train)
    coef_path.append(lasso.coef_)

coef_path = np.array(coef_path)

# Regularization path grafiği
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
for i in range(coef_path.shape[1]):
    plt.plot(alphas, coef_path[:, i], label=f'Özellik {i+1}')
plt.xscale('log')
plt.xlabel('Alpha (log scale)')
plt.ylabel('Katsayı Değeri')
plt.title('LASSO Regularization Path')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
# Her alpha için aktif özellik sayısı
active_features = [np.sum(np.abs(coef) > 1e-6) for coef in coef_path]
plt.plot(alphas, active_features, 'g-', linewidth=2)
plt.xscale('log')
plt.xlabel('Alpha (log scale)')
plt.ylabel('Aktif Özellik Sayısı')
plt.title('Özellik Seçimi Eğrisi')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n🎯 LASSO'nun temel avantajları:")
print("   • Otomatik özellik seçimi")
print("   • Modelin karmaşıklığını azaltır")
print("   • Aşırı öğrenmeyi (overfitting) önler")
print("   • Yorumlanabilir modeller oluşturur")
print("\n💡 Alpha parametresi:")
print("   • Alpha arttıkça -> daha az özellik seçilir")
print("   • Alpha azaldıkça -> daha fazla özellik seçilir")
print("   • Alpha = 0 -> Normal linear regresyon")