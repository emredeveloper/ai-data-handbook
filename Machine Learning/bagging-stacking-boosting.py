import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier, 
                             AdaBoostClassifier, GradientBoostingClassifier,
                             VotingClassifier, StackingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Veri hazırla
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("="*50)
print("ENSEMBLE YÖNTEMLERİ - BASIT AÇIKLAMALAR")
print("="*50)

# =============================================================================
# 1. BAGGING (TORBALAMA)
# =============================================================================
print("\n1. BAGGING (TORBALAMA)")
print("-" * 30)
print("Mantık: Aynı veriyi farklı şekillerde örnekleyerek birden fazla model eğit")
print("Sonuç: Modellerin ortalamasını/oylamasını al")

# Tek karar ağacı
single_tree = DecisionTreeClassifier(random_state=42)
single_tree.fit(X_train, y_train)
single_pred = single_tree.predict(X_test)
single_acc = accuracy_score(y_test, single_pred)

# Bagging
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=10,  # 10 farklı model
    random_state=42
)
bagging.fit(X_train, y_train)
bagging_pred = bagging.predict(X_test)
bagging_acc = accuracy_score(y_test, bagging_pred)

print(f"Tek ağaç doğruluğu: {single_acc:.3f}")
print(f"Bagging doğruluğu:  {bagging_acc:.3f}")
print(f"İyileştirme: +{bagging_acc - single_acc:.3f}")

# Random Forest (Bagging + rastgele özellik seçimi)
rf = RandomForestClassifier(n_estimators=10, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"Random Forest:       {rf_acc:.3f}")

# =============================================================================
# 2. BOOSTING (GÜÇLENDİRME)
# =============================================================================
print("\n\n2. BOOSTING (GÜÇLENDİRME)")
print("-" * 30)
print("Mantık: Modelleri sırayla eğit, her yeni model öncekinin hatalarını düzeltsin")
print("Sonuç: Zayıf modelleri güçlü bir model haline getir")

# AdaBoost
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # Çok basit ağaçlar
    n_estimators=10,
    random_state=42
)
ada.fit(X_train, y_train)
ada_pred = ada.predict(X_test)
ada_acc = accuracy_score(y_test, ada_pred)

# Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=10, random_state=42)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
gb_acc = accuracy_score(y_test, gb_pred)

print(f"AdaBoost doğruluğu:        {ada_acc:.3f}")
print(f"Gradient Boosting doğruluğu: {gb_acc:.3f}")

# =============================================================================
# 3. VOTING (OYLAMA)
# =============================================================================
print("\n\n3. VOTING (OYLAMA)")
print("-" * 30)
print("Mantık: Farklı algoritmaları eğit, sonuçları oyla")
print("Hard Voting: Çoğunluk oyunu al")
print("Soft Voting: Olasılıkların ortalamasını al")

# Farklı modeller
model1 = LogisticRegression(random_state=42, max_iter=1000)
model2 = DecisionTreeClassifier(random_state=42)
model3 = GaussianNB()

# Hard Voting
hard_voting = VotingClassifier(
    estimators=[('lr', model1), ('dt', model2), ('nb', model3)],
    voting='hard'
)
hard_voting.fit(X_train, y_train)
hard_pred = hard_voting.predict(X_test)
hard_acc = accuracy_score(y_test, hard_pred)

# Soft Voting
soft_voting = VotingClassifier(
    estimators=[('lr', model1), ('dt', model2), ('nb', model3)],
    voting='soft'
)
soft_voting.fit(X_train, y_train)
soft_pred = soft_voting.predict(X_test)
soft_acc = accuracy_score(y_test, soft_pred)

# Tekil model performansları
model1.fit(X_train, y_train)
model2.fit(X_train, y_train) 
model3.fit(X_train, y_train)

m1_acc = accuracy_score(y_test, model1.predict(X_test))
m2_acc = accuracy_score(y_test, model2.predict(X_test))
m3_acc = accuracy_score(y_test, model3.predict(X_test))

print(f"Logistic Regression: {m1_acc:.3f}")
print(f"Decision Tree:       {m2_acc:.3f}")
print(f"Naive Bayes:         {m3_acc:.3f}")
print(f"Hard Voting:         {hard_acc:.3f}")
print(f"Soft Voting:         {soft_acc:.3f}")

# =============================================================================
# 4. STACKING (İSTİFLEME)
# =============================================================================
print("\n\n4. STACKING (İSTİFLEME)")
print("-" * 30)
print("Mantık: Base modellerin tahminlerini başka bir modelle birleştir")
print("Meta-learner: Base modellerin çıktılarını öğrenir")

# Stacking
stacking = StackingClassifier(
    estimators=[
        ('lr', LogisticRegression(random_state=42, max_iter=1000)),
        ('dt', DecisionTreeClassifier(random_state=42)),
        ('nb', GaussianNB())
    ],
    final_estimator=LogisticRegression(),  # Meta-learner
    cv=3  # Cross-validation
)
stacking.fit(X_train, y_train)
stacking_pred = stacking.predict(X_test)
stacking_acc = accuracy_score(y_test, stacking_pred)

print(f"Stacking doğruluğu: {stacking_acc:.3f}")

# =============================================================================
# 5. EL İLE VOTING ÖRNEĞİ
# =============================================================================
print("\n\n5. EL İLE VOTING ÖRNEĞİ")
print("-" * 30)

# Üç modeli eğit
lr = LogisticRegression(random_state=42, max_iter=1000)
dt = DecisionTreeClassifier(random_state=42)
nb = GaussianNB()

lr.fit(X_train, y_train)
dt.fit(X_train, y_train)
nb.fit(X_train, y_train)

# Tahminleri al
pred1 = lr.predict(X_test)
pred2 = dt.predict(X_test)
pred3 = nb.predict(X_test)

# Hard voting manuel
manual_hard_voting = []
for i in range(len(pred1)):
    votes = [pred1[i], pred2[i], pred3[i]]
    # En çok oy alan sınıfı seç
    manual_hard_voting.append(max(set(votes), key=votes.count))

manual_hard_acc = accuracy_score(y_test, manual_hard_voting)
print(f"Manuel Hard Voting: {manual_hard_acc:.3f}")

# Soft voting manuel (olasılıklarla)
prob1 = lr.predict_proba(X_test)
prob2 = dt.predict_proba(X_test)
prob3 = nb.predict_proba(X_test)

# Olasılıkların ortalaması
avg_probs = (prob1 + prob2 + prob3) / 3
manual_soft_pred = np.argmax(avg_probs, axis=1)
manual_soft_acc = accuracy_score(y_test, manual_soft_pred)
print(f"Manuel Soft Voting: {manual_soft_acc:.3f}")

# =============================================================================
# 6. SONUÇLAR ÖZETİ
# =============================================================================
print("\n\n6. TÜM SONUÇLAR")
print("-" * 30)
results = {
    'Tek Karar Ağacı': single_acc,
    'Bagging': bagging_acc,
    'Random Forest': rf_acc,
    'AdaBoost': ada_acc,
    'Gradient Boosting': gb_acc,
    'Hard Voting': hard_acc,
    'Soft Voting': soft_acc,
    'Stacking': stacking_acc
}

# Sırala
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

print("Sıralama (En iyi -> En kötü):")
for i, (method, score) in enumerate(sorted_results, 1):
    print(f"{i}. {method:<20}: {score:.3f}")

# =============================================================================
# 7. ÖZET AÇIKLAMALAR
# =============================================================================
print("\n\n7. KISA ÖZETLEr")
print("-" * 30)
print("BAGGING:")
print("  • Aynı algoritmayı farklı veri parçalarında eğit")
print("  • Overfitting'i azaltır")
print("  • Paralel çalışır")

print("\nBOOSTING:")
print("  • Hataları düzelterek sırayla öğren")
print("  • Zayıf modelleri güçlü yapar")
print("  • Sıralı çalışır")

print("\nVOTING:")
print("  • Farklı algoritmaları birleştir")
print("  • Hard: Çoğunluk oyu")
print("  • Soft: Olasılık ortalaması")

print("\nSTACKING:")
print("  • Base modellerin tahminlerini başka model öğrenir")
print("  • En sofistike yöntem")
print("  • Genelde en iyi sonucu verir")

print("\n" + "="*50)
print("ENSEMBLE YÖNTEMLERİ ÖZETİ TAMAMLANDI")
print("="*50)