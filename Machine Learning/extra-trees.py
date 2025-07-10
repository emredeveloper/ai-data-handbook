from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np

# Iris veri setini yükle
iris = load_iris()
X, y = iris.data, iris.target

# Özellikleri ölçeklendir
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Veriyi böl
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Ayarlanmış parametrelerle Extra Trees Sınıflandırıcısını başlat
clf = ExtraTreesClassifier(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42)

# Çapraz doğrulama gerçekleştir
cv_scores = cross_val_score(clf, X_scaled, y, cv=5)
print(f"Cross-validation scores: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")

# Modeli eğit
clf.fit(X_train, y_train)

# Tahmin yap ve değerlendir
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Özellik önemliliği
feature_importance = clf.feature_importances_
for feature, importance in zip(iris.feature_names, feature_importance):
    print(f"{feature}: {importance:.4f}")