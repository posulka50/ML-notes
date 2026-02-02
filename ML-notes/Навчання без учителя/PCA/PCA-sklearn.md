# PCA — sklearn практика

Повний практичний гайд по використанню PCA в scikit-learn з прикладами коду.

---

## 📦 Основні імпорти

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# PCA
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA

# Preprocessing
from sklearn.preprocessing import StandardScaler

# Metrics
from sklearn.metrics import (
    mean_squared_error,
    r2_score
)

# ML models (для порівняння)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Data
from sklearn.datasets import load_iris, load_digits, make_classification
```

---

## 1️⃣ PCA — основний клас

### Параметри

```python
PCA(
    n_components=None,         # Кількість компонент або variance to keep
    copy=True,                 # Копіювати дані
    whiten=False,              # Нормалізувати компоненти до од. variance
    svd_solver='auto',         # 'auto', 'full', 'arpack', 'randomized'
    tol=0.0,                   # Толерантність для 'arpack'
    iterated_power='auto',     # Кількість ітерацій для 'randomized'
    n_oversamples=10,          # Для 'randomized' solver
    power_iteration_normalizer='auto',  # Для 'randomized'
    random_state=None          # Seed для відтворюваності
)
```

**n_components варіанти:**

```python
# 1. Integer — точна кількість компонент
pca = PCA(n_components=2)  # тільки 2 компоненти

# 2. Float (0.0 - 1.0) — variance to keep
pca = PCA(n_components=0.95)  # зберегти 95% variance

# 3. None — всі компоненти
pca = PCA()  # max(n_samples, n_features) компонент

# 4. String 'mle' — автоматичний вибір через MLE
pca = PCA(n_components='mle')  # експериментально
```

---

### Атрибути після fit

```python
pca = PCA(n_components=2)
pca.fit(X)

# Доступні атрибути:
pca.components_              # Головні компоненти (n_components, n_features)
pca.explained_variance_      # Variance кожної компоненти (n_components,)
pca.explained_variance_ratio_  # % variance кожної компоненти (n_components,)
pca.singular_values_         # Сингулярні значення
pca.mean_                    # Середнє значення для центрування
pca.n_components_            # Кількість компонент (після fit)
pca.n_features_              # Кількість ознак в X
pca.n_features_in_           # Кількість вхідних ознак
pca.n_samples_               # Кількість зразків
pca.noise_variance_          # Оцінка variance шуму
```

---

### Методи

```python
# Навчання
pca.fit(X)

# Трансформація (зменшення розмірності)
X_pca = pca.transform(X)

# Навчання + трансформація
X_pca = pca.fit_transform(X)

# Зворотна трансформація (відновлення)
X_restored = pca.inverse_transform(X_pca)

# Score (середня log-likelihood)
score = pca.score(X)

# Score для кожного зразка
scores = pca.score_samples(X)
```

---

## 2️⃣ Базовий приклад

```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# 1. Завантажити дані (4D)
iris = load_iris()
X = iris.data
y = iris.target

print(f"Оригінальна розмірність: {X.shape}")  # (150, 4)

# 2. Масштабування (ОБОВ'ЯЗКОВО!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. PCA
pca = PCA(n_components=2)  # 4D → 2D
X_pca = pca.fit_transform(X_scaled)

print(f"Нова розмірність: {X_pca.shape}")  # (150, 2)

# 4. Explained variance
print(f"\nExplained variance ratio:")
print(f"PC1: {pca.explained_variance_ratio_[0]:.2%}")
print(f"PC2: {pca.explained_variance_ratio_[1]:.2%}")
print(f"Total: {pca.explained_variance_ratio_.sum():.2%}")

# 5. Візуалізація
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=50, alpha=0.6)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('PCA of Iris Dataset')
plt.colorbar(scatter, label='Species')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 3️⃣ Вибір кількості компонент

### Метод 1: Explained Variance Ratio

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Масштабування
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA з усіма компонентами
pca = PCA()
pca.fit(X_scaled)

# Explained variance
explained_var = pca.explained_variance_ratio_
cumsum_var = np.cumsum(explained_var)

print("Explained Variance по компонентам:")
for i, (var, cum) in enumerate(zip(explained_var, cumsum_var)):
    print(f"PC{i+1}: {var:.3f} (cumulative: {cum:.3f})")

# Візуалізація
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Individual variance
axes[0].bar(range(1, len(explained_var)+1), explained_var, alpha=0.7)
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Explained Variance Ratio')
axes[0].set_title('Explained Variance per Component')
axes[0].grid(True, alpha=0.3)

# Cumulative variance
axes[1].plot(range(1, len(cumsum_var)+1), cumsum_var, 'o-', linewidth=2)
axes[1].axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
axes[1].axhline(y=0.90, color='orange', linestyle='--', label='90% threshold')
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Explained Variance')
axes[1].set_title('Cumulative Explained Variance')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Вибір кількості компонент для 95% variance
n_components_95 = np.argmax(cumsum_var >= 0.95) + 1
print(f"\nДля 95% variance потрібно {n_components_95} компонент")
```

---

### Метод 2: Elbow Method (Scree Plot)

```python
# Scree plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_var)+1), explained_var, 'o-', linewidth=2, markersize=8)
plt.xlabel('Component Number')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
plt.grid(True, alpha=0.3)
plt.show()

# Шукай "лікоть" — точку де графік різко падає
```

---

### Метод 3: Автоматичний вибір (0.95 variance)

```python
# Автоматично вибрати компоненти для 95% variance
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

print(f"Автоматично вибрано компонент: {pca.n_components_}")
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
```

---

## 4️⃣ Візуалізація компонент

### Loadings (вклад ознак у компоненти)

```python
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Дані
iris = load_iris()
X = iris.data
feature_names = iris.feature_names

# Масштабування + PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
pca.fit(X_scaled)

# Loadings (компоненти)
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

# DataFrame для зручності
loadings_df = pd.DataFrame(
    loadings,
    columns=['PC1', 'PC2'],
    index=feature_names
)

print("Loadings (вклад ознак у компоненти):")
print(loadings_df)

# Візуалізація
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Heatmap
sns.heatmap(loadings_df.T, annot=True, fmt='.2f', cmap='RdBu_r', 
            center=0, ax=axes[0], cbar_kws={'label': 'Loading'})
axes[0].set_title('Feature Loadings on Principal Components')

# Biplot
axes[1].scatter(loadings[:, 0], loadings[:, 1], s=100, alpha=0.7)
for i, txt in enumerate(feature_names):
    axes[1].annotate(txt, (loadings[i, 0], loadings[i, 1]), 
                    fontsize=10, ha='center')
axes[1].axhline(0, color='gray', linestyle='--', alpha=0.5)
axes[1].axvline(0, color='gray', linestyle='--', alpha=0.5)
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
axes[1].set_title('Feature Loadings Biplot')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

### Biplot (дані + loadings разом)

```python
def biplot(X_pca, loadings, labels=None, feature_names=None):
    """
    Biplot: показує і дані і loadings на одному графіку
    """
    plt.figure(figsize=(12, 8))
    
    # Точки даних
    if labels is not None:
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, 
                            cmap='viridis', s=50, alpha=0.6)
        plt.colorbar(scatter, label='Class')
    else:
        plt.scatter(X_pca[:, 0], X_pca[:, 1], s=50, alpha=0.6)
    
    # Стрілки для ознак
    if feature_names is not None:
        for i, feature in enumerate(feature_names):
            plt.arrow(0, 0, loadings[i, 0]*5, loadings[i, 1]*5,
                     color='red', alpha=0.5, head_width=0.1)
            plt.text(loadings[i, 0]*5.5, loadings[i, 1]*5.5, feature,
                    color='red', fontsize=10, ha='center')
    
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA Biplot')
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
    plt.show()

# Приклад
biplot(X_pca, loadings, labels=y, feature_names=feature_names)
```

---

## 5️⃣ Практичні приклади

### Приклад 1: Візуалізація MNIST цифр

```python
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Завантажити MNIST (64D: 8x8 зображення)
digits = load_digits()
X = digits.data  # (1797, 64)
y = digits.target

print(f"Оригінальна розмірність: {X.shape}")

# 2. Масштабування
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. PCA: 64D → 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"Нова розмірність: {X_pca.shape}")
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")

# 4. Візуалізація
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', s=30, alpha=0.6)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.title('PCA of MNIST Digits (64D → 2D)')
plt.colorbar(scatter, label='Digit', ticks=range(10))
plt.grid(True, alpha=0.3)
plt.show()

# 5. Показати декілька цифр
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='gray')
    ax.set_title(f'Label: {y[i]}\nPC1: {X_pca[i,0]:.1f}')
    ax.axis('off')
plt.tight_layout()
plt.show()
```

---

### Приклад 2: Реконструкція зображень (шумоочищення)

```python
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 1. Завантажити дані
digits = load_digits()
X = digits.data / 16.0  # нормалізувати [0, 1]

# 2. Додати шум
np.random.seed(42)
noise = np.random.normal(0, 0.5, X.shape)
X_noisy = X + noise
X_noisy = np.clip(X_noisy, 0, 1)  # обмежити [0, 1]

# 3. PCA з малою кількістю компонент (шумоочищення)
n_components = 20  # замість 64
pca = PCA(n_components=n_components)
X_compressed = pca.fit_transform(X_noisy)

# 4. Реконструкція (inverse transform)
X_restored = pca.inverse_transform(X_compressed)

# 5. Візуалізація
fig, axes = plt.subplots(3, 10, figsize=(15, 5))

for i in range(10):
    # Оригінал
    axes[0, i].imshow(X[i].reshape(8, 8), cmap='gray')
    axes[0, i].axis('off')
    if i == 0:
        axes[0, i].set_ylabel('Original', fontsize=12)
    
    # Зашумлений
    axes[1, i].imshow(X_noisy[i].reshape(8, 8), cmap='gray')
    axes[1, i].axis('off')
    if i == 0:
        axes[1, i].set_ylabel('Noisy', fontsize=12)
    
    # Відновлений
    axes[2, i].imshow(X_restored[i].reshape(8, 8), cmap='gray')
    axes[2, i].axis('off')
    if i == 0:
        axes[2, i].set_ylabel(f'Restored\n({n_components} PCs)', fontsize=12)

plt.suptitle('PCA Denoising', fontsize=14)
plt.tight_layout()
plt.show()

# MSE
from sklearn.metrics import mean_squared_error
mse_noisy = mean_squared_error(X, X_noisy)
mse_restored = mean_squared_error(X, X_restored)

print(f"MSE (оригінал vs зашумлений): {mse_noisy:.4f}")
print(f"MSE (оригінал vs відновлений): {mse_restored:.4f}")
print(f"Покращення: {(1 - mse_restored/mse_noisy)*100:.1f}%")
```

---

### Приклад 3: Прискорення ML алгоритмів

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time

# 1. Генерація складних даних (100 ознак)
X, y = make_classification(
    n_samples=1000,
    n_features=100,
    n_informative=20,
    n_redundant=60,
    n_repeated=20,
    random_state=42
)

print(f"Дані: {X.shape}")

# 2. Масштабування
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Модель без PCA
clf = RandomForestClassifier(n_estimators=100, random_state=42)

start = time.time()
score_before = cross_val_score(clf, X_scaled, y, cv=5).mean()
time_before = time.time() - start

print(f"\n=== Без PCA ===")
print(f"Точність: {score_before:.3f}")
print(f"Час: {time_before:.2f} сек")

# 4. З PCA (95% variance)
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

print(f"\nPCA: {X.shape[1]} → {pca.n_components_} компонент")
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")

start = time.time()
score_after = cross_val_score(clf, X_pca, y, cv=5).mean()
time_after = time.time() - start

print(f"\n=== З PCA ===")
print(f"Точність: {score_after:.3f}")
print(f"Час: {time_after:.2f} сек")

# 5. Порівняння
print(f"\n=== Результат ===")
print(f"Зміна точності: {(score_after - score_before):.3f}")
print(f"Прискорення: {time_before/time_after:.1f}x")
print(f"Зменшення розмірності: {(1 - pca.n_components_/X.shape[1])*100:.0f}%")
```

---

### Приклад 4: Eigenfaces (розпізнавання облич)

```python
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 1. Завантажити дані облич
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
X = lfw_people.data
y = lfw_people.target
target_names = lfw_people.target_names

n_samples, h, w = lfw_people.images.shape
n_features = X.shape[1]

print(f"Зображень: {n_samples}")
print(f"Розмір зображення: {h}x{w}")
print(f"Ознак: {n_features}")

# 2. PCA для eigenfaces
n_components = 150
pca = PCA(n_components=n_components, whiten=True, random_state=42)
X_pca = pca.fit_transform(X)

print(f"\nPCA: {n_features} → {n_components}")
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")

# 3. Візуалізація eigenfaces (головних компонент)
eigenfaces = pca.components_.reshape((n_components, h, w))

fig, axes = plt.subplots(3, 5, figsize=(12, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(eigenfaces[i], cmap='gray')
    ax.set_title(f'Eigenface {i+1}\n{pca.explained_variance_ratio_[i]:.1%}')
    ax.axis('off')

plt.suptitle('Top 15 Eigenfaces', fontsize=14)
plt.tight_layout()
plt.show()

# 4. Реконструкція обличчя
def reconstruct_face(idx, n_components_range):
    """Показати реконструкцію з різною кількістю компонент"""
    original = lfw_people.images[idx]
    
    fig, axes = plt.subplots(1, len(n_components_range)+1, figsize=(15, 3))
    
    # Оригінал
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Реконструкції
    for i, n_comp in enumerate(n_components_range):
        pca_temp = PCA(n_components=n_comp)
        X_temp = pca_temp.fit_transform(X)
        X_reconstructed = pca_temp.inverse_transform(X_temp[idx:idx+1])
        
        axes[i+1].imshow(X_reconstructed.reshape(h, w), cmap='gray')
        axes[i+1].set_title(f'{n_comp} PCs\n({pca_temp.explained_variance_ratio_.sum():.0%})')
        axes[i+1].axis('off')
    
    plt.tight_layout()
    plt.show()

# Приклад
reconstruct_face(0, [10, 50, 100, 150])
```

---

## 6️⃣ Порівняння з/без PCA

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
import time

# 1. Дані
digits = load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 2. Масштабування
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === БЕЗ PCA ===
print("=== БЕЗ PCA ===")
print(f"Розмірність: {X_train_scaled.shape}")

clf = LogisticRegression(max_iter=1000, random_state=42)

start = time.time()
clf.fit(X_train_scaled, y_train)
time_train_before = time.time() - start

start = time.time()
y_pred_before = clf.predict(X_test_scaled)
time_pred_before = time.time() - start

from sklearn.metrics import accuracy_score
acc_before = accuracy_score(y_test, y_pred_before)

print(f"Точність: {acc_before:.3f}")
print(f"Час навчання: {time_train_before:.3f} сек")
print(f"Час передбачення: {time_pred_before:.4f} сек")

# === З PCA ===
print("\n=== З PCA (95% variance) ===")

pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"Розмірність: {X_train_pca.shape}")
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")

clf_pca = LogisticRegression(max_iter=1000, random_state=42)

start = time.time()
clf_pca.fit(X_train_pca, y_train)
time_train_after = time.time() - start

start = time.time()
y_pred_after = clf_pca.predict(X_test_pca)
time_pred_after = time.time() - start

acc_after = accuracy_score(y_test, y_pred_after)

print(f"Точність: {acc_after:.3f}")
print(f"Час навчання: {time_train_after:.3f} сек")
print(f"Час передбачення: {time_pred_after:.4f} сек")

# Порівняння
print("\n=== ПОРІВНЯННЯ ===")
print(f"Зміна точності: {acc_after - acc_before:+.3f}")
print(f"Прискорення навчання: {time_train_before/time_train_after:.1f}x")
print(f"Прискорення передбачення: {time_pred_before/time_pred_after:.1f}x")
print(f"Зменшення розмірності: {(1 - pca.n_components_/X.shape[1])*100:.0f}%")
```

---

## 7️⃣ Інші варіації PCA

### IncrementalPCA (для великих даних)

```python
from sklearn.decomposition import IncrementalPCA
import numpy as np

# Дані, що не влазять у пам'ять
n_samples = 10000
n_features = 1000

# IncrementalPCA обробляє батчами
ipca = IncrementalPCA(n_components=50, batch_size=200)

# Імітація батчів
for i in range(0, n_samples, 200):
    X_batch = np.random.randn(200, n_features)  # генеруємо батч
    ipca.partial_fit(X_batch)

# Трансформація
X_new = np.random.randn(100, n_features)
X_transformed = ipca.transform(X_new)

print(f"Explained variance: {ipca.explained_variance_ratio_.sum():.2%}")
```

---

### KernelPCA (нелінійний)

```python
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_moons

# Нелінійні дані
X, y = make_moons(n_samples=300, noise=0.05, random_state=42)

# Звичайний PCA (лінійний)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Kernel PCA (нелінійний)
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=10)
X_kpca = kpca.fit_transform(X)

# Візуалізація
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Оригінал
axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50)
axes[0].set_title('Original Data')
axes[0].grid(True, alpha=0.3)

# PCA
axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=50)
axes[1].set_title('Linear PCA')
axes[1].grid(True, alpha=0.3)

# Kernel PCA
axes[2].scatter(X_kpca[:, 0], X_kpca[:, 1], c=y, cmap='viridis', s=50)
axes[2].set_title('Kernel PCA (RBF)')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 8️⃣ PCA для кластеризації

```python
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 1. Дані (високовимірні)
X, y_true = make_blobs(n_samples=300, n_features=50, centers=3, random_state=42)

# 2. Масштабування
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === БЕЗ PCA ===
kmeans_before = KMeans(n_clusters=3, random_state=42)
labels_before = kmeans_before.fit_predict(X_scaled)
sil_before = silhouette_score(X_scaled, labels_before)

print("=== БЕЗ PCA ===")
print(f"Розмірність: {X_scaled.shape}")
print(f"Silhouette: {sil_before:.3f}")

# === З PCA ===
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

kmeans_after = KMeans(n_clusters=3, random_state=42)
labels_after = kmeans_after.fit_predict(X_pca)
sil_after = silhouette_score(X_pca, labels_after)

print(f"\n=== З PCA ===")
print(f"Розмірність: {X_pca.shape}")
print(f"Компонент: {pca.n_components_}")
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
print(f"Silhouette: {sil_after:.3f}")

# Візуалізація (проекція на 2D)
pca_viz = PCA(n_components=2)
X_viz = pca_viz.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

axes[0].scatter(X_viz[:, 0], X_viz[:, 1], c=labels_before, cmap='viridis', s=50)
axes[0].set_title(f'K-Means без PCA\nSilhouette: {sil_before:.3f}')

axes[1].scatter(X_viz[:, 0], X_viz[:, 1], c=labels_after, cmap='viridis', s=50)
axes[1].set_title(f'K-Means з PCA\nSilhouette: {sil_after:.3f}')

plt.tight_layout()
plt.show()
```

---

## 9️⃣ Збереження та завантаження

```python
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Навчання
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# 2. Збереження
model_data = {
    'pca': pca,
    'scaler': scaler,
    'n_components': pca.n_components_,
    'explained_variance': pca.explained_variance_ratio_.sum()
}

joblib.dump(model_data, 'pca_model.pkl')

# 3. Завантаження
loaded_data = joblib.load('pca_model.pkl')
loaded_pca = loaded_data['pca']
loaded_scaler = loaded_data['scaler']

print(f"Компонент: {loaded_data['n_components']}")
print(f"Explained variance: {loaded_data['explained_variance']:.2%}")

# 4. Використання на нових даних
X_new = np.random.randn(10, X.shape[1])
X_new_scaled = loaded_scaler.transform(X_new)
X_new_pca = loaded_pca.transform(X_new_scaled)

print(f"\nНові дані: {X_new.shape} → {X_new_pca.shape}")
```

---

## 🔟 Поради та best practices

### 1. Завжди масштабуй дані перед PCA

```python
# ПОГАНО (різний масштаб ознак)
pca = PCA(n_components=2)
pca.fit(X)

# ДОБРЕ
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
pca.fit(X_scaled)
```

---

### 2. Перевіряй explained variance

```python
# Скільки інформації зберігаємо?
pca = PCA()
pca.fit(X_scaled)

cumsum = np.cumsum(pca.explained_variance_ratio_)
print(f"95% variance at component: {np.argmax(cumsum >= 0.95) + 1}")
```

---

### 3. PCA для візуалізації vs для ML

```python
# Для візуалізації: 2-3 компоненти (завжди)
pca_viz = PCA(n_components=2)
X_viz = pca_viz.fit_transform(X_scaled)
plt.scatter(X_viz[:, 0], X_viz[:, 1])

# Для ML: зберегти 95% variance
pca_ml = PCA(n_components=0.95)
X_ml = pca_ml.fit_transform(X_scaled)
```

---

### 4. Не завжди покращує ML

```python
# ЗАВЖДИ порівнюй з baseline (без PCA)

# Baseline
score_before = cross_val_score(model, X_scaled, y, cv=5).mean()

# З PCA
X_pca = pca.fit_transform(X_scaled)
score_after = cross_val_score(model, X_pca, y, cv=5).mean()

if score_after < score_before:
    print("⚠️ PCA погіршило результат!")
```

---

### 5. Використовуй whiten для деяких алгоритмів

```python
# whiten=True нормалізує компоненти до одиничної variance
# Корисно для neural networks, SVM

pca = PCA(n_components=50, whiten=True)
X_pca = pca.fit_transform(X_scaled)
```

---

### 6. IncrementalPCA для великих даних

```python
# Якщо дані не влазять у пам'ять
if X.shape[0] > 10000:
    from sklearn.decomposition import IncrementalPCA
    ipca = IncrementalPCA(n_components=50, batch_size=1000)
    
    # Обробляємо батчами
    for i in range(0, len(X), 1000):
        batch = X[i:i+1000]
        ipca.partial_fit(batch)
```

---

## Чек-лист для PCA

```python
# ✅ 1. Завантажити дані
X = load_data()

# ✅ 2. EDA
print(X.shape)
print(pd.DataFrame(X).describe())

# ✅ 3. Масштабування (ОБОВ'ЯЗКОВО!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ 4. Визначити оптимальну кількість компонент
pca_temp = PCA()
pca_temp.fit(X_scaled)

cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
n_components_95 = np.argmax(cumsum >= 0.95) + 1
print(f"Для 95%: {n_components_95} компонент")

# ✅ 5. Навчання PCA
pca = PCA(n_components=0.95)  # або n_components_95
X_pca = pca.fit_transform(X_scaled)

# ✅ 6. Перевірка
print(f"Розмірність: {X.shape} → {X_pca.shape}")
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")

# ✅ 7. Візуалізація (якщо потрібно)
if pca.n_components_ >= 2:
    plt.scatter(X_pca[:, 0], X_pca[:, 1])
    plt.show()

# ✅ 8. Порівняння ML з/без PCA
score_before = evaluate_model(X_scaled, y)
score_after = evaluate_model(X_pca, y)
print(f"Без PCA: {score_before:.3f}")
print(f"З PCA: {score_after:.3f}")

# ✅ 9. Збереження
joblib.dump({'pca': pca, 'scaler': scaler}, 'pca_model.pkl')
```

---

## Корисні посилання

- [sklearn PCA docs](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [sklearn IncrementalPCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html)
- [sklearn KernelPCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html)
- [Decomposition Guide](https://scikit-learn.org/stable/modules/decomposition.html)

---

**Створено для практичного використання PCA в проєктах** 🚀
