# t-SNE та UMAP — sklearn практика

Повний практичний гайд по використанню t-SNE та UMAP з прикладами коду.

---

## 📦 Основні імпорти

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# t-SNE
from sklearn.manifold import TSNE

# UMAP (потрібно встановити: pip install umap-learn)
import umap

# PCA (для порівняння)
from sklearn.decomposition import PCA

# Preprocessing
from sklearn.preprocessing import StandardScaler

# Metrics
from sklearn.metrics import silhouette_score

# Data
from sklearn.datasets import (
    load_iris, 
    load_digits, 
    load_wine,
    make_blobs,
    make_moons
)

# Для великих даних
from sklearn.datasets import fetch_openml
```

---

## 1️⃣ t-SNE — sklearn.manifold.TSNE

### Параметри

```python
TSNE(
    n_components=2,            # Розмірність виходу (2 для візуалізації)
    perplexity=30.0,           # Кількість найближчих сусідів (5-50)
    early_exaggeration=12.0,   # Коефіцієнт на ранніх ітераціях
    learning_rate=200.0,       # Швидкість навчання (10-1000, auto='auto')
    n_iter=1000,               # Кількість ітерацій (мін 250, рекомендовано 1000+)
    n_iter_without_progress=300,  # Зупинка якщо немає прогресу
    min_grad_norm=1e-7,        # Мінімальний градієнт для зупинки
    metric='euclidean',        # Метрика: 'euclidean', 'cosine', 'manhattan', etc.
    metric_params=None,        # Додаткові параметри метрики
    init='random',             # Ініціалізація: 'random', 'pca'
    verbose=0,                 # Виводити прогрес (0, 1, 2)
    random_state=None,         # Seed для відтворюваності
    method='barnes_hut',       # 'barnes_hut' (швидший) або 'exact'
    angle=0.5,                 # Для barnes_hut (0.2-0.8)
    n_jobs=None                # Кількість ядер (тільки для exact method)
)
```

---

### Атрибути після fit

```python
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Доступні атрибути:
tsne.embedding_          # Результат (n_samples, n_components)
tsne.kl_divergence_      # Фінальне значення KL divergence
tsne.n_iter_             # Кількість ітерацій до збіжності
tsne.n_features_in_      # Кількість вхідних ознак
```

---

### Методи

```python
# Навчання + трансформація
X_tsne = tsne.fit_transform(X)

# ВАЖЛИВО: t-SNE НЕ має окремих методів fit() та transform()!
# Не можна передбачити для нових точок
```

---

## 2️⃣ Базовий приклад t-SNE

```python
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# 1. Завантажити дані (4D)
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

print(f"Оригінальна розмірність: {X.shape}")  # (150, 4)

# 2. Масштабування (ОБОВ'ЯЗКОВО!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. t-SNE
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    n_iter=1000,
    random_state=42,
    verbose=1
)
X_tsne = tsne.fit_transform(X_scaled)

print(f"Нова розмірність: {X_tsne.shape}")  # (150, 2)
print(f"KL divergence: {tsne.kl_divergence_:.4f}")
print(f"Ітерацій: {tsne.n_iter_}")

# 4. Візуалізація
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                     c=y, cmap='viridis', s=50, alpha=0.7)
plt.colorbar(scatter, label='Species', ticks=[0, 1, 2])
plt.title('t-SNE visualization of Iris dataset')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.grid(True, alpha=0.3)

# Додати легенду з назвами
for i, name in enumerate(target_names):
    plt.scatter([], [], c=plt.cm.viridis(i/2), label=name, s=50)
plt.legend()

plt.show()
```

---

## 3️⃣ UMAP — umap.UMAP

### Параметри

```python
umap.UMAP(
    n_neighbors=15,            # Кількість сусідів (2-100)
    n_components=2,            # Розмірність виходу
    metric='euclidean',        # Метрика: 'euclidean', 'cosine', 'manhattan', etc.
    metric_kwds=None,          # Додаткові параметри метрики
    output_metric='euclidean', # Метрика для виходу
    n_epochs=None,             # Кількість epochs (auto = 200-500)
    learning_rate=1.0,         # Швидкість навчання
    init='spectral',           # Ініціалізація: 'spectral', 'random'
    min_dist=0.1,              # Мінімальна відстань між точками (0.0-0.99)
    spread=1.0,                # Розкид точок
    low_memory=False,          # Режим низької пам'яті
    set_op_mix_ratio=1.0,      # Баланс fuzzy union/intersection
    local_connectivity=1.0,    # Локальна зв'язаність
    repulsion_strength=1.0,    # Сила відштовхування
    negative_sample_rate=5,    # Частота негативних семплів
    transform_queue_size=4.0,  # Розмір черги для transform
    a=None,                    # Параметр кривої (auto)
    b=None,                    # Параметр кривої (auto)
    random_state=None,         # Seed
    angular_rp_forest=False,   # Angular random projection forest
    target_n_neighbors=-1,     # Для supervised UMAP
    target_metric='categorical',  # Метрика для target
    target_weight=0.5,         # Вага target у supervised
    transform_seed=42,         # Seed для transform
    force_approximation_algorithm=False,
    verbose=False,             # Виводити прогрес
    unique=False               # Видалити дублікати
)
```

---

### Атрибути після fit

```python
reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X)

# Доступні атрибути:
reducer.embedding_           # Результат (n_samples, n_components)
reducer.graph_               # Граф сусідства
reducer.transform_            # Функція для нових точок
```

---

### Методи

```python
# Навчання
reducer.fit(X)

# Трансформація (для навчених даних)
X_umap = reducer.transform(X)

# Навчання + трансформація
X_umap = reducer.fit_transform(X)

# ПЕРЕВАГА над t-SNE: можна для нових точок!
X_new_umap = reducer.transform(X_new)
```

---

## 4️⃣ Базовий приклад UMAP

```python
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# 1. Завантажити дані
iris = load_iris()
X = iris.data
y = iris.target

# 2. Масштабування
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. UMAP
reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    random_state=42
)
X_umap = reducer.fit_transform(X_scaled)

print(f"Нова розмірність: {X_umap.shape}")

# 4. Візуалізація
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], 
                     c=y, cmap='viridis', s=50, alpha=0.7)
plt.colorbar(scatter, label='Species')
plt.title('UMAP visualization of Iris dataset')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 5️⃣ Порівняння PCA vs t-SNE vs UMAP

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# 1. Завантажити складні дані (64D: 8x8 зображення цифр)
digits = load_digits()
X = digits.data
y = digits.target

print(f"Оригінальна розмірність: {X.shape}")

# 2. Масштабування
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")

# 4. t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42, verbose=0)
X_tsne = tsne.fit_transform(X_scaled)
print(f"t-SNE KL divergence: {tsne.kl_divergence_:.4f}")

# 5. UMAP
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X_scaled)

# 6. Візуалізація
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# PCA
axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', s=20, alpha=0.6)
axes[0].set_title(f'PCA ({pca.explained_variance_ratio_.sum():.1%} variance)')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')
axes[0].grid(True, alpha=0.3)

# t-SNE
axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=20, alpha=0.6)
axes[1].set_title('t-SNE')
axes[1].set_xlabel('t-SNE 1')
axes[1].set_ylabel('t-SNE 2')
axes[1].grid(True, alpha=0.3)

# UMAP
axes[2].scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='tab10', s=20, alpha=0.6)
axes[2].set_title('UMAP')
axes[2].set_xlabel('UMAP 1')
axes[2].set_ylabel('UMAP 2')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 6️⃣ Вплив параметрів t-SNE

### Perplexity

```python
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# Дані
digits = load_digits()
X = digits.data
y = digits.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Різні perplexity
perplexities = [5, 10, 30, 50]

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

for idx, perplexity in enumerate(perplexities):
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        verbose=0
    )
    X_tsne = tsne.fit_transform(X_scaled)
    
    axes[idx].scatter(X_tsne[:, 0], X_tsne[:, 1], 
                     c=y, cmap='tab10', s=20, alpha=0.6)
    axes[idx].set_title(f'Perplexity = {perplexity}')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Очікувані результати:**
- `perplexity=5`: Багато малих, фрагментованих кластерів
- `perplexity=30`: Збалансовано (рекомендовано)
- `perplexity=50`: Більші, об'єднані кластери

---

### Learning rate

```python
learning_rates = [10, 100, 200, 1000]

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

for idx, lr in enumerate(learning_rates):
    tsne = TSNE(
        n_components=2,
        learning_rate=lr,
        random_state=42,
        verbose=0
    )
    X_tsne = tsne.fit_transform(X_scaled)
    
    axes[idx].scatter(X_tsne[:, 0], X_tsne[:, 1], 
                     c=y, cmap='tab10', s=20, alpha=0.6)
    axes[idx].set_title(f'Learning Rate = {lr}')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Очікувані результати:**
- `lr=10`: Повільна збіжність, може не зійтись
- `lr=200`: Добре (за замовчуванням)
- `lr=1000`: Дуже швидко, може бути нестабільно

---

### Iterations

```python
n_iters = [250, 500, 1000, 5000]

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

for idx, n_iter in enumerate(n_iters):
    tsne = TSNE(
        n_components=2,
        n_iter=n_iter,
        random_state=42,
        verbose=0
    )
    X_tsne = tsne.fit_transform(X_scaled)
    
    axes[idx].scatter(X_tsne[:, 0], X_tsne[:, 1], 
                     c=y, cmap='tab10', s=20, alpha=0.6)
    axes[idx].set_title(f'Iterations = {n_iter}')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 7️⃣ Вплив параметрів UMAP

### n_neighbors

```python
import umap
import matplotlib.pyplot as plt

# Різні n_neighbors
n_neighbors_list = [5, 15, 30, 50]

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

for idx, n_neighbors in enumerate(n_neighbors_list):
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=0.1,
        random_state=42
    )
    X_umap = reducer.fit_transform(X_scaled)
    
    axes[idx].scatter(X_umap[:, 0], X_umap[:, 1], 
                     c=y, cmap='tab10', s=20, alpha=0.6)
    axes[idx].set_title(f'n_neighbors = {n_neighbors}')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Очікувані результати:**
- `n_neighbors=5`: Фокус на локальній структурі, багато малих кластерів
- `n_neighbors=15`: Баланс (за замовчуванням)
- `n_neighbors=50`: Фокус на глобальній структурі, великі кластери

---

### min_dist

```python
min_dists = [0.0, 0.1, 0.5, 0.99]

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

for idx, min_dist in enumerate(min_dists):
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=min_dist,
        random_state=42
    )
    X_umap = reducer.fit_transform(X_scaled)
    
    axes[idx].scatter(X_umap[:, 0], X_umap[:, 1], 
                     c=y, cmap='tab10', s=20, alpha=0.6)
    axes[idx].set_title(f'min_dist = {min_dist}')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Очікувані результати:**
- `min_dist=0.0`: Максимально щільно упаковані точки
- `min_dist=0.1`: Нормально (за замовчуванням)
- `min_dist=0.99`: Розріджені кластери

---

## 8️⃣ Практичні приклади

### Приклад 1: Візуалізація результатів кластеризації

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
import umap
import matplotlib.pyplot as plt

# 1. Генерація високовимірних даних (50D)
X, y_true = make_blobs(
    n_samples=500,
    n_features=50,
    centers=5,
    cluster_std=1.0,
    random_state=42
)

# 2. Масштабування
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Кластеризація
kmeans = KMeans(n_clusters=5, random_state=42)
y_pred = kmeans.fit_predict(X_scaled)

# 4. Візуалізація через UMAP
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X_scaled)

# 5. Порівняння
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Справжні мітки
axes[0].scatter(X_umap[:, 0], X_umap[:, 1], c=y_true, cmap='tab10', s=50, alpha=0.6)
axes[0].set_title('True Labels (UMAP)')
axes[0].grid(True, alpha=0.3)

# Передбачені мітки
axes[1].scatter(X_umap[:, 0], X_umap[:, 1], c=y_pred, cmap='tab10', s=50, alpha=0.6)
axes[1].set_title('K-Means Predictions (UMAP)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Метрики
from sklearn.metrics import adjusted_rand_score
ari = adjusted_rand_score(y_true, y_pred)
print(f"Adjusted Rand Index: {ari:.3f}")
```

---

### Приклад 2: Зменшення розмірності перед ML

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import umap
import time

# 1. Дані (64D)
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

# === БЕЗ зменшення розмірності ===
print("=== БЕЗ зменшення розмірності ===")
clf = LogisticRegression(max_iter=1000, random_state=42)

start = time.time()
clf.fit(X_train_scaled, y_train)
time_train = time.time() - start

score = clf.score(X_test_scaled, y_test)
print(f"Розмірність: {X_train_scaled.shape}")
print(f"Точність: {score:.3f}")
print(f"Час навчання: {time_train:.3f} сек")

# === З UMAP ===
print("\n=== З UMAP (n_components=10) ===")

# ВАЖЛИВО: fit на train, transform на test
reducer = umap.UMAP(n_components=10, random_state=42)
X_train_umap = reducer.fit_transform(X_train_scaled)
X_test_umap = reducer.transform(X_test_scaled)

clf_umap = LogisticRegression(max_iter=1000, random_state=42)

start = time.time()
clf_umap.fit(X_train_umap, y_train)
time_train_umap = time.time() - start

score_umap = clf_umap.score(X_test_umap, y_test)
print(f"Розмірність: {X_train_umap.shape}")
print(f"Точність: {score_umap:.3f}")
print(f"Час навчання: {time_train_umap:.3f} сек")

print(f"\n=== Порівняння ===")
print(f"Зміна точності: {score_umap - score:+.3f}")
print(f"Прискорення: {time_train/time_train_umap:.1f}x")
```

---

### Приклад 3: Supervised UMAP (з мітками)

```python
import umap
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Дані
digits = load_digits()
X = digits.data
y = digits.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Unsupervised UMAP
reducer_unsup = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
X_unsup = reducer_unsup.fit_transform(X_scaled)

# 3. Supervised UMAP (з мітками)
reducer_sup = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
X_sup = reducer_sup.fit_transform(X_scaled, y=y)

# 4. Порівняння
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Unsupervised
axes[0].scatter(X_unsup[:, 0], X_unsup[:, 1], c=y, cmap='tab10', s=20, alpha=0.6)
axes[0].set_title('Unsupervised UMAP')
axes[0].grid(True, alpha=0.3)

# Supervised
axes[1].scatter(X_sup[:, 0], X_sup[:, 1], c=y, cmap='tab10', s=20, alpha=0.6)
axes[1].set_title('Supervised UMAP (with labels)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Supervised UMAP краще розділяє класи!
```

---

### Приклад 4: 3D візуалізація

```python
from mpl_toolkits.mplot3d import Axes3D
import umap
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# 1. Дані
digits = load_digits()
X = digits.data
y = digits.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. UMAP в 3D
reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42)
X_umap_3d = reducer.fit_transform(X_scaled)

# 3. Візуалізація
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    X_umap_3d[:, 0], 
    X_umap_3d[:, 1], 
    X_umap_3d[:, 2],
    c=y, 
    cmap='tab10', 
    s=20, 
    alpha=0.6
)

ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_zlabel('UMAP 3')
ax.set_title('UMAP 3D visualization')
plt.colorbar(scatter, ax=ax, label='Digit')

plt.show()
```

---

### Приклад 5: Стабільність результатів

```python
from sklearn.datasets import load_digits
import umap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Дані
digits = load_digits()
X = digits.data[:500]  # підвибірка для швидкості
y = digits.target[:500]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# t-SNE: кілька запусків
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for i in range(3):
    # t-SNE
    tsne = TSNE(n_components=2, random_state=i*42, verbose=0)
    X_tsne = tsne.fit_transform(X_scaled)
    
    axes[0, i].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=20, alpha=0.6)
    axes[0, i].set_title(f't-SNE (seed={i*42})')
    axes[0, i].grid(True, alpha=0.3)
    
    # UMAP
    reducer = umap.UMAP(n_components=2, random_state=i*42)
    X_umap = reducer.fit_transform(X_scaled)
    
    axes[1, i].scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='tab10', s=20, alpha=0.6)
    axes[1, i].set_title(f'UMAP (seed={i*42})')
    axes[1, i].grid(True, alpha=0.3)

plt.suptitle('Stability: t-SNE vs UMAP (different random seeds)', fontsize=14)
plt.tight_layout()
plt.show()

# UMAP більш стабільний (схожі результати при різних seeds)
```

---

## 9️⃣ PCA перед t-SNE/UMAP (для швидкості)

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import time

# Дані з багатьма ознаками
from sklearn.datasets import make_classification
X, y = make_classification(
    n_samples=1000,
    n_features=200,  # багато ознак
    n_informative=50,
    random_state=42
)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Без PCA ===
print("=== БЕЗ PCA ===")

start = time.time()
tsne = TSNE(n_components=2, random_state=42, verbose=0)
X_tsne_direct = tsne.fit_transform(X_scaled)
time_direct = time.time() - start

print(f"Час: {time_direct:.2f} сек")

# === З PCA ===
print("\n=== З PCA (200D → 50D → 2D) ===")

start = time.time()

# PCA спочатку
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")

# Потім t-SNE
tsne = TSNE(n_components=2, random_state=42, verbose=0)
X_tsne_pca = tsne.fit_transform(X_pca)

time_pca = time.time() - start

print(f"Час: {time_pca:.2f} сек")
print(f"Прискорення: {time_direct/time_pca:.1f}x")

# Візуалізація
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

axes[0].scatter(X_tsne_direct[:, 0], X_tsne_direct[:, 1], c=y, cmap='viridis', s=20, alpha=0.6)
axes[0].set_title(f't-SNE напряму ({time_direct:.1f}s)')

axes[1].scatter(X_tsne_pca[:, 0], X_tsne_pca[:, 1], c=y, cmap='viridis', s=20, alpha=0.6)
axes[1].set_title(f'PCA → t-SNE ({time_pca:.1f}s)')

plt.tight_layout()
plt.show()
```

---

## 🔟 Збереження та завантаження

### t-SNE (НЕ можна зберегти для нових точок)

```python
import joblib
from sklearn.manifold import TSNE

# Навчання
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Збереження результату (але не моделі!)
result = {
    'X_tsne': X_tsne,
    'scaler': scaler,
    'kl_divergence': tsne.kl_divergence_
}

joblib.dump(result, 'tsne_result.pkl')

# Завантаження
loaded = joblib.load('tsne_result.pkl')
X_tsne_loaded = loaded['X_tsne']

# ВАЖЛИВО: Не можна застосувати до нових даних!
# Треба заново робити fit_transform для всього датасету
```

---

### UMAP (МОЖНА зберегти модель)

```python
import joblib
import umap

# Навчання
reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X_scaled)

# Збереження моделі
model = {
    'umap': reducer,
    'scaler': scaler
}

joblib.dump(model, 'umap_model.pkl')

# Завантаження
loaded_model = joblib.load('umap_model.pkl')
loaded_reducer = loaded_model['umap']
loaded_scaler = loaded_model['scaler']

# Використання на нових даних
X_new = np.random.randn(10, X.shape[1])
X_new_scaled = loaded_scaler.transform(X_new)
X_new_umap = loaded_reducer.transform(X_new_scaled)

print(f"Нові точки: {X_new.shape} → {X_new_umap.shape}")
```

---

## 1️⃣1️⃣ Інтерактивна візуалізація (Plotly)

```python
import plotly.express as px
import pandas as pd
import umap
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

# 1. Дані
digits = load_digits()
X = digits.data
y = digits.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. UMAP
reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X_scaled)

# 3. DataFrame
df = pd.DataFrame({
    'UMAP_1': X_umap[:, 0],
    'UMAP_2': X_umap[:, 1],
    'Digit': y.astype(str)
})

# 4. Інтерактивний scatter plot
fig = px.scatter(
    df, 
    x='UMAP_1', 
    y='UMAP_2', 
    color='Digit',
    title='Interactive UMAP visualization of Digits',
    width=900, 
    height=700
)

fig.update_traces(marker=dict(size=5, opacity=0.7))
fig.show()

# Можна зберегти в HTML
# fig.write_html('umap_digits.html')
```

---

## 1️⃣2️⃣ Поради та best practices

### 1. Завжди масштабуй дані

```python
# ПОГАНО
tsne = TSNE()
X_tsne = tsne.fit_transform(X)

# ДОБРЕ
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
tsne = TSNE()
X_tsne = tsne.fit_transform(X_scaled)
```

---

### 2. Експериментуй з параметрами

```python
# t-SNE: спробуй різні perplexity
for perp in [5, 10, 30, 50]:
    tsne = TSNE(perplexity=perp, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    # Візуалізуй

# UMAP: спробуй різні n_neighbors
for n_neigh in [5, 15, 30, 50]:
    reducer = umap.UMAP(n_neighbors=n_neigh, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)
    # Візуалізуй
```

---

### 3. Використовуй PCA перед t-SNE/UMAP для >100 ознак

```python
# Якщо багато ознак
if X.shape[1] > 100:
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X_scaled)
    
    # Потім t-SNE/UMAP
    tsne = TSNE()
    X_tsne = tsne.fit_transform(X_pca)
```

---

### 4. Перевіряй стабільність (кілька запусків)

```python
# Запусти 3-5 разів з різними seeds
results = []
for seed in [42, 123, 456]:
    tsne = TSNE(random_state=seed)
    X_tsne = tsne.fit_transform(X_scaled)
    results.append(X_tsne)

# Якщо результати дуже різні → погані параметри або дані
```

---

### 5. t-SNE тільки для візуалізації!

```python
# ПОГАНО (не використовуй для ML)
X_tsne = tsne.fit_transform(X_train)
model.fit(X_tsne, y_train)  # ✗

# ДОБРЕ (використовуй PCA або UMAP)
X_pca = pca.fit_transform(X_train)
model.fit(X_pca, y_train)  # ✓
```

---

### 6. Для великих даних: sampling

```python
# Якщо >100,000 точок
if len(X) > 100000:
    # Візуалізувати підвибірку
    idx = np.random.choice(len(X), size=10000, replace=False)
    X_sample = X_scaled[idx]
    
    tsne = TSNE()
    X_tsne = tsne.fit_transform(X_sample)
```

---

### 7. Фіксуй random_state

```python
# Для відтворюваності
tsne = TSNE(random_state=42)
reducer = umap.UMAP(random_state=42)
```

---

## Чек-лист для t-SNE/UMAP

```python
# ✅ 1. Завантажити дані
X = load_data()

# ✅ 2. EDA
print(X.shape)
print(pd.DataFrame(X).describe())

# ✅ 3. Масштабування (ОБОВ'ЯЗКОВО!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ 4. Якщо >100 ознак → спочатку PCA
if X.shape[1] > 100:
    pca = PCA(n_components=50)
    X_scaled = pca.fit_transform(X_scaled)

# ✅ 5. Вибір методу
# Для візуалізації: t-SNE або UMAP
# Для ML preprocessing: PCA або UMAP

# ✅ 6. t-SNE
tsne = TSNE(
    n_components=2,
    perplexity=30,  # експериментуй: 5-50
    random_state=42
)
X_tsne = tsne.fit_transform(X_scaled)

# або UMAP
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,  # експериментуй: 5-50
    min_dist=0.1,    # експериментуй: 0.0-0.5
    random_state=42
)
X_umap = reducer.fit_transform(X_scaled)

# ✅ 7. Візуалізація
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, alpha=0.6)
plt.title('t-SNE')
plt.show()

# ✅ 8. Перевірка стабільності (кілька запусків)
for seed in [42, 123, 456]:
    # Запусти знову з іншим seed
    # Порівняй результати

# ✅ 9. Збереження (тільки для UMAP, не для t-SNE)
if using_umap:
    joblib.dump(reducer, 'umap_model.pkl')
```

---

## Порівняльна таблиця

| Характеристика | PCA | t-SNE | UMAP |
|----------------|-----|-------|------|
| **Швидкість (10k точок)** | ~1 сек | ~5 хв | ~30 сек |
| **Візуалізація** | ✗ Погано | ✓✓✓ Відмінно | ✓✓✓ Відмінно |
| **Для ML** | ✓✓✓ Так | ✗ Ні | ✓ Можна |
| **Нові точки** | ✓✓✓ transform() | ✗ Ні | ✓ transform() |
| **Детермінованість** | ✓✓✓ Так | ✗ Ні | ✗ Ні |
| **Великі дані (>100k)** | ✓✓✓ Так | ✗ Ні | ✓✓ Так |
| **Рекомендовані параметри** | n_components=2 | perplexity=30 | n_neighbors=15, min_dist=0.1 |

---

## Корисні посилання

- [sklearn t-SNE docs](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
- [UMAP docs](https://umap-learn.readthedocs.io/)
- [How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/)
- [Understanding UMAP](https://pair-code.github.io/understanding-umap/)

---

**Створено для практичної візуалізації високовимірних даних з t-SNE та UMAP** 🚀
