# Manifold Learning (Навчання Многовидів)

## Що це?

**Manifold Learning** — це сімейство **нелінійних** методів dimensionality reduction, які припускають що високорозмірні дані лежать на або близько низькорозмірного **manifold** (многовиду) вбудованого у високорозмірний простір.

**Головна ідея:** "розгорнути" складну нелінійну структуру в низькорозмірний простір, зберігаючи важливі геометричні властивості.

## Що таке Manifold?

**Manifold (многовид)** — це простір, який локально виглядає як Euclidean, але глобально може бути згорнутим.

### Приклади

**1D manifold в 3D (крива):**
```
    z
    |  /\
    | /  \
    |/    \___
    |________ y
   /
  /
 x

Локально: пряма лінія
Глобально: складна крива
```

**2D manifold в 3D (поверхня):**
```
Swiss Roll:
    z
    |  ╱╲╱╲
    | ╱  X  ╲
    |╱   ║   ╲
    |    ║____╲_ y
   /
  x

Згорнутий аркуш паперу
```

### Інтуїція в ML

**Приклад:** Зображення облич
- **Високорозмірний простір:** 1000×1000 pixels = 1,000,000D
- **Фактичний manifold:** ~10-50D (пози, освітлення, емоції)

Більшість високорозмірних даних насправді мають низькорозмірну структуру!

---

## Навіщо потрібні?

- 🌀 **Нелінійні структури** — розгортання складних manifolds
- 📊 **Візуалізація** — 2D/3D проекції складних даних
- 🎯 **Локальна геометрія** — зберігає сусідство
- 🗺️ **Geodesic distances** — відстані вздовж manifold
- 🧬 **Exploratory analysis** — розуміння структури
- 🎨 **Різні методи** — різні властивості зберігаються

## Коли використовувати?

**Потрібно:**
- **Нелінійні дані** — складні згортки, криві поверхні
- **Дані лежать на manifold** — low intrinsic dimensionality
- **Візуалізація** — розуміння структури
- **Exploratory analysis** — перший погляд на дані
- **Середні дані** (100-10,000 точок)

**Не потрібно:**
- **Лінійні дані** → PCA
- **Дуже великі дані** (> 50,000) → UMAP
- **Downstream ML** → UMAP, PCA (стабільніші)
- **Швидкість** критична → PCA
- **Нові дані** потрібно трансформувати → PCA, UMAP

---

## Методи Manifold Learning

### Огляд

| Метод | Що зберігає | Швидкість | Для візуалізації | Transform нових |
|-------|-------------|-----------|------------------|-----------------|
| **MDS** | Глобальні відстані | ⭐⭐ | ✅ | ❌ |
| **Isomap** | Geodesic відстані | ⭐⭐ | ✅ | ⚠️ |
| **LLE** | Локальну геометрію | ⭐⭐⭐ | ✅ | ❌ |
| **Spectral Embedding** | Graph structure | ⭐⭐⭐ | ✅ | ❌ |
| **t-SNE** | Локальну структуру | ⭐ | ✅ | ❌ |
| **UMAP** | Локальну + глобальну | ⭐⭐⭐⭐ | ✅ | ✅ |

---

## 1. MDS (Multidimensional Scaling)

### Що це?

**MDS** — знаходить low-dimensional представлення, що зберігає **попарні відстані** між точками.

### Математика

**Мета:** Мінімізувати stress function:

$$\text{Stress} = \sqrt{\sum_{i<j} (d_{ij} - \hat{d}_{ij})^2}$$

де:
- $d_{ij}$ — відстань між точками $i$ та $j$ у високорозмірному просторі
- $\hat{d}_{ij}$ — відстань у низькорозмірному просторі

### Код

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

# Завантажити дані
digits = load_digits()
X = digits.data
y = digits.target

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# MDS
mds = MDS(
    n_components=2,
    metric=True,        # Metric MDS (зберігає відстані)
    n_init=4,           # Кількість ініціалізацій
    max_iter=300,
    random_state=42
)

X_mds = mds.fit_transform(X_scaled)

print(f"Original shape: {X.shape}")   # (1797, 64)
print(f"MDS shape: {X_mds.shape}")    # (1797, 2)
print(f"Stress: {mds.stress_:.2f}")

# Візуалізація
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_mds[:, 0], X_mds[:, 1], 
                     c=y, cmap='tab10', s=20, alpha=0.6)
plt.colorbar(scatter, label='Digit')
plt.title('MDS Projection', fontsize=14, fontweight='bold')
plt.xlabel('MDS 1')
plt.ylabel('MDS 2')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Варіації

**Metric MDS:**
- Зберігає абсолютні відстані
- Працює з Euclidean distances

**Non-metric MDS:**
- Зберігає порядок відстаней (rankings)
- Більш гнучкий

```python
# Non-metric MDS
nmds = MDS(n_components=2, metric=False, random_state=42)
X_nmds = nmds.fit_transform(X_scaled)
```

### Застосування

- Візуалізація схожості даних
- Психологічні дослідження (similarity judgments)
- Географічні дані

---

## 2. Isomap (Isometric Mapping)

### Що це?

**Isomap** — розширення MDS, що використовує **geodesic distances** (найкоротші шляхи вздовж manifold) замість Euclidean.

### Інтуїція

**Проблема з Euclidean distance:**

```
Swiss Roll:

Точки A та B:
- Euclidean distance: коротка (наскрізь)
- Geodesic distance: довга (вздовж поверхні)

    B
   /|
  / |  ← Euclidean (неправильно)
 /  |
A___| ← Geodesic (правильно, вздовж поверхні)
```

### Алгоритм

1. **Побудувати граф k-nearest neighbors**
2. **Обчислити shortest paths** між усіма точками (Floyd-Warshall або Dijkstra)
3. **MDS на geodesic distances**

### Код

```python
from sklearn.manifold import Isomap

# Isomap
isomap = Isomap(
    n_neighbors=5,      # k для kNN graph
    n_components=2,     # Вихідна розмірність
    metric='minkowski',
    p=2                 # Euclidean distance
)

X_isomap = isomap.fit_transform(X_scaled)

print(f"Isomap shape: {X_isomap.shape}")
print(f"Reconstruction error: {isomap.reconstruction_error():.4f}")

# Візуалізація
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_isomap[:, 0], X_isomap[:, 1],
                     c=y, cmap='tab10', s=20, alpha=0.6)
plt.colorbar(scatter, label='Digit')
plt.title('Isomap Projection', fontsize=14, fontweight='bold')
plt.xlabel('Isomap 1')
plt.ylabel('Isomap 2')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Приклад: Swiss Roll

```python
from sklearn.datasets import make_swiss_roll

# Створити Swiss Roll
X_swiss, t = make_swiss_roll(n_samples=1500, noise=0.1, random_state=42)

# PCA (linear) - погано
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_swiss)

# Isomap (nonlinear) - добре
isomap = Isomap(n_neighbors=10, n_components=2)
X_isomap = isomap.fit_transform(X_swiss)

# Порівняння
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 3D Original
axes[0] = fig.add_subplot(131, projection='3d')
axes[0].scatter(X_swiss[:, 0], X_swiss[:, 1], X_swiss[:, 2],
               c=t, cmap='viridis', s=10)
axes[0].set_title('Original Swiss Roll (3D)', fontsize=13, fontweight='bold')

# PCA
axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=t, cmap='viridis', s=10)
axes[1].set_title('PCA (Linear) ❌', fontsize=13, fontweight='bold')
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')

# Isomap
axes[2].scatter(X_isomap[:, 0], X_isomap[:, 1], c=t, cmap='viridis', s=10)
axes[2].set_title('Isomap (Nonlinear) ✓', fontsize=13, fontweight='bold')
axes[2].set_xlabel('Isomap 1')
axes[2].set_ylabel('Isomap 2')

plt.tight_layout()
plt.show()

print("Isomap успішно 'розгорнув' Swiss Roll!")
```

### Переваги та недоліки

**✅ Переваги:**
- Зберігає geodesic distances
- Добре розгортає manifolds
- Теоретично обґрунтований

**❌ Недоліки:**
- Чутливий до n_neighbors
- Повільний на великих даних (O(n³))
- Погано з "holes" у manifold

---

## 3. LLE (Locally Linear Embedding)

### Що це?

**LLE** — зберігає **локальну лінійну структуру**: кожна точка виражається як лінійна комбінація сусідів.

### Інтуїція

**Ідея:**
1. Кожна точка ≈ weighted sum сусідів (локально linear)
2. Знайти ваги в high-dim
3. Зберегти ті ж ваги в low-dim

```
High-dimensional:
    x₃
    ↗ ↑ ↖
   /  |  \
  x₁  x  x₂
      ↓
x ≈ w₁x₁ + w₂x₂ + w₃x₃

Low-dimensional:
Зберегти ці ваги!
```

### Математика

**Крок 1:** Знайти ваги $w_{ij}$ що мінімізують:

$$\sum_i \left\| x_i - \sum_j w_{ij} x_j \right\|^2$$

з умовою $\sum_j w_{ij} = 1$

**Крок 2:** Знайти $y_i$ що мінімізують:

$$\sum_i \left\| y_i - \sum_j w_{ij} y_j \right\|^2$$

### Код

```python
from sklearn.manifold import LocallyLinearEmbedding

# LLE
lle = LocallyLinearEmbedding(
    n_neighbors=10,
    n_components=2,
    method='standard',  # 'standard', 'modified', 'hessian', 'ltsa'
    random_state=42
)

X_lle = lle.fit_transform(X_scaled)

print(f"LLE shape: {X_lle.shape}")
print(f"Reconstruction error: {lle.reconstruction_error_:.4f}")

# Візуалізація
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_lle[:, 0], X_lle[:, 1],
                     c=y, cmap='tab10', s=20, alpha=0.6)
plt.colorbar(scatter, label='Digit')
plt.title('LLE Projection', fontsize=14, fontweight='bold')
plt.xlabel('LLE 1')
plt.ylabel('LLE 2')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Варіації LLE

**Standard LLE:**
- Оригінальний метод

**Modified LLE:**
- Більш стабільний
- Краще для малих n_neighbors

**Hessian LLE (HLLE):**
- Враховує локальну кривизну
- Краще для криволінійних manifolds

**LTSA (Local Tangent Space Alignment):**
- Використовує tangent space approximation

```python
# Modified LLE
mlle = LocallyLinearEmbedding(
    n_neighbors=10,
    n_components=2,
    method='modified',
    random_state=42
)

X_mlle = mlle.fit_transform(X_scaled)
```

### Застосування

- Face recognition (eigenfaces)
- Spectroscopy data
- Image analysis

---

## 4. Spectral Embedding (Laplacian Eigenmaps)

### Що це?

**Spectral Embedding** — використовує **graph Laplacian** для знаходження embedding.

### Математика

**1. Побудувати граф схожості:**
$$W_{ij} = \exp\left(-\frac{\|x_i - x_j\|^2}{2\sigma^2}\right) \text{ якщо } j \in \text{neighbors}(i)$$

**2. Graph Laplacian:**
$$L = D - W$$

де $D$ — diagonal degree matrix.

**3. Знайти власні вектори $L$:**

Embedding = власні вектори з найменшими власними значеннями.

### Код

```python
from sklearn.manifold import SpectralEmbedding

# Spectral Embedding
spectral = SpectralEmbedding(
    n_components=2,
    n_neighbors=10,
    affinity='nearest_neighbors',  # або 'rbf'
    random_state=42
)

X_spectral = spectral.fit_transform(X_scaled)

# Візуалізація
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_spectral[:, 0], X_spectral[:, 1],
                     c=y, cmap='tab10', s=20, alpha=0.6)
plt.colorbar(scatter, label='Digit')
plt.title('Spectral Embedding', fontsize=14, fontweight='bold')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Застосування

- Clustering (spectral clustering)
- Graph partitioning
- Semi-supervised learning

---

## Порівняння всіх методів

### Swiss Roll Test

```python
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import PCA
from sklearn.manifold import (
    MDS, Isomap, LocallyLinearEmbedding, 
    SpectralEmbedding, TSNE
)
import umap

# Створити Swiss Roll
X_swiss, t = make_swiss_roll(n_samples=1500, noise=0.1, random_state=42)

# Всі методи
methods = {
    'PCA': PCA(n_components=2),
    'MDS': MDS(n_components=2, max_iter=100, n_init=1),
    'Isomap': Isomap(n_neighbors=10, n_components=2),
    'LLE': LocallyLinearEmbedding(n_neighbors=10, n_components=2),
    'Spectral': SpectralEmbedding(n_neighbors=10, n_components=2),
    't-SNE': TSNE(n_components=2, random_state=42),
    'UMAP': umap.UMAP(n_components=2, random_state=42)
}

# Візуалізація
fig = plt.figure(figsize=(18, 12))

# Original 3D
ax = fig.add_subplot(3, 3, 1, projection='3d')
ax.scatter(X_swiss[:, 0], X_swiss[:, 1], X_swiss[:, 2],
          c=t, cmap='viridis', s=10)
ax.set_title('Original Swiss Roll (3D)', fontsize=12, fontweight='bold')

# Всі методи
for idx, (name, method) in enumerate(methods.items(), start=2):
    print(f"Running {name}...")
    
    X_transformed = method.fit_transform(X_swiss)
    
    ax = fig.add_subplot(3, 3, idx)
    scatter = ax.scatter(X_transformed[:, 0], X_transformed[:, 1],
                        c=t, cmap='viridis', s=10, alpha=0.6)
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xlabel('Component 1', fontsize=9)
    ax.set_ylabel('Component 2', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n=== Swiss Roll Results ===")
print("✅ Good unrolling: Isomap, LLE, t-SNE, UMAP")
print("❌ Poor unrolling: PCA (linear)")
```

### S-Curve Test

```python
from sklearn.datasets import make_s_curve

# S-Curve
X_s, t_s = make_s_curve(n_samples=1500, noise=0.1, random_state=42)

# Повторити для S-curve
# ... (аналогічний код)
```

---

## Практичне порівняння

### Digits Dataset

```python
import time

# Digits dataset
digits = load_digits()
X = digits.data
y = digits.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Порівняти всі методи
methods = {
    'PCA': PCA(n_components=2),
    'MDS': MDS(n_components=2, max_iter=100, n_init=1),
    'Isomap': Isomap(n_neighbors=10, n_components=2),
    'LLE': LocallyLinearEmbedding(n_neighbors=10, n_components=2),
    'Spectral': SpectralEmbedding(n_neighbors=10, n_components=2),
    't-SNE': TSNE(n_components=2, random_state=42),
    'UMAP': umap.UMAP(n_components=2, random_state=42)
}

results = {}

for name, method in methods.items():
    print(f"\nRunning {name}...")
    
    start = time.time()
    X_transformed = method.fit_transform(X_scaled)
    elapsed = time.time() - start
    
    results[name] = {
        'time': elapsed,
        'embedding': X_transformed
    }
    
    print(f"  Time: {elapsed:.2f}s")

# Візуалізація всіх
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.ravel()

for idx, (name, result) in enumerate(results.items()):
    axes[idx].scatter(
        result['embedding'][:, 0],
        result['embedding'][:, 1],
        c=y, cmap='tab10', s=10, alpha=0.6
    )
    axes[idx].set_title(
        f"{name}\nTime: {result['time']:.2f}s",
        fontsize=11, fontweight='bold'
    )
    axes[idx].grid(True, alpha=0.3)

# Hide last subplot
axes[-1].axis('off')

plt.tight_layout()
plt.show()

# Порівняльна таблиця
print("\n=== Performance Comparison ===")
print(f"{'Method':<15} {'Time (s)':<10}")
print("-" * 25)
for name, result in sorted(results.items(), key=lambda x: x[1]['time']):
    print(f"{name:<15} {result['time']:<10.2f}")
```

---

## Вибір параметрів

### n_neighbors (критичний параметр!)

**Вплив:**

```python
# Занадто мало (3-5)
# → Шумні результати
# → Розриви в manifold

# Оптимально (10-30)
# → Збалансовано
# → Гладкі результати

# Занадто багато (50+)
# → Втрата локальної структури
# → Наближається до глобальних методів
```

**Експеримент:**

```python
n_neighbors_values = [5, 10, 20, 50]

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

for idx, n_neighbors in enumerate(n_neighbors_values):
    isomap = Isomap(n_neighbors=n_neighbors, n_components=2)
    X_transformed = isomap.fit_transform(X_scaled)
    
    axes[idx].scatter(X_transformed[:, 0], X_transformed[:, 1],
                     c=y, cmap='tab10', s=15, alpha=0.6)
    axes[idx].set_title(f'n_neighbors = {n_neighbors}',
                       fontsize=12, fontweight='bold')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Рекомендації:**
- **Малі дані** (< 500): n_neighbors = 5-10
- **Середні дані** (500-5000): n_neighbors = 10-20
- **Великі дані** (> 5000): n_neighbors = 20-50

---

## Переваги та недоліки

### Загальні переваги manifold methods ✓

- ✅ Нелінійні transformations
- ✅ Виявляють складні структури
- ✅ Добре для exploratory analysis
- ✅ Різні методи для різних задач

### Загальні недоліки ✗

- ❌ Повільні на великих даних
- ❌ Чутливі до параметрів
- ❌ Немає .transform() для нових даних (крім Isomap частково)
- ❌ Локальні мінімуми (деякі методи)
- ❌ Важко інтерпретувати осі

### Порівняльна таблиця

| Метод | Швидкість | Стабільність | Глобальна структура | Локальна структура |
|-------|-----------|--------------|---------------------|-------------------|
| **MDS** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Isomap** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **LLE** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Spectral** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **t-SNE** | ⭐ | ⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ |
| **UMAP** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## Коли який метод використовувати

### Decision Tree

```
Чи дані лінійні?
├─ Так → PCA
└─ Ні (нелінійні)
   │
   Чи потрібна швидкість?
   ├─ Так → UMAP
   └─ Ні
      │
      Що важливіше?
      ├─ Глобальна структура → Isomap або MDS
      ├─ Локальна структура → LLE або Spectral
      ├─ Візуалізація кластерів → t-SNE або UMAP
      └─ Exploratory analysis → Спробувати кілька!
```

### Рекомендації за типом даних

**Images:**
- Великі: UMAP
- Середні: Isomap, t-SNE
- Малі: LLE

**Text (TF-IDF):**
- UMAP (найкраще)
- t-SNE
- PCA (baseline)

**Biological data (gene expression):**
- UMAP
- Spectral Embedding
- Diffusion Maps

**Graph data:**
- Spectral Embedding
- Graph Neural Networks

---

## Практичні поради 💡

### 1. Завжди почни з PCA baseline

```python
# Спочатку PCA для порівняння
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Потім manifold methods
isomap = Isomap(n_components=2)
X_isomap = isomap.fit_transform(X_scaled)

# Порівняй візуально
```

### 2. Scaling обов'язковий

```python
# ✅ ЗАВЖДИ
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Потім manifold learning
```

### 3. Спробуй різні n_neighbors

```python
# Експериментуй
for n in [5, 10, 20, 30]:
    isomap = Isomap(n_neighbors=n, n_components=2)
    X_transformed = isomap.fit_transform(X_scaled)
    # Візуалізуй та порівняй
```

### 4. Subsampling для великих даних

```python
# Якщо > 10,000 точок
if len(X) > 10000:
    indices = np.random.choice(len(X), 5000, replace=False)
    X_sample = X[indices]
else:
    X_sample = X

# Manifold learning на sample
```

### 5. PCA preprocessing для прискорення

```python
# Якщо багато features
if X.shape[1] > 50:
    # PCA спочатку
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X_scaled)
    
    # Потім manifold
    isomap = Isomap(n_components=2)
    X_isomap = isomap.fit_transform(X_pca)
```

### 6. Порівняй кілька методів

```python
# Не обмежуйся одним методом!
methods = [
    ('Isomap', Isomap(n_components=2)),
    ('LLE', LocallyLinearEmbedding(n_components=2)),
    ('t-SNE', TSNE(n_components=2))
]

for name, method in methods:
    X_transformed = method.fit_transform(X_scaled)
    # Візуалізуй
```

### 7. Перевіряй reconstruction error

```python
# Для методів що мають це
isomap = Isomap(n_components=2)
X_isomap = isomap.fit_transform(X_scaled)

error = isomap.reconstruction_error()
print(f"Reconstruction error: {error:.4f}")

# Менше = краще
```

### 8. Використовуй для exploratory, не production

```python
# ✅ Exploratory analysis
# Використай manifold learning для розуміння структури

# ❌ Production ML pipeline
# Краще PCA або UMAP (мають .transform())
```

### 9. Візуалізуй 3D для кращого розуміння

```python
# 2D може бути недостатньо
isomap = Isomap(n_components=3)
X_3d = isomap.fit_transform(X_scaled)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2],
                    c=y, cmap='tab10', s=20)
plt.colorbar(scatter)
plt.show()
```

### 10. Документуй параметри

```python
# Запиши які параметри працювали найкраще
best_params = {
    'method': 'Isomap',
    'n_neighbors': 15,
    'n_components': 2,
    'reconstruction_error': 0.234
}
```

---

## Поширені помилки ❌

### 1. Використовувати на великих даних без sampling

```python
# ❌ Буде працювати годинами
X_huge = np.random.randn(100000, 100)
isomap = Isomap()
X_isomap = isomap.fit_transform(X_huge)  # Дуже повільно!

# ✅ Sample спочатку
indices = np.random.choice(len(X_huge), 5000)
X_sample = X_huge[indices]
X_isomap = isomap.fit_transform(X_sample)
```

### 2. Не робити scaling

```python
# ❌ Features в різних масштабах
isomap.fit_transform(X)

# ✅ Scaling
X_scaled = StandardScaler().fit_transform(X)
isomap.fit_transform(X_scaled)
```

### 3. Один n_neighbors без експериментів

```python
# ❌ Default може бути поганим
isomap = Isomap()  # n_neighbors=5 default

# ✅ Спробуй різні
for n in [5, 10, 20]:
    isomap = Isomap(n_neighbors=n)
    # Порівняй результати
```

### 4. Очікувати .transform() для нових даних

```python
# ❌ Більшість методів не мають .transform()
lle = LocallyLinearEmbedding()
lle.fit(X_train)
# X_test_transformed = lle.transform(X_test)  # AttributeError!

# ✅ Використовуй UMAP або PCA якщо потрібен transform
```

### 5. Інтерпретувати осі

```python
# ❌ "Вісь 1 означає..."
# Осі manifold methods не мають значення!

# ✅ Інтерпретуй тільки кластери та відстані
```

### 6. Використовувати для production ML

```python
# ❌ В production pipeline
# Manifold methods важко re-apply до нових даних

# ✅ Для exploratory або використовуй UMAP/PCA
```

---

## Реальні застосування

### 1. Face Recognition

**Задача:** Розпізнавання облич з різними позами/освітленням.

**Підхід:**
```python
# Isomap на face images
# Manifold captures: pose, lighting, expression

isomap = Isomap(n_neighbors=10, n_components=50)
face_embeddings = isomap.fit_transform(face_images)

# Використати для nearest neighbor matching
```

### 2. Gene Expression Analysis

**Задача:** Візуалізація клітин за експресією генів.

**Підхід:**
```python
# Spectral Embedding або UMAP
spectral = SpectralEmbedding(n_neighbors=15, n_components=2)
cell_embedding = spectral.fit_transform(gene_expression)

# Виявлення клітинних типів (кластери)
```

### 3. Text Visualization

**Задача:** Візуалізація документів за схожістю.

**Підхід:**
```python
from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
X_tfidf = vectorizer.fit_transform(documents)

# UMAP (найкраще для sparse text)
umap_model = umap.UMAP(n_components=2, metric='cosine')
doc_embedding = umap_model.fit_transform(X_tfidf.toarray())
```

### 4. Audio Feature Learning

**Задача:** Знайти структуру в audio features.

**Підхід:**
```python
# Extract MFCC features
# Apply manifold learning

lle = LocallyLinearEmbedding(n_components=3)
audio_embedding = lle.fit_transform(mfcc_features)
```

---

## Пов'язані теми

- [[01_PCA]] — лінійна альтернатива
- [[02_t-SNE]] — популярний manifold method
- [[03_UMAP]] — сучасна швидка альтернатива
- [[Graph_Theory]] — основа для spectral methods
- [[Dimensionality_Reduction]] — загальний огляд

## Ресурси

- [Scikit-learn: Manifold Learning](https://scikit-learn.org/stable/modules/manifold.html)
- [Original Isomap Paper (Tenenbaum et al., 2000)](https://www.science.org/doi/10.1126/science.290.5500.2319)
- [LLE Paper (Roweis & Saul, 2000)](https://www.science.org/doi/10.1126/science.290.5500.2323)
- [A Tutorial on Spectral Clustering (von Luxburg, 2007)](https://arxiv.org/abs/0711.0189)

---

## Ключові висновки

> Manifold Learning — це сімейство нелінійних методів dimensionality reduction, які "розгортають" складні високорозмірні структури (manifolds) в низькорозмірний простір, зберігаючи важливі геометричні властивості.

**Основна ідея:**
- Високорозмірні дані часто лежать на низькорозмірному manifold
- Мета: знайти це низькорозмірне представлення
- Різні методи зберігають різні властивості

**Основні методи:**

**MDS:**
- Зберігає глобальні Euclidean distances
- Повільний, але стабільний
- Для similarity visualization

**Isomap:**
- Зберігає geodesic distances (вздовж manifold)
- Добре розгортає Swiss Roll
- Чутливий до n_neighbors

**LLE:**
- Зберігає локальну лінійну структуру
- Швидше за Isomap
- Може мати проблеми з instability

**Spectral Embedding:**
- Використовує graph Laplacian
- Зв'язок зі spectral clustering
- Добре для graph-structured data

**Сучасні альтернативи:**
- **t-SNE:** Краща візуалізація, дуже повільний
- **UMAP:** Швидкий, має .transform(), рекомендовано!

**Коли використовувати:**
- Exploratory analysis + нелінійні дані = Manifold methods ✓
- Візуалізація + швидкість → UMAP ✓
- Production ML → PCA або UMAP (має transform) ✓
- Лінійні дані → PCA ✓

**Ключові параметри:**
- **n_neighbors:** Баланс локальної/глобальної структури (10-30)
- **n_components:** Розмірність виходу (2 для viz, більше для ML)

**Найважливіше:**
- **Scaling обов'язковий**
- **Експериментуй з n_neighbors**
- **Порівнюй кілька методів**
- **Subsampling для великих даних**
- **Для production краще UMAP або PCA**
- **Manifold methods = exploratory tools**

---

#ml #unsupervised-learning #dimensionality-reduction #manifold-learning #isomap #lle #mds #spectral-embedding #nonlinear #visualization
