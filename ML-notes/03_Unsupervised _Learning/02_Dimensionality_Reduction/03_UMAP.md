# UMAP (Uniform Manifold Approximation and Projection)

## Що це?

**UMAP (Uniform Manifold Approximation and Projection)** — це сучасний **нелінійний** алгоритм dimensionality reduction, який працює **швидше за t-SNE** і зберігає як **локальну**, так і **глобальну** структуру даних. Підходить як для візуалізації, так і для downstream ML tasks.

**Головна ідея:** побудувати граф зв'язків у високорозмірному просторі, потім оптимізувати low-dimensional представлення щоб зберегти цю топологічну структуру.

## Навіщо потрібен?

- ⚡ **Швидкість** — 10-100x швидше за t-SNE
- 🎨 **Візуалізація** — чудові 2D/3D проекції
- 🔄 **Downstream ML** — можна використати для supervised learning
- 🌍 **Глобальна структура** — зберігає великомасштабні паттерни
- 🎯 **Локальна структура** — також зберігає близькість
- 📊 **Масштабованість** — працює на мільйонах точок
- 🔧 **Гнучкість** — custom metrics, supervised mode

## Коли використовувати?

**Потрібно:**
- **Візуалізація + downstream ML** — універсальний метод
- **Великі дані** (> 10,000 точок) — швидше за t-SNE
- **Глобальна структура** важлива — відстані між кластерами
- **Нелінійні структури** — складні manifolds
- Потрібна **новизна** — cutting-edge метод
- **Різні типи даних** — числові, категоріальні, змішані

**Не потрібно:**
- **Лінійні дані** — PCA простіший
- **Інтерпретація компонентів** важлива → PCA, LDA
- **Максимальна стабільність** → PCA (UMAP частково стохастичний)
- **Дуже малі дані** (< 100) — t-SNE може бути кращим

---

## Як працює UMAP?

### Інтуїція: Топологічний підхід

**Крок 1:** Уяви дані як точки на деформованому manifold (поверхні):

```
High-dimensional manifold:
    
    ●──●──●
   /       \
  ●         ●
   \       /
    ●──●──●
    
Складна поверхня
```

**Крок 2:** Побудуй граф найближчих сусідів:

```
    ●──●──●
    │\ │ /│
    ● \●/ ●
    │ /│\ │
    ●──●──●
    
Edges = зв'язки
```

**Крок 3:** Знайди low-dimensional представлення, що зберігає ці зв'язки:

```
2D projection:
    
    ●──●──●
    │  │  │
    ●──●──●
    │  │  │
    ●──●──●
    
Розгорнуто з збереженням структури
```

### Математичний процес

**Етап 1: High-dimensional graph**

Для кожної точки $x_i$:
1. Знайти k найближчих сусідів
2. Обчислити локальну метрику відстані
3. Створити fuzzy simplicial set (нечіткий граф)

**Вага ребра:**
$$w_{ij} = \exp\left(-\frac{d(x_i, x_j) - \rho_i}{\sigma_i}\right)$$

де:
- $\rho_i$ — відстань до найближчого сусіда
- $\sigma_i$ — локальна масштабуюча константа

**Етап 2: Low-dimensional optimization**

Мінімізувати різницю між high-dim та low-dim графами:

$$CE = \sum_{ij} w_{ij}^{high} \log\frac{w_{ij}^{high}}{w_{ij}^{low}} + (1-w_{ij}^{high})\log\frac{1-w_{ij}^{high}}{1-w_{ij}^{low}}$$

**Low-dimensional вага:**
$$w_{ij}^{low} = \frac{1}{1 + a \|y_i - y_j\|_2^{2b}}$$

**Оптимізація:** Stochastic gradient descent

### Відмінності від t-SNE

| Аспект | t-SNE | UMAP |
|--------|-------|------|
| **Математична основа** | Ймовірності (KL-divergence) | Топологія (cross-entropy) |
| **Швидкість** | O(n²) → O(n log n) | O(n log n) |
| **Глобальна структура** | ❌ Втрачається | ✅ Зберігається |
| **Transform нових даних** | ❌ Немає | ✅ Є (.transform()) |
| **Для ML tasks** | ❌ Ні | ✅ Так |

---

## Математика

### Fuzzy Topological Representation

**High-dimensional fuzzy set:**

Для точки $x_i$, ймовірність зв'язку з $x_j$:

$$v_i(x_j) = \exp\left(-\frac{\max(0, d(x_i, x_j) - \rho_i)}{\sigma_i}\right)$$

де:
- $\rho_i$ = відстань до 1-го сусіда (local connectivity)
- $\sigma_i$ вибирається через fixed perplexity

**Симетризація:**

$$w_{ij} = v_i(x_j) + v_j(x_i) - v_i(x_j) \cdot v_j(x_i)$$

### Low-dimensional вага

**Функція схожа на t-розподіл:**

$$\psi(y_i, y_j) = \frac{1}{1 + a\|y_i - y_j\|^{2b}}$$

Типово: $a \approx 1.58$, $b \approx 0.88$ (підібрані емпірично)

### Cross-entropy loss

$$CE = \sum_{i,j} \left[w_{ij} \log\left(\frac{w_{ij}}{\psi_{ij}}\right) + (1-w_{ij})\log\left(\frac{1-w_{ij}}{1-\psi_{ij}}\right)\right]$$

**Інтуїція:**
- Перший член: притягує близькі точки
- Другий член: відштовхує далекі точки

### Gradient

$$\nabla_{y_i} CE = \sum_j \left[2ab\|y_i - y_j\|^{2b-2}w_{ij}(y_i - y_j) - 2b(1-w_{ij})\frac{\psi_{ij}}{1+a\|y_i-y_j\|^{2b}}(y_i - y_j)\right]$$

**Оптимізація:** Stochastic gradient descent з momentum

---

## Простий приклад: Iris Dataset

### Дані

Iris: 150 квітів, 4 features, 3 види.

### Код

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import umap

# Завантажити дані
iris = load_iris()
X = iris.data
y = iris.target

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# UMAP
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    random_state=42
)

X_umap = reducer.fit_transform(X_scaled)

# Візуалізація
plt.figure(figsize=(10, 7))
scatter = plt.scatter(
    X_umap[:, 0], 
    X_umap[:, 1],
    c=y,
    cmap='viridis',
    s=50,
    alpha=0.7,
    edgecolors='black',
    linewidths=0.5
)
plt.colorbar(scatter, label='Species', ticks=[0, 1, 2])
plt.title('UMAP Projection of Iris Dataset', fontsize=14, fontweight='bold')
plt.xlabel('UMAP 1', fontsize=12)
plt.ylabel('UMAP 2', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Результат

```
UMAP 2D:
    
    Setosa
      ●●●
       ●●
      ●●●
    
        Versicolor
          ■■■
           ■■
          ■■■
    
              Virginica
                ▲▲▲
                 ▲▲
                ▲▲▲

Три види чітко розділені!
```

---

## Складний приклад: MNIST

### Задача

MNIST: 70,000 цифр, 784 features (28×28 пікселі).

**Мета:** Візуалізувати + створити features для класифікатора.

### Код

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import umap
import time

# Завантажити MNIST
print("Loading MNIST...")
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X = mnist.data.to_numpy()
y = mnist.target.to_numpy().astype(int)

# Підмножина для швидкості
n_samples = 10000
indices = np.random.RandomState(42).choice(len(X), n_samples, replace=False)
X_sample = X[indices]
y_sample = y[indices]

print(f"Using {n_samples} samples")
print(f"Original shape: {X_sample.shape}")

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sample)

# UMAP
print("\nRunning UMAP...")
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    metric='euclidean',
    random_state=42,
    verbose=True
)

start = time.time()
X_umap = reducer.fit_transform(X_scaled)
umap_time = time.time() - start

print(f"UMAP time: {umap_time:.2f}s")

# Візуалізація
plt.figure(figsize=(12, 10))
scatter = plt.scatter(
    X_umap[:, 0],
    X_umap[:, 1],
    c=y_sample,
    cmap='tab10',
    s=10,
    alpha=0.6
)
plt.colorbar(scatter, label='Digit', ticks=range(10))
plt.title('UMAP Visualization of MNIST', fontsize=14, fontweight='bold')
plt.xlabel('UMAP 1', fontsize=12)
plt.ylabel('UMAP 2', fontsize=12)
plt.grid(True, alpha=0.3)

# Додати labels
for digit in range(10):
    mask = y_sample == digit
    center = X_umap[mask].mean(axis=0)
    plt.annotate(
        str(digit),
        center,
        fontsize=16,
        fontweight='bold',
        color='white',
        bbox=dict(boxstyle='circle', facecolor='black', alpha=0.8)
    )

plt.tight_layout()
plt.show()
```

### Використання для ML

```python
# UMAP як feature extraction для classifier

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_sample, test_size=0.3, random_state=42
)

# UMAP на train
reducer = umap.UMAP(n_components=50, random_state=42)
X_train_umap = reducer.fit_transform(X_train)

# Transform test (на відміну від t-SNE!)
X_test_umap = reducer.transform(X_test)

# Classifier на UMAP features
print("\nTraining classifier on UMAP features...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_umap, y_train)

# Predict
y_pred = clf.predict(X_test_umap)
accuracy_umap = accuracy_score(y_test, y_pred)

# Порівняння з original features
clf_original = RandomForestClassifier(n_estimators=100, random_state=42)
clf_original.fit(X_train, y_train)
y_pred_original = clf_original.predict(X_test)
accuracy_original = accuracy_score(y_test, y_pred_original)

print(f"\n=== Classification Results ===")
print(f"Original features (784D): {accuracy_original:.4f}")
print(f"UMAP features (50D): {accuracy_umap:.4f}")
print(f"Dimension reduction: {784/50:.1f}x")
```

---

## Код (Python + umap-learn)

### Встановлення

```bash
pip install umap-learn
```

### Базовий приклад

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
import umap

# Завантажити дані
digits = load_digits()
X = digits.data  # (1797, 64)
y = digits.target

# Scaling (рекомендовано)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# UMAP
reducer = umap.UMAP(
    n_components=2,       # Розмірність виходу
    n_neighbors=15,       # Локальна структура (5-50)
    min_dist=0.1,         # Мінімальна відстань (0.0-0.99)
    metric='euclidean',   # Метрика відстані
    random_state=42
)

X_umap = reducer.fit_transform(X_scaled)

print(f"Original shape: {X.shape}")
print(f"UMAP shape: {X_umap.shape}")

# Візуалізація
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], 
                     c=y, cmap='tab10', s=20, alpha=0.7)
plt.colorbar(scatter, label='Digit', ticks=range(10))
plt.title('UMAP Projection', fontsize=14, fontweight='bold')
plt.xlabel('UMAP 1', fontsize=12)
plt.ylabel('UMAP 2', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Transform нових даних (ключова перевага!)

```python
# Fit на train
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

reducer = umap.UMAP(n_components=2, random_state=42)
X_train_umap = reducer.fit_transform(X_train)

# Transform test (НА ВІДМІНУ ВІД t-SNE!)
X_test_umap = reducer.transform(X_test)

# Візуалізація train + test
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_train_umap[:, 0], X_train_umap[:, 1], 
           c=y_train, cmap='tab10', s=20, alpha=0.6)
plt.title('Train Set', fontsize=13, fontweight='bold')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(X_test_umap[:, 0], X_test_umap[:, 1],
           c=y_test, cmap='tab10', s=20, alpha=0.6)
plt.title('Test Set (transformed)', fontsize=13, fontweight='bold')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Порівняння PCA, t-SNE, UMAP

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import time

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
print("Running PCA...")
start = time.time()
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
time_pca = time.time() - start

# t-SNE
print("Running t-SNE...")
start = time.time()
tsne = TSNE(n_components=2, random_state=42, verbose=0)
X_tsne = tsne.fit_transform(X_scaled)
time_tsne = time.time() - start

# UMAP
print("Running UMAP...")
start = time.time()
reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X_scaled)
time_umap = time.time() - start

# Візуалізація
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# PCA
axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', s=15, alpha=0.6)
axes[0].set_title(f'PCA\nTime: {time_pca:.2f}s', fontsize=13, fontweight='bold')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')
axes[0].grid(True, alpha=0.3)

# t-SNE
axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=15, alpha=0.6)
axes[1].set_title(f't-SNE\nTime: {time_tsne:.2f}s', fontsize=13, fontweight='bold')
axes[1].set_xlabel('t-SNE 1')
axes[1].set_ylabel('t-SNE 2')
axes[1].grid(True, alpha=0.3)

# UMAP
scatter = axes[2].scatter(X_umap[:, 0], X_umap[:, 1], 
                         c=y, cmap='tab10', s=15, alpha=0.6)
axes[2].set_title(f'UMAP\nTime: {time_umap:.2f}s', fontsize=13, fontweight='bold')
axes[2].set_xlabel('UMAP 1')
axes[2].set_ylabel('UMAP 2')
axes[2].grid(True, alpha=0.3)

plt.colorbar(scatter, ax=axes, label='Digit', ticks=range(10))
plt.tight_layout()
plt.show()

print(f"\n=== Speed Comparison ===")
print(f"PCA: {time_pca:.2f}s (fastest, linear)")
print(f"UMAP: {time_umap:.2f}s (fast, nonlinear)")
print(f"t-SNE: {time_tsne:.2f}s (slow, nonlinear)")
print(f"\nUMAP is {time_tsne/time_umap:.1f}x faster than t-SNE!")
```

### Вплив параметрів

```python
# n_neighbors: локальна vs глобальна структура
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

n_neighbors_values = [5, 15, 50, 100]

for idx, n_neighbors in enumerate(n_neighbors_values):
    print(f"Running UMAP with n_neighbors={n_neighbors}")
    
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        random_state=42
    )
    
    X_umap = reducer.fit_transform(X_scaled)
    
    axes[idx].scatter(X_umap[:, 0], X_umap[:, 1],
                     c=y, cmap='tab10', s=15, alpha=0.6)
    axes[idx].set_title(f'n_neighbors = {n_neighbors}', 
                       fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('UMAP 1')
    axes[idx].set_ylabel('UMAP 2')
    axes[idx].grid(True, alpha=0.3)

plt.suptitle('Effect of n_neighbors Parameter', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n=== n_neighbors Effects ===")
print("Low (5): Focus on very local structure")
print("Medium (15): Balanced (default, recommended)")
print("High (50-100): Focus on global structure")

# min_dist: щільність vs розподілення
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

min_dist_values = [0.0, 0.1, 0.5, 0.9]

for idx, min_dist in enumerate(min_dist_values):
    print(f"Running UMAP with min_dist={min_dist}")
    
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=min_dist,
        random_state=42
    )
    
    X_umap = reducer.fit_transform(X_scaled)
    
    axes[idx].scatter(X_umap[:, 0], X_umap[:, 1],
                     c=y, cmap='tab10', s=15, alpha=0.6)
    axes[idx].set_title(f'min_dist = {min_dist}', 
                       fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('UMAP 1')
    axes[idx].set_ylabel('UMAP 2')
    axes[idx].grid(True, alpha=0.3)

plt.suptitle('Effect of min_dist Parameter', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n=== min_dist Effects ===")
print("0.0: Dense clusters, points can overlap")
print("0.1: Balanced (default)")
print("0.5-0.9: More spread out, better separation")
```

### Supervised UMAP

```python
# Використати labels для кращої візуалізації

# Unsupervised
reducer_unsup = umap.UMAP(n_components=2, random_state=42)
X_umap_unsup = reducer_unsup.fit_transform(X_scaled)

# Supervised (використовує y)
reducer_sup = umap.UMAP(n_components=2, random_state=42)
X_umap_sup = reducer_sup.fit_transform(X_scaled, y=y)

# Порівняння
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

axes[0].scatter(X_umap_unsup[:, 0], X_umap_unsup[:, 1],
               c=y, cmap='tab10', s=15, alpha=0.6)
axes[0].set_title('Unsupervised UMAP', fontsize=13, fontweight='bold')
axes[0].set_xlabel('UMAP 1')
axes[0].set_ylabel('UMAP 2')
axes[0].grid(True, alpha=0.3)

scatter = axes[1].scatter(X_umap_sup[:, 0], X_umap_sup[:, 1],
                         c=y, cmap='tab10', s=15, alpha=0.6)
axes[1].set_title('Supervised UMAP (uses labels)', 
                 fontsize=13, fontweight='bold')
axes[1].set_xlabel('UMAP 1')
axes[1].set_ylabel('UMAP 2')
axes[1].grid(True, alpha=0.3)

plt.colorbar(scatter, ax=axes, label='Digit', ticks=range(10))
plt.tight_layout()
plt.show()

print("Supervised UMAP uses labels to create better separation!")
```

### Custom metrics

```python
# UMAP підтримує багато метрик!

# Euclidean (default)
reducer_euclidean = umap.UMAP(metric='euclidean', random_state=42)

# Cosine (для text data)
reducer_cosine = umap.UMAP(metric='cosine', random_state=42)

# Manhattan
reducer_manhattan = umap.UMAP(metric='manhattan', random_state=42)

# Hamming (для binary data)
reducer_hamming = umap.UMAP(metric='hamming', random_state=42)

# Custom metric function
def custom_metric(x, y):
    return np.sum(np.abs(x - y))

reducer_custom = umap.UMAP(metric=custom_metric, random_state=42)

# Visualize different metrics
metrics = ['euclidean', 'cosine', 'manhattan']
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, metric in enumerate(metrics):
    reducer = umap.UMAP(metric=metric, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)
    
    axes[idx].scatter(X_umap[:, 0], X_umap[:, 1],
                     c=y, cmap='tab10', s=15, alpha=0.6)
    axes[idx].set_title(f'Metric: {metric}', fontsize=13, fontweight='bold')
    axes[idx].set_xlabel('UMAP 1')
    axes[idx].set_ylabel('UMAP 2')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Параметри UMAP

### Основні параметри

```python
umap.UMAP(
    n_components=2,         # Розмірність виходу
    n_neighbors=15,         # Локальна структура
    min_dist=0.1,           # Мінімальна відстань
    metric='euclidean',     # Метрика відстані
    random_state=None,      # Відтворюваність
    n_epochs=None,          # Кількість епох оптимізації
    learning_rate=1.0,      # Швидкість навчання
    init='spectral',        # Ініціалізація
    verbose=False           # Показувати прогрес
)
```

| Параметр | Опис | Типові значення | Рекомендації |
|----------|------|-----------------|--------------|
| **n_components** | Розмірність виходу | 2, 3, 10-100 | 2 для візуалізації, >2 для ML |
| **n_neighbors** | Кількість сусідів | 5-100 | 15 (default), більше для глобальної |
| **min_dist** | Мін. відстань в embedding | 0.0-0.99 | 0.1 (default), 0.0 для щільних |
| **metric** | Метрика відстані | 'euclidean', 'cosine', ... | Залежить від даних |

### n_neighbors (найважливіший)

**Що це:** Баланс між локальною та глобальною структурою.

**Вплив:**

```python
# Low n_neighbors (2-5): дуже локальна структура
reducer_local = umap.UMAP(n_neighbors=5)
# → Багато дрібних кластерів
# → Локальні деталі зберігаються
# → Глобальна структура може бути шумною

# Medium n_neighbors (10-20): збалансовано
reducer_balanced = umap.UMAP(n_neighbors=15)  # ← Рекомендовано
# → Баланс локальної та глобальної

# High n_neighbors (50-100): глобальна структура
reducer_global = umap.UMAP(n_neighbors=100)
# → Менше деталей, більш гладко
# → Фокус на великих паттернах
```

**Правило:**
- **Малі дані** (< 1000): n_neighbors = 5-10
- **Середні дані** (1000-10000): n_neighbors = 15-30
- **Великі дані** (> 10000): n_neighbors = 30-100

### min_dist

**Що це:** Мінімальна дозволена відстань між точками в embedding.

**Вплив:**

```python
# min_dist = 0.0: максимально щільні кластери
reducer_dense = umap.UMAP(min_dist=0.0)
# → Точки можуть накладатись
# → Чіткі компактні кластери
# → Краще для topology

# min_dist = 0.1: збалансовано
reducer_balanced = umap.UMAP(min_dist=0.1)  # ← Рекомендовано
# → Помірна щільність

# min_dist = 0.5-0.99: розподілені точки
reducer_spread = umap.UMAP(min_dist=0.8)
# → Більше простору між точками
# → Легше бачити окремі points
```

**Коли що:**
- **Візуалізація кластерів** → min_dist = 0.0-0.1
- **Розглядати окремі точки** → min_dist = 0.3-0.5
- **Downstream ML** → min_dist = 0.1 (default)

### metric

**Доступні метрики:**

| Метрика | Використання |
|---------|--------------|
| **euclidean** | Числові features (default) |
| **manhattan** | Robust до outliers |
| **cosine** | Text, high-dimensional sparse |
| **correlation** | Gene expression |
| **hamming** | Binary/categorical |
| **jaccard** | Set data |
| Custom function | Будь-яка власна |

---

## Supervised UMAP

### Що це?

**Supervised UMAP** використовує labels (якщо є) для кращого розділення класів.

### Як використовувати

```python
# Unsupervised
reducer_unsup = umap.UMAP(n_components=2)
X_umap_unsup = reducer_unsup.fit_transform(X)

# Supervised (передай y!)
reducer_sup = umap.UMAP(n_components=2)
X_umap_sup = reducer_sup.fit_transform(X, y=y)

# Semi-supervised (частково labeled)
# y містить -1 для unlabeled points
y_partial = y.copy()
y_partial[np.random.rand(len(y)) < 0.5] = -1

reducer_semi = umap.UMAP(n_components=2)
X_umap_semi = reducer_semi.fit_transform(X, y=y_partial)
```

### Порівняння

```python
from sklearn.metrics import silhouette_score

# Unsupervised
sil_unsup = silhouette_score(X_umap_unsup, y)

# Supervised
sil_sup = silhouette_score(X_umap_sup, y)

print(f"Unsupervised Silhouette: {sil_unsup:.4f}")
print(f"Supervised Silhouette: {sil_sup:.4f}")
print(f"Improvement: {(sil_sup - sil_unsup)/sil_unsup*100:.1f}%")
```

**Коли використовувати:**
- ✅ Є partial labels (semi-supervised)
- ✅ Хочеш краще розділення класів
- ✅ Classification task downstream

**Коли НЕ використовувати:**
- ❌ Unsupervised clustering (немає labels)
- ❌ Exploratory analysis (може bias results)

---

## Переваги та недоліки

### Переваги ✓

| Перевага | Пояснення |
|----------|-----------|
| **Швидкість** | 10-100x швидше за t-SNE |
| **Глобальна структура** | Зберігає відстані між кластерами |
| **Transform нових даних** | Є .transform() метод! |
| **Для ML tasks** | Можна використати для supervised learning |
| **Масштабованість** | Працює на мільйонах точок |
| **Гнучкість** | Custom metrics, supervised mode |
| **Локальна структура** | Також зберігає близькість |

### Недоліки ✗

| Недолік | Пояснення |
|---------|-----------|
| **Стохастичність** | Трохи різні результати при кожному запуску |
| **Новіший метод** | Менше перевірений часом ніж PCA/t-SNE |
| **Складність** | Складніше зрозуміти математику |
| **Чутливість до параметрів** | n_neighbors впливає на результат |
| **Осі без значення** | Як t-SNE, не інтерпретовні |
| **Потребує tuning** | Треба підбирати параметри |

---

## Порівняння з іншими методами

### Детальна таблиця

| Критерій | PCA | t-SNE | UMAP | LDA |
|----------|-----|-------|------|-----|
| **Швидкість** | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Локальна структура** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Глобальна структура** | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Візуалізація** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Для ML** | ✅ | ❌ | ✅ | ✅ |
| **Transform test** | ✅ | ❌ | ✅ | ✅ |
| **Детермінізм** | ✅ | ❌ | ⚠️ | ✅ |
| **Інтерпретованість** | ⭐⭐⭐⭐⭐ | ⭐ | ⭐ | ⭐⭐⭐⭐⭐ |

### UMAP vs t-SNE (детально)

**UMAP переваги:**
- ✅ Набагато швидше (10-100x)
- ✅ Зберігає глобальну структуру
- ✅ .transform() для нових даних
- ✅ Можна використати для ML
- ✅ Масштабується краще

**t-SNE переваги:**
- ✅ Більш перевірений часом
- ✅ Трохи краща локальна структура
- ✅ Більше матеріалів та прикладів

**Рекомендація:**
- **Візуалізація:** спробуй обидва, UMAP частіше краще
- **ML preprocessing:** тільки UMAP
- **Великі дані:** тільки UMAP
- **Публікація:** можна показати обидва для порівняння

---

## Коли використовувати UMAP

### Ідеально підходить ✓

- **Візуалізація + downstream ML** — універсальний
- **Великі дані** (> 10,000 точок) — швидше за t-SNE
- **Потрібен .transform()** — нові дані
- **Глобальна структура** важлива — відстані між кластерами
- **Швидкість критична** — набагато швидше t-SNE
- **Custom metrics** — text, graphs, тощо
- **Semi-supervised** — є частково labeled дані

### Краще використати інше ✗

- **Інтерпретація осей** → PCA, LDA
- **Максимальна стабільність** → PCA
- **Лінійні дані** → PCA швидше та простіше
- **Дуже малі дані** (< 100) → t-SNE може бути кращим
- **Потрібні точні probability scores** → t-SNE

---

## Практичні поради 💡

### 1. Почни з default параметрів

```python
# ✅ Default параметри добрі для більшості випадків
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1
)
```

### 2. Експериментуй з n_neighbors

```python
# Спробуй 3-5 значень
for n_neighbors in [5, 15, 30, 100]:
    reducer = umap.UMAP(n_neighbors=n_neighbors)
    X_umap = reducer.fit_transform(X)
    
    # Візуалізуй та порівняй
    plt.figure()
    plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y)
    plt.title(f'n_neighbors = {n_neighbors}')
    plt.show()
```

### 3. Використовуй supervised якщо є labels

```python
# Якщо є labels для кращого розділення
reducer = umap.UMAP(n_components=2)
X_umap = reducer.fit_transform(X, y=y)  # ← Передай y!
```

### 4. Scaling для числових даних

```python
# Рекомендовано (хоча UMAP менш чутливий ніж PCA)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

reducer = umap.UMAP()
X_umap = reducer.fit_transform(X_scaled)
```

### 5. Підбирай metric до типу даних

```python
# Euclidean для числових
reducer_num = umap.UMAP(metric='euclidean')

# Cosine для text/TF-IDF
reducer_text = umap.UMAP(metric='cosine')

# Hamming для binary
reducer_bin = umap.UMAP(metric='hamming')
```

### 6. Використовуй для preprocessing перед ML

```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('umap', umap.UMAP(n_components=20)),
    ('classifier', RandomForestClassifier())
])

pipeline.fit(X_train, y_train)
score = pipeline.score(X_test, y_test)
```

### 7. min_dist залежить від задачі

```python
# Візуалізація кластерів → 0.0-0.1 (щільні)
reducer_vis = umap.UMAP(min_dist=0.0)

# Розглядати окремі точки → 0.3-0.5
reducer_points = umap.UMAP(min_dist=0.4)

# ML tasks → 0.1 (default)
reducer_ml = umap.UMAP(min_dist=0.1)
```

### 8. Зберігай trained model

```python
import pickle

# Зберегти
with open('umap_model.pkl', 'wb') as f:
    pickle.dump(reducer, f)

# Завантажити
with open('umap_model.pkl', 'rb') as f:
    reducer = pickle.load(f)

# Transform нових даних
X_new_umap = reducer.transform(X_new)
```

### 9. Перевіряй кілька random_state

```python
# UMAP частково стохастичний
results = []

for seed in range(5):
    reducer = umap.UMAP(random_state=seed)
    X_umap = reducer.fit_transform(X)
    
    # Оцінка якості
    score = silhouette_score(X_umap, y)
    results.append((seed, score, X_umap))

# Вибери найкращий
best_seed, best_score, best_X_umap = max(results, key=lambda x: x[1])
print(f"Best random_state: {best_seed} (score: {best_score:.4f})")
```

### 10. Візуалізуй і якісно оціни

```python
# Не покладайся тільки на метрики!
# Подивись візуально чи має сенс

plt.figure(figsize=(12, 10))
scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], 
                     c=y, cmap='tab10', s=20, alpha=0.6)
plt.colorbar(scatter)
plt.title('UMAP Visualization')
plt.show()

# Запитай себе:
# - Чи кластери мають сенс?
# - Чи відповідає domain knowledge?
# - Чи є несподівані паттерни?
```

---

## Реальні застосування

### 1. Single-cell RNA-seq Analysis

**Задача:** Візуалізувати та кластеризувати клітини за експресією генів.

**Дані:**
- 50,000 клітин × 20,000 генів
- Дуже високорозмірні, розріджені

**Підхід:**
```python
# 1. Preprocessing
# Normalize, log-transform

# 2. Feature selection (top variable genes)
from sklearn.feature_selection import SelectKBest
selector = SelectKBest(k=2000)
X_selected = selector.fit_transform(gene_expression, cell_types)

# 3. UMAP
reducer = umap.UMAP(
    n_neighbors=30,
    min_dist=0.3,
    metric='correlation',  # Для gene expression
    random_state=42
)

cell_umap = reducer.fit_transform(X_selected)

# 4. Clustering на UMAP space
from sklearn.cluster import HDBSCAN
clusterer = HDBSCAN(min_cluster_size=50)
clusters = clusterer.fit_predict(cell_umap)

# Візуалізація
plt.scatter(cell_umap[:, 0], cell_umap[:, 1], 
           c=clusters, cmap='tab20', s=1)
plt.title('Cell Types (UMAP)')
```

**Результат:**
- Виявлення типів клітин
- Траєкторії диференціації
- Рідкісні популяції

### 2. Text Document Clustering

**Задача:** Візуалізувати та кластеризувати документи.

**Дані:**
- 100,000 документів
- TF-IDF vectors (10,000D)

**Підхід:**
```python
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(documents)

# 2. UMAP з cosine metric
reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    metric='cosine',  # ← Важливо для text!
    random_state=42
)

doc_umap = reducer.fit_transform(X_tfidf)

# 3. Clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=20)
topics = kmeans.fit_predict(doc_umap)

# Візуалізація
plt.scatter(doc_umap[:, 0], doc_umap[:, 1], 
           c=topics, cmap='tab20', s=5, alpha=0.5)
plt.title('Document Topics (UMAP)')
```

### 3. Recommendation Systems

**Задача:** Embedding користувачів/товарів для рекомендацій.

**Підхід:**
```python
# User-item interaction matrix
# (sparse, high-dimensional)

# UMAP embedding
reducer_users = umap.UMAP(
    n_components=50,  # ← Не тільки 2D!
    n_neighbors=20,
    metric='cosine'
)

user_embeddings = reducer_users.fit_transform(user_item_matrix)

# Використати embeddings для nearest neighbors
from sklearn.neighbors import NearestNeighbors

nn = NearestNeighbors(n_neighbors=10, metric='cosine')
nn.fit(user_embeddings)

# Знайти схожих користувачів
distances, indices = nn.kneighbors([user_embeddings[user_id]])
similar_users = indices[0]

# Рекомендації з їхніх уподобань
```

### 4. Image Similarity Search

**Задача:** Навігація по великій колекції зображень.

**Підхід:**
```python
# 1. CNN features (ResNet, VGG)
from torchvision import models
resnet = models.resnet50(pretrained=True)
# Extract features: (n_images, 2048)

# 2. UMAP для швидкого пошуку
reducer = umap.UMAP(
    n_components=128,  # Зменшити для швидкості
    n_neighbors=15,
    metric='cosine'
)

image_embeddings = reducer.fit_transform(cnn_features)

# 3. Approximate nearest neighbors (для мільйонів)
from annoy import AnnoyIndex

index = AnnoyIndex(128, 'angular')
for i, emb in enumerate(image_embeddings):
    index.add_item(i, emb)
index.build(10)

# 4. Пошук схожих зображень
similar_images = index.get_nns_by_item(image_id, 10)
```

### 5. Fraud Detection

**Задача:** Виявити аномальні транзакції.

**Підхід:**
```python
# 1. Features: amount, time, location, merchant, etc.
X_scaled = scaler.fit_transform(transaction_features)

# 2. UMAP embedding
reducer = umap.UMAP(n_components=2, random_state=42)
trans_umap = reducer.fit_transform(X_scaled)

# 3. Density-based outlier detection
from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(n_neighbors=20)
outlier_labels = lof.fit_predict(trans_umap)

# Візуалізація
plt.scatter(trans_umap[:, 0], trans_umap[:, 1],
           c=(outlier_labels == -1), cmap='RdYlGn',
           s=10, alpha=0.5)
plt.title('Fraud Detection (UMAP)')
# Outliers = potential fraud
```

---

## Поширені помилки ❌

### 1. Не налаштовувати параметри

```python
# ❌ Завжди default без експериментів
reducer = umap.UMAP()
X_umap = reducer.fit_transform(X)

# ✅ Спробуй різні n_neighbors та min_dist
for n_neighbors in [5, 15, 50]:
    for min_dist in [0.0, 0.1, 0.5]:
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
        # Візуалізуй та порівняй
```

### 2. Неправильна метрика для типу даних

```python
# ❌ Euclidean для text/TF-IDF
reducer = umap.UMAP(metric='euclidean')
X_umap = reducer.fit_transform(tfidf_matrix)

# ✅ Cosine для sparse text
reducer = umap.UMAP(metric='cosine')
X_umap = reducer.fit_transform(tfidf_matrix)
```

### 3. Fit на всіх даних (train+test)

```python
# ❌ DATA LEAKAGE
X_all = np.vstack([X_train, X_test])
reducer = umap.UMAP()
reducer.fit(X_all)  # ← Leakage!

# ✅ Fit тільки на train
reducer = umap.UMAP()
reducer.fit(X_train)

X_train_umap = reducer.transform(X_train)
X_test_umap = reducer.transform(X_test)
```

### 4. Забути про scaling

```python
# ❌ Без scaling (особливо для euclidean)
reducer = umap.UMAP(metric='euclidean')
X_umap = reducer.fit_transform(X)

# ✅ Зі scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_umap = reducer.fit_transform(X_scaled)
```

### 5. Інтерпретувати осі

```python
# ❌ "UMAP axis 1 представляє..."
# Осі не мають значення!

# ✅ "Кластери показують..."
# Інтерпретуй тільки групи та відстані
```

### 6. Не використовувати supervised якщо є labels

```python
# ❌ Unsupervised коли є labels
reducer = umap.UMAP()
X_umap = reducer.fit_transform(X)

# ✅ Supervised для кращого розділення
reducer = umap.UMAP()
X_umap = reducer.fit_transform(X, y=y)
```

### 7. Занадто мало n_neighbors для великих даних

```python
# ❌ n_neighbors=5 для 100,000 точок
# Втрачається глобальна структура

# ✅ Збільш для великих даних
n_neighbors = min(100, len(X) // 100)
reducer = umap.UMAP(n_neighbors=n_neighbors)
```

### 8. Не зберігати trained model

```python
# ❌ Fit знову для нових даних
reducer = umap.UMAP()
reducer.fit(X_new)  # Втрачає consistency!

# ✅ Transform на вже fitted model
X_new_umap = reducer.transform(X_new)
```

---

## Пов'язані теми

- [[01_PCA]] — лінійна альтернатива
- [[02_t-SNE]] — попередник UMAP
- [[04_LDA]] — supervised reduction
- [[05_Autoencoders]] — neural network approach
- [[06_Manifold_Learning]] — інші методи
- [[Clustering_Methods]] — для downstream tasks

## Ресурси

- [UMAP Documentation](https://umap-learn.readthedocs.io/)
- [Original Paper: McInnes et al. (2018)](https://arxiv.org/abs/1802.03426)
- [Understanding UMAP (Andy Coenen & Adam Pearce)](https://pair-code.github.io/understanding-umap/)
- [How UMAP Works (Leland McInnes)](https://www.youtube.com/watch?v=nq6iPZVUxZU)

---

## Ключові висновки

> UMAP — це швидкий нелінійний алгоритм dimensionality reduction на основі топології, який зберігає як локальну так і глобальну структуру, підходить для візуалізації та downstream ML tasks.

**Основні принципи:**
- **Топологічний підхід:** граф сусідства → optimization
- **Швидкість:** 10-100x швидше за t-SNE
- **Універсальність:** візуалізація + ML + transform
- **Баланс:** локальна + глобальна структура

**Алгоритм:**
1. Побудувати fuzzy simplicial set (граф) у high-dim
2. Ініціалізувати low-dim представлення
3. Оптимізувати cross-entropy між графами
4. Stochastic gradient descent

**Ключові параметри:**
- **n_neighbors** (15 default) — баланс локальної/глобальної
- **min_dist** (0.1 default) — щільність кластерів
- **metric** — залежить від типу даних
- **supervised mode** — використай labels якщо є

**Переваги над t-SNE:**
- ⚡ Набагато швидше
- 🌍 Зберігає глобальну структуру
- 🔄 Є .transform() метод
- 📊 Для ML tasks

**Коли використовувати:**
- Візуалізація + ML + великі дані = UMAP ✓
- Максимальна швидкість → UMAP ✓
- Тільки візуалізація + малі дані → t-SNE теж OK
- Лінійні дані → PCA простіше ✓

**Найважливіше:**
- **Default параметри добрі** — почни з них
- **Експериментуй з n_neighbors** — найважливіший параметр
- **Використовуй supervised** якщо є labels
- **Правильна метрика** для типу даних
- **Масштабується** на мільйони точок
- **.transform()** працює для нових даних!

---

#ml #unsupervised-learning #dimensionality-reduction #umap #visualization #manifold-learning #nonlinear #topology #fast
