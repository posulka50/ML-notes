# t-SNE (t-Distributed Stochastic Neighbor Embedding)

## Що це?

**t-SNE (t-Distributed Stochastic Neighbor Embedding)** — це **нелінійний** алгоритм dimensionality reduction, який спеціалізується на **візуалізації** високорозмірних даних у 2D або 3D, зберігаючи **локальну структуру** (близькі точки залишаються близькими).

**Головна ідея:** перетворити високорозмірні дані в низькорозмірний простір так, щоб **подібні точки були близько, а різні — далеко**, оптимізуючи вірогідності сусідства.

## Навіщо потрібен?

- 🎨 **Візуалізація** — головне призначення! Побачити структуру високорозмірних даних
- 🔍 **Виявлення кластерів** — подібні об'єкти групуються разом
- 🧬 **Exploratory Data Analysis** — розуміння структури даних
- 📊 **Якість embedding** — кращий за PCA для нелінійних структур
- 🎯 **Перевірка гіпотез** — чи справді існують кластери?
- 🌈 **Красиві візуалізації** — для презентацій, статей

## Коли використовувати?

**Потрібно:**

- **Візуалізація** високорозмірних даних (головне застосування!)
- **Нелінійні структури** — складні manifolds
- **Виявлення кластерів** візуально
- **Exploratory analysis** — подивитись на дані
- Розмірність **10-1000** features
- Дані **не занадто великі** (< 10,000 точок)

**Не потрібно:**

- **Downstream ML tasks** — t-SNE НЕ зберігає глобальну структуру → UMAP
- **Інтерпретація осей** — вісі не мають значення → PCA
- **Великі дані** (> 100,000 точок) → UMAP або PCA
- **Лінійні дані** — PCA швидше та краще
- **Швидкість критична** → PCA

---
### Що робить t-SNE?

**Крок 1:** У високорозмірному просторі обчислює **ймовірності** того, що точка $x_i$ вибере точку $x_j$ як "сусіда":

$$p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}$$

**Інтуїція:** Близькі точки → висока ймовірність, далекі → низька.

**Крок 2:** У низькорозмірному просторі (2D) обчислює **аналогічні ймовірності** використовуючи t-розподіл:

$$q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|y_k - y_l\|^2)^{-1}}$$

**Крок 3:** Мінімізує різницю між $p_{ij}$ та $q_{ij}$ (KL-divergence):

$$KL(P||Q) = \sum_i \sum_j p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

**Крок 4:** Використовує gradient descent для оптимізації позицій $y_i$.

### Чому t-розподіл?

**Проблема Gaussian:** "crowding problem" — у низькорозмірному просторі не вистачає місця для всіх точок.

```
High-dim: відстані 1, 2, 3, 4, 5, ...
Low-dim (2D): всі точки змушені бути близько!

Gaussian kernel:
     ∩
    / \
   /   \
  /     \___________
Швидко спадає → all points clumped

t-distribution:
     ∩
    / \
   /   \
  /     \_____
 /           \___
Heavy tails → moderate distances preserved
```

**t-розподіл має "важкі хвости"** → дозволяє точкам бути на помірних відстанях.

---

## Математика

### Умовна ймовірність (high-dimensional)

**Gaussian kernel з адаптивною шириною:**

$$p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}$$

де $\sigma_i$ вибирається так, щоб досягти заданої **perplexity**.

### Perplexity

**Perplexity** — це міра "ефективної кількості сусідів":

$$\text{Perplexity}(P_i) = 2^{H(P_i)}$$

де $H(P_i) = -\sum_j p_{j|i} \log_2 p_{j|i}$ — ентропія.

**Типові значення:** 5-50 (зазвичай 30)

**Інтуїція:**
- Perplexity = 5 → кожна точка має ~5 "ефективних сусідів"
- Perplexity = 50 → ~50 сусідів
- Більше perplexity → більш глобальна структура

### Симетризація

**Зробити ймовірності симетричними:**

$$p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}$$

### Low-dimensional ймовірності (Student t-distribution)

$$q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|y_k - y_l\|^2)^{-1}}$$

**Чому (1 + d²)⁻¹?** Це Student t-розподіл з 1 ступенем свободи.

### Градієнт (для оптимізації)

$$\frac{\partial KL}{\partial y_i} = 4 \sum_j (p_{ij} - q_{ij})(y_i - y_j)(1 + \|y_i - y_j\|^2)^{-1}$$

**Інтуїція:**
- $(p_{ij} - q_{ij})$ — наскільки далеко від цільової ймовірності
- $(y_i - y_j)$ — напрямок руху
- $(1 + \|y_i - y_j\|^2)^{-1}$ — важкі хвости (модулює силу)

---

## Простий приклад: Swiss Roll → 2D

### Дані

**Swiss Roll** — класичний нелінійний manifold:

```
3D Swiss Roll (side view):
    z
    |  ●●●
    | ●   ●
    |●     ●
    |●     ●
    | ●   ●
    |  ●●●
    |_______ x
    
Згорнутий аркуш паперу
```

### PCA (провал)

```
PCA проекція (лінійна):
    PC2
     |
     |●●●●●●●
     |●●●●●●●
     |●●●●●●●
     |_______ PC1
     
Розгорнути не може! ❌
```

### t-SNE (успіх)

```
t-SNE 2D:
    
    ●●●●●●●●●●●●●●●
    ●            ●
    ●            ●
    ●●●●●●●●●●●●●●●
    
Розгорнув аркуш! ✓
```

**Результат:** t-SNE знаходить нелінійну структуру та "розгортає" її.

---

## Складний приклад: MNIST

### Задача

MNIST: 70,000 рукописних цифр, кожна 28×28 = 784 пікселі.

**Мета:** Візуалізувати 784D → 2D, щоб побачити кластери цифр.

### Результат t-SNE

```
t-SNE 2D проекція:

        3   2
    5   3  222
   555 333 22
    55  3   2
    
  1111    4444
   11      444
   11       44
   
    0000   9  8
    000   999 888
    000    99  88
   0000   9   8
   
      6    7777
    666     77
    6666     7
     66      7
```

**Спостереження:**
- ✅ Кожна цифра формує чіткий кластер
- ✅ Подібні цифри близько (3 та 8, 4 та 9)
- ✅ Різні стилі написання в межах кластера
- ✅ 784 розмірності → 2D з збереженням структури!

### Порівняння з PCA

| Метод | Кластери | Перекриття | Структура |
|-------|----------|------------|-----------|
| PCA | Розмиті | Багато | Лінійна проекція |
| t-SNE | Чіткі | Мало | Нелінійне розгортання |

---

## Код (Python + scikit-learn)

### Базовий приклад

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

# 1. Завантажити дані
digits = load_digits()
X = digits.data  # (1797, 64) - 8x8 зображення
y = digits.target

print(f"Original shape: {X.shape}")

# 2. Scaling (рекомендовано)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. t-SNE
tsne = TSNE(
    n_components=2,      # 2D візуалізація
    perplexity=30,       # ~30 ефективних сусідів
    learning_rate=200,   # швидкість навчання
    n_iter=1000,         # кількість ітерацій
    random_state=42,
    verbose=1            # показувати прогрес
)

X_tsne = tsne.fit_transform(X_scaled)

print(f"t-SNE shape: {X_tsne.shape}")
print(f"KL divergence: {tsne.kl_divergence_:.4f}")

# 4. Візуалізація
plt.figure(figsize=(12, 10))

# Scatter plot з кольорами для кожної цифри
scatter = plt.scatter(
    X_tsne[:, 0], 
    X_tsne[:, 1],
    c=y,
    cmap='tab10',
    s=20,
    alpha=0.7,
    edgecolors='black',
    linewidths=0.5
)

plt.colorbar(scatter, label='Digit', ticks=range(10))
plt.title('t-SNE Visualization of Digits Dataset', 
         fontsize=14, fontweight='bold')
plt.xlabel('t-SNE Component 1', fontsize=12)
plt.ylabel('t-SNE Component 2', fontsize=12)
plt.grid(True, alpha=0.3)

# Додати мітки кластерів
for digit in range(10):
    mask = y == digit
    center = X_tsne[mask].mean(axis=0)
    plt.annotate(
        str(digit),
        center,
        fontsize=20,
        fontweight='bold',
        color='white',
        bbox=dict(boxstyle='circle', facecolor='black', alpha=0.7)
    )

plt.tight_layout()
plt.show()
```

### Порівняння PCA vs t-SNE

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# t-SNE
tsne = TSNE(n_components=2, random_state=42, verbose=1)
X_tsne = tsne.fit_transform(X_scaled)

# Візуалізація порівняння
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# PCA
scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], 
                          c=y, cmap='tab10', s=20, alpha=0.7,
                          edgecolors='black', linewidths=0.5)
axes[0].set_title(
    f'PCA (Explained Variance: {pca.explained_variance_ratio_.sum():.1%})',
    fontsize=13, fontweight='bold'
)
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=11)
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=11)
axes[0].grid(True, alpha=0.3)
plt.colorbar(scatter1, ax=axes[0], label='Digit')

# t-SNE
scatter2 = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1],
                          c=y, cmap='tab10', s=20, alpha=0.7,
                          edgecolors='black', linewidths=0.5)
axes[1].set_title(
    f't-SNE (KL divergence: {tsne.kl_divergence_:.2f})',
    fontsize=13, fontweight='bold'
)
axes[1].set_xlabel('t-SNE Component 1', fontsize=11)
axes[1].set_ylabel('t-SNE Component 2', fontsize=11)
axes[1].grid(True, alpha=0.3)
plt.colorbar(scatter2, ax=axes[1], label='Digit')

plt.tight_layout()
plt.show()

print("\n=== Comparison ===")
print("PCA: Linear projection, fast, interpretable axes")
print("t-SNE: Nonlinear, better clusters, axes not interpretable")
```

### Вплив Perplexity

```python
# Перевірити різні perplexity
perplexity_values = [5, 30, 50, 100]

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

for idx, perplexity in enumerate(perplexity_values):
    print(f"\nRunning t-SNE with perplexity={perplexity}...")
    
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=200,
        n_iter=1000,
        random_state=42,
        verbose=0
    )
    
    X_tsne = tsne.fit_transform(X_scaled)
    
    # Візуалізація
    scatter = axes[idx].scatter(
        X_tsne[:, 0], X_tsne[:, 1],
        c=y, cmap='tab10', s=15, alpha=0.7
    )
    
    axes[idx].set_title(
        f'Perplexity = {perplexity}\nKL div: {tsne.kl_divergence_:.2f}',
        fontsize=12, fontweight='bold'
    )
    axes[idx].set_xlabel('t-SNE 1', fontsize=10)
    axes[idx].set_ylabel('t-SNE 2', fontsize=10)
    axes[idx].grid(True, alpha=0.3)

plt.colorbar(scatter, ax=axes, label='Digit', 
            orientation='horizontal', pad=0.02)
plt.tight_layout()
plt.show()

print("\n=== Perplexity Effects ===")
print("Low (5-10): Local structure, many small clusters")
print("Medium (30-50): Balanced, recommended")
print("High (100+): Global structure, larger clusters")
```

### Повний приклад: MNIST з PCA preprocessing

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import time

# Завантажити MNIST (може зайняти час)
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X = mnist.data.to_numpy()
y = mnist.target.to_numpy().astype(int)

# Вибрати підмножину (t-SNE повільний на 70K точок!)
n_samples = 5000
indices = np.random.choice(len(X), n_samples, replace=False)
X_sample = X[indices]
y_sample = y[indices]

print(f"Using {n_samples} samples")
print(f"Original shape: {X_sample.shape}")

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sample)

# Стратегія: PCA спочатку (784D → 50D), потім t-SNE (50D → 2D)
# Це набагато швидше ніж пряме t-SNE на 784D!

print("\n=== Step 1: PCA (784D → 50D) ===")
pca = PCA(n_components=50)
start = time.time()
X_pca = pca.fit_transform(X_scaled)
pca_time = time.time() - start

print(f"PCA time: {pca_time:.2f}s")
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")

print("\n=== Step 2: t-SNE (50D → 2D) ===")
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate='auto',
    n_iter=1000,
    random_state=42,
    verbose=1
)

start = time.time()
X_tsne = tsne.fit_transform(X_pca)
tsne_time = time.time() - start

print(f"t-SNE time: {tsne_time:.2f}s")
print(f"Total time: {pca_time + tsne_time:.2f}s")
print(f"KL divergence: {tsne.kl_divergence_:.4f}")

# Візуалізація
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# PCA 2D
axes[0].scatter(X_pca[:, 0], X_pca[:, 1], 
               c=y_sample, cmap='tab10', s=5, alpha=0.6)
axes[0].set_title('PCA (784D → 2D)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('PC1', fontsize=11)
axes[0].set_ylabel('PC2', fontsize=11)
axes[0].grid(True, alpha=0.3)

# t-SNE 2D
scatter = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1],
                         c=y_sample, cmap='tab10', s=5, alpha=0.6)
axes[1].set_title('t-SNE (PCA 50D → 2D)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('t-SNE 1', fontsize=11)
axes[1].set_ylabel('t-SNE 2', fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.colorbar(scatter, ax=axes, label='Digit', ticks=range(10))
plt.tight_layout()
plt.show()

# Density plot для кожної цифри
fig, axes = plt.subplots(2, 5, figsize=(16, 7))
axes = axes.ravel()

for digit in range(10):
    mask = y_sample == digit
    
    axes[digit].scatter(
        X_tsne[~mask, 0], X_tsne[~mask, 1],
        c='lightgray', s=1, alpha=0.3
    )
    axes[digit].scatter(
        X_tsne[mask, 0], X_tsne[mask, 1],
        c='red', s=10, alpha=0.7
    )
    
    axes[digit].set_title(f'Digit {digit} (n={np.sum(mask)})',
                         fontsize=11, fontweight='bold')
    axes[digit].axis('off')

plt.suptitle('t-SNE: Individual Digit Clusters', 
            fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Оцінка кластеризації
from sklearn.metrics import silhouette_score, calinski_harabasz_score

print("\n=== Clustering Quality Metrics ===")

# На PCA embedding
sil_pca = silhouette_score(X_pca[:, :2], y_sample)
ch_pca = calinski_harabasz_score(X_pca[:, :2], y_sample)

# На t-SNE embedding
sil_tsne = silhouette_score(X_tsne, y_sample)
ch_tsne = calinski_harabasz_score(X_tsne, y_sample)

print(f"\nPCA 2D:")
print(f"  Silhouette Score: {sil_pca:.4f}")
print(f"  Calinski-Harabasz: {ch_pca:.2f}")

print(f"\nt-SNE 2D:")
print(f"  Silhouette Score: {sil_tsne:.4f}")
print(f"  Calinski-Harabasz: {ch_tsne:.2f}")

print("\nHigher = better separation")
print("t-SNE shows much better cluster separation!")
```

### Interactive 3D t-SNE

```python
# t-SNE в 3D (для інтерактивної візуалізації)
tsne_3d = TSNE(n_components=3, random_state=42, verbose=1)
X_tsne_3d = tsne_3d.fit_transform(X_pca)

# Plotly для інтерактивної візуалізації
import plotly.graph_objects as go

fig = go.Figure(data=[go.Scatter3d(
    x=X_tsne_3d[:, 0],
    y=X_tsne_3d[:, 1],
    z=X_tsne_3d[:, 2],
    mode='markers',
    marker=dict(
        size=3,
        color=y_sample,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title="Digit")
    ),
    text=[f'Digit: {d}' for d in y_sample],
    hovertemplate='<b>%{text}</b><br>' +
                  't-SNE1: %{x:.2f}<br>' +
                  't-SNE2: %{y:.2f}<br>' +
                  't-SNE3: %{z:.2f}<br>' +
                  '<extra></extra>'
)])

fig.update_layout(
    title='Interactive 3D t-SNE Visualization',
    scene=dict(
        xaxis_title='t-SNE Component 1',
        yaxis_title='t-SNE Component 2',
        zaxis_title='t-SNE Component 3'
    ),
    width=900,
    height=700
)

fig.show()
```

---

## Параметри t-SNE

### Основні параметри

```python
TSNE(
    n_components=2,         # Розмірність виходу (2 або 3)
    perplexity=30.0,        # Ефективна кількість сусідів
    learning_rate=200.0,    # Швидкість навчання
    n_iter=1000,            # Кількість ітерацій
    metric='euclidean',     # Метрика відстані
    init='pca',             # Ініціалізація ('random' або 'pca')
    random_state=42,        # Відтворюваність
    verbose=0               # Виводити прогрес
)
```

| Параметр | Опис | Типові значення | Рекомендації |
|----------|------|-----------------|--------------|
| **n_components** | Розмірність виходу | 2 (візуалізація), 3 (інтерактивна) | Завжди 2 для статичних графіків |
| **perplexity** | Кількість сусідів | 5-50 | 30 (default), 5-10 для малих даних, 50 для великих |
| **learning_rate** | Швидкість оптимізації | 10-1000 | 'auto' (=n_samples/12) або 200 |
| **n_iter** | Кількість ітерацій | 250-2000 | Мінімум 1000 для convergence |
| **init** | Початкові позиції | 'pca', 'random' | 'pca' швидше та стабільніше |
| **metric** | Метрика відстані | 'euclidean', 'cosine' | 'euclidean' для більшості задач |

### Perplexity (найважливіший параметр)

**Що це:** Балансує між локальною та глобальною структурою.

**Вплив:**

```python
# Low perplexity (5-10): фокус на локальній структурі
tsne_low = TSNE(perplexity=5)
# → Багато дрібних кластерів
# → Локальні відношення зберігаються
# → Глобальна структура може бути втрачена

# Medium perplexity (20-50): збалансовано
tsne_medium = TSNE(perplexity=30)  # ← Рекомендовано
# → Баланс локальної та глобальної структури

# High perplexity (50-100): фокус на глобальній структурі
tsne_high = TSNE(perplexity=100)
# → Менше кластерів
# → Глобальні відношення
# → Локальні деталі можуть згладжуватись
```

**Правило:**
- **Малі дані** (< 1000): perplexity = 5-20
- **Середні дані** (1000-10000): perplexity = 30-50
- **Великі дані** (> 10000): perplexity = 50-100

**Перевірка:**
```python
# Perplexity не може бути більшим за n_samples - 1
max_perplexity = len(X) - 1
recommended = min(30, max_perplexity)
```

### Learning Rate

**Що це:** Швидкість gradient descent оптимізації.

**Вплив:**
- **Занадто низька** (< 10): дуже повільна конвергенція
- **Оптимальна** (10-1000): нормальна конвергенція
- **Занадто висока** (> 1000): нестабільність, погані результати

**Рекомендації:**
```python
# Automatic (рекомендовано)
tsne = TSNE(learning_rate='auto')  # = n_samples / 12

# Або типові значення
tsne = TSNE(learning_rate=200)  # Conservative
tsne = TSNE(learning_rate=500)  # Faster
tsne = TSNE(learning_rate=1000) # Aggressive
```

### Ініціалізація

**'pca' vs 'random':**

```python
# PCA initialization (рекомендовано)
tsne_pca = TSNE(init='pca')
# ✅ Швидша конвергенція
# ✅ Більш стабільні результати
# ✅ Менше локальних мінімумів

# Random initialization
tsne_random = TSNE(init='random')
# ❌ Повільніше
# ❌ Більше варіативності між запусками
# ✅ Іноді знаходить цікаві структури
```

---

## Оптимізація та прискорення

### 1. PCA Preprocessing (КРИТИЧНО для великих даних!)

**Проблема:** t-SNE має складність O(n²), дуже повільний на високій розмірності.

**Рішення:** PCA спочатку для зменшення розмірності.

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ❌ Повільно: t-SNE на 784D
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)  # Дуже повільно!

# ✅ Швидко: PCA → t-SNE
pca = PCA(n_components=50)  # 784D → 50D
X_pca = pca.fit_transform(X)

tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X_pca)  # Набагато швидше!
```

**Правило:**
- Якщо **d > 50** → спочатку PCA до 50D
- Зберігає 90-95% variance
- t-SNE працює на 50D замість 784D

**Порівняння:**
```python
import time

# Без PCA
start = time.time()
tsne = TSNE(n_components=2, verbose=0)
X_tsne_direct = tsne.fit_transform(X[:1000])
time_direct = time.time() - start

# З PCA
start = time.time()
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X[:1000])
tsne = TSNE(n_components=2, verbose=0)
X_tsne_pca = tsne.fit_transform(X_pca)
time_pca = time.time() - start

print(f"Direct t-SNE: {time_direct:.2f}s")
print(f"PCA + t-SNE: {time_pca:.2f}s")
print(f"Speedup: {time_direct/time_pca:.1f}x")
```

### 2. Subsampling для дуже великих даних

```python
# Якщо > 10,000 точок, вибрати підмножину
n_samples = 5000
indices = np.random.choice(len(X), n_samples, replace=False)
X_sample = X[indices]
y_sample = y[indices]

# t-SNE на sample
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X_sample)
```

### 3. Barnes-Hut approximation (вбудовано)

**scikit-learn автоматично використовує Barnes-Hut** для прискорення:

- **Exact t-SNE:** O(n²) — дуже повільно
- **Barnes-Hut:** O(n log n) — набагато швидше

```python
# За замовчуванням використовує Barnes-Hut
tsne = TSNE(n_components=2, method='barnes_hut')

# Або точний метод (повільний, для малих даних)
tsne = TSNE(n_components=2, method='exact')
```

### 4. Multicore t-SNE (сторонні бібліотеки)

**scikit-learn t-SNE не підтримує паралелізм!**

**Альтернативи:**

```python
# MulticoreTSNE (швидше на багатоядерних CPU)
from MulticoreTSNE import MulticoreTSNE as TSNE

tsne = TSNE(n_jobs=4)  # Використати 4 ядра
X_tsne = tsne.fit_transform(X)

# Або openTSNE (найшвидша реалізація)
from openTSNE import TSNE

tsne = TSNE(n_jobs=-1)  # Всі ядра
X_tsne = tsne.fit(X)
```

---

## Інтерпретація результатів t-SNE

### Що означають осі?

**ВАЖЛИВО:** Осі t-SNE **НЕ МАЮТЬ значення**!

```python
# ❌ НЕПРАВИЛЬНО
print(f"t-SNE axis 1 represents...")  # Безглуздо!

# ✅ ПРАВИЛЬНО
print("Clusters in t-SNE space:")
# Важливі тільки відстані між точками, не абсолютні позиції
```

**Чому:**
- t-SNE оптимізує тільки **локальні відстані**
- Глобальні відстані та орієнтація довільні
- Обертання/відображення не змінює інтерпретацію

### Що можна інтерпретувати?

**✅ Можна:**
1. **Кластери** — щільні групи точок
2. **Відносні відстані** — близькі vs далекі точки
3. **Локальна структура** — сусідство
4. **Outliers** — ізольовані точки

**❌ Не можна:**
1. **Абсолютні координати** — без значення
2. **Відстані між кластерами** — не зберігаються
3. **Розміри кластерів** — можуть бути оманливими
4. **Щільність** — може бути артефактом perplexity

### Типові паттерни

**1. Чіткі кластери:**
```
    ●●●        ■■■
     ●●        ■■
    ●●●        ■■■
    
Добре розділені класи
```

**2. Перекриття:**
```
    ●●●
     ●●■■■
    ●●■■■
    
Класи з подібними характеристиками
```

**3. Manifold (континуум):**
```
    ●●●●●●●●●
    ●        ●
    ●●●●●●●●●
    
Безперервна варіація (наприклад, обертання об'єкта)
```

**4. Outliers:**
```
    ●●●        ■■■
     ●●        ■■
    ●●●   •   ■■■
          ↑
        outlier
```

---

## Переваги та недоліки

### Переваги ✓

| Перевага | Пояснення |
|----------|-----------|
| **Нелінійне зменшення** | Знаходить складні manifolds |
| **Краща візуалізація** | Кластери чіткіші ніж PCA |
| **Локальна структура** | Зберігає сусідство точок |
| **Працює з будь-якими даними** | Потрібна тільки матриця відстаней |
| **Виявлення кластерів** | Візуально чіткі групи |
| **Відомий та популярний** | Багато матеріалів, прикладів |

### Недоліки ✗

| Недолік | Пояснення |
|---------|-----------|
| **Тільки візуалізація** | НЕ для downstream ML tasks |
| **Дуже повільний** | O(n²), проблеми на > 10K точок |
| **Недетерміністичний** | Різні результати при кожному запуску |
| **Глобальна структура втрачена** | Відстані між кластерами безглуздові |
| **Чутливість до параметрів** | Perplexity сильно впливає |
| **Осі без значення** | Не інтерпретовні |
| **Crowding problem** | Іноді занадто стискає кластери |

---

## Порівняння з іншими методами

| Метод | Швидкість | Глобальна структура | Downstream ML | Детермінізм | Використання |
|-------|-----------|---------------------|---------------|-------------|--------------|
| **t-SNE** | ⭐ | ❌ | ❌ | ❌ | Тільки візуалізація |
| **PCA** | ⭐⭐⭐⭐⭐ | ✅ | ✅ | ✅ | Загальне зменшення |
| **UMAP** | ⭐⭐⭐⭐ | ✅ | ✅ | ⚠️ | Візуалізація + ML |
| **LDA** | ⭐⭐⭐⭐ | ✅ | ✅ | ✅ | Supervised tasks |
| **Autoencoders** | ⭐⭐ | ⚠️ | ✅ | ⚠️ | Складні дані |

### t-SNE vs PCA

**t-SNE:**
- ✅ Нелінійний (складні структури)
- ✅ Кращі кластери візуально
- ❌ Повільний
- ❌ Тільки для візуалізації
- ❌ Глобальна структура втрачена

**PCA:**
- ✅ Швидкий
- ✅ Зберігає глобальну структуру
- ✅ Для downstream tasks
- ❌ Тільки лінійний

**Коли що:**
- **Візуалізація кластерів** → t-SNE ✓
- **Preprocessing для ML** → PCA ✓
- **Інтерпретація компонентів** → PCA ✓

### t-SNE vs UMAP

**t-SNE:**
- ✅ Більш відомий, перевірений часом
- ✅ Краща локальна структура
- ❌ Дуже повільний
- ❌ Тільки візуалізація

**UMAP:**
- ✅ Набагато швидше (10-100x)
- ✅ Зберігає глобальну структуру
- ✅ Для downstream ML
- ⚠️ Новіший (менше перевірений)

**Рекомендація:** Спробуй UMAP спочатку, потім t-SNE для порівняння.

---

## Коли використовувати t-SNE

### Ідеально підходить ✓

- **Візуалізація** високорозмірних даних (головне призначення!)
- **Exploratory analysis** — подивитись на структуру
- **Виявлення кластерів** візуально
- **Презентації, статті** — красиві графіки
- **Перевірка гіпотез** — чи є кластери?
- **Невеликі дані** (< 10,000 точок)
- **Нелінійні структури** — manifolds

### Краще використати інше ✗

- **Downstream ML** (класифікація, регресія) → UMAP, PCA
- **Великі дані** (> 50,000 точок) → UMAP або PCA → t-SNE
- **Швидкість критична** → PCA, UMAP
- **Потрібна глобальна структура** → PCA, UMAP
- **Інтерпретація осей** важлива → PCA, LDA
- **Лінійні дані** — PCA простіший та швидший

---

## Практичні поради 💡

### 1. ЗАВЖДИ PCA preprocessing для d > 50

```python
# ❌ Повільно
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X_784d)  # Години!

# ✅ Швидко
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_784d)

tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X_pca)  # Хвилини!
```

### 2. Перевіряй різні perplexity

```python
# Спробуй 3-5 значень
perplexities = [5, 30, 50, 100]

for perp in perplexities:
    tsne = TSNE(perplexity=perp, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    # Візуалізуй та порівняй
    plt.figure()
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y)
    plt.title(f'Perplexity = {perp}')
    plt.show()
```

### 3. Запускай кілька разів (недетерміністичний!)

```python
# t-SNE дає різні результати!
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx in range(3):
    tsne = TSNE(random_state=idx)  # Різні random_state
    X_tsne = tsne.fit_transform(X)
    
    axes[idx].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y)
    axes[idx].set_title(f'Run {idx+1}')

plt.show()

# Вибери найкращий візуально
```

### 4. Використовуй init='pca' для стабільності

```python
# Рекомендовано
tsne = TSNE(init='pca', random_state=42)
# ✅ Швидша конвергенція
# ✅ Більш стабільні результати
```

### 5. n_iter мінімум 1000

```python
# ❌ Замало ітерацій
tsne = TSNE(n_iter=250)  # Може не зійтись!

# ✅ Достатньо
tsne = TSNE(n_iter=1000)  # Рекомендовано

# Для складних даних
tsne = TSNE(n_iter=2000)
```

### 6. Перевіряй KL divergence

```python
tsne = TSNE(verbose=1)
X_tsne = tsne.fit_transform(X)

print(f"Final KL divergence: {tsne.kl_divergence_:.4f}")

# Нижче = краще (зазвичай 1-5)
# Якщо > 10 → погана конвергенція, спробуй:
#   - Більше n_iter
#   - Інший learning_rate
#   - Інший perplexity
```

### 7. Scaling перед t-SNE

```python
# Рекомендовано (хоча не критично як для PCA)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

tsne = TSNE()
X_tsne = tsne.fit_transform(X_scaled)
```

### 8. Використовуй для initial exploration, не для остаточних висновків

```python
# ✅ Добре
"t-SNE показує що дані мають ~3 кластери"
"Візуально схоже що класи A та B подібні"

# ❌ Неправильно
"Відстань між кластерами А та Б = 5 одиниць"
"Кластер C більший за кластер D"

# Для кількісного аналізу використовуй інші методи!
```

### 9. Subsampling для > 10K точок

```python
# Якщо дуже багато даних
if len(X) > 10000:
    indices = np.random.choice(len(X), 10000, replace=False)
    X_sample = X[indices]
    y_sample = y[indices]
else:
    X_sample = X
    y_sample = y

tsne = TSNE()
X_tsne = tsne.fit_transform(X_sample)
```

### 10. Комбінуй з кластеризацією для validation

```python
# t-SNE для візуалізації + clustering для кількісного аналізу
from sklearn.cluster import KMeans

# 1. t-SNE візуалізація
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)

# 2. K-Means на оригінальних даних
kmeans = KMeans(n_clusters=3)
labels = kmeans.fit_predict(X)

# 3. Візуалізувати t-SNE з мітками кластерів
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels)
plt.title('t-SNE colored by K-Means clusters')
plt.show()

# Якщо кластери на t-SNE співпадають з K-Means → добре!
```

---

## Реальні застосування

### 1. Genomics (аналіз генів)

**Задача:** Візуалізувати експресію генів у клітинах.

**Дані:**
- Single-cell RNA-seq
- 10,000 клітин × 20,000 генів

**Підхід:**
```python
# 1. PCA preprocessing (20K → 50 компонентів)
pca = PCA(n_components=50)
X_pca = pca.fit_transform(gene_expression)

# 2. t-SNE візуалізація
tsne = TSNE(n_components=2, perplexity=30)
X_tsne = tsne.fit_transform(X_pca)

# 3. Колір по типу клітин (якщо відомо) або кластерам
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cell_types)
```

**Результат:**
- Виявлення підтипів клітин
- Траєкторії диференціації
- Рідкісні популяції

### 2. Word Embeddings

**Задача:** Візуалізувати word vectors (Word2Vec, GloVe).

**Дані:**
- 50,000 слів × 300D vectors

**Підхід:**
```python
# t-SNE на word embeddings
tsne = TSNE(n_components=2, perplexity=50)
word_tsne = tsne.fit_transform(word_vectors)

# Візуалізація з підписами
plt.figure(figsize=(20, 20))
plt.scatter(word_tsne[:, 0], word_tsne[:, 1], alpha=0.3)

# Підписати цікаві слова
for i, word in enumerate(interesting_words):
    idx = word_to_idx[word]
    plt.annotate(word, word_tsne[idx], fontsize=12)

plt.show()
```

**Спостереження:**
- Семантично подібні слова кластеризуються
- Синоніми близько один до одного
- Категорії (країни, тварини) формують групи

### 3. Image Retrieval

**Задача:** Навігація по великій колекції зображень.

**Дані:**
- 100,000 зображень
- CNN features (ResNet) 2048D

**Підхід:**
```python
# 1. Extract CNN features
features = resnet_model.predict(images)  # (100K, 2048)

# 2. PCA preprocessing
pca = PCA(n_components=50)
features_pca = pca.fit_transform(features)

# 3. t-SNE на підмножині
sample_idx = np.random.choice(len(features), 5000)
features_sample = features_pca[sample_idx]

tsne = TSNE(n_components=2)
features_tsne = tsne.fit_transform(features_sample)

# 4. Інтерактивна візуалізація з thumbnail зображеннями
```

**Результат:**
- Візуальна навігація по колекції
- Подібні зображення групуються
- Виявлення дублікатів

### 4. Customer Segmentation

**Задача:** Візуалізувати сегменти клієнтів.

**Дані:**
- 50,000 клієнтів
- 100 features (поведінка, демографія)

**Підхід:**
```python
# 1. Feature engineering + scaling
X_scaled = scaler.fit_transform(customer_features)

# 2. PCA preprocessing
pca = PCA(n_components=30)
X_pca = pca.fit_transform(X_scaled)

# 3. t-SNE
tsne = TSNE(n_components=2, perplexity=50)
X_tsne = tsne.fit_transform(X_pca)

# 4. Кластеризація на t-SNE просторі
from sklearn.cluster import DBSCAN
clusters = DBSCAN(eps=0.5).fit_predict(X_tsne)

# 5. Візуалізація
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=clusters, cmap='tab10')
```

### 5. Drug Discovery

**Задача:** Візуалізація хімічних сполук.

**Дані:**
- Molecular fingerprints (1024-bit vectors)
- Тисячі молекул

**Підхід:**
```python
# t-SNE на molecular fingerprints
tsne = TSNE(n_components=2, metric='jaccard')  # Jaccard для бінарних
mol_tsne = tsne.fit_transform(fingerprints)

# Колір по активності
plt.scatter(mol_tsne[:, 0], mol_tsne[:, 1], 
           c=biological_activity, cmap='RdYlGn')
```

**Результат:**
- Схожі молекули групуються
- Виявлення нових кандидатів
- Structure-activity relationships

---

## Поширені помилки ❌

### 1. Використовувати t-SNE для downstream ML

```python
# ❌ НЕПРАВИЛЬНО
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X_train)

# Навчити classifier на t-SNE features
classifier.fit(X_tsne, y_train)

# На test:
X_test_tsne = tsne.fit_transform(X_test)  # ПОМИЛКА!
# t-SNE НЕ має transform для нових даних!

# ✅ ПРАВИЛЬНО - використовуй PCA або UMAP
pca = PCA(n_components=50)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)  # ✓ Працює
```

### 2. Інтерпретувати відстані між кластерами

```python
# ❌ "Кластер A та B близькі, тому подібні"
# ❌ "Кластер C далеко, тому дуже відмінний"

# t-SNE НЕ зберігає глобальні відстані!
# Тільки локальна структура має значення.

# ✅ ПРАВИЛЬНО
# "Кластери A та B існують"
# "Точки всередині кластера C подібні між собою"
```

### 3. Не робити PCA preprocessing

```python
# ❌ t-SNE на 784D MNIST
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X_784d)  # Дуже повільно!

# ✅ PCA спочатку
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_784d)
X_tsne = tsne.fit_transform(X_pca)  # Набагато швидше!
```

### 4. Один запуск без перевірки стабільності

```python
# ❌ Один запуск
tsne = TSNE(random_state=42)
X_tsne = tsne.fit_transform(X)
# Можливо, поганий локальний мінімум!

# ✅ Кілька запусків
best_kl = float('inf')
best_result = None

for seed in range(5):
    tsne = TSNE(random_state=seed)
    X_tsne = tsne.fit_transform(X)
    
    if tsne.kl_divergence_ < best_kl:
        best_kl = tsne.kl_divergence_
        best_result = X_tsne

# Використовуй найкращий
```

### 5. Не перевіряти різні perplexity

```python
# ❌ Тільки default perplexity=30
tsne = TSNE(perplexity=30)

# ✅ Спробувати кілька
for perp in [5, 30, 50]:
    tsne = TSNE(perplexity=perp)
    # Візуалізувати та порівняти
```

### 6. Очікувати швидкість

```python
# ❌ t-SNE на 100,000 точках
# Може працювати години/дні!

# ✅ Sampling спочатку
n_samples = min(10000, len(X))
indices = np.random.choice(len(X), n_samples)
X_sample = X[indices]

tsne = TSNE()
X_tsne = tsne.fit_transform(X_sample)
```

### 7. Порівнювати розміри кластерів

```python
# ❌ "Кластер A більший за кластер B"
# Розміри на t-SNE можуть бути оманливими!

# Perplexity впливає на щільність
# Великі кластери можуть здаватись маленькими і навпаки

# ✅ Порахуй точки
cluster_sizes = np.bincount(labels)
print(f"Actual sizes: {cluster_sizes}")
```

### 8. Не налаштовувати learning_rate

```python
# ❌ Default може не працювати для всіх даних
tsne = TSNE()

# Якщо бачиш "gradient descent did not converge"
# ✅ Налаштуй learning_rate
tsne = TSNE(learning_rate='auto')  # Або
tsne = TSNE(learning_rate=500)
```

---

## Пов'язані теми

- [[01_PCA]] — лінійна альтернатива
- [[03_UMAP]] — швидша альтернатива t-SNE
- [[04_LDA]] — supervised dimensionality reduction
- [[05_Autoencoders]] — neural network approach
- [[06_Manifold_Learning]] — інші нелінійні методи
- [[Clustering_Evaluation]] — оцінка кластерів

## Ресурси

- [Original Paper: van der Maaten & Hinton (2008)](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)
- [How to Use t-SNE Effectively (Distill)](https://distill.pub/2016/misread-tsne/)
- [Scikit-learn: t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
- [StatQuest: t-SNE](https://www.youtube.com/watch?v=NEaUSP4YerM)

---

## Ключові висновки

> t-SNE — це нелінійний алгоритм для візуалізації високорозмірних даних, який зберігає локальну структуру (близькі точки залишаються близькими) мінімізуючи KL-divergence між ймовірностями в high-dimensional та low-dimensional просторах.

**Основні принципи:**
- **Нелінійний:** знаходить складні manifolds
- **Локальна структура:** зберігає сусідство
- **Probabilistic:** базується на ймовірностях сусідства
- **t-розподіл:** вирішує crowding problem

**Алгоритм:**
1. Обчислити ймовірності в high-dim (Gaussian)
2. Обчислити ймовірності в low-dim (t-distribution)
3. Мінімізувати KL-divergence gradient descent
4. Повторювати до конвергенції

**Ключові параметри:**
- **Perplexity** (5-50, зазвичай 30) — баланс локальної/глобальної структури
- **Learning rate** ('auto' або 200) — швидкість оптимізації
- **n_iter** (мінімум 1000) — кількість ітерацій
- **init** ('pca' рекомендовано) — початкова позиція

**Коли використовувати:**
- Візуалізація + нелінійні дані + exploratory = t-SNE ✓
- Downstream ML → UMAP, PCA ✓
- Великі дані + швидкість → UMAP ✓
- Лінійні дані → PCA ✓

**Найважливіше:**
- **Тільки для візуалізації!** НЕ для ML tasks
- **PCA preprocessing** якщо d > 50 (критично!)
- **Perplexity** — найважливіший параметр
- **Недетерміністичний** — запускай кілька разів
- **Осі без значення** — інтерпретуй тільки кластери
- **Sampling** для > 10K точок
- Глобальна структура НЕ зберігається

---

#ml #unsupervised-learning #dimensionality-reduction #tsne #visualization #manifold-learning #nonlinear #exploratory-analysis
